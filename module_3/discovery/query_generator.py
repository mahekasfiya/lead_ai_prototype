from __future__ import annotations

import json
import logging
from collections import Counter, deque
from pathlib import Path
from typing import Any, Optional

from module_2.config import settings


logger = logging.getLogger(__name__)


QUERY_STRATEGIES: dict[str, dict[str, Any]] = {
    "procurement_web": {
        "source_type": "procurement",
        "platform": "web",
        "intent_type": "procurement",
        "priority": 1,
        "template": (
            '{phrase} '
            '("request for proposal" OR RFP OR RFQ OR tender '
            'OR "procurement notice" OR "invitation to bid") '
            '-template -guide -tutorial -"how to" -blog -jobs'
        ),
    },
    "partner_search": {
        "source_type": "general_web",
        "platform": "web",
        "intent_type": "partner_search",
        "priority": 2,
        "template": (
            '{phrase} '
            '("seeking implementation partner" OR "looking for technology partner" '
            'OR "seeking vendor" OR "inviting service providers" '
            'OR "looking for consultants") '
            '-jobs -careers -hiring -template -guide -blog'
        ),
    },
    "technology_requirement": {
        "source_type": "general_web",
        "platform": "web",
        "intent_type": "technology_requirement",
        "priority": 3,
        "template": (
            '{phrase} '
            '("implementation required" OR "migration required" '
            'OR "integration required" OR "project requirement" '
            'OR "required solution provider") '
            '-jobs -careers -hiring -template -guide -tutorial -blog'
        ),
    },
    "freelancer_marketplace": {
        "source_type": "marketplace",
        "platform": "freelancer",
        "intent_type": "direct_project",
        "priority": 4,
        "template": (
            'site:freelancer.com/projects {phrase} '
            '-profile -contest -freelancers'
        ),
    },
    "peopleperhour_marketplace": {
        "source_type": "marketplace",
        "platform": "peopleperhour",
        "intent_type": "direct_project",
        "priority": 5,
        "template": (
            'site:peopleperhour.com/freelance-jobs {phrase} '
            '-freelancer -profile'
        ),
    },
}


UNIVERSAL_STRATEGY_ORDER = [
    "procurement_web",
    "partner_search",
    "technology_requirement",
]

MARKETPLACE_STRATEGY_ORDER = [
    "freelancer_marketplace",
    "peopleperhour_marketplace",
]


MARKETPLACE_FRIENDLY_TERMS = {
    "api",
    "application development",
    "automation",
    "cloud migration",
    "database",
    "devops",
    "e-invoicing",
    "erp implementation",
    "generative ai",
    "mobile",
    "network automation",
    "software testing",
    "ui/ux",
    "ui ux",
    "web development",
}

ENTERPRISE_ONLY_TERMS = {
    "banking analytics",
    "cyberark",
    "financial crime",
    "identity security",
    "iso 27001",
    "managed it",
    "pam",
    "payment hub",
    "privileged access",
    "soc as a service",
    "temenos",
    "t24",
}


class QueryGenerator:
    """Generate deterministic, source-aware buying-signal queries.

    ``queries_per_service`` controls how many query strategies are requested
    for each eligible service. ``max_total_queries`` is a hard safety limit
    across the complete run.
    """

    def __init__(
        self,
        knowledge_base_path: Path | None = None,
        use_llm: bool = False,
        llm_model=None,
        use_gemini: bool = False,
        planner_prompt_path: Optional[Path] = None,
    ):
        self.knowledge_base_path = (
            knowledge_base_path or settings.knowledge_base_path
        )
        self.services = self._load_services()

        self.use_llm = False
        self.use_gemini = False
        self.planner = None

        if use_llm or use_gemini or llm_model is not None:
            logger.info(
                "QueryGenerator is running in deterministic rule-only mode; "
                "LLM query planning is disabled."
            )

    def _load_services(self) -> list[dict[str, Any]]:
        with self.knowledge_base_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        services = data.get("services", [])

        if not isinstance(services, list):
            raise ValueError("Knowledge base must contain a services list.")

        return services

    @staticmethod
    def _list_values(
        service: dict[str, Any],
        field_name: str,
    ) -> list[str]:
        values = service.get(field_name, [])

        if not isinstance(values, list):
            return []

        cleaned: list[str] = []
        seen: set[str] = set()

        for value in values:
            text = str(value).strip()
            key = text.casefold()

            if text and key not in seen:
                cleaned.append(text)
                seen.add(key)

        return cleaned

    def _service_search_terms(
        self,
        service: dict[str, Any],
        max_terms: int = 3,
    ) -> list[str]:
        """Select concise, distinctive terms for web search."""

        service_name = str(service.get("service_name", "")).strip()

        candidates = [
            service_name,
            *self._list_values(service, "search_keywords"),
            *self._list_values(service, "evidence_phrases"),
            *self._list_values(service, "technologies"),
        ]

        terms: list[str] = []
        seen: set[str] = set()

        for candidate in candidates:
            text = str(candidate).strip()
            key = text.casefold()

            if not text or len(text) > 80 or key in seen:
                continue

            terms.append(text)
            seen.add(key)

            if len(terms) >= max_terms:
                break

        return terms

    def _service_search_phrase(
        self,
        service: dict[str, Any],
    ) -> str | None:
        terms = self._service_search_terms(service)

        if not terms:
            return None

        if len(terms) == 1:
            return f'"{terms[0]}"'

        return "(" + " OR ".join(f'"{term}"' for term in terms) + ")"

    @staticmethod
    def _normalised_service_text(
        service: dict[str, Any],
    ) -> str:
        values: list[str] = [str(service.get("service_name", ""))]

        for field_name in ("search_keywords", "technologies"):
            field_values = service.get(field_name, [])
            if isinstance(field_values, list):
                values.extend(str(value) for value in field_values)

        return " ".join(values).casefold()

    def _is_marketplace_friendly(
        self,
        service: dict[str, Any],
    ) -> bool:
        text = self._normalised_service_text(service)

        if any(term in text for term in ENTERPRISE_ONLY_TERMS):
            return False

        return any(term in text for term in MARKETPLACE_FRIENDLY_TERMS)

    def _strategies_for_service(
        self,
        service: dict[str, Any],
        queries_per_service: int,
    ) -> list[str]:
        """Return ordered strategies for one service."""

        strategies = list(UNIVERSAL_STRATEGY_ORDER)

        if self._is_marketplace_friendly(service):
            strategies.extend(MARKETPLACE_STRATEGY_ORDER)

        return strategies[:queries_per_service]

    def _build_query(
        self,
        service: dict[str, Any],
        strategy_name: str,
        strategy_order: int,
    ) -> dict[str, Any] | None:
        service_id = str(service.get("service_id", "")).strip()
        service_name = str(service.get("service_name", "")).strip()
        phrase = self._service_search_phrase(service)
        strategy = QUERY_STRATEGIES.get(strategy_name)

        if not service_id or not phrase or not strategy:
            return None

        return {
            "service_id": service_id,
            "service_name": service_name,
            "query": strategy["template"].format(phrase=phrase),
            "source_type": strategy["source_type"],
            "platform": strategy["platform"],
            "intent_type": strategy["intent_type"],
            "strategy": strategy_name,
            "strategy_order": strategy_order,
            "priority": int(strategy["priority"]),
        }

    def generate(
        self,
        queries_per_service: int,
        max_total_queries: int,
        selected_service_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate queries fairly across eligible services.

        Queries are emitted round-robin so every eligible service receives its
        first strategy before any service receives its second strategy.
        """

        if queries_per_service <= 0:
            logger.warning(
                "Query generation skipped because queries_per_service=%s.",
                queries_per_service,
            )
            return []

        if max_total_queries <= 0:
            logger.warning(
                "Query generation skipped because max_total_queries=%s.",
                max_total_queries,
            )
            return []

        selected_ids = {
            str(value).strip()
            for value in (selected_service_ids or [])
            if str(value).strip()
        }

        eligible_services: list[dict[str, Any]] = []

        for service in self.services:
            service_id = str(service.get("service_id", "")).strip()

            if not service_id:
                continue

            if selected_ids and service_id not in selected_ids:
                continue

            if not self._service_search_phrase(service):
                continue

            eligible_services.append(service)

        if not eligible_services:
            logger.info("Generated 0 source-aware buying-signal queries.")
            return []

        service_queues: dict[str, deque[dict[str, Any]]] = {}

        for service in eligible_services:
            service_id = str(service.get("service_id", "")).strip()
            queue: deque[dict[str, Any]] = deque()

            strategy_names = self._strategies_for_service(
                service=service,
                queries_per_service=queries_per_service,
            )

            for strategy_order, strategy_name in enumerate(
                strategy_names,
                start=1,
            ):
                candidate = self._build_query(
                    service=service,
                    strategy_name=strategy_name,
                    strategy_order=strategy_order,
                )

                if candidate:
                    queue.append(candidate)

            if queue:
                service_queues[service_id] = queue

        requested_total = len(service_queues) * queries_per_service
        effective_limit = min(requested_total, max_total_queries)

        results: list[dict[str, Any]] = []

        while len(results) < effective_limit:
            added_this_round = False

            for service in eligible_services:
                service_id = str(service.get("service_id", "")).strip()
                queue = service_queues.get(service_id)

                if not queue:
                    continue

                results.append(queue.popleft())
                added_this_round = True

                if len(results) >= effective_limit:
                    break

            if not added_this_round:
                break

        source_counts = Counter(item["source_type"] for item in results)
        platform_counts = Counter(item["platform"] for item in results)
        strategy_counts = Counter(item["strategy"] for item in results)
        covered_services = {item["service_id"] for item in results}

        logger.info(
            "Query generation complete. Eligible services: %s | "
            "Queries per service: %s | Requested total: %s | "
            "Maximum total: %s | Generated: %s | "
            "Services covered: %s/%s | Source types: %s | "
            "Platforms: %s | Strategies: %s",
            len(eligible_services),
            queries_per_service,
            requested_total,
            max_total_queries,
            len(results),
            len(covered_services),
            len(eligible_services),
            dict(source_counts),
            dict(platform_counts),
            dict(strategy_counts),
        )

        if requested_total > max_total_queries:
            logger.info(
                "The maximum total query cap reduced the run from %s "
                "requested queries to %s generated queries.",
                requested_total,
                len(results),
            )

        for item in results:
            logger.debug(
                "Generated query | Service: %s (%s) | "
                "Strategy: %s (#%s) | Source: %s | Platform: %s | "
                "Intent: %s | Query: %s",
                item["service_name"],
                item["service_id"],
                item["strategy"],
                item["strategy_order"],
                item["source_type"],
                item["platform"],
                item["intent_type"],
                item["query"],
            )

        return results

    def _fallback_generate(
        self,
        service: dict[str, Any],
        queries_per_service: int,
    ) -> list[dict[str, Any]]:
        """Generate queries for one service using the same strategies."""

        if queries_per_service <= 0:
            return []

        queries: list[dict[str, Any]] = []
        strategy_names = self._strategies_for_service(
            service=service,
            queries_per_service=queries_per_service,
        )

        for strategy_order, strategy_name in enumerate(
            strategy_names,
            start=1,
        ):
            candidate = self._build_query(
                service=service,
                strategy_name=strategy_name,
                strategy_order=strategy_order,
            )

            if candidate:
                queries.append(candidate)

        return queries