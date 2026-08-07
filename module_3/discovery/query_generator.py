from __future__ import annotations

import json
import logging
from collections import Counter, deque
from datetime import date
from pathlib import Path
from typing import Any, Optional

from module_2.config import settings
from module_3.discovery.knowledge_base import ServiceKnowledge
from module_3.discovery.models import PlannedSearchQuery
from module_3.discovery.query_planner import QueryPlanner


logger = logging.getLogger(__name__)


QUERY_NEGATIVES = (
    '-template -guide -tutorial -"how to" -course -training '
    '-blog -webinar -conference -summit -seminar -agenda '
    '-award -awarded -closed -expired -archived -results '
    '-"case study" -"market report" -"market size" '
    '-"market forecast" -"industry report" -research'
)


QUERY_STRATEGIES: dict[str, dict[str, Any]] = {
    "procurement_web": {
        "source_type": "procurement",
        "platform": "web",
        "intent_type": "formal_procurement",
        "priority": 1,
        "template": (
            '{phrase} '
            '("request for proposal" OR RFP OR RFQ OR tender '
            'OR procurement OR "call for bids") '
            '{region} {year} {negatives}'
        ),
    },
    "partner_search": {
        "source_type": "general_web",
        "platform": "web",
        "intent_type": "partner_request",
        "priority": 2,
        "template": (
            '{phrase} '
            '("seeking implementation partner" OR "seeking technology partner" '
            'OR "seeking vendor" OR "inviting service providers" '
            'OR "appointing consultant") '
            '{region} {year} {negatives}'
        ),
    },
    "implementation_action": {
        "source_type": "general_web",
        "platform": "web",
        "intent_type": "implementation_announcement",
        "priority": 3,
        "template": (
            '{phrase} '
            '("plans to implement" OR "is implementing" '
            'OR "implementation project" OR "integration project" '
            'OR "migration project" OR "supplier onboarding") '
            '{region} {year} {negatives}'
        ),
    },
    "regulatory_implementation": {
        "source_type": "general_web",
        "platform": "web",
        "intent_type": "regulatory_trigger",
        "priority": 4,
        "template": (
            '{phrase} '
            '(mandate OR regulation OR compliance) '
            '("plans to implement" OR "implementation project" '
            'OR "ERP integration" OR "supplier onboarding" '
            'OR "seeking implementation partner") '
            '{region} {year} {negatives}'
        ),
    },
    "freelancer_marketplace": {
        "source_type": "marketplace",
        "platform": "freelancer",
        "intent_type": "direct_project",
        "priority": 5,
        "template": (
            'site:freelancer.com/projects {phrase} '
            '-profile -contest -freelancers'
        ),
    },
    "peopleperhour_marketplace": {
        "source_type": "marketplace",
        "platform": "peopleperhour",
        "intent_type": "direct_project",
        "priority": 6,
        "template": (
            'site:peopleperhour.com/freelance-jobs {phrase} '
            '-freelancer -profile'
        ),
    },
}


UNIVERSAL_STRATEGY_ORDER = [
    "procurement_web",
    "partner_search",
    "implementation_action",
    "regulatory_implementation",
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


BUYER_ACTION_TERMS = {
    "request for proposal",
    "request for quotation",
    "invitation to bid",
    "rfp",
    "rfq",
    "tender",
    "procurement",
    "call for bids",
    "seeking implementation partner",
    "seeking technology partner",
    "seeking vendor",
    "inviting service providers",
    "appointing consultant",
    "plans to implement",
    "is implementing",
    "implementation project",
    "integration project",
    "migration project",
    "supplier onboarding",
    "system replacement",
    "budget approved",
}


class QueryGenerator:
    """
    Generate LLM-planned or deterministic buyer-action search queries.

    The generator performs run-level service selection, fair round-robin
    distribution, validation, and deterministic fallback.
    """

    def __init__(
        self,
        knowledge_base_path: Path | None = None,
        use_llm: bool = False,
        llm_model: Any | None = None,
        planner_prompt_path: Optional[Path] = None,
    ) -> None:
        self.knowledge_base_path = (
            knowledge_base_path or settings.knowledge_base_path
        )
        self.services, self.allowed_regions = self._load_knowledge_base()
        self.use_llm = bool(use_llm and llm_model is not None)
        self.llm_model = llm_model
        self.planner: QueryPlanner | None = None

        if self.use_llm:
            try:
                knowledge = ServiceKnowledge(self.knowledge_base_path)
                self.planner = QueryPlanner(
                    kb=knowledge,
                    llm_model=llm_model,
                    use_llm=True,
                    prompt_path=planner_prompt_path,
                    allowed_regions=self.allowed_regions,
                )
                logger.info(
                    "LLM query planning enabled with %s allowed regions.",
                    len(self.allowed_regions),
                )
            except Exception:
                logger.exception(
                    "Failed to initialize the LLM query planner. "
                    "Deterministic query generation will be used."
                )
                self.use_llm = False
                self.planner = None
        else:
            logger.info(
                "LLM query planning disabled. "
                "Deterministic query generation will be used."
            )

    def _load_knowledge_base(
        self,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        with self.knowledge_base_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        services = data.get("services", [])
        regions = data.get("service_regions", [])

        if not isinstance(services, list):
            raise ValueError("Knowledge base must contain a services list.")

        allowed_regions: list[str] = []
        seen: set[str] = set()

        if isinstance(regions, list):
            for region in regions:
                value = (
                    region.get("region")
                    if isinstance(region, dict)
                    else region
                )
                text = str(value or "").strip()
                key = text.casefold()

                if text and key not in seen:
                    allowed_regions.append(text)
                    seen.add(key)

        return services, allowed_regions

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
            return f'("{terms[0]}")'

        return "(" + " OR ".join(
            f'"{term}"'
            for term in terms
        ) + ")"

    @staticmethod
    def _normalised_service_text(
        service: dict[str, Any],
    ) -> str:
        values: list[str] = [
            str(service.get("service_name", ""))
        ]

        for field_name in (
            "search_keywords",
            "technologies",
        ):
            field_values = service.get(field_name, [])

            if isinstance(field_values, list):
                values.extend(
                    str(value)
                    for value in field_values
                )

        return " ".join(values).casefold()

    def _is_marketplace_friendly(
        self,
        service: dict[str, Any],
    ) -> bool:
        text = self._normalised_service_text(service)

        if any(
            term in text
            for term in ENTERPRISE_ONLY_TERMS
        ):
            return False

        return any(
            term in text
            for term in MARKETPLACE_FRIENDLY_TERMS
        )

    def _region_expression(self) -> str:
        if not self.allowed_regions:
            return ""

        return "(" + " OR ".join(
            f'"{region}"'
            for region in self.allowed_regions
        ) + ")"

    def _strategies_for_service(
        self,
        service: dict[str, Any],
        queries_per_service: int,
    ) -> list[str]:
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

        query = strategy["template"].format(
            phrase=phrase,
            region=self._region_expression(),
            year=date.today().year,
            negatives=QUERY_NEGATIVES,
        )

        return {
            "service_id": service_id,
            "service_name": service_name,
            "query": " ".join(query.split()),
            "source_type": strategy["source_type"],
            "platform": strategy["platform"],
            "intent_type": strategy["intent_type"],
            "strategy": strategy_name,
            "strategy_order": strategy_order,
            "priority": int(strategy["priority"]),
            "target_country": None,
        }

    @staticmethod
    def _planned_query_to_dict(
        planned: PlannedSearchQuery,
    ) -> dict[str, Any]:
        if hasattr(planned, "model_dump"):
            return planned.model_dump()

        return {
            "service_id": planned.service_id,
            "service_name": planned.service_name,
            "query": planned.query,
            "source_type": planned.source_type,
            "platform": planned.platform,
            "intent_type": planned.intent_type,
            "strategy": planned.strategy,
            "strategy_order": planned.strategy_order,
            "priority": planned.priority,
            "target_country": planned.target_country,
        }

    @staticmethod
    def _parentheses_balanced(query: str) -> bool:
        depth = 0

        for char in query:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return False

        return depth == 0

    def _query_has_region_signal(
        self,
        query: str,
        target_country: str | None,
    ) -> bool:
        normalized = query.casefold()

        if "site:" in normalized:
            return True

        if target_country:
            if target_country.casefold() in {
                region.casefold()
                for region in self.allowed_regions
            }:
                return True

        return any(
            region.casefold() in normalized
            for region in self.allowed_regions
        )

    def _validate_query(
        self,
        item: dict[str, Any],
    ) -> tuple[bool, str | None]:
        query = str(item.get("query") or "").strip()
        normalized = " ".join(query.casefold().split())

        required_fields = {
            "service_id",
            "service_name",
            "source_type",
            "platform",
            "intent_type",
            "strategy",
            "strategy_order",
            "priority",
        }

        missing = [
            field
            for field in required_fields
            if item.get(field) in (None, "")
        ]
        if missing:
            return False, f"missing required fields: {missing}"

        if not query:
            return False, "empty query"

        if len(query) > 1800:
            return False, "query too long"

        if not self._parentheses_balanced(query):
            return False, "unbalanced parentheses"

        if self.allowed_regions and not self._query_has_region_signal(
            query,
            item.get("target_country"),
        ):
            return False, "no allowed-region or site restriction"

        if not any(
            term in normalized
            for term in BUYER_ACTION_TERMS
        ):
            return False, "no organisation-level buyer action"

        return True, None

    def _generate_with_llm(
        self,
        service: dict[str, Any],
        queries_per_service: int,
    ) -> list[dict[str, Any]]:
        if not self.use_llm or self.planner is None:
            return self._fallback_generate(
                service=service,
                queries_per_service=queries_per_service,
            )

        service_id = str(service.get("service_id", "")).strip()
        if not service_id:
            return []

        try:
            planner_candidates = max(
                queries_per_service,
                5,
            )
            planned = self.planner.plan_queries(
                service_id,
                desired_queries=planner_candidates,
            )

            raw_results = [
                self._planned_query_to_dict(item)
                for item in planned
            ]

            results: list[dict[str, Any]] = []
            for item in raw_results:
                valid, reason = self._validate_query(item)

                if not valid:
                    logger.warning(
                        "Discarding planned query for %s: %s | Query: %s",
                        service_id,
                        reason,
                        item.get("query"),
                    )
                    continue

                results.append(item)

            results = self._deduplicate_queries(results)

            if results:
                logger.info(
                    "LLM generated %s queries for service %s.",
                    len(results),
                    service_id,
                )
                results.sort(
                    key=lambda q: (
                        -q.get("final_query_score", 0.0),
                        q.get("rank", 999),
                    )
                )
                return results

            logger.warning(
                "LLM generated no usable queries for service %s. "
                "Using deterministic fallback.",
                service_id,
            )

        except Exception:
            logger.exception(
                "LLM query generation failed for service %s. "
                "Using deterministic fallback.",
                service_id,
            )

        return self._fallback_generate(
            service=service,
            queries_per_service=queries_per_service,
        )

    def generate(
        self,
        queries_per_service: int,
        max_total_queries: int,
        selected_service_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
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

        service_capacity = max(
            1,
            (max_total_queries + queries_per_service - 1)
            // queries_per_service,
        )
        services_for_run = eligible_services[:service_capacity]

        if len(services_for_run) < len(eligible_services):
            logger.info(
                "Query budget selected %s of %s eligible services before "
                "LLM planning.",
                len(services_for_run),
                len(eligible_services),
            )

        service_queries = {}

        for service in services_for_run:
            service_id = str(service["service_id"])
            queries = self._generate_with_llm(
                service,
                queries_per_service,
            )
            if queries:
                queries.sort(
                    key=lambda q: (
                        -q.get("final_query_score", 0.0),
                        q.get("rank", 999),
                    )
                )
                service_queries[service_id] = queries

        selected = []
        for queries in service_queries.values():
            if len(selected) >= max_total_queries:
                break
            selected.append(queries[0])

        remaining = []
        for queries in service_queries.values():
            remaining.extend(queries[1:])
            remaining.sort(
                key=lambda q: (
                    -q.get("final_query_score", 0.0),
                    q.get("rank", 999),
                )
            )
        for query in remaining:
            if len(selected) >= max_total_queries:
                break
            selected.append(query)
        results = selected

        logger.info("Selected queries:")
        for index, query in enumerate(results, start=1):
            logger.info(
                "%2d | %.3f | %s | %s",
                index,
                query.get("final_query_score", 0.0),
                query["service_id"],
                query["strategy"],
            )

        source_counts = Counter(
            item["source_type"]
            for item in results
        )
        platform_counts = Counter(
            item["platform"]
            for item in results
        )
        strategy_counts = Counter(
            item["strategy"]
            for item in results
        )
        covered_services = {
            item["service_id"]
            for item in results
        }
        requested_total = sum(
            len(queries)
            for queries in service_queries.values()
        )

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
            len(services_for_run),
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

        return results

    @staticmethod
    def _deduplicate_queries(
        queries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()

        for item in queries:
            query = str(item.get("query") or "").strip()

            if not query:
                continue

            normalized = " ".join(
                query.casefold().split()
            )

            if normalized in seen:
                continue

            seen.add(normalized)
            unique.append(item)

        return unique

    def _fallback_generate(
        self,
        service: dict[str, Any],
        queries_per_service: int,
    ) -> list[dict[str, Any]]:
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

            if not candidate:
                continue

            valid, reason = self._validate_query(candidate)

            if not valid:
                logger.warning(
                    "Discarding deterministic query for %s: %s | Query: %s",
                    service.get("service_id"),
                    reason,
                    candidate.get("query"),
                )
                continue

            queries.append(candidate)

        return queries