from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from module_3.discovery.knowledge_base import ServiceKnowledge
from module_3.discovery.models import PlannedSearchQuery


logger = logging.getLogger(__name__)


ALLOWED_INTENT_TYPES = {
    "formal_procurement",
    "partner_request",
    "implementation_announcement",
    "digital_transformation",
    "modernization_project",
    "hiring_activity",
}


class QueryPlanner:
    """
    Generate search queries using an LLM, with deterministic fallback queries.

    The supplied LLM adapter may expose one of the following interfaces:

    1. generate_content(prompt)
    2. generate(prompt)
    3. __call__(prompt)

    The response may be a plain string or an object exposing a `text` attribute.
    """

    def __init__(
        self,
        kb: ServiceKnowledge,
        llm_model: Any | None,
        use_llm: bool = True,
        prompt_path: Path | None = None,
        min_queries: int = 6,
        max_queries: int = 10,
    ) -> None:
        self.kb = kb
        self.llm_model = llm_model
        self.use_llm = use_llm
        self.prompt_path = prompt_path
        self.min_queries = max(1, min_queries)
        self.max_queries = max(self.min_queries, max_queries)

    def plan_queries(
        self,
        service_id: str,
    ) -> list[PlannedSearchQuery]:
        """
        Generate search queries for one service.

        If LLM planning is unavailable or produces an invalid response,
        deterministic fallback queries are returned.
        """

        service = self.kb.get_service(service_id)

        if not service:
            raise ValueError(f"Service {service_id} not found")

        if not self.use_llm or self.llm_model is None:
            logger.info(
                "LLM query planning is disabled for service %s. "
                "Using deterministic fallback queries.",
                service_id,
            )
            return self._fallback_queries(service)

        prompt = self._build_prompt(service)

        try:
            raw_response = self._call_model(prompt)
            response_text = self._extract_response_text(raw_response)
            response_data = self._parse_json_response(response_text)

            planned_queries = self._build_planned_queries(
                response_data=response_data,
                service=service,
            )

            if not planned_queries:
                raise ValueError(
                    "The LLM response did not contain usable search queries."
                )

            logger.info(
                "LLM query planning generated %s valid queries for %s.",
                len(planned_queries),
                service_id,
            )

            return planned_queries[: self.max_queries]

        except Exception:
            logger.exception(
                "LLM query planning failed for service %s. "
                "Using deterministic fallback queries.",
                service_id,
            )
            return self._fallback_queries(service)

    def _call_model(self, prompt: str) -> Any:
        """Call the configured LLM adapter."""

        if hasattr(self.llm_model, "generate_content"):
            return self.llm_model.generate_content(prompt)

        if hasattr(self.llm_model, "generate"):
            return self.llm_model.generate(prompt)

        if callable(self.llm_model):
            return self.llm_model(prompt)

        raise TypeError(
            "Unsupported LLM model object. Expected generate_content(), "
            "generate(), or a callable."
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text from a plain string or model response object."""

        if isinstance(response, str):
            text = response
        else:
            text = getattr(response, "text", None)

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "The LLM response did not contain usable text."
            )

        return text.strip()

    @staticmethod
    def _source_type_for_intent(intent_type: str) -> str:
        if intent_type == "formal_procurement":
            return "procurement"
        if intent_type == "hiring_activity":
            return "job_board"
        return "general_web"

    @staticmethod
    def _parse_json_response(content: str) -> list[dict[str, Any]]:
        """
        Parse a JSON list from the LLM response.

        Supports:
        - plain JSON;
        - JSON enclosed in Markdown code fences;
        - a surrounding object containing a `queries` list.
        """

        cleaned = content.strip()

        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced_match:
            cleaned = fenced_match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            list_start = cleaned.find("[")
            list_end = cleaned.rfind("]")

            if list_start == -1 or list_end == -1:
                raise ValueError(
                    "The LLM response was not valid JSON."
                )

            data = json.loads(cleaned[list_start : list_end + 1])

        if isinstance(data, dict):
            data = data.get("queries")

        if not isinstance(data, list):
            raise ValueError(
                "The query-planner response must be a JSON list "
                "or an object containing a `queries` list."
            )

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    def _build_planned_queries(
        self,
        *,
        response_data: list[dict[str, Any]],
        service: dict[str, Any],
    ) -> list[PlannedSearchQuery]:
        """
        Validate, normalize and deduplicate LLM-generated queries.
        """

        planned_queries: list[PlannedSearchQuery] = []
        seen_queries: set[str] = set()

        expected_service_id = str(service["service_id"])
        expected_service_name = str(service["service_name"])

        for item in response_data:
            query = str(item.get("query") or "").strip()
            intent_type = str(
                item.get("intent_type") or ""
            ).strip().lower()

            if not query:
                logger.warning(
                    "Ignoring planned query with empty query text."
                )
                continue

            if intent_type not in ALLOWED_INTENT_TYPES:
                logger.warning(
                    "Ignoring planned query with invalid intent type: %s",
                    intent_type,
                )
                continue

            normalized_query = " ".join(query.lower().split())

            if normalized_query in seen_queries:
                continue

            seen_queries.add(normalized_query)

            target_country = item.get("target_country")

            if target_country is not None:
                target_country = str(target_country).strip() or None

            planned_queries.append(
                PlannedSearchQuery(
                    service_id=expected_service_id,
                    service_name=expected_service_name,
                    query=query,
                    source_type=self._source_type_for_intent(intent_type),
                    platform="web",
                    intent_type=intent_type,
                    strategy=f"llm_{intent_type}",
                    strategy_order=len(planned_queries) + 1,
                    priority=len(planned_queries) + 1,
                    target_country=target_country,
                )
            )

            if len(planned_queries) >= self.max_queries:
                break

        return planned_queries

    def _build_prompt(
        self,
        service: dict[str, Any],
    ) -> str:
        service_id = str(service["service_id"])
        service_name = str(service["service_name"])
        description = str(service.get("description") or "")

        search_keywords = [
            str(keyword)
            for keyword in service.get("search_keywords", [])
            if keyword
        ]

        keywords_text = ", ".join(search_keywords[:8])

        return f"""
You are the search-query-planning agent for a B2B IT-services
lead-discovery system.

Your only responsibility is to generate search-engine queries.
Do not assess, qualify, summarize, or invent leads.

Service details:
- Service ID: {service_id}
- Service name: {service_name}
- Description: {description}
- Search keywords: {keywords_text}

Generate between {self.min_queries} and {self.max_queries} distinct
Google-compatible search queries that can discover publicly available
evidence of organizations that may require this service.

Prioritize organizations that are buyers or potential buyers, not IT
vendors advertising their own services.

Cover a useful mixture of these intent types:

1. formal_procurement
   Examples: RFP, RFQ, RFI, EOI, tender, procurement notice,
   invitation to bid, scope of work.

2. partner_request
   Examples: seeking an implementation partner, integration partner,
   technology partner, consulting partner, managed-service partner.

3. implementation_announcement
   Examples: announces implementation, rollout, deployment,
   migration, adoption or programme launch.

4. digital_transformation
   Examples: digital-transformation programme, IT transformation,
   infrastructure transformation.

5. modernization_project
   Examples: legacy modernization, cloud migration, system upgrade,
   infrastructure refresh.

6. hiring_activity
   Examples: hiring roles that may indicate an active or upcoming
   requirement for this service.

Requirements:

- Every query must be directly relevant to the supplied service.
- Use quoted phrases and OR groups where useful.
- Include buyer-intent terminology.
- Avoid generic queries such as "companies needing IT services".
- Do not return URLs or lead records.
- Do not include explanations.
- Do not claim that a company has a requirement.
- Use negative terms conservatively. Do not create invalid phrases
  such as "-company that provides".
- Return valid JSON only.

Return this exact structure:

{{
  "queries": [
    {{
      "service_id": "{service_id}",
      "service_name": "{service_name}",
      "intent_type": "formal_procurement",
      "query": "\\"{service_name}\\" (\\"RFP\\" OR \\"tender\\") -training -course",
      "target_country": null
    }}
  ]
}}

Allowed intent_type values:

- formal_procurement
- partner_request
- implementation_announcement
- digital_transformation
- modernization_project
- hiring_activity
""".strip()

    def _fallback_queries(
        self,
        service: dict[str, Any],
    ) -> list[PlannedSearchQuery]:
        """
        Generate deterministic queries when LLM planning is unavailable.
        """

        service_id = str(service["service_id"])
        service_name = str(service["service_name"])

        keywords = [
            str(keyword).strip()
            for keyword in service.get("search_keywords", [])
            if str(keyword).strip()
        ]

        base_terms = [service_name, *keywords[:2]]

        intent_patterns = {
            "formal_procurement": (
                '"RFP" OR "RFQ" OR "tender" OR "procurement notice"'
            ),
            "implementation_announcement": (
                '"announces" OR "implements" OR "deploys" OR "rolls out"'
            ),
            "hiring_activity": (
                '"hiring" OR "recruiting" OR "job opening"'
            ),
        }

        planned_queries: list[PlannedSearchQuery] = []
        seen_queries: set[str] = set()

        for term in base_terms:
            for intent_type, intent_expression in intent_patterns.items():
                query = (
                    f'"{term}" ({intent_expression}) '
                    f'-training -course'
                )

                normalized_query = " ".join(query.lower().split())

                if normalized_query in seen_queries:
                    continue

                seen_queries.add(normalized_query)

                planned_queries.append(
                    PlannedSearchQuery(
                        service_id=service_id,
                        service_name=service_name,
                        query=query,
                        intent_type=intent_type,
                        target_country=None,
                    )
                )

                if len(planned_queries) >= self.max_queries:
                    return planned_queries

        return planned_queries