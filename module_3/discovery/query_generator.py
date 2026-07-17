from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional
from module_2.config import settings
from module_3.discovery.query_planner import QueryPlanner
from module_3.discovery.models import PlannedSearchQuery

BUYING_INTENT_SUFFIXES = [
    "RFP",
    "tender",
    "procurement",
    "seeking implementation partner",
    "implementation project",
]

class QueryGenerator:
    def __init__(
        self,
        knowledge_base_path: Path | None = None,
        use_llm: bool = False,
        llm_model=None,          # Gemini or OpenAI model instance
        use_gemini: bool = False,
        planner_prompt_path: Optional[Path] = None,
    ):
        self.knowledge_base_path = knowledge_base_path or settings.knowledge_base_path
        self.services = self._load_services()
        self.use_llm = use_llm
        self.use_gemini = use_gemini
        if use_llm:
            # For Gemini, we don't need a prompt path; we have it in the planner.
            # For OpenAI, we might need a prompt path, but we can also use default.
            if llm_model is None:
                raise ValueError("LLM model is required when use_llm is True")
            from module_3.discovery.knowledge_base import ServiceKnowledge
            kb = ServiceKnowledge(self.knowledge_base_path)
            self.planner = QueryPlanner(
                kb=kb,
                llm_model=llm_model,
                use_gemini=use_gemini,
                prompt_path=planner_prompt_path,  # optional, used only if not Gemini
            )
        else:
            self.planner = None

    def _load_services(self) -> list[dict[str, Any]]:
        with self.knowledge_base_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        services = data.get("services", [])
        if not isinstance(services, list):
            raise ValueError("Knowledge base must contain a services list.")
        return services

    @staticmethod
    def _first_non_empty(service: dict[str, Any], field_name: str) -> str | None:
        values = service.get(field_name, [])
        if not isinstance(values, list):
            return None
        for value in values:
            text = str(value).strip()
            if text:
                return text
        return None

    def generate(
        self,
        max_queries: int,
        selected_service_ids: list[str] | None = None,
    ) -> list[dict[str, str]]:
        selected_ids = set(selected_service_ids or [])

        if self.use_llm and self.planner:
            all_planned = []
            for service in self.services:
                service_id = service.get("service_id", "")
                if selected_ids and service_id not in selected_ids:
                    continue
                try:
                    planned = self.planner.plan_queries(service_id)
                    for p in planned:
                        all_planned.append({
                            "service_id": p.service_id,
                            "service_name": p.service_name,
                            "query": p.query,
                            "intent_type": p.intent_type,
                        })
                except Exception as e:
                    logger.warning(f"LLM planning failed for {service_id}, using fallback: {e}")
                    candidates = self._fallback_generate(service, max_queries)
                    all_planned.extend(candidates)
            # limit to max_queries
            return all_planned[:max_queries]

        # Original logic
        candidates: list[dict[str, str]] = []
        for service in self.services:
            service_id = str(service.get("service_id", "")).strip()
            if selected_ids and service_id not in selected_ids:
                continue
            service_name = str(service.get("service_name", "")).strip()
            phrase = (
                self._first_non_empty(service, "evidence_phrases")
                or self._first_non_empty(service, "search_keywords")
                or service_name
            )
            if not phrase:
                continue
            suffix = BUYING_INTENT_SUFFIXES[len(candidates) % len(BUYING_INTENT_SUFFIXES)]
            query = f'"{phrase}" {suffix}'
            candidates.append({
                "service_id": service_id,
                "service_name": service_name,
                "query": query,
            })
            if len(candidates) >= max_queries:
                break
        return candidates

    def _fallback_generate(self, service: dict, max_queries: int) -> list[dict[str, str]]:
        service_id = service.get("service_id", "")
        service_name = service.get("service_name", "")
        phrase = self._first_non_empty(service, "evidence_phrases") or service_name
        queries = []
        for suffix in BUYING_INTENT_SUFFIXES[:3]:
            queries.append({
                "service_id": service_id,
                "service_name": service_name,
                "query": f'"{phrase}" {suffix}',
            })
        return queries[:max_queries]