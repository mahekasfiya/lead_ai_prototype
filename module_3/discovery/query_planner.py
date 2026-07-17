from __future__ import annotations
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from module_3.discovery.models import PlannedSearchQuery
from module_3.discovery.knowledge_base import ServiceKnowledge

logger = logging.getLogger(__name__)

class QueryPlanner:
    def __init__(self, kb: ServiceKnowledge, llm_model, use_gemini: bool = True, prompt_path: Path = None):
        self.kb = kb
        self.llm_model = llm_model
        self.use_gemini = use_gemini
        self.prompt_path = prompt_path  # not used for Gemini

    def plan_queries(self, service_id: str) -> List[PlannedSearchQuery]:
        service = self.kb.get_service(service_id)
        if not service:
            raise ValueError(f"Service {service_id} not found")

        if self.use_gemini and self.llm_model:
            prompt = self._build_prompt(service)
            try:
                response = self.llm_model.generate_content(prompt)
                content = response.text
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                data = json.loads(content)
                queries = []
                for item in data:
                    queries.append(PlannedSearchQuery(
                        service_id=item["service_id"],
                        service_name=item["service_name"],
                        query=item["query"],
                        intent_type=item["intent_type"],
                        target_country=item.get("target_country")
                    ))
                return queries
            except Exception as e:
                logger.error(f"Gemini query planning failed: {e}, using fallback")
                return self._fallback_queries(service)
        else:
            # If not using Gemini, we could use a file-based prompt (e.g., OpenAI), but for now fallback
            logger.warning("No Gemini model available, using fallback queries")
            return self._fallback_queries(service)

    def _build_prompt(self, service: Dict[str, Any]) -> str:
        return f"""
You are a search query planner for a B2B IT services lead discovery system.

Company A provides the following service:
- Service Name: {service['service_name']}
- Description: {service.get('description', '')}
- Keywords: {', '.join(service.get('search_keywords', [])[:5])}

Your task: Generate 6 to 10 search queries to find PUBLICLY AVAILABLE EVIDENCE that an organization is planning, procuring, implementing, or actively seeking external assistance for services like this.

Include queries targeting these signal types:
1. formal_procurement - RFP, RFQ, EOI, tender, procurement notice, invitation to bid
2. partner_request - "seeking implementation partner", "looking for integration partner"
3. implementation_announcement - "announces implementation", "rolls out", "deploys"
4. digital_transformation - "digital transformation", "modernization initiative", "IT transformation"
5. modernization_project - "legacy modernization", "cloud migration project", "upgrade project"
6. hiring_activity - "hiring", "looking for", "recruiting" + role (e.g., cloud architect, DevOps)

IMPORTANT: These queries must find BUYERS (organizations requesting the service), NOT providers (vendors selling the service).

Add negative keywords to exclude providers: -services -solutions -offers -provider -vendor -company that provides -we provide

Output a JSON list with keys:
- service_id: (same as input)
- service_name: (same as input)
- intent_type: one of the 6 types above
- query: the actual Google search query string
- target_country: null (or country if specified in the service)

Example:
[
  {{"service_id": "SVC-001", "service_name": "Cloud Migration", "intent_type": "formal_procurement", "query": "\"cloud migration\" (\"RFP\" OR \"tender\") -services -provider", "target_country": null}},
  {{"service_id": "SVC-001", "service_name": "Cloud Migration", "intent_type": "implementation_announcement", "query": "\"announces\" \"cloud migration\" -services -provider", "target_country": null}},
  {{"service_id": "SVC-001", "service_name": "Cloud Migration", "intent_type": "hiring_activity", "query": "\"hiring\" \"cloud architect\" -services -provider", "target_country": null}}
]
"""
    def _fallback_queries(self, service: Dict[str, Any]) -> List[PlannedSearchQuery]:
        name = service["service_name"]
        keywords = service.get("search_keywords", [])
        base_terms = [name] + keywords[:2]
        intent_types = ["formal_procurement", "implementation_announcement", "hiring_activity"]
        queries = []
        for term in base_terms:
            for intent in intent_types:
                if intent == "formal_procurement":
                    suffix = "RFP OR tender OR procurement"
                elif intent == "implementation_announcement":
                    suffix = "announces OR deploys OR implements"
                else:
                    suffix = "hiring OR recruiting OR looking for"
                q = f'"{term}" ({suffix}) -services -provider'
                queries.append(PlannedSearchQuery(
                    service_id=service["service_id"],
                    service_name=service["service_name"],
                    query=q,
                    intent_type=intent,
                    target_country=None
                ))
        return queries[:10]