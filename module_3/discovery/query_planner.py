from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from module_3.discovery.knowledge_base import ServiceKnowledge
from module_3.discovery.models import PlannedSearchQuery


logger = logging.getLogger(__name__)


ALLOWED_INTENT_TYPES = {
    "formal_procurement",
    "official_procurement",
    "partner_request",
    "implementation_announcement",
    "digital_transformation",
    "modernization_project",
    "regulatory_trigger",
    "technology_requirement",
    "industry_requirement",
    "hiring_activity",
}


BUYER_ACTION_TERMS = {
    "request for proposal",
    "request for quotation",
    "invitation to bid",
    "expression of interest",
    "rfp",
    "rfq",
    "tender",
    "procurement",
    "call for bids",
    "call for proposals",
    "seeking implementation partner",
    "seeking technology partner",
    "seeking vendor",
    "inviting service providers",
    "appointing consultant",
    "appointing service provider",
    "plans to implement",
    "planning to implement",
    "is implementing",
    "implementation project",
    "implementation programme",
    "implementation program",
    "integration project",
    "migration project",
    "migration programme",
    "migration program",
    "modernization project",
    "modernisation programme",
    "modernization programme",
    "system replacement",
    "platform replacement",
    "supplier onboarding",
    "vendor onboarding",
    "budget approved",
    "approved budget",
    "project launch",
    "programme launch",
    "program launch",
}

REGULATORY_SIGNAL_TERMS = {
    "sama",
    "nca",
    "nesa",
    "ncema",
    "zatca",
    "pdpl",
    "mandate",
    "regulation",
    "regulatory",
    "compliance",
    "remediation",
    "audit requirement",
    "rollout",
}

INDUSTRY_QUERY_ALIASES = {
    "healthcare": {
        "healthcare",
        "hospital",
        "hospitals",
        "clinic",
        "healthcare provider",
    },
    "healthtech": {
        "healthtech",
        "hospital",
        "healthcare provider",
        "healthcare company",
    },
    "banking": {
        "bank",
        "banks",
        "banking",
        "financial institution",
    },
    "financial services": {
        "financial services",
        "financial institution",
        "bank",
        "fintech",
        "insurance company",
    },
    "finance": {
        "finance",
        "financial institution",
        "bank",
        "fintech",
    },
    "fintech": {
        "fintech",
        "financial institution",
        "bank",
        "payment company",
    },
    "education": {
        "education",
        "school",
        "schools",
        "university",
        "universities",
        "educational institution",
    },
    "manufacturing": {
        "manufacturing",
        "manufacturer",
        "manufacturers",
        "factory",
        "industrial company",
    },
    "automation": {
        "automation company",
        "industrial automation",
        "automation",
        "industrial company",
    },
    "oil and gas": {
        "oil and gas",
        "oil and gas company",
        "oil company",
        "gas company",
        "petroleum company",
        "energy company",
    },
    "oil & gas": {
        "oil and gas",
        "oil and gas company",
        "oil company",
        "gas company",
        "petroleum company",
        "energy company",
    },
    "petroleum": {
        "petroleum",
        "petroleum company",
        "oil company",
        "energy company",
    },
    "real estate": {
        "real estate",
        "real estate developer",
        "property developer",
        "property company",
    },
    "government": {
        "government",
        "government authority",
        "ministry",
        "municipality",
        "public sector",
    },
    "retail": {
        "retail",
        "retailer",
        "retail company",
        "supermarket",
        "ecommerce company",
    },
    "logistics": {
        "logistics",
        "logistics company",
        "transport company",
        "shipping company",
        "supply chain company",
    },
    "aviation": {
        "aviation",
        "airline",
        "airport",
        "aviation company",
    },
    "insurance": {
        "insurance",
        "insurance company",
        "insurer",
    },
}

QUERY_NEGATIVES = (
    '-template -training -course -jobs '
    '-"market report" -"market size" -"case study"'
)

class QueryPlanner:
    """
    Generate region-aware, buyer-action-focused search queries with an LLM.

    Deterministic fallback queries are returned whenever LLM planning is
    disabled, fails, or produces no acceptable output.
    """

    def __init__(
        self,
        kb: ServiceKnowledge,
        llm_model: Any | None,
        use_llm: bool = True,
        prompt_path: Path | None = None,
        min_queries: int = 3,
        max_queries: int = 10,
        allowed_regions: list[str] | None = None,
    ) -> None:
        self.kb = kb
        self.llm_model = llm_model
        self.use_llm = use_llm
        self.prompt_path = prompt_path
        self.min_queries = max(1, min_queries)
        self.max_queries = max(self.min_queries, max_queries)
        self.allowed_regions = self._clean_values(allowed_regions or [])

    @staticmethod
    def _clean_values(values: list[Any]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for value in values:
            text = str(value or "").strip()
            key = text.casefold()

            if text and key not in seen:
                cleaned.append(text)
                seen.add(key)

        return cleaned

    def plan_queries(
        self,
        service_id: str,
        desired_queries: int | None = None,
    ) -> list[PlannedSearchQuery]:
        service = self.kb.get_service(service_id)

        if not service:
            raise ValueError(f"Service {service_id} not found")

        requested_count = max(
            1,
            min(desired_queries or self.max_queries, self.max_queries),
        )

        if not self.use_llm or self.llm_model is None:
            logger.info(
                "LLM query planning is disabled for service %s. "
                "Using deterministic fallback queries.",
                service_id,
            )
            return self._fallback_queries(service, requested_count)

        prompt = self._build_prompt(service, requested_count)

        try:
            raw_response = self._call_model(prompt)
            response_text = self._extract_response_text(raw_response)
            response_data = self._parse_json_response(response_text)

            planned_queries = self._build_planned_queries(
                response_data=response_data,
                service=service,
                limit=requested_count,
            )

            if not planned_queries:
                raise ValueError(
                    "The LLM response did not contain acceptable search queries."
                )

            logger.info(
                "LLM query planning generated %s valid queries for %s.",
                len(planned_queries),
                service_id,
            )
            for query in sorted(
                planned_queries,
                key=lambda q: q.final_query_score,
                reverse=True,
            ):
                logger.info(
                    "Query Rank %s | Score %.3f | Strategy=%s | %s",
                    query.rank,
                    query.final_query_score,
                    query.strategy,
                    query.query,
                )
            return planned_queries

        except Exception:
            logger.exception(
                "LLM query planning failed for service %s. "
                "Using deterministic fallback queries.",
                service_id,
            )
            return self._fallback_queries(service, requested_count)

    def _call_model(self, prompt: str) -> Any:
        if hasattr(self.llm_model, "generate_content"):
            return self.llm_model.generate_content(
                prompt,
                max_tokens=2500,
            )

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
        if intent_type in {
            "formal_procurement",
            "official_procurement",
        }:
            return "procurement"

        if intent_type == "hiring_activity":
            return "job_board"

        return "general_web"

    @staticmethod
    def _parse_json_response(content: str) -> list[dict[str, Any]]:
        cleaned = content.strip()

        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced_match:
            cleaned = fenced_match.group(1).strip()

        candidates = [
            cleaned,
            cleaned.replace("“", '"').replace("”", '"').replace("’", "'"),
            re.sub(r",\s*([}\]])", r"\1", cleaned),
        ]

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    data = data.get("queries")
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
            except json.JSONDecodeError:
                continue

        list_start = cleaned.find("[")
        list_end = cleaned.rfind("]")
        if list_start == -1 or list_end == -1 or list_end <= list_start:
            raise ValueError("The LLM response was not valid JSON.")

        repaired = re.sub(
            r",\s*([}\]])",
            r"\1",
            cleaned[list_start:list_end + 1],
        )

        data = json.loads(repaired)
        if not isinstance(data, list):
            raise ValueError("The query planner output must be a JSON list.")

        return [item for item in data if isinstance(item, dict)]

    def _service_terms(
        self,
        service: dict[str, Any],
    ) -> list[str]:
        values = [
            service.get("service_name"),
            *(service.get("search_keywords") or []),
            *(service.get("technologies") or []),
            *(service.get("evidence_phrases") or []),
        ]
        return self._clean_values(values)[:16]

    def _industry_terms(
            self,
            service: dict[str, Any],
    ) -> list[str]:
        return self._clean_values(
            service.get("industries") or []
        )

    def _query_has_industry_signal(
            self,
            *,
            query: str,
            service: dict[str, Any],
    ) -> bool:
        normalized = " ".join(
            query.casefold().split()
        )
        industries = self._industry_terms(service)
        # If the service has no industries configured, do not reject the query.
        if not industries:
            return True
        for industry in industries:
            normalized_industry = industry.casefold().strip()
            aliases = INDUSTRY_QUERY_ALIASES.get(
                normalized_industry,
                {normalized_industry},
            )
            if any(
                alias.casefold() in normalized
                for alias in aliases
            ):
                return True
        return False

    def _has_region_signal(
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

    def _query_is_acceptable(
            self,
            *,
            query: str,
            intent_type: str,
            service: dict[str, Any],
            target_country: str | None,
    ) -> tuple[bool, str | None]:
        normalized = " ".join(
            query.casefold().split()
        )
        # --------------------------------
        # 1. Basic query validation
        # --------------------------------
        if not query:
            return False, "query is empty"
        if len(query) > 1200:
            return False, "query is too long"
        if intent_type not in ALLOWED_INTENT_TYPES:
            return False, "intent type is invalid"
        if not self._parentheses_balanced(query):
            return False, "query contains unbalanced parentheses"
        # --------------------------------
        # 2. Service relevance
        # --------------------------------
        service_terms = self._service_terms(service)
        if service_terms and not any(
            term.casefold() in normalized
            for term in service_terms
            ):
            return False, (
                "query contains no service-specific term"
            )
        # --------------------------------
        # 3. Industry / buyer targeting
        # --------------------------------
        if not self._query_has_industry_signal(
            query=query,
            service=service,
        ):
            return False, (
                "query contains no supported industry "
                "or buyer-organisation signal"
            )
        # --------------------------------
        # 4. Region validation
        # --------------------------------
        if (
            self.allowed_regions
            and not self._has_region_signal(
                query,
                target_country,
            )
        ):
            return False, (
                "query has no allowed-region "
                "or official-domain signal"
            )
        # --------------------------------
        # 5. Buyer / opportunity signal
        # --------------------------------
        has_buyer_action = any(
            action in normalized
            for action in BUYER_ACTION_TERMS
        )
        has_regulatory_signal = (
            intent_type == "regulatory_trigger"
            and any(
                signal in normalized
                for signal in REGULATORY_SIGNAL_TERMS
            )
        )
        if not has_buyer_action and not has_regulatory_signal:
            return False, (
                "query has no buyer action or "
                "regulatory opportunity signal"
            )
        # --------------------------------
        # Everything passed
        # --------------------------------
        return True, None

    def _build_planned_queries(
        self,
        *,
        response_data: list[dict[str, Any]],
        service: dict[str, Any],
        limit: int,
    ) -> list[PlannedSearchQuery]:
        planned_queries: list[PlannedSearchQuery] = []
        seen_queries: set[str] = set()
        seen_strategies: set[str] = set()

        expected_service_id = str(service["service_id"])
        expected_service_name = str(service["service_name"])

        for item in response_data:
            query = str(item.get("query") or "").strip()
            intent_type = str(
                item.get("intent_type") or ""
            ).strip().casefold()
            strategy_name = str(
                item.get("strategy")
                or intent_type
                or "general"
            ).strip().casefold()
            target_country = item.get("target_country")

            if target_country is not None:
                target_country = str(target_country).strip() or None

            acceptable, reason = self._query_is_acceptable(
                query=query,
                intent_type=intent_type,
                service=service,
                target_country=target_country,
            )
            if not acceptable:
                logger.warning(
                    "Ignoring planned query for %s: %s | Query: %s",
                    expected_service_id,
                    reason,
                    query,
                )
                continue

            normalized_query = " ".join(query.casefold().split())

            if normalized_query in seen_queries:
                continue

            if (
                strategy_name in seen_strategies
                and len(seen_strategies) < min(3, limit)
            ):
                logger.warning(
                    "Ignoring duplicate query strategy '%s' for %s.",
                    strategy_name,
                    expected_service_id,
                )
                continue

            seen_queries.add(normalized_query)
            seen_strategies.add(strategy_name)

            buyer_specificity = float(
                item.get("buyer_specificity_score", 0.0)
            )
            current_opportunity = float(
                item.get("current_opportunity_score", 0.0)
            )
            service_relevance = float(
                item.get("service_relevance_score", 0.0)
            )
            regional_precision = float(
                item.get("regional_precision_score", 0.0)
            )
            source_quality = float(
                item.get("source_quality_score", 0.0)
            )
            noise_risk = float(
                item.get("noise_risk_score", 1.0)
            )
            final_score = round(
                buyer_specificity * 0.25
                + current_opportunity * 0.20
                + service_relevance * 0.20
                + regional_precision * 0.15
                + source_quality * 0.10
                + (1.0 - noise_risk) * 0.10,
                4,
            )
            planned_queries.append(
                PlannedSearchQuery(
                    service_id=expected_service_id,
                    service_name=expected_service_name,
                    query=query,
                    source_type=self._source_type_for_intent(intent_type),
                    platform="web",
                    intent_type=intent_type,
                    strategy=f"llm_{strategy_name}",
                    strategy_order=len(planned_queries) + 1,
                    priority=int(
                        item.get("priority")
                        or len(planned_queries) + 1
                    ),
                    target_country=target_country,
                    rank=int(item.get("rank", 999)),
                    buyer_specificity_score=buyer_specificity,
                    current_opportunity_score=current_opportunity,
                    service_relevance_score=service_relevance,
                    regional_precision_score=regional_precision,
                    source_quality_score=source_quality,
                    noise_risk_score=noise_risk,
                    final_query_score=final_score,
                    ranking_reason=item.get("ranking_reason"),
                )
            )

            if len(planned_queries) >= limit:
                break

        return planned_queries

    def _build_prompt(
        self,
        service: dict[str, Any],
        requested_count: int,
    ) -> str:
        service_id = str(service["service_id"])
        service_name = str(service["service_name"])
        description = str(service.get("description") or "")

        search_keywords = self._clean_values(
            service.get("search_keywords") or []
        )[:12]
        technologies = self._clean_values(
            service.get("technologies") or []
        )[:10]
        evidence_phrases = self._clean_values(
            service.get("evidence_phrases") or []
        )[:10]
        industries = self._clean_values(
            service.get("industries") or []
        )[:8]

        regions_text = (
            "\n".join(f"- {region}" for region in self.allowed_regions)
            if self.allowed_regions
            else "- No region restrictions supplied"
        )

        return f"""
You are the search-query-planning agent for a B2B enterprise IT-services
lead-discovery system.

Your task is to generate compact Google-compatible search queries that find
potential buyer opportunities.

Do not browse URLs.
Do not qualify leads.
Do not invent companies.
Do not return explanations or Markdown.

Current date: {date.today().isoformat()}
Current year: {date.today().year}

Allowed buyer regions:
{regions_text}

Service:
- Service ID: {service_id}
- Service name: {service_name}
- Description: {description}
- Search keywords: {", ".join(search_keywords)}
- Technologies: {", ".join(technologies)}
- Evidence phrases: {", ".join(evidence_phrases)}
- Relevant industries: {", ".join(industries)}

Generate exactly {requested_count} distinct search queries.

The objective is to find specific organisations, buyers, procurement notices,
implementation programmes, modernization projects, regulatory-driven projects,
partner searches, and other commercially meaningful technology requirements.

The downstream pipeline already performs:
- source validation;
- listing-page detection;
- deadline checks;
- requirement qualification;
- contradiction detection;
- similarity matching;
- final Claude lead validation.

Therefore, optimize for useful discovery recall.
Do not try to prove that a page is definitely a valid lead inside the query.

INDUSTRY TARGETING

Every query must target exactly one relevant industry from the supplied
Relevant industries list.

The industry must appear explicitly in the query, either as the original
industry name or as a realistic buyer-organisation term derived from it.

Prefer concrete buyer-organisation language instead of abstract industry
labels.

Examples:

Healthcare:
- hospital
- healthcare provider
- clinic

Banking / Financial Services:
- bank
- financial institution
- insurance company
- fintech

Education:
- school
- university
- educational institution

Manufacturing:
- manufacturer
- manufacturing company
- factory
- industrial company

Oil and Gas / Petroleum:
- oil and gas company
- petroleum company
- energy company

Real Estate:
- real estate developer
- property developer
- property company

Government:
- government authority
- ministry
- municipality
- public sector organisation

Retail:
- retailer
- retail company
- supermarket
- ecommerce company

Logistics:
- logistics company
- shipping company
- transport company
- supply-chain organisation

Aviation:
- airline
- airport
- aviation company

Use only industries supported by the supplied service record.
Do not invent unrelated industries.

When several relevant industries are available, diversify the queries across
different industries where possible.

The purpose of industry targeting is to find likely BUYERS.

For example:

Better:

("e-invoicing" OR "invoice automation")
(manufacturer OR "manufacturing company")
("ERP integration" OR "implementation project" OR procurement)
("Saudi Arabia" OR KSA)

Less useful:

("e-invoicing" OR "invoice automation")
(manufacturing)
("Saudi Arabia" OR KSA)

QUERY DESIGN RULES

Each query should normally contain four positive components:

1. SERVICE GROUP
   Two or three service, capability, technology, regulation, or business
   problem terms.

2. INDUSTRY / BUYER GROUP
   Exactly one relevant industry expressed using one or two realistic buyer
   organisation terms.

3. OPPORTUNITY-SIGNAL GROUP
   One compact procurement, implementation, modernization, partner,
   regulatory, project, rollout, migration, integration, or transformation
   signal.

4. COUNTRY GROUP
   Exactly one target country, optionally with one common abbreviation.

Use no more than five negative terms.

Keep queries compact.
Prefer recall over perfect precision.

Do not try to encode every possible requirement into a single query.

BOOLEAN RULES

- Put every OR group inside parentheses.
- Use no more than three alternatives in one OR group.
- Use no more than four positive Boolean groups:
  service, industry, opportunity signal, and country.
- Do not use nested Boolean expressions.
- Do not repeat the same country in several groups.
- Do not combine all allowed regions in one query.
- Select exactly one target country per query.
- Different generated queries may target different allowed countries.
- Different queries should use different industries or strategies where
  possible.

Preferred structure:

("service term 1" OR "service term 2")
("buyer organisation 1" OR "buyer organisation 2")
("opportunity signal 1" OR "opportunity signal 2" OR "opportunity signal 3")
("Country Name" OR abbreviation)
-negative1 -negative2 -negative3

Do not generate very long queries containing many service terms, many
industries, many buyer actions, all regions, or a large exclusion list.

STRATEGY RULES

When requested_count is 1:
- Generate the query most likely to find a specific buyer opportunity.

When requested_count is 2:
- Include one procurement/official procurement/partner query.
- Include one implementation, modernization, regulatory, technology,
  industry, or transformation query.
- Prefer different relevant industries when possible.

When requested_count is 3 or more:
- Include at least one procurement query.
- Include at least one implementation, partner, or modernization query.
- Use remaining queries for regulatory, technology, industry, or
  transformation strategies.
- Diversify industries and countries when useful.

PROCUREMENT QUERIES

Use compact procurement signals such as:

- RFP
- RFQ
- tender
- procurement
- request for proposal
- invitation to bid
- call for bids
- expression of interest

Example:

("cloud migration" OR "data center migration")
(bank OR "financial institution")
(RFP OR tender OR procurement)
("Saudi Arabia" OR KSA)
-template -training -jobs -"market report"

IMPLEMENTATION AND MODERNIZATION QUERIES

Use organisation-level project language such as:

- implementation project
- migration project
- system replacement
- platform upgrade
- rollout
- integration project
- modernization programme
- plans to implement
- is implementing
- digital transformation
- seeking implementation partner

Example:

("cloud migration" OR "workload migration")
(hospital OR "healthcare provider")
("migration project" OR "modernization programme" OR "is implementing")
("United Arab Emirates" OR UAE)
-jobs -training -"market report"

REGULATORY QUERIES

A named regulation, authority, mandate, compliance framework, implementation
wave, remediation requirement, or rollout can be a useful discovery signal.

Do not force a regulatory query to contain explicit vendor-request language.

Pair the regulation with an industry likely to be affected.

Example:

("penetration testing" OR VAPT)
(bank OR "financial institution")
(SAMA OR NCA OR "security compliance")
("Saudi Arabia" OR KSA)
-jobs -training -"market report"

Example:

("e-invoicing" OR "electronic invoicing")
(manufacturer OR "manufacturing company")
(ZATCA OR "ERP integration" OR "implementation project")
("Saudi Arabia" OR KSA)
-training -jobs -"market report"

INDUSTRY-SPECIFIC EXAMPLES

Cloud migration:
- hospital + data-center migration
- bank + cloud-modernization programme
- manufacturer + workload migration
- government authority + cloud-transformation tender

E-invoicing:
- manufacturer + ERP integration
- hospital + invoice automation
- retailer + e-invoicing rollout
- government authority + digital-invoicing procurement

VAPT:
- bank + penetration-testing tender
- hospital + application-security assessment
- government authority + vulnerability-assessment procurement
- oil-and-gas company + OT security testing

SOC / Managed Security:
- bank + managed SOC procurement
- hospital + SOC outsourcing
- government authority + security operations tender
- manufacturer + managed detection and response project

Temenos:
- bank + Temenos upgrade
- financial institution + core banking modernization
- bank + T24 migration
- bank + Temenos support procurement

PAM / CyberArk:
- bank + PAM implementation
- government authority + privileged-access project
- hospital + identity-security modernization
- oil-and-gas company + privileged-access deployment

NEGATIVE TERMS

Use only a small relevant subset of:

- -template
- -training
- -course
- -jobs
- -"market report"
- -"market size"
- -"case study"

Do not automatically include every negative term.

Do not globally exclude:

- news
- research
- results
- award
- conference
- summit
- seminar
- agenda
- thought leadership

Those words can appear on legitimate buyer or procurement pages.

QUERY RANKING

Rank every generated query from highest to lowest expected lead-discovery
value.

Rank 1 must be the query most likely to discover a specific, current and
actionable buyer opportunity.

Score every query from 0.0 to 1.0 for:

buyer_specificity_score:
Likelihood that the search results identify a specific buyer organisation.

current_opportunity_score:
Likelihood that the search results describe an active or upcoming requirement.

service_relevance_score:
Likelihood that the search results directly match the supplied service.

regional_precision_score:
Likelihood that the results belong to the selected supported country.

source_quality_score:
Likelihood that the query returns an official buyer page, procurement notice,
specific tender, credible organisation announcement, or buyer-owned source.

noise_risk_score:
Likelihood of returning vendor marketing, generic blogs, directories,
aggregators, market reports, generic regulation explainers, or unrelated
content.

A higher noise_risk_score is worse.

Prefer queries that combine:

specific service
+ concrete buyer industry
+ strong opportunity signal
+ supported country

Do not automatically rank procurement first if another query is more likely to
find a specific buyer.

Keep ranking_reason to one short sentence.

QUALITY RULES

Every query must:

- contain at least one supplied service-related term;
- explicitly contain one supplied relevant industry or realistic
  buyer-organisation type derived from that industry;
- contain one procurement, project, implementation, modernization,
  regulatory, migration, rollout, integration, transformation, or partner
  signal;
- contain exactly one allowed target country;
- target likely buyer organisations rather than service providers;
- remain compact;
- be materially different from the other generated queries.

Do not generate:

- generic informational searches;
- generic technology searches;
- market research searches;
- market sizing searches;
- broad global queries;
- near-duplicate queries;
- queries containing multiple unrelated industries;
- queries containing all allowed countries;
- queries with more than four positive Boolean groups;
- queries with more than five negative filters;
- URLs;
- lead records;
- prose outside the JSON.

Return valid JSON only using this exact structure:

{{
  "queries": [
    {{
      "service_id": "{service_id}",
      "service_name": "{service_name}",
      "intent_type": "formal_procurement",
      "strategy": "regional_industry_procurement",
      "priority": 1,
      "rank": 1,
      "buyer_specificity_score": 0.90,
      "current_opportunity_score": 0.85,
      "service_relevance_score": 0.95,
      "regional_precision_score": 0.95,
      "source_quality_score": 0.85,
      "noise_risk_score": 0.15,
      "ranking_reason": "Targets a specific buyer industry with strong procurement intent in a supported region.",
      "query": "(\\\"service term 1\\\" OR \\\"service term 2\\\") (\\\"hospital\\\" OR \\\"healthcare provider\\\") (RFP OR tender OR procurement) (\\\"Country Name\\\" OR abbreviation) -template -training -jobs -\\\"market report\\\"",
      "target_country": "Country Name"
    }}
  ]
}}

Allowed intent_type values:

- formal_procurement
- official_procurement
- partner_request
- implementation_announcement
- digital_transformation
- modernization_project
- regulatory_trigger
- technology_requirement
- industry_requirement
- hiring_activity
""".strip()

    def _region_expression(self) -> str:
        if not self.allowed_regions:
            return ""

        return "(" + " OR ".join(
            f'"{region}"'
            for region in self.allowed_regions
        ) + ")"

    def _fallback_queries(
        self,
        service: dict[str, Any],
        requested_count: int,
    ) -> list[PlannedSearchQuery]:
        service_id = str(service["service_id"])
        service_name = str(service["service_name"])

        keywords = self._clean_values(
            service.get("search_keywords") or []
        )
        technologies = self._clean_values(
            service.get("technologies") or []
        )
        industries = self._clean_values(
            service.get("industries") or []
        )

        terms = self._clean_values(
            [service_name, *keywords[:2], *technologies[:1]]
        )
        primary = terms[0] if terms else service_name
        secondary = terms[1] if len(terms) > 1 else primary
        industry = industries[0] if industries else "enterprise"

        region_expression = self._region_expression()
        year = date.today().year

        patterns = [
            (
                "formal_procurement",
                "deterministic_regional_procurement",
                f'("{primary}" OR "{secondary}") '
                f'(RFP OR RFQ OR tender OR procurement OR "call for bids") '
                f'{region_expression} {year} {QUERY_NEGATIVES}',
            ),
            (
                "partner_request",
                "deterministic_partner_request",
                f'("{primary}" OR "{secondary}") '
                f'("seeking implementation partner" OR "seeking vendor" '
                f'OR "inviting service providers" OR "appointing consultant") '
                f'{region_expression} {year} {QUERY_NEGATIVES}',
            ),
            (
                "implementation_announcement",
                "deterministic_implementation_action",
                f'("{primary}" OR "{secondary}") '
                f'("plans to implement" OR "is implementing" '
                f'OR "implementation project" OR "integration project" '
                f'OR "supplier onboarding") '
                f'{region_expression} {year} {QUERY_NEGATIVES}',
            ),
            (
                "modernization_project",
                "deterministic_modernization_action",
                f'("{primary}" OR "{secondary}") '
                f'("modernization project" OR "modernisation programme" '
                f'OR "system replacement" OR "migration project") '
                f'("seeking partner" OR "implementation partner" '
                f'OR RFP OR tender OR procurement) '
                f'{region_expression} {year} {QUERY_NEGATIVES}',
            ),
            (
                "regulatory_trigger",
                "deterministic_regulatory_implementation",
                f'("{primary}" OR "{secondary}") '
                f'(mandate OR regulation OR compliance) '
                f'("plans to implement" OR "implementation project" '
                f'OR "ERP integration" OR "supplier onboarding" '
                f'OR "seeking implementation partner") '
                f'{region_expression} {year} {QUERY_NEGATIVES}',
            ),
            (
                "industry_requirement",
                "deterministic_industry_requirement",
                f'("{primary}" OR "{secondary}") '
                f'("{industry}") '
                f'("implementation project" OR "migration project" '
                f'OR "seeking implementation partner" OR RFP OR tender) '
                f'{region_expression} {year} {QUERY_NEGATIVES}',
            ),
        ]

        planned_queries: list[PlannedSearchQuery] = []

        for index, (intent_type, strategy, query) in enumerate(
            patterns,
            start=1,
        ):
            planned_queries.append(
                PlannedSearchQuery(
                    service_id=service_id,
                    service_name=service_name,
                    query=" ".join(query.split()),
                    source_type=self._source_type_for_intent(intent_type),
                    platform="web",
                    intent_type=intent_type,
                    strategy=strategy,
                    strategy_order=index,
                    priority=index,
                    target_country=None,
                )
            )

            if len(planned_queries) >= requested_count:
                break

        return planned_queries