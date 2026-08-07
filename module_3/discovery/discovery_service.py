from __future__ import annotations

# =============================================================================
# MERGE NOTES (read before editing further)
# -----------------------------------------------------------------------------
# This file combines two branches of LeadDiscoveryService:
#
#   "Search stage" (query generation -> candidate collection -> qualification
#   gate) is taken from the branch with queries_per_service/max_total_queries,
#   ListingPageDetector, and richer content-truncation helpers.
#
#   "Extraction / email-generation stage" (local analysis -> LeadProfile ->
#   LLM final validation -> region filter -> response) is taken from the
#   branch with the region filter, country-override-from-LLM, and the
#   raw full_text/cleaned_content/page_text metadata (kept because the
#   email-generation endpoint likely reads these fields).
#
# Things to double check against your actual schemas.py / model classes
# before running this in production:
#   1. DiscoverLeadsRequest must expose `queries_per_service` and
#      `max_total_queries` (not `max_queries`) -- this matches the frontend
#      merge done earlier.
#   2. SearchCandidate must accept the extra fields (source_type, platform,
#      intent_type, strategy, strategy_order, priority) passed below.
#   3. DiscoverLeadsResponse -- I added `listing_page_rejections` and
#      `expired_rejections` to the returned object since those stats are now
#      produced by the merged pipeline. Remove them if your schema doesn't
#      define those fields yet (or add the fields to the schema).
#   4. LeadIntelligenceService.build_report() and LeadValidationCandidate
#      are NOT given the extra deadline_status/deadline/deadline_reason
#      kwargs, even though that data is now available in `context`, because
#      I couldn't confirm those classes accept them. If they do, it's a
#      quick follow-up to thread `context.get("deadline_status")` etc. in.
# =============================================================================

import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from datetime import date

from app.search.serpapi import search

from module_3.discovery.contradiction_checker import ContradictionChecker
from module_3.discovery.deadline_checker import (
    DeadlineChecker,
    DeadlineAssessment,
)
from module_3.discovery.document_fetcher import DocumentFetcher
from module_3.discovery.llm_lead_validator import (
    LLMLeadValidator,
    LeadValidationCandidate,
    LeadValidationDecision,
)
from module_3.discovery.listing_page_detector import ListingPageDetector
from module_3.discovery.metadata_extractor import MetadataExtractor
from module_3.discovery.models import SearchCandidate
from module_3.discovery.qualification_gate import QualificationGate
from module_3.discovery.query_generator import QueryGenerator
from module_3.discovery.requirement_classifier import RequirementClassifier
from module_3.intelligence.service import LeadIntelligenceService
from module_3.schemas import (
    AnalyzeLeadRequest,
    DiscoverLeadsRequest,
    DiscoverLeadsResponse,
    DiscoveredLeadResponse,
    LeadProfile,
    ManualReviewLead,
)
from module_3.service import LeadAnalysisService

logger = logging.getLogger(__name__)


TRACKING_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "srsltid",
}


BLOCKED_TITLE_TERMS = {
    "rfp template",
    "proposal template",
    "request for proposal template",
    "sample rfp",
    "sample proposal",
    "how to write an rfp",
    "how to respond to an rfp",
    "how to win an rfp",
    "rfp response guide",
    "proposal writing guide",
    "boost win rates",
    "winning proposals",
    "vendor best practices",
    "vendor guide",
    "buyer guide",
    "tutorial",
    "webinar",
    "white paper",
    "ebook",
    "case study",
    "job opening",
    "job description",
    "career opportunity",
    "we are hiring",
    "hiring now",
}


BLOCKED_URL_TERMS = {
    "/blog",
    "/blogs",
    "/blog-post",
    "/blog-posts",
    "/article",
    "/articles",
    "/career",
    "/careers",
    "/job",
    "/jobs",
    "/template",
    "/templates",
    "/guide",
    "/guides",
    "/tutorial",
    "/tutorials",
    "/webinar",
    "/webinars",
    "/ebook",
    "/ebooks",
    "/whitepaper",
    "/whitepapers",
    "/case-study",
    "/case-studies",
}


BLOCKED_GENERAL_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "instagram.com",
    "medium.com",
    "behance.net",
    "reddit.com",
    "x.com",
    "twitter.com",
}


REAL_PROCUREMENT_TERMS = {
    "request for proposal",
    "request for quotation",
    "invitation to tender",
    "invitation to bid",
    "invitation for bids",
    "expression of interest",
    "solicitation number",
    "solicitation reference",
    "tender number",
    "tender reference",
    "procurement reference",
    "contract notice",
    "procurement notice",
    "submission deadline",
    "proposal deadline",
    "bid deadline",
    "tender deadline",
    "closing date",
    "closing time",
    "scope of work",
    "statement of work",
    "terms of reference",
    "instructions to bidders",
    "instructions for bidders",
    "contracting authority",
    "issuing authority",
    "procuring entity",
    "submit proposals",
    "submit bids",
    "technical proposal",
    "financial proposal",
    "eligibility criteria",
    "bid security",
}


WEAK_PROCUREMENT_TERMS = {
    "evaluation criteria",
    "supplier",
    "vendor",
    "proposal",
    "procurement",
    "contract",
}


NON_OPPORTUNITY_CONTENT_TERMS = {
    "rfp template",
    "proposal template",
    "request for proposal template",
    "sample rfp",
    "sample proposal",
    "how to respond to an rfp",
    "how to write an rfp",
    "how to win an rfp",
    "boost win rates",
    "winning proposal",
    "help vendors",
    "vendor best practices",
    "sales teams",
    "download the template",
    "free template",
    "proposal writing tips",
    "responding to rfps",
    "best rfp software",
    "rfp management platform",
}


MARKETPLACE_REQUIREMENT_TERMS = {
    "looking for",
    "need a",
    "need an",
    "we need",
    "seeking",
    "required",
    "requirements",
    "project details",
    "project description",
    "submit a proposal",
    "place a bid",
    "send proposal",
    "budget",
    "fixed price",
    "hourly rate",
    "deadline",
    "deliverables",
    "scope",
}


MARKETPLACE_NEGATIVE_TERMS = {
    "freelancer profile",
    "hire me",
    "my portfolio",
    "services i offer",
    "available for work",
    "course assignment",
    "homework",
    "student project",
    "academic assignment",
}


PARTNER_REQUIREMENT_TERMS = {
    "seeking implementation partner",
    "looking for technology partner",
    "seeking vendor",
    "inviting service providers",
    "looking for consultants",
    "requesting proposals",
    "external supplier",
    "implementation partner required",
}

BUYING_SIGNAL_TERMS = {
    # Regulatory and compliance signals
    "mandatory compliance",
    "compliance deadline",
    "regulatory requirement",
    "regulatory mandate",
    "new regulation",
    "compliance programme",
    "must comply",
    "required to comply",

    # Implementation and rollout signals
    "implementation programme",
    "implementation project",
    "planned implementation",
    "system implementation",
    "platform implementation",
    "integration project",
    "planned integration",
    "rollout programme",
    "technology rollout",
    "deployment programme",

    # Transformation and modernization signals
    "digital transformation",
    "technology transformation",
    "modernization programme",
    "modernisation programme",
    "system modernization",
    "legacy modernization",
    "platform replacement",
    "system replacement",
    "infrastructure upgrade",
    "technology upgrade",

    # Commercial/project signals
    "project scope",
    "project requirements",
    "programme launch",
    "budget approved",
    "approved budget",
    "implementation timeline",
    "implementation deadline",
    "project deadline",
    "seeking solution",
    "solution required",
}


BUYING_SIGNAL_NEGATIVE_TERMS = {
    "services we offer",
    "our services",
    "contact us",
    "book a demo",
    "request a demo",
    "our solution",
    "our platform",
    "leading provider",
    "trusted provider",
    "case study",
    "customer success story",
    "market report",
    "industry report",
    "training course",
    "webinar",
    "template",
    "guide",
}


def normalize_url(url: str) -> str:
    parts = urlsplit(url)

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMETERS
    ]

    normalized_path = parts.path.rstrip("/") or "/"

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower().removeprefix("www."),
            normalized_path,
            urlencode(filtered_query),
            "",
        )
    )


LEAD_CONTENT_MAX_CHARS = 50_000


def truncate_text(
    value: str | None,
    max_chars: int = LEAD_CONTENT_MAX_CHARS,
) -> str:
    """
    Safely truncate extracted content before passing it into LeadProfile.

    Keeps the beginning and end of long procurement documents because
    requirements are usually near the beginning while deadlines and
    submission instructions may appear near the end.
    """
    text = (value or "").strip()

    if len(text) <= max_chars:
        return text

    separator = (
        "\n\n[... CONTENT TRUNCATED FOR ANALYSIS ...]\n\n"
    )

    available_chars = max_chars - len(separator)

    beginning_chars = int(available_chars * 0.75)
    ending_chars = available_chars - beginning_chars

    return (
        text[:beginning_chars]
        + separator
        + text[-ending_chars:]
    )


def build_llm_excerpt(
    value: str | None,
    max_chars: int = 1800,
) -> str:
    """
    Build a compact LLM excerpt that preserves both the beginning
    and end of a document.

    Procurement scope is commonly near the beginning, while deadlines
    and submission instructions may appear near the end.
    """
    text = (value or "").strip()

    if len(text) <= max_chars:
        return text

    separator = (
        "\n\n[... DOCUMENT CONTENT OMITTED ...]\n\n"
    )

    available_chars = max_chars - len(separator)

    beginning_chars = int(
        available_chars * 0.65
    )
    ending_chars = (
        available_chars - beginning_chars
    )

    return (
        text[:beginning_chars]
        + separator
        + text[-ending_chars:]
    )


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def normalized_domain(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def domain_matches(domain: str, blocked_domain: str) -> bool:
    return domain == blocked_domain or domain.endswith(f".{blocked_domain}")


def url_contains_blocked_term(url: str, blocked_term: str) -> bool:
    path = urlsplit(url).path.lower().rstrip("/")
    term = blocked_term.lower().rstrip("/")

    if not term.startswith("/"):
        term = f"/{term}"

    return path == term or path.startswith(f"{term}/") or f"{term}/" in path


def should_skip_candidate(
    candidate: SearchCandidate,
    source_type: str,
    platform: str,
    intent_type: str | None = None,
) -> tuple[bool, str | None]:
    """
    Apply lightweight pre-fetch filtering.

    Formal procurement and partner-search results use strict URL filtering.
    Buying-signal strategies may keep article pages because regulatory,
    implementation, and transformation announcements are often published
    under /article or /news paths.

    This function only removes obvious noise. It does not qualify leads.
    """

    title = normalize_text(candidate.source_title)
    snippet = normalize_text(candidate.source_snippet)
    combined_search_text = f"{title} {snippet}".strip()

    candidate_url = str(candidate.source_url)
    domain = normalized_domain(candidate_url)

    normalized_source = str(
        source_type or ""
    ).strip().casefold()

    normalized_platform = str(
        platform or ""
    ).strip().casefold()

    normalized_intent = str(
        intent_type or ""
    ).strip().casefold()

    buying_signal_intents = {
        "regulatory_trigger",
        "implementation_announcement",
        "digital_transformation",
        "modernization_project",
        "technology_requirement",
        "industry_requirement",
    }

    hiring_intents = {
        "hiring_activity",
        "hiring_signal",
    }

    is_marketplace = normalized_source == "marketplace"
    is_buying_signal = normalized_intent in buying_signal_intents
    is_hiring_signal = normalized_intent in hiring_intents

    # Marketplace URLs legitimately contain /jobs and similar paths.
    if not is_marketplace:
        for term in BLOCKED_TITLE_TERMS:
            # Job wording may be expected for a deliberate hiring signal.
            if (
                is_hiring_signal
                and term in {
                    "job opening",
                    "job description",
                    "career opportunity",
                    "we are hiring",
                    "hiring now",
                }
            ):
                continue

            if term in combined_search_text:
                return (
                    True,
                    f"title/snippet contains blocked term '{term}'",
                )

        for term in BLOCKED_URL_TERMS:
            # Regulatory and transformation announcements are frequently
            # published under article paths. Keep them for deeper validation.
            if (
                is_buying_signal
                and term in {
                    "/article",
                    "/articles",
                }
            ):
                continue

            # Deliberate hiring strategies may legitimately return job URLs.
            if (
                is_hiring_signal
                and term in {
                    "/career",
                    "/careers",
                    "/job",
                    "/jobs",
                }
            ):
                continue

            if url_contains_blocked_term(
                candidate_url,
                term,
            ):
                return (
                    True,
                    f"URL contains blocked path term '{term}'",
                )

        for blocked_domain in BLOCKED_GENERAL_DOMAINS:
            if domain_matches(domain, blocked_domain):
                return (
                    True,
                    f"domain '{domain}' is not an approved "
                    "opportunity source",
                )

    if normalized_platform == "freelancer":
        if not domain_matches(domain, "freelancer.com"):
            return (
                True,
                "result does not belong to Freelancer",
            )

    if normalized_platform == "peopleperhour":
        if not domain_matches(domain, "peopleperhour.com"):
            return (
                True,
                "result does not belong to PeoplePerHour",
            )

    return False, None

def validate_procurement_content(
    *,
    title: str,
    snippet: str,
    text: str,
) -> tuple[bool, list[str], list[str]]:
    combined = normalize_text("\n".join([title, snippet, text]))
    reasons: list[str] = []

    negative_matches = sorted(
        term for term in NON_OPPORTUNITY_CONTENT_TERMS if term in combined
    )
    if negative_matches:
        reasons.append(
            "Content appears to be guidance, a template, vendor marketing, "
            "or proposal advice."
        )

    strong_matches = sorted(
        term for term in REAL_PROCUREMENT_TERMS if term in combined
    )
    weak_matches = sorted(
        term for term in WEAK_PROCUREMENT_TERMS if term in combined
    )

    # One weak phrase such as "evaluation criteria" is not sufficient.
    if not strong_matches:
        reasons.append("No strong procurement indicator was found.")

    return not reasons, reasons, strong_matches + weak_matches


def validate_marketplace_content(
    *,
    platform: str,
    url: str,
    title: str,
    snippet: str,
    text: str,
) -> tuple[bool, list[str], list[str]]:
    combined = normalize_text("\n".join([title, snippet, text]))
    reasons: list[str] = []

    matched_positive = sorted(
        term for term in MARKETPLACE_REQUIREMENT_TERMS if term in combined
    )
    matched_negative = sorted(
        term for term in MARKETPLACE_NEGATIVE_TERMS if term in combined
    )

    domain = normalized_domain(url)

    expected_domain = {
        "freelancer": "freelancer.com",
        "peopleperhour": "peopleperhour.com",
    }.get(platform)

    if expected_domain and not domain_matches(domain, expected_domain):
        reasons.append(f"Page is not hosted on the expected {platform} domain.")

    if matched_negative:
        reasons.append(
            "Marketplace page appears to be a provider profile, portfolio, "
            "or non-commercial academic request."
        )

    if not matched_positive:
        reasons.append(
            "Marketplace page does not contain enough direct project-request language."
        )

    return not reasons, reasons, matched_positive


def validate_partner_content(
    *,
    title: str,
    snippet: str,
    text: str,
) -> tuple[bool, list[str], list[str]]:
    combined = normalize_text("\n".join([title, snippet, text]))
    reasons: list[str] = []

    matched = sorted(
        term for term in PARTNER_REQUIREMENT_TERMS if term in combined
    )

    if not matched:
        reasons.append(
            "No explicit external partner, vendor, consultant, or service-provider "
            "request was found."
        )

    negative_matches = sorted(
        term for term in NON_OPPORTUNITY_CONTENT_TERMS if term in combined
    )
    if negative_matches:
        reasons.append(
            "Content appears educational, promotional, or template-oriented."
        )

    return not reasons, reasons, matched

def validate_buying_signal_content(
    *,
    title: str,
    snippet: str,
    text: str,
) -> tuple[bool, list[str], list[str]]:
    combined = normalize_text(
        "\n".join([title, snippet, text])
    )

    reasons: list[str] = []

    positive_matches = sorted(
        term
        for term in BUYING_SIGNAL_TERMS
        if term in combined
    )

    procurement_matches = sorted(
        term
        for term in REAL_PROCUREMENT_TERMS
        if term in combined
    )

    partner_matches = sorted(
        term
        for term in PARTNER_REQUIREMENT_TERMS
        if term in combined
    )

    negative_matches = sorted(
        term
        for term in BUYING_SIGNAL_NEGATIVE_TERMS
        if term in combined
    )

    matched_indicators = sorted(
        set(
            positive_matches
            + procurement_matches
            + partner_matches
        )
    )

    if not matched_indicators:
        reasons.append(
            "No implementation, regulatory, transformation, "
            "modernization, procurement, or partner-search "
            "signal was found."
        )

    # Do not reject merely because one marketing phrase appears.
    # Reject only when marketing evidence is present and there is
    # no genuine buying signal.
    if negative_matches and not matched_indicators:
        reasons.append(
            "Content appears promotional, educational, or "
            "provider-generated rather than buyer-driven."
        )

    return (
        not reasons,
        reasons,
        matched_indicators,
    )


def validate_by_source(
    *,
    source_type: str,
    platform: str,
    intent_type: str | None,
    title: str,
    snippet: str,
    text: str,
    url: str,
) -> tuple[bool, list[str], list[str]]:
    normalized_source = (
        str(source_type or "")
        .strip()
        .casefold()
    )

    normalized_intent = (
        str(intent_type or "")
        .strip()
        .casefold()
    )

    if normalized_source == "marketplace":
        return validate_marketplace_content(
            platform=platform,
            url=url,
            title=title,
            snippet=snippet,
            text=text,
        )

    if normalized_intent in {
        "formal_procurement",
        "official_procurement",
        "procurement",
    }:
        return validate_procurement_content(
            title=title,
            snippet=snippet,
            text=text,
        )

    if normalized_intent in {
        "partner_request",
        "partner_search",
    }:
        return validate_partner_content(
            title=title,
            snippet=snippet,
            text=text,
        )

    if normalized_intent in {
        "regulatory_trigger",
        "implementation_announcement",
        "digital_transformation",
        "modernization_project",
        "technology_requirement",
        "industry_requirement",
    }:
        return validate_buying_signal_content(
            title=title,
            snippet=snippet,
            text=text,
        )

    if normalized_intent in {
        "hiring_activity",
        "hiring_signal",
    }:
        return validate_buying_signal_content(
            title=title,
            snippet=snippet,
            text=text,
        )

    # Legacy fallback behavior.
    if normalized_source == "general_web":
        return validate_buying_signal_content(
            title=title,
            snippet=snippet,
            text=text,
        )

    return validate_procurement_content(
        title=title,
        snippet=snippet,
        text=text,
    )


class LeadDiscoveryService:
    def __init__(
        self,
        analysis_service: LeadAnalysisService,
        config: dict,
    ):
        self.analysis_service = analysis_service
        self.intelligence_service = LeadIntelligenceService()
        self.listing_page_detector = ListingPageDetector()

        llm_model = config.get("llm_model")
        use_llm = config.get("use_llm", False)
        self.llm_model = llm_model

        self.query_generator = QueryGenerator(
            knowledge_base_path=config.get("knowledge_base_path"),
            use_llm=use_llm,
            llm_model=llm_model,
            planner_prompt_path=config.get("planner_prompt_path"),
        )

        # Requirement classification remains rule-based for now.
        # Claude is used only for final validation at this stage.
        self.classifier = RequirementClassifier(
            llm_model=None,
            use_gemini=False,
            max_chunks=config.get("max_chunks", 3),
        )

        self.config = config

        # Load the canonical service-region names before creating the LLM
        # validator. Claude receives these exact names and must copy one of
        # them when it marks a buyer region as supported.
        self.knowledge_base_path = config.get("knowledge_base_path")
        if self.knowledge_base_path:
            self.knowledge_base_path = Path(self.knowledge_base_path)

        self.allowed_regions: list[str] = []
        self.region_names: set[str] = set()

        if self.knowledge_base_path and self.knowledge_base_path.exists():
            try:
                with open(
                    self.knowledge_base_path,
                    "r",
                    encoding="utf-8",
                ) as file_handle:
                    data = json.load(file_handle)

                regions = data.get("service_regions", [])

                self.allowed_regions = sorted(
                    {
                        str(region.get("region") or "").strip()
                        for region in regions
                        if str(region.get("region") or "").strip()
                    }
                )

                self.region_names = {
                    region.casefold()
                    for region in self.allowed_regions
                }

                logger.info(
                    "Loaded %s allowed regions from knowledge base.",
                    len(self.allowed_regions),
                )

            except Exception as exc:
                logger.warning(
                    "Could not load region names from knowledge base: %s",
                    exc,
                )
                self.allowed_regions = []
                self.region_names = set()
        else:
            logger.warning(
                "Knowledge base path not provided or missing; "
                "LLM region validation will use unknown status."
            )

        # The LLM is used as the final verifier for locally shortlisted
        # opportunities. It also identifies the buyer country and maps it
        # against the canonical region list loaded above.
        self.llm_validator = None

        if use_llm and llm_model is not None:
            self.llm_validator = LLMLeadValidator(
                model=llm_model,
                allowed_regions=self.allowed_regions,
                batch_size=config.get("llm_batch_size", 8),
                max_candidates=config.get("llm_max_candidates", 20),
                max_excerpt_chars=config.get(
                    "llm_max_excerpt_chars",
                    3000,
                ),
            )

            logger.info(
                "LLM lead validator enabled. Batch size: %s | "
                "Max candidates: %s | Allowed regions: %s",
                config.get("llm_batch_size", 8),
                config.get("llm_max_candidates", 20),
                len(self.allowed_regions),
            )
        else:
            logger.info(
                "LLM lead validator disabled. "
                "Locally validated leads will be returned."
            )

        self.fetcher = DocumentFetcher(
            timeout=config.get("fetch_timeout", 20),
            max_size=config.get("fetch_max_size", 10 * 1024 * 1024),
        )

        self.contradiction = ContradictionChecker()
        self.deadline_checker = DeadlineChecker(
            grace_days=config.get("deadline_grace_days", 0),
        )

        self.gate = QualificationGate(
            {
                "min_buyer_score": config.get("min_buyer_score", 0.6),
                "max_provider_prob": config.get("max_provider_prob", 0.4),
            }
        )

        # Concurrency settings
        self.query_workers = config.get("query_workers", 5)
        self.fetch_workers = config.get("fetch_workers", 10)

    def _should_recheck_deadline_with_llm(
            self,
            deadline_assessment,
    ) -> bool:
        """
        Decide whether the rule-based deadline result needs LLM extraction.
        Claude is used only for ambiguous or potentially unsafe cases.
        """
        if self.llm_model is None:
            return False
        if deadline_assessment.status == "unknown":
            return True
        if deadline_assessment.deadline is None:
            return True
        matched_text = (
            deadline_assessment.matched_text or ""
        ).casefold()
        ambiguous_labels = {
            "proposal opening",
            "bid opening",
            "question deadline",
            "clarification deadline",
            "pre-bid meeting",
            "award date",
            "contract start",
        }
        return any(
            label in matched_text
            for label in ambiguous_labels
        )

    def _extract_deadline_with_llm(
            self,
            *,
            title: str,
            snippet: str,
            text: str,
    ) -> dict[str, Any] | None:
        """
        Ask the LLM to extract the final submission deadline only.
        The LLM extracts evidence; application code decides active/expired.
        """
        if self.llm_model is None:
            return None
        excerpt = build_llm_excerpt(
            "\n\n".join(
                part
                for part in [title, snippet, text]
                if part
            ),
            max_chars=5000,
        )
        prompt = f"""
        You are extracting a procurement deadline from a document.
        Identify only the final deadline for submitting the bid, proposal,
        quotation, tender response, application, or RFI response.
        Do not select:
        - publication date
        - issue date
        - clarification deadline
        - question deadline
        - pre-bid meeting date
        - proposal opening date
        - bid opening date
        - award date
        - contract start date
        - implementation date
        If no final submission deadline is clearly supported, return null.
        Return valid JSON only:
        {{
            "deadline": "YYYY-MM-DD or null",
            "evidence": "exact supporting text or null",
            "confidence": 0.0,
            "reason": "brief explanation"
        }}
        Document:
        {excerpt}
""".strip()
        try:
            response = self.llm_model.generate_content(
                prompt,
                max_tokens=800,
            )
            raw = getattr(response, "text", None)
            if not isinstance(raw, str) or not raw.strip():
                return None

            cleaned = raw.strip()

            fenced_match = re.search(
                r"```(?:json)?\s*(.*?)\s*```",
                cleaned,
                flags=re.DOTALL | re.IGNORECASE,
            )

            if fenced_match:
                cleaned = fenced_match.group(1).strip()

            data = json.loads(cleaned)

            if not isinstance(data, dict):
                return None

            return data

        except Exception:
            logger.exception(
                "Claude deadline extraction failed."
            )
            return None
    
    def discover(
        self,
        request: DiscoverLeadsRequest,
    ) -> DiscoverLeadsResponse:
        # =====================================================================
        # SEARCH STAGE (from friend's branch): query generation, candidate
        # collection, source validation, listing-page detection, deadline
        # check, and qualification gate.
        # =====================================================================
        query_records = self.query_generator.generate(
            queries_per_service=request.queries_per_service,
            max_total_queries=request.max_total_queries,
            selected_service_ids=request.selected_service_ids,
        )

        requested_query_total = (
            len(
                {
                    record["service_id"]
                    for record in query_records
                }
            )
            * request.queries_per_service
        )

        logger.info(
            "Query generation complete. Queries per service: %s | "
            "Requested total: %s | Maximum total: %s | Generated: %s",
            request.queries_per_service,
            requested_query_total,
            request.max_total_queries,
            len(query_records),
        )

        listing_page_rejections = 0
        collected_candidates: List[SearchCandidate] = []
        candidate_context: dict[str, dict[str, Any]] = {}
        seen_urls: set[str] = set()
        prefiltered_count = 0

        # -----------------------------------------------------------------
        # PARALLEL QUERY EXECUTION (SerpAPI calls)
        # -----------------------------------------------------------------
        def execute_query(record: dict) -> tuple[dict, List[Any]]:
            logger.info(
                "Executing query | Service: %s (%s) | Source: %s | "
                "Platform: %s | Strategy: %s | Query: %s",
                record.get("service_name", ""),
                record["service_id"],
                record.get("source_type", "procurement"),
                record.get("platform", "web"),
                record.get("strategy", "legacy"),
                record["query"],
            )
            results = search(record["query"], num_results=request.results_per_query)
            return record, results

        with ThreadPoolExecutor(max_workers=self.query_workers) as query_executor:
            query_futures = {
                query_executor.submit(execute_query, rec): rec
                for rec in query_records
            }

            for future in as_completed(query_futures):
                try:
                    record, results = future.result()
                except Exception as exc:
                    logger.error("Query execution failed: %s", exc, exc_info=True)
                    continue

                for result in results:
                    normalized_url = normalize_url(result.url)

                    if normalized_url in seen_urls:
                        logger.debug("Duplicate URL skipped: %s", result.url)
                        continue

                    seen_urls.add(normalized_url)

                    candidate = SearchCandidate(
                        source_url=result.url,
                        source_title=result.title,
                        source_snippet=result.snippet,
                        source_domain=urlsplit(result.url).netloc,
                        search_query=record["query"],
                        service_id=record["service_id"],
                        service_name=record.get("service_name"),
                        source_type=record.get("source_type"),
                        platform=record.get("platform"),
                        intent_type=record.get("intent_type"),
                        strategy=record.get("strategy"),
                        strategy_order=record.get("strategy_order"),
                        priority=record.get("priority"),
                    )

                    context = {
                        "source_type": record.get("source_type", "procurement"),
                        "platform": record.get("platform", "web"),
                        "intent_type": record.get("intent_type", "procurement"),
                        "strategy": record.get("strategy", "legacy"),
                        "priority": record.get("priority", 99),
                        "strategy_order": record.get("strategy_order"),
                        "service_name": record.get("service_name", ""),
                    }

                    skip_candidate, skip_reason = should_skip_candidate(
                        candidate,
                        source_type=context["source_type"],
                        platform=context["platform"],
                        intent_type=context.get("intent_type"),
                    )

                    if skip_candidate:
                        prefiltered_count += 1
                        logger.info(
                            "⏭️ PRE-FILTERED: %s | Source: %s/%s | %s",
                            candidate.source_url,
                            context["source_type"],
                            context["platform"],
                            skip_reason,
                        )
                        continue

                    candidate_context[normalized_url] = context
                    collected_candidates.append(candidate)

        context_counts = Counter(
            context["source_type"] for context in candidate_context.values()
        )

        logger.info(
            "Candidate collection complete. Accepted for fetching: %s | "
            "Pre-filtered: %s | Source mix: %s",
            len(collected_candidates),
            prefiltered_count,
            dict(context_counts),
        )

        # -----------------------------------------------------------------
        # PARALLEL DOCUMENT FETCHING AND PROCESSING
        # -----------------------------------------------------------------
        qualified_candidates: list[dict[str, Any]] = []
        successful_fetches = 0
        failed_fetches = 0
        empty_content_count = 0
        validation_rejections = 0
        expired_rejections = 0
        gate_rejections = 0

        # Use a thread-local lock to update counters safely
        counter_lock = threading.Lock()

        def fetch_and_process(candidate: SearchCandidate) -> None:
            nonlocal successful_fetches, failed_fetches, empty_content_count
            nonlocal validation_rejections, expired_rejections, gate_rejections
            nonlocal listing_page_rejections

            context = candidate_context.get(
                normalize_url(str(candidate.source_url)),
                {
                    "source_type": "procurement",
                    "platform": "web",
                    "intent_type": "procurement",
                    "strategy": "legacy",
                    "priority": 99,
                    "strategy_order": None,
                    "service_name": "",
                },
            )

            doc = self.fetcher.fetch(str(candidate.source_url))

            if doc.fetch_status != "success":
                with counter_lock:
                    failed_fetches += 1
                logger.warning(
                    "Fetch failed for %s: %s",
                    candidate.source_url,
                    doc.fetch_error,
                )
                return

            with counter_lock:
                successful_fetches += 1

            if not normalize_text(doc.text):
                with counter_lock:
                    empty_content_count += 1
                logger.info("❌ EMPTY CONTENT: %s", candidate.source_url)
                return

            valid, rejection_reasons, matched_terms = validate_by_source(
                source_type=context["source_type"],
                platform=context["platform"],
                intent_type=context.get("intent_type"),
                title=candidate.source_title or "",
                snippet=candidate.source_snippet or "",
                text=doc.text or "",
                url=str(candidate.source_url),
            )

            if not valid:
                with counter_lock:
                    validation_rejections += 1
                logger.info(
                    "❌ SOURCE VALIDATION FAILED: %s | Source: %s/%s",
                    candidate.source_url,
                    context["source_type"],
                    context["platform"],
                )
                logger.info("   Reasons: %s", rejection_reasons)
                logger.info("   Matched indicators: %s", matched_terms)
                return

            logger.info(
                "✅ SOURCE VALIDATION PASSED: %s | Source: %s/%s",
                candidate.source_url,
                context["source_type"],
                context["platform"],
            )
            logger.info("   Matched indicators: %s", matched_terms)

            listing_assessment = self.listing_page_detector.assess(
                title=candidate.source_title or "",
                url=str(candidate.source_url),
                text="\n\n".join(
                    part
                    for part in [
                        candidate.source_snippet or "",
                        doc.text or "",
                    ]
                    if part
                ),
            )
            logger.info(
                "Listing-page assessment | URL: %s | Listing: %s | "
                "Confidence: %.2f | Reason: %s",
                candidate.source_url,
                listing_assessment.is_listing_page,
                listing_assessment.confidence,
                listing_assessment.reason,
            )
            if listing_assessment.is_listing_page:
                with counter_lock:
                    listing_page_rejections += 1
                logger.info(
                    "❌ LISTING PAGE REJECTED: %s | Indicators: %s",
                    candidate.source_url,
                    listing_assessment.matched_indicators,
                )
                return

            deadline_assessment = self.deadline_checker.assess(
                title=candidate.source_title or "",
                snippet=candidate.source_snippet or "",
                text=doc.text or "",
            )

            logger.info(
                "Deadline assessment | URL: %s | Status: %s | "
                "Deadline: %s | Confidence: %.2f | Reason: %s",
                candidate.source_url,
                deadline_assessment.status,
                deadline_assessment.deadline.isoformat() if deadline_assessment.deadline else None,
                deadline_assessment.confidence,
                deadline_assessment.reason,
            )

            if self._should_recheck_deadline_with_llm(
                deadline_assessment
            ):
                llm_deadline = self._extract_deadline_with_llm(
                    title=candidate.source_title or "",
                    snippet=candidate.source_snippet or "",
                    text=doc.text or "",
                )
                if llm_deadline:
                    deadline_value = llm_deadline.get("deadline")
                    if deadline_value:
                        try:
                            extracted_date = date.fromisoformat(
                                deadline_value
                            )
                            status_value = (
                                "expired"
                                if extracted_date < date.today()
                                else "active"
                            )
                            deadline_assessment = DeadlineAssessment(
                                status=status_value,
                                deadline=extracted_date,
                                matched_text=llm_deadline.get(
                                    "evidence"
                                ),
                                reason=llm_deadline.get(
                                    "reason"
                                )
                                or (
                                    "Deadline extracted by Claude and "
                                    "classified deterministically."
                                ),
                                confidence=float(
                                    llm_deadline.get(
                                        "confidence",
                                        0.0,
                                    )
                                ),
                            )
                            logger.info(
                                "Claude deadline extraction | URL: %s | "
                                "Status: %s | Deadline: %s | Confidence: %.2f",
                                candidate.source_url,
                                deadline_assessment.status,
                                extracted_date.isoformat(),
                                deadline_assessment.confidence,
                            )
                        except ValueError:
                            logger.warning(
                                "Claude returned an invalid deadline format "
                                "for %s: %s",
                                candidate.source_url,
                                deadline_value,
                            )
            if deadline_assessment.is_expired:
                with counter_lock:
                    expired_rejections += 1
                logger.info(
                    "❌ EXPIRED OPPORTUNITY: %s | Deadline: %s | %s",
                    candidate.source_url,
                    (
                        deadline_assessment.deadline.isoformat()
                        if deadline_assessment.deadline
                        else "not explicitly dated"
                    ),
                    deadline_assessment.reason,
                )
                return

            context = dict(context)
            context.update(
                {
                    "deadline_status": deadline_assessment.status,
                    "deadline": (
                        deadline_assessment.deadline.isoformat()
                        if deadline_assessment.deadline
                        else None
                    ),
                    "deadline_reason": deadline_assessment.reason,
                    "deadline_confidence": deadline_assessment.confidence,
                    "deadline_matched_text": getattr(
                        deadline_assessment,
                        "matched_text",
                        None,
                    ),
                }
            )

            if deadline_assessment.requires_manual_review:
                logger.info(
                    "⚠️ DEADLINE UNKNOWN — CONTINUING: %s | %s",
                    candidate.source_url,
                    deadline_assessment.reason,
                )

            qualification = self.classifier.classify(
                doc.text,
                doc.text_chunks,
                source_type=context["source_type"],
                platform=context["platform"],
            )

            contradiction_decision = self.contradiction.check(
                text=doc.text,
                qual=qualification,
                source_type=context["source_type"],
                platform=context["platform"],
                title=candidate.source_title or "",
                snippet=candidate.source_snippet or "",
            )
            contradiction_passed = contradiction_decision.passed
            contradiction_reasons = contradiction_decision.messages

            logger.info(
                "Contradiction assessment | URL: %s | Source: %s | "
                "Buyer score: %.2f | Provider score: %.2f | Passed: %s",
                candidate.source_url,
                contradiction_decision.source_type,
                contradiction_decision.buyer_signal_score,
                contradiction_decision.provider_signal_score,
                contradiction_decision.passed,
            )

            gate_decision = self.gate.apply(
                candidate,
                doc,
                qualification,
                contradiction_passed,
                source_type=context["source_type"],
            )

            if gate_decision.accepted:
                # Append to qualified_candidates (need lock because list append is not thread-safe)
                with counter_lock:
                    qualified_candidates.append(
                        {
                            "candidate": candidate,
                            "document": doc,
                            "qualification": qualification,
                            "context": context,
                            "deadline": deadline_assessment,
                            "listing": listing_assessment,
                        }
                    )
                logger.info(
                    "✅ QUALIFIED: %s | Source: %s/%s",
                    candidate.source_url,
                    context["source_type"],
                    context["platform"],
                )
                return

            with counter_lock:
                gate_rejections += 1

            logger.info(
                "❌ QUALIFICATION REJECTED: %s | Source: %s | Reason: %s",
                candidate.source_url,
                gate_decision.source_type,
                gate_decision.reason or "No reason provided",
            )

            rejection_reasons = list(
                getattr(qualification, "rejection_reasons", []) or []
            )
            if not rejection_reasons:
                rejection_reasons.append(
                    "Candidate did not meet one or more configured qualification "
                    "thresholds."
                )

            logger.info("❌ REJECTED: %s", candidate.source_url)
            logger.info("   Source: %s/%s", context["source_type"], context["platform"])
            logger.info("   Document type: %s", qualification.document_type)
            logger.info(
                "   is_service_requirement: %s",
                qualification.is_service_requirement,
            )
            logger.info(
                "   organization_role: %s",
                qualification.organization_role,
            )
            logger.info(
                "   buyer_intent_score: %s",
                qualification.buyer_intent_score,
            )
            logger.info(
                "   provider_probability: %s",
                qualification.provider_probability,
            )
            logger.info(
                "   explicit_requirement: %s",
                qualification.explicit_requirement,
            )
            logger.info(
                "   requires_external_supplier: %s",
                qualification.requires_external_supplier,
            )
            logger.info("   contradiction_passed: %s", contradiction_passed)
            logger.info("   contradiction_reasons: %s", contradiction_reasons)
            logger.info("   rejection_reasons: %s", rejection_reasons)

        # Execute fetch-and-process in parallel
        with ThreadPoolExecutor(max_workers=self.fetch_workers) as fetch_executor:
            fetch_futures = {
                fetch_executor.submit(fetch_and_process, cand): cand
                for cand in collected_candidates
            }
            # Wait for all to complete
            for future in as_completed(fetch_futures):
                candidate = fetch_futures[future]

                try:
                    future.result()
                except Exception:
                    logger.exception(
                        "Unexpected document-processing failure "
                        "for candidate: %s",
                        candidate.source_url,
                    )

        # qualified_candidates is now fully populated.
        logger.info(
            "Qualification stage complete. Collected: %s | Successful fetches: %s | "
            "Failed fetches: %s | Empty content: %s | Source-validation rejections: %s | "
            "Listing-page rejections: %s | Expired rejections: %s | "
            "Gate rejections: %s | Qualified candidates: %s",
            len(collected_candidates),
            successful_fetches,
            failed_fetches,
            empty_content_count,
            validation_rejections,
            listing_page_rejections,
            expired_rejections,
            gate_rejections,
            len(qualified_candidates),
        )

        # =====================================================================
        # EXTRACTION / EMAIL-GENERATION SUPPORT STAGE (from your branch):
        # LeadProfile construction, similarity-threshold manual review,
        # LLM final validation, country override, region filter.
        # =====================================================================
        discovered_leads: List[DiscoveredLeadResponse] = []
        local_shortlist: list[dict[str, Any]] = []
        similarity_manual_review: list[ManualReviewLead] = []
        manual_review: list[ManualReviewLead] = []

        for item in qualified_candidates:
            candidate = item["candidate"]
            doc = item["document"]
            qualification = item["qualification"]
            context = item["context"]

            combined_content = "\n\n".join(
                part
                for part in [
                    candidate.source_title or "",
                    candidate.source_snippet or "",
                    doc.text or "",
                ]
                if part
            ).strip()

            if not combined_content:
                logger.warning(
                    "Skipping %s because no usable content was found.",
                    candidate.source_url,
                )
                continue

            metadata = MetadataExtractor.extract_all(
                url=str(candidate.source_url),
                title=candidate.source_title or "",
                snippet=candidate.source_snippet or "",
                text=doc.text or "",
            )

            country = metadata.get("country")

            # Smart truncation (keeps both ends) for the content actually
            # sent to the analysis/embedding model...
            analysis_content = truncate_text(
                combined_content,
                max_chars=LEAD_CONTENT_MAX_CHARS,
            )
            if len(combined_content) > LEAD_CONTENT_MAX_CHARS:
                logger.info(
                    "Lead content truncated for analysis | URL: %s | "
                    "Original chars: %s | Analysis chars: %s",
                    candidate.source_url,
                    len(combined_content),
                    len(analysis_content),
                )

            # ...while the FULL raw text is preserved in metadata for
            # downstream use (e.g. the email-generation endpoint).
            lead = LeadProfile(
                company_name=metadata.get("company_name"),
                industry=metadata.get("industry"),
                country=country,
                source_url=str(candidate.source_url),
                summary=candidate.source_snippet or candidate.source_title or "",
                content=analysis_content,
                technologies=[],
                projects=[],
                signals=[],
                keywords=[],
                metadata={
                    "search_query": candidate.search_query,
                    "source_title": candidate.source_title,
                    "source_snippet": candidate.source_snippet,
                    "source_domain": candidate.source_domain,
                    "service_id": candidate.service_id,
                    "source_type": context["source_type"],
                    "platform": context["platform"],
                    "intent_type": context["intent_type"],
                    "strategy": context["strategy"],
                    "query_priority": context["priority"],
                    "query_strategy_order": context.get("strategy_order"),
                    "query_service_name": context["service_name"],
                    "extracted_emails": metadata.get("emails", []),
                    "original_content_length": len(doc.text or ""),
                    "analysis_content_length": len(analysis_content),
                    "content_was_truncated": (
                        len(combined_content) > LEAD_CONTENT_MAX_CHARS
                    ),
                    "deadline_status": context.get("deadline_status", "unknown"),
                    "deadline": context.get("deadline"),
                    "deadline_reason": context.get("deadline_reason", ""),
                    "deadline_confidence": context.get("deadline_confidence", 0.0),
                    "deadline_matched_text": context.get("deadline_matched_text"),
                    # Raw text, kept for email generation / drafting.
                    "full_text": doc.text or "",
                    "cleaned_content": doc.text or "",
                    "page_text": doc.text or "",
                },
            )

            analysis_response = self.analysis_service.analyze(
                AnalyzeLeadRequest(
                    lead=lead,
                    top_k=3,
                    minimum_similarity=request.minimum_similarity,
                )
            )

            fallback_service_match: dict[str, Any] | None = None

            if not analysis_response.matched_services:
                originating_service_id = str(
                    candidate.service_id or ""
                ).strip()

                originating_service_name = str(
                    candidate.service_name
                    or context.get("service_name")
                    or ""
                ).strip()

                if not originating_service_id:
                    logger.warning(
                        "🟠 MANUAL REVIEW — NO SERVICE CONTEXT: %s | "
                        "Embedding similarity found no match and the "
                        "originating query did not contain a service ID.",
                        candidate.source_url,
                    )

                    similarity_manual_review.append(
                        ManualReviewLead(
                            source_title=candidate.source_title or "",
                            source_url=str(candidate.source_url),
                            source_snippet=candidate.source_snippet,
                            search_query=candidate.search_query,
                            reason=(
                                "No service exceeded the similarity threshold "
                                "and no originating service context was available."
                            ),
                            country=lead.country,
                            review_type="similarity",
                        )
                    )
                    continue

                fallback_service_match = {
                    "service_id": originating_service_id,
                    "service_name": (
                        originating_service_name
                        or originating_service_id
                    ),
                    "similarity_percentage": 0.0,
                    "service_match_percentage": 0.0,
                    "match_source": "originating_search_query",
                    "similarity_status": "below_threshold",
                }

                logger.info(
                    "Low-similarity candidate retained for LLM validation: "
                    "%s | Originating service: %s (%s)",
                    candidate.source_url,
                    fallback_service_match["service_name"],
                    fallback_service_match["service_id"],
                )

            local_shortlist.append(
                {
                    "candidate": candidate,
                    "document": doc,
                    "qualification": qualification,
                    "context": context,
                    "lead": lead,
                    "analysis": analysis_response,
                    "deadline": item["deadline"],
                    "fallback_service_match": fallback_service_match,
                }
            )

        logger.info(
            "Local analysis complete. Analysed: %s | "
            "Sent to LLM validation: %s | "
            "Unroutable similarity reviews: %s",
            len(qualified_candidates),
            len(local_shortlist),
            len(similarity_manual_review),
        )

        llm_results = []

        if self.llm_validator and local_shortlist:
            llm_candidates: list[LeadValidationCandidate] = []

            for index, item in enumerate(local_shortlist):
                candidate = item["candidate"]
                doc = item["document"]
                qualification = item["qualification"]
                analysis_response = item["analysis"]

                matched_services: list[dict[str, Any]] = []

                for match in analysis_response.matched_services:
                    if hasattr(match, "model_dump"):
                        match_payload = match.model_dump()
                    elif hasattr(match, "dict"):
                        match_payload = match.dict()
                    else:
                        match_payload = {
                            "service_id": getattr(match, "service_id", None),
                            "service_name": getattr(match, "service_name", None),
                            "similarity_percentage": getattr(
                                match,
                                "similarity_percentage",
                                0.0,
                            ),
                            "service_match_percentage": getattr(
                                match,
                                "service_match_percentage",
                                0.0,
                            ),
                        }

                    matched_services.append(match_payload)
                if (
                    not matched_services
                    and item.get("fallback_service_match")
                ):
                    matched_services.append(
                        item["fallback_service_match"]
                    )
                document_type = getattr(
                    qualification.document_type,
                    "value",
                    str(qualification.document_type),
                )

                llm_candidates.append(
                    LeadValidationCandidate(
                        candidate_id=str(index),
                        title=candidate.source_title or "",
                        url=str(candidate.source_url),
                        snippet=candidate.source_snippet or "",
                        content_excerpt=build_llm_excerpt(
                            doc.text,
                            max_chars=self.config.get(
                                "llm_max_excerpt_chars",
                                3000,
                            ),
                        ),
                        preliminary_company=analysis_response.company_name,
                        preliminary_signal_type=document_type,
                        preliminary_confidence=float(
                            qualification.confidence or 0.0
                        ),
                        matched_services=matched_services,
                        evidence=list(
                            qualification.evidence_quotes or []
                        ),
                        uncertainty_reasons=list(
                            qualification.rejection_reasons or []
                        ),
                        deadline_status=item["context"].get(
                            "deadline_status",
                            "unknown",
                        ),
                        deadline=item["context"].get("deadline"),
                        deadline_reason=item["context"].get(
                            "deadline_reason",
                            "",
                        ),
                        deadline_confidence=float(
                            item["context"].get(
                                "deadline_confidence",
                                0.0,
                            )
                        ),
                    )
                )

            logger.info(
                "Starting LLM validation. Candidates: %s",
                len(llm_candidates),
            )

            llm_results = self.llm_validator.validate_candidates(
                llm_candidates
            )

        llm_result_map = {
            result.candidate_id: result
            for result in llm_results
        }

        llm_rejected_count = 0
        llm_manual_review_count = 0

        for index, item in enumerate(local_shortlist):
            candidate = item["candidate"]
            qualification = item["qualification"]
            lead = item["lead"]
            analysis_response = item["analysis"]

            if analysis_response.matched_services:
                top_match = analysis_response.matched_services[0]
            else:
                fallback = item.get("fallback_service_match")

                if not fallback:
                    logger.warning(
                        "No matched or fallback service available for %s",
                        candidate.source_url,
                    )
                    continue

                top_match = SimpleNamespace(
                    service_id=fallback["service_id"],
                    service_name=fallback["service_name"],
                    similarity_percentage=0.0,
                    service_match_percentage=0.0,
                )

            llm_result = llm_result_map.get(str(index))

            if self.llm_validator is None:
                decision = LeadValidationDecision.VALID_LEAD
                validation_reason = (
                    "LLM validation was disabled; local validation was used."
                )
            elif llm_result is None:
                decision = LeadValidationDecision.MANUAL_REVIEW
                validation_reason = (
                    "LLM returned no validation result for this candidate."
                )
            else:
                decision = llm_result.decision
                validation_reason = llm_result.reason

            # Enforce region eligibility before ordinary decision routing.
            # Claude extracts the buyer country and region_status; Python
            # applies the hard business policy.
            if llm_result is not None:
                enforced_region_status = getattr(
                    llm_result,
                    "region_status",
                    "unknown",
                )
                enforced_country = (
                    getattr(
                        llm_result,
                        "canonical_country",
                        None,
                    )
                    or getattr(
                        llm_result,
                        "buyer_country_raw",
                        None,
                    )
                )

                if enforced_region_status == "unsupported":
                    decision = LeadValidationDecision.NOT_A_LEAD
                    validation_reason = (
                        "Buyer is outside the allowed service regions"
                        + (
                            f": {enforced_country}."
                            if enforced_country
                            else "."
                        )
                    )

                elif (
                    enforced_region_status == "unknown"
                    and decision == LeadValidationDecision.VALID_LEAD
                ):
                    decision = LeadValidationDecision.MANUAL_REVIEW
                    validation_reason = (
                        "Buyer country could not be established reliably."
                    )

                elif (
                    enforced_region_status == "supported"
                    and not getattr(
                        llm_result,
                        "canonical_country",
                        None,
                    )
                    and decision == LeadValidationDecision.VALID_LEAD
                ):
                    decision = LeadValidationDecision.MANUAL_REVIEW
                    validation_reason = (
                        "Claude marked the region as supported but did not "
                        "return a canonical allowed-region value."
                    )

            if decision == LeadValidationDecision.NOT_A_LEAD:
                llm_rejected_count += 1
                logger.info(
                    "❌ LLM REJECTED: %s | Reason: %s",
                    candidate.source_url,
                    validation_reason,
                )
                continue

            if decision == LeadValidationDecision.MANUAL_REVIEW:
                llm_manual_review_count += 1
                logger.info(
                    "🟠 LLM MANUAL REVIEW: %s | Reason: %s",
                    candidate.source_url,
                    validation_reason,
                )
                manual_review.append(
                    ManualReviewLead(
                        source_title=candidate.source_title or "",
                        source_url=str(candidate.source_url),
                        source_snippet=candidate.source_snippet,
                        search_query=candidate.search_query,
                        company_name=analysis_response.company_name,
                        industry=analysis_response.industry,
                        country=lead.country,
                        suggested_service_id=top_match.service_id,
                        suggested_service_name=top_match.service_name,
                        suggested_similarity=top_match.service_match_percentage,
                        review_type="llm",
                        reason=validation_reason,
                    )
                )
                continue

            # A valid lead must still have at least one service confirmed by
            # Claude. This is especially important for candidates that reached
            # Claude through originating-query fallback rather than embeddings.
            if (
                llm_result is not None
                and decision == LeadValidationDecision.VALID_LEAD
                and not llm_result.matched_service_ids
            ):
                llm_manual_review_count += 1

                logger.warning(
                    "LLM returned valid_lead without confirming "
                    "a Triway service: %s",
                    candidate.source_url,
                )

                manual_review.append(
                    ManualReviewLead(
                        source_title=candidate.source_title or "",
                        source_url=str(candidate.source_url),
                        source_snippet=candidate.source_snippet,
                        search_query=candidate.search_query,
                        company_name=analysis_response.company_name,
                        industry=analysis_response.industry,
                        country=lead.country,
                        suggested_service_id=top_match.service_id,
                        suggested_service_name=top_match.service_name,
                        suggested_similarity=(
                            top_match.service_match_percentage
                        ),
                        review_type="llm",
                        reason=(
                            "Claude considered the candidate potentially valid "
                            "but did not confirm a matching Triway service."
                        ),
                    )
                )
                continue

            fallback_service = item.get("fallback_service_match")

            if (
                fallback_service
                and llm_result is not None
                and decision == LeadValidationDecision.VALID_LEAD
            ):
                confirmed_service_ids = set(
                    llm_result.matched_service_ids
                )
                originating_service_id = (
                    fallback_service["service_id"]
                )

                if originating_service_id not in confirmed_service_ids:
                    logger.info(
                        "LLM did not confirm originating service %s "
                        "for %s. Confirmed services: %s",
                        originating_service_id,
                        candidate.source_url,
                        sorted(confirmed_service_ids),
                    )

            # Claude makes the semantic buyer-region decision. Python only
            # routes that structured decision and verifies that a supposedly
            # supported country was copied from the supplied canonical list.
            if self.llm_validator is not None:
                region_status = (
                    getattr(
                        llm_result,
                        "region_status",
                        "unknown",
                    )
                    if llm_result is not None
                    else "unknown"
                )

                canonical_country = (
                    getattr(
                        llm_result,
                        "canonical_country",
                        None,
                    )
                    if llm_result is not None
                    else None
                )

                buyer_country_raw = (
                    getattr(
                        llm_result,
                        "buyer_country_raw",
                        None,
                    )
                    if llm_result is not None
                    else None
                )

                region_evidence = (
                    getattr(
                        llm_result,
                        "region_evidence",
                        None,
                    )
                    if llm_result is not None
                    else None
                )

                if region_status == "unsupported":
                    llm_rejected_count += 1

                    logger.info(
                        "❌ REGION REJECTED: %s | Buyer country: %s | "
                        "Evidence: %s",
                        candidate.source_url,
                        buyer_country_raw
                        or canonical_country
                        or "unknown",
                        region_evidence or "none",
                    )
                    continue

                if region_status == "unknown":
                    llm_manual_review_count += 1

                    logger.info(
                        "🟠 REGION UNKNOWN — MANUAL REVIEW: %s",
                        candidate.source_url,
                    )

                    manual_review.append(
                        ManualReviewLead(
                            source_title=(
                                candidate.source_title or ""
                            ),
                            source_url=str(
                                candidate.source_url
                            ),
                            source_snippet=(
                                candidate.source_snippet
                            ),
                            search_query=(
                                candidate.search_query
                            ),
                            company_name=(
                                analysis_response.company_name
                            ),
                            industry=(
                                analysis_response.industry
                            ),
                            country=buyer_country_raw,
                            suggested_service_id=(
                                top_match.service_id
                            ),
                            suggested_service_name=(
                                top_match.service_name
                            ),
                            suggested_similarity=(
                                top_match.service_match_percentage
                            ),
                            review_type="llm",
                            reason=(
                                "Claude could not reliably establish "
                                "the buyer organization's country."
                                + (
                                    f" Evidence: {region_evidence}"
                                    if region_evidence
                                    else ""
                                )
                            ),
                        )
                    )
                    continue

                if region_status == "supported":
                    canonical_key = (
                        canonical_country.casefold()
                        if canonical_country
                        else None
                    )

                    if (
                        not canonical_key
                        or canonical_key not in self.region_names
                    ):
                        llm_manual_review_count += 1

                        logger.warning(
                            "Claude marked the region as supported but "
                            "returned an invalid canonical country: %s",
                            canonical_country,
                        )

                        manual_review.append(
                            ManualReviewLead(
                                source_title=(
                                    candidate.source_title or ""
                                ),
                                source_url=str(
                                    candidate.source_url
                                ),
                                source_snippet=(
                                    candidate.source_snippet
                                ),
                                search_query=(
                                    candidate.search_query
                                ),
                                company_name=(
                                    analysis_response.company_name
                                ),
                                industry=(
                                    analysis_response.industry
                                ),
                                country=canonical_country,
                                suggested_service_id=(
                                    top_match.service_id
                                ),
                                suggested_service_name=(
                                    top_match.service_name
                                ),
                                suggested_similarity=(
                                    top_match.service_match_percentage
                                ),
                                review_type="llm",
                                reason=(
                                    "Claude marked the buyer region as "
                                    "supported but did not return a valid "
                                    "canonical country from the supplied "
                                    "allowed-region list."
                                ),
                            )
                        )
                        continue

                    lead.country = canonical_country

                    logger.info(
                        "Buyer region validated by LLM: %s",
                        lead.country,
                    )

                else:
                    # Defensive fallback for an unexpected region_status.
                    llm_manual_review_count += 1

                    logger.warning(
                        "Unexpected LLM region status for %s: %s",
                        candidate.source_url,
                        region_status,
                    )

                    manual_review.append(
                        ManualReviewLead(
                            source_title=(
                                candidate.source_title or ""
                            ),
                            source_url=str(
                                candidate.source_url
                            ),
                            source_snippet=(
                                candidate.source_snippet
                            ),
                            search_query=(
                                candidate.search_query
                            ),
                            company_name=(
                                analysis_response.company_name
                            ),
                            industry=(
                                analysis_response.industry
                            ),
                            country=buyer_country_raw,
                            suggested_service_id=(
                                top_match.service_id
                            ),
                            suggested_service_name=(
                                top_match.service_name
                            ),
                            suggested_similarity=(
                                top_match.service_match_percentage
                            ),
                            review_type="llm",
                            reason=(
                                "The LLM returned an unsupported "
                                "region-status value."
                            ),
                        )
                    )
                    continue

            logger.info(
                "✅ LLM VALIDATED: %s | Reason: %s",
                candidate.source_url,
                validation_reason,
            )

            intelligence_report = self.intelligence_service.build_report(
                lead=lead,
                qualification=qualification,
                analysis=analysis_response,
                deadline=item["deadline"],
            )

            # DiscoveredLeadResponse.matched_services expects the full
            # ServiceMatchResponse schema produced by the embedding service.
            # The originating-service fallback is only provisional context
            # for Claude and must not be returned as a partial service match.
            response_matched_services = analysis_response.matched_services

            discovered_leads.append(
                DiscoveredLeadResponse(
                    source_title=candidate.source_title or "",
                    source_url=str(candidate.source_url),
                    source_snippet=candidate.source_snippet,
                    search_query=candidate.search_query,
                    company_name=analysis_response.company_name,
                    industry=analysis_response.industry,
                    country=lead.country,
                    matched_services=response_matched_services,
                    top_service_id=top_match.service_id,
                    top_service_name=top_match.service_name,
                    top_service_match_percentage=(
                        top_match.service_match_percentage
                    ),
                    qualification=qualification,
                    intelligence=intelligence_report,
                )
            )

        logger.info(
            "LLM validation complete. Valid: %s | Rejected: %s | "
            "LLM manual review: %s | Similarity manual review: %s",
            len(discovered_leads),
            llm_rejected_count,
            llm_manual_review_count,
            len(similarity_manual_review),
        )

        discovered_leads.sort(
            key=lambda lead: lead.top_service_match_percentage or 0.0,
            reverse=True,
        )
        discovered_leads = discovered_leads[: request.max_leads]

        return DiscoverLeadsResponse(
            queries_executed=[record["query"] for record in query_records],
            sources_collected=len(collected_candidates),
            sources_analyzed=successful_fetches,
            leads_found=len(discovered_leads),
            leads=discovered_leads,
            manual_review_count=(
                len(similarity_manual_review) + len(manual_review)
            ),
            manual_review=(
                similarity_manual_review + manual_review
            ),
            # NOTE: remove these two kwargs if DiscoverLeadsResponse doesn't
            # define them yet -- see merge notes at the top of this file.
            listing_page_rejections=listing_page_rejections,
            expired_rejections=expired_rejections,
        )