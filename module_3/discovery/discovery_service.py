from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.search.serpapi import search

from module_3.discovery.contradiction_checker import ContradictionChecker
from module_3.discovery.deadline_checker import DeadlineChecker
from module_3.discovery.document_fetcher import DocumentFetcher
from module_3.discovery.gemini_lead_validator import (
    GeminiLeadValidator,
    LeadValidationCandidate,
    LeadValidationDecision,
)
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
from module_3.discovery.listing_page_detector import ListingPageDetector

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

def build_gemini_excerpt(
    value: str | None,
    max_chars: int = 1800,
) -> str:
    """
    Build a compact Gemini excerpt that preserves both the beginning
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
) -> tuple[bool, str | None]:
    title = normalize_text(candidate.source_title)
    snippet = normalize_text(candidate.source_snippet)
    combined_search_text = f"{title} {snippet}".strip()
    candidate_url = str(candidate.source_url)
    domain = normalized_domain(candidate_url)

    # Marketplace URLs legitimately contain /jobs or related paths, so the
    # procurement/general-web path filters must not be applied to them.
    if source_type != "marketplace":
        for term in BLOCKED_TITLE_TERMS:
            if term in combined_search_text:
                return True, f"title/snippet contains blocked term '{term}'"

        for term in BLOCKED_URL_TERMS:
            if url_contains_blocked_term(candidate_url, term):
                return True, f"URL contains blocked path term '{term}'"

        for blocked_domain in BLOCKED_GENERAL_DOMAINS:
            if domain_matches(domain, blocked_domain):
                return True, f"domain '{domain}' is not an approved opportunity source"

    if platform == "freelancer":
        if not domain_matches(domain, "freelancer.com"):
            return True, "result does not belong to Freelancer"

    if platform == "peopleperhour":
        if not domain_matches(domain, "peopleperhour.com"):
            return True, "result does not belong to PeoplePerHour"

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


def validate_by_source(
    *,
    source_type: str,
    platform: str,
    title: str,
    snippet: str,
    text: str,
    url: str,
) -> tuple[bool, list[str], list[str]]:
    if source_type == "marketplace":
        return validate_marketplace_content(
            platform=platform,
            url=url,
            title=title,
            snippet=snippet,
            text=text,
        )

    if source_type == "general_web":
        return validate_partner_content(
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
        use_gemini = config.get("use_gemini", False)

        self.query_generator = QueryGenerator(
            knowledge_base_path=config.get("knowledge_base_path"),
            use_llm=False,
            llm_model=None,
            use_gemini=False,
            planner_prompt_path=config.get("planner_prompt_path"),
        )

        # Gemini can remain disabled while all deterministic pipeline edits
        # are tested. RequirementClassifier receives no remote model when off.
        self.classifier = RequirementClassifier(
            llm_model=None,
            use_gemini=False,
            max_chunks=config.get("max_chunks", 3),
        )

        # Gemini is used only as the final verifier for locally shortlisted
        # opportunities. It is deliberately not used for query generation or
        # requirement classification.
        self.gemini_validator = None

        if use_gemini and llm_model is not None:
            self.gemini_validator = GeminiLeadValidator(
                model=llm_model,
                batch_size=config.get("gemini_batch_size", 4),
                max_candidates=config.get("gemini_max_candidates"),
                max_excerpt_chars=config.get(
                    "gemini_max_excerpt_chars",
                    1800,
                ),
            )

            logger.info(
                "Gemini lead validator enabled. Batch size: %s | "
                "Max candidates: %s",
                config.get("gemini_batch_size", 4),
                config.get("gemini_max_candidates"),
            )
        else:
            logger.info(
                "Gemini lead validator disabled. "
                "Qualified candidates will be routed to manual review."
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

        self.config = config

    def discover(
        self,
        request: DiscoverLeadsRequest,
    ) -> DiscoverLeadsResponse:
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

        for record in query_records:
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

            results = search(
                record["query"],
                num_results=request.results_per_query,
            )

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

        qualified_candidates: list[dict[str, Any]] = []
        successful_fetches = 0
        failed_fetches = 0
        empty_content_count = 0
        validation_rejections = 0
        expired_rejections = 0
        gate_rejections = 0
        manual_review: list[ManualReviewLead] = []

        for candidate in collected_candidates:
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
                failed_fetches += 1
                logger.warning(
                    "Fetch failed for %s: %s",
                    candidate.source_url,
                    doc.fetch_error,
                )
                continue

            successful_fetches += 1

            if not normalize_text(doc.text):
                empty_content_count += 1
                logger.info("❌ EMPTY CONTENT: %s", candidate.source_url)
                continue

            valid, rejection_reasons, matched_terms = validate_by_source(
                source_type=context["source_type"],
                platform=context["platform"],
                title=candidate.source_title or "",
                snippet=candidate.source_snippet or "",
                text=doc.text or "",
                url=str(candidate.source_url),
            )

            if not valid:
                validation_rejections += 1
                logger.info(
                    "❌ SOURCE VALIDATION FAILED: %s | Source: %s/%s",
                    candidate.source_url,
                    context["source_type"],
                    context["platform"],
                )
                logger.info("   Reasons: %s", rejection_reasons)
                logger.info("   Matched indicators: %s", matched_terms)
                continue

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
                listing_page_rejections += 1
                logger.info(
                    "❌ LISTING PAGE REJECTED: %s | Indicators: %s",
                    candidate.source_url,
                    listing_assessment.matched_indicators,
                )
                continue

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

            if deadline_assessment.is_expired:
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
                continue

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
                    "⚠️ DEADLINE UNKNOWN — CONTINUING TO GEMINI: %s | %s",
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
                continue

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

        discovered_leads: List[DiscoveredLeadResponse] = []
        local_shortlist: list[dict[str, Any]] = []
        # Local analysis produces service evidence. It is not a rejection
        # gate; Gemini makes the final decision for every qualified candidate.
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

            lead = LeadProfile(
                company_name=metadata.get("company_name"),
                industry=metadata.get("industry"),
                country=metadata.get("country"),
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
                    "deadline_status": context.get(
                        "deadline_status",
                        "unknown",
                    ),
                    "deadline": context.get("deadline"),
                    "deadline_reason": context.get(
                        "deadline_reason",
                        "",
                    ),
                    "deadline_confidence": context.get(
                        "deadline_confidence",
                        0.0,
                    ),
                    "deadline_matched_text": context.get(
                        "deadline_matched_text",
                    ),
                },
            )

            analysis_response = self.analysis_service.analyze(
                AnalyzeLeadRequest(
                    lead=lead,
                    top_k=3,
                    minimum_similarity=0.0,
                )
            )

            top_similarity = 0.0
            if analysis_response.matched_services:
                top_match_for_evidence = analysis_response.matched_services[0]
                top_similarity = float(
                    getattr(
                        top_match_for_evidence,
                        "service_match_score",
                        getattr(
                            top_match_for_evidence,
                            "similarity_score",
                            0.0,
                        ),
                    )
                    or 0.0
                )

            similarity_uncertainty = None
            if top_similarity < request.minimum_similarity:
                similarity_uncertainty = (
                    "No service exceeded the configured similarity threshold "
                    f"of {request.minimum_similarity:.2f}. "
                    f"Best local score was {top_similarity:.4f}. "
                    "Similarity is supporting evidence only."
                )

            context = dict(context)
            context["top_similarity"] = top_similarity
            context["similarity_uncertainty"] = similarity_uncertainty

            local_shortlist.append(
                {
                    "candidate": candidate,
                    "document": doc,
                    "qualification": qualification,
                    "context": context,
                    "deadline": item["deadline"],
                    "listing": item["listing"],
                    "lead": lead,
                    "analysis": analysis_response,
                }
            )

        logger.info(
            "Local analysis complete. Analysed: %s | "
            "Sent to Gemini final validation: %s",
            len(qualified_candidates),
            len(local_shortlist),
        )

        gemini_results = []

        if self.gemini_validator and local_shortlist:
            gemini_candidates: list[LeadValidationCandidate] = []

            for index, item in enumerate(local_shortlist):
                candidate = item["candidate"]
                doc = item["document"]
                qualification = item["qualification"]
                analysis_response = item["analysis"]
                context = item["context"]

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

                document_type = getattr(
                    qualification.document_type,
                    "value",
                    str(qualification.document_type),
                )

                gemini_candidates.append(
                    LeadValidationCandidate(
                        candidate_id=str(index),
                        title=candidate.source_title or "",
                        url=str(candidate.source_url),
                        snippet=candidate.source_snippet or "",
                        content_excerpt=build_gemini_excerpt(
                            doc.text,
                            max_chars=self.config.get(
                                "gemini_max_excerpt_chars",
                                1800,
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
                        uncertainty_reasons=[
                            value
                            for value in [
                                *list(
                                    qualification.rejection_reasons or []
                                ),
                                context.get("similarity_uncertainty"),
                                (
                                    context.get("deadline_reason")
                                    if context.get("deadline_status") == "unknown"
                                    else None
                                ),
                            ]
                            if value
                        ],
                        deadline_status=context.get(
                            "deadline_status",
                            "unknown",
                        ),
                        deadline=context.get("deadline"),
                        deadline_reason=context.get(
                            "deadline_reason",
                            "",
                        ),
                        deadline_confidence=float(
                            context.get("deadline_confidence", 0.0)
                            or 0.0
                        ),
                    )
                )

            logger.info(
                "Starting Gemini validation. Candidates: %s",
                len(gemini_candidates),
            )

            gemini_results = self.gemini_validator.validate_candidates(
                gemini_candidates
            )

        gemini_result_map = {
            result.candidate_id: result
            for result in gemini_results
        }

        gemini_rejected_count = 0
        gemini_manual_review_count = 0

        for index, item in enumerate(local_shortlist):
            candidate = item["candidate"]
            qualification = item["qualification"]
            lead = item["lead"]
            analysis_response = item["analysis"]
            context = item["context"]

            gemini_result = gemini_result_map.get(str(index))

            if self.gemini_validator is None:
                decision = LeadValidationDecision.MANUAL_REVIEW
                validation_reason = (
                    "Gemini final validation was disabled, so this candidate "
                    "cannot be automatically validated."
                )
            elif gemini_result is None:
                decision = LeadValidationDecision.MANUAL_REVIEW
                validation_reason = (
                    "Gemini returned no validation result for this candidate."
                )
            else:
                decision = gemini_result.decision
                validation_reason = gemini_result.reason

            if decision == LeadValidationDecision.NOT_A_LEAD:
                gemini_rejected_count += 1
                logger.info(
                    "❌ GEMINI REJECTED: %s | Reason: %s",
                    candidate.source_url,
                    validation_reason,
                )
                continue
            if decision == LeadValidationDecision.MANUAL_REVIEW:
                gemini_manual_review_count += 1
                top_match = (
                    analysis_response.matched_services[0]
                    if analysis_response.matched_services
                    else None
                )

                logger.info(
                    "🟠 GEMINI MANUAL REVIEW: %s | Reason: %s",
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
                        country=analysis_response.country,
                        suggested_service_id=(
                            top_match.service_id if top_match else None
                        ),
                        suggested_service_name=(
                            top_match.service_name if top_match else None
                        ),
                        suggested_similarity=(
                            top_match.service_match_percentage
                            if top_match
                            else None
                        ),
                        review_type="gemini",
                        reason=validation_reason,
                        gemini_confidence=(
                            gemini_result.confidence
                            if gemini_result
                            else None
                        ),
                        deadline_status=context.get("deadline_status"),
                        deadline=context.get("deadline"),
                        best_similarity=context.get("top_similarity"),
                    )
                )
                continue

            logger.info(
                "✅ GEMINI VALIDATED: %s | Reason: %s",
                candidate.source_url,
                validation_reason,
            )

            if not analysis_response.matched_services:
                gemini_manual_review_count += 1
                manual_review.append(
                    ManualReviewLead(
                        source_title=candidate.source_title or "",
                        source_url=str(candidate.source_url),
                        source_snippet=candidate.source_snippet,
                        search_query=candidate.search_query,
                        company_name=analysis_response.company_name,
                        industry=analysis_response.industry,
                        country=analysis_response.country,
                        review_type="gemini",
                        reason=(
                            "Gemini marked the candidate valid, but local "
                            "analysis produced no service match."
                        ),
                        gemini_confidence=(
                            gemini_result.confidence
                            if gemini_result
                            else None
                        ),
                        deadline_status=context.get("deadline_status"),
                        deadline=context.get("deadline"),
                        best_similarity=context.get("top_similarity"),
                    )
                )
                continue

            intelligence_report = self.intelligence_service.build_report(
                lead=lead,
                qualification=qualification,
                analysis=analysis_response,
                deadline=item["deadline"],
            )

            top_match = analysis_response.matched_services[0]

            discovered_leads.append(
                DiscoveredLeadResponse(
                    source_title=candidate.source_title or "",
                    source_url=str(candidate.source_url),
                    source_snippet=candidate.source_snippet,
                    search_query=candidate.search_query,
                    company_name=analysis_response.company_name,
                    industry=analysis_response.industry,
                    country=analysis_response.country,
                    matched_services=analysis_response.matched_services,
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
            "Gemini final validation complete. Evaluated: %s | "
            "Valid: %s | Rejected: %s | Manual review: %s",
            len(local_shortlist),
            len(discovered_leads),
            gemini_rejected_count,
            gemini_manual_review_count,
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
            manual_review_count=len(manual_review),
            manual_review=manual_review,
            listing_page_rejections=listing_page_rejections,
            expired_rejections=expired_rejections,
            qualification_rejections=gate_rejections,
            gemini_evaluated=len(local_shortlist),
            gemini_validated=len(discovered_leads),
            gemini_rejected=gemini_rejected_count,
            gemini_manual_review=gemini_manual_review_count,
        )