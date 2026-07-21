from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlsplit


logger = logging.getLogger(__name__)


class BuyingSignalType(str, Enum):
    OPEN_PROCUREMENT = "open_procurement"
    PARTNERSHIP_SEARCH = "partnership_search"
    TRANSFORMATION_ANNOUNCEMENT = "transformation_announcement"
    TECHNOLOGY_MIGRATION = "technology_migration"
    BUDGET_APPROVAL = "budget_approval"
    EXECUTIVE_HIRING = "executive_hiring"
    TECHNICAL_HIRING = "technical_hiring"
    TECHNOLOGY_PRESSURE = "technology_pressure"
    DATA_CENTER_EXIT = "data_center_exit"
    COMPLETED_PROJECT = "completed_project"
    CONTRACT_AWARD = "contract_award"
    VENDOR_MARKETING = "vendor_marketing"
    EDUCATIONAL_CONTENT = "educational_content"
    IRRELEVANT = "irrelevant"


class ContentType(str, Enum):
    PROCUREMENT = "procurement"
    BUYER_EVENT = "buyer_event"
    COMPANY_ANNOUNCEMENT = "company_announcement"
    EXECUTIVE_JOB_POST = "executive_job_post"
    TECHNICAL_JOB_POST = "technical_job_post"
    RECRUITMENT_AGGREGATOR = "recruitment_aggregator"
    CONTRACT_AWARD = "contract_award"
    COMPLETED_PROJECT = "completed_project"
    VENDOR_MARKETING = "vendor_marketing"
    CASE_STUDY = "case_study"
    RESEARCH = "research"
    EDUCATIONAL_ARTICLE = "educational_article"
    WEBINAR_OR_EVENT = "webinar_or_event"
    SOCIAL_POST = "social_post"
    VIDEO = "video"
    NEWS_REPORT = "news_report"
    UNKNOWN = "unknown"


@dataclass
class BuyingSignalResult:
    signal_type: BuyingSignalType
    signal_strength: float
    is_buyer_signal: bool
    company_name: str | None = None
    evidence: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    is_historical: bool = False
    is_actionable: bool = False
    confidence: float = 0.0
    rejection_reasons: list[str] = field(default_factory=list)
    raw_llm_response: str | None = None
    content_type: ContentType = ContentType.UNKNOWN
    buyer_event_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["signal_type"] = self.signal_type.value
        result["content_type"] = self.content_type.value
        return result


SIGNAL_RULES: dict[BuyingSignalType, dict[str, Any]] = {
    BuyingSignalType.OPEN_PROCUREMENT: {
        "weight": 1.00,
        "terms": {
            "request for proposal", "request for quotation",
            "request for information", "request for tender",
            "rfp", "rfq", "rfi", "itt",
            "tender", "solicitation", "procurement",
            "invitation to tender", "invitation to bid",
            "invites proposals", "seeking proposals",
            "requesting proposals", "requesting bids",
            "procurement notice", "solicitation notice",
            "tender notice", "bid notice",
            "submission deadline", "proposal deadline",
            "bid deadline", "tender deadline", "closing date",
            "scope of work", "terms of reference",
            "contracting authority", "procuring entity",
            "procurement identifier", "tender reference",
        },
    },
    BuyingSignalType.PARTNERSHIP_SEARCH: {
        "weight": 0.90,
        "terms": {
            "seeking implementation partner",
            "seeking technology partner", "seeking strategic partner",
            "seeking vendor", "looking for a partner",
            "looking for a vendor", "partner selection",
            "vendor selection", "inviting technology partners",
        },
    },
    BuyingSignalType.BUDGET_APPROVAL: {
        "weight": 0.85,
        "terms": {
            "budget approved", "approved funding", "funding approved",
            "capital expenditure approved", "investment approved",
            "allocated budget", "modernization budget",
            "modernisation budget", "transformation budget",
        },
    },
    BuyingSignalType.DATA_CENTER_EXIT: {
        "weight": 0.83,
        "terms": {
            "data center exit", "data centre exit",
            "closing data center", "closing data centre",
            "data center consolidation", "data centre consolidation",
            "exit on-premises", "exit on premises",
        },
    },
    BuyingSignalType.TECHNOLOGY_PRESSURE: {
        "weight": 0.80,
        "terms": {
            "end of support", "end-of-support", "end of life",
            "end-of-life", "unsupported platform",
            "legacy infrastructure", "licensing changes",
            "technical debt", "security exposure",
        },
    },
    BuyingSignalType.TECHNOLOGY_MIGRATION: {
        "weight": 0.78,
        "terms": {
            "cloud migration", "migrating to cloud",
            "migration program", "migration programme",
            "platform migration", "application migration",
            "infrastructure migration", "core system migration",
            "moving workloads", "transitioning workloads",
            "modernization program", "modernisation programme",
        },
    },
    BuyingSignalType.TRANSFORMATION_ANNOUNCEMENT: {
        "weight": 0.72,
        "terms": {
            "digital transformation programme",
            "digital transformation program",
            "transformation initiative", "modernization initiative",
            "modernisation initiative", "cloud-first strategy",
            "technology transformation programme",
            "technology transformation program",
            "infrastructure modernization initiative",
            "infrastructure modernisation initiative",
        },
    },
    BuyingSignalType.EXECUTIVE_HIRING: {
        "weight": 0.68,
        "terms": {
            "head of cloud", "head of transformation",
            "director of cloud", "director of transformation",
            "chief digital officer", "chief technology officer",
            "cloud program director", "cloud programme director",
            "transformation director", "migration program manager",
            "migration programme manager", "vp of transformation",
            "vice president of transformation",
        },
    },
    BuyingSignalType.TECHNICAL_HIRING: {
        "weight": 0.42,
        "terms": {
            "cloud architect", "migration architect", "cloud engineer",
            "platform engineer", "devops engineer",
            "site reliability engineer", "implementation consultant",
            "data migration consultant", "migration specialist",
        },
    },
    BuyingSignalType.CONTRACT_AWARD: {
        "weight": 0.35,
        "terms": {
            "contract awarded", "contract award", "award notice",
            "selected supplier", "selected vendor", "awarded to",
            "won the tender", "tender won by",
        },
    },
    BuyingSignalType.COMPLETED_PROJECT: {
        "weight": 0.25,
        "terms": {
            "successfully completed", "completed migration",
            "project completed", "went live", "go-live completed",
            "implementation completed", "success story",
            "customer story", "case study",
        },
    },
}


CONTENT_MARKERS: dict[ContentType, set[str]] = {
    ContentType.CASE_STUDY: {
        "case study", "customer story", "success story",
        "client story", "our work", "project spotlight",
    },
    ContentType.RESEARCH: {
        "research report", "accepted paper", "conference paper",
        "academic paper", "journal article", "white paper",
        "whitepaper", "key insights", "study investigates",
        "we investigate", "research findings",
    },
    ContentType.EDUCATIONAL_ARTICLE: {
        "how to", "best practices", "guide to", "complete guide",
        "tutorial", "tips for", "what is", "benefits of",
        "challenges of", "smart strategies", "essential steps",
        "key considerations", "everything you need to know",
    },
    ContentType.WEBINAR_OR_EVENT: {
        "webinar", "register now", "on-demand webinar",
        "watch on demand", "upcoming event", "conference session",
        "panel discussion", "a discussion on", "event recording",
    },
    ContentType.VENDOR_MARKETING: {
        "our services", "contact us", "book a demo", "request a demo",
        "we help companies", "our cloud migration services",
        "our consulting services", "why choose us",
        "managed services provider", "technology solutions provider",
        "talk to an expert", "get started today",
    },
}


HISTORICAL_TERMS = {
    "contract awarded", "contract award", "award notice",
    "selected supplier", "selected vendor", "awarded to",
    "won the tender", "tender won by", "completed migration",
    "project completed", "successfully completed", "went live",
    "go-live completed", "implementation completed", "case study",
    "customer story", "success story", "previously completed",
}


INACTIVE_TERMS = HISTORICAL_TERMS | {
    "cancelled", "canceled", "closed", "expired",
    "deadline has passed", "no longer accepting",
    "procurement completed", "withdrawn", "archived tender",
}


ACTION_VERBS = {
    "announced", "announces", "plans", "planning", "launched",
    "launches", "began", "begins", "started", "starts",
    "is migrating", "will migrate", "is moving", "will move",
    "is modernizing", "is modernising", "will modernize",
    "will modernise", "approved", "allocated", "funded",
    "is seeking", "seeks", "invites", "issued", "published",
    "is replacing", "will replace", "is consolidating",
    "will consolidate", "is transitioning", "will transition",
}


PROJECT_TERMS = {
    "migration", "modernization", "modernisation", "transformation",
    "implementation", "upgrade", "replacement", "cloud adoption",
    "data center exit", "data centre exit", "consolidation",
    "core banking", "erp", "crm", "cybersecurity", "automation",
    "platform programme", "platform program", "technology initiative",
}


ORG_SUFFIXES = {
    "bank", "university", "college", "ministry", "authority",
    "government", "department", "agency", "council", "group",
    "corporation", "company", "holdings", "airways", "airlines",
    "hospital", "health", "insurance", "telecom", "telecommunications",
    "energy", "airport", "ports", "port", "municipality", "limited",
    "ltd", "llc", "inc", "plc", "pjsc", "foundation", "association",
}


JOB_AGGREGATOR_DOMAINS = {
    "indeed.com", "glassdoor.com", "ziprecruiter.com", "jobtarget.com",
    "jobgether.com", "haystackapp.io", "jaabz.com", "adzuna.com",
    "monster.com", "bebee.com", "salesjobs.com", "tealhq.com",
    "lever.co", "greenhouse.io", "smartrecruiters.com",
}


SOCIAL_DOMAINS = {
    "linkedin.com", "facebook.com", "x.com", "twitter.com",
    "reddit.com", "medium.com",
}


VIDEO_DOMAINS = {"youtube.com", "youtu.be", "vimeo.com"}


TITLE_LIKE_PREFIXES = {
    "how ", "why ", "what ", "when ", "where ", "nine ", "ten ",
    "top ", "best ", "guide ", "overview", "partner program",
    "digital transformation:", "cloud migration:", "application modernization",
    "scaling geopolitics", "research", "webinar", "conference",
}


PROCUREMENT_BUYER_PATTERNS = [
    re.compile(
        r"\bcontracting authority\s*[:\-]\s*([^\n|]{3,120})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bprocuring entit(?:y|ies)\s*[:\-]\s*([^\n|]{3,120})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbuyer(?: name)?\s*[:\-]\s*([^\n|]{3,120})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\borganisation name\s*[:\-]\s*([^\n|]{3,120})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\borganization name\s*[:\-]\s*([^\n|]{3,120})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bissued by\s+([A-Z][A-Za-z0-9&.,'’()\- ]{3,100})",
    ),
    re.compile(
        r"\bpublished by\s+([A-Z][A-Za-z0-9&.,'’()\- ]{3,100})",
    ),
    re.compile(
        r"\bprocurement(?: identifier)?[^.\n]{0,120}?\bfor\s+"
        r"([A-Z][A-Za-z0-9&.,'’()\- ]{3,100})",
        re.IGNORECASE,
    ),
]


PROCUREMENT_PATH_MARKERS = {
    "/procurement", "/tender", "/tenders", "/solicitation",
    "/solicitations", "/rfp", "/rfq", "/eprocure", "/epublish",
}


PROCUREMENT_FIELD_KEYS = (
    "buyer_name", "buyer", "procuring_entity", "contracting_authority",
    "issuer", "authority", "organisation_name", "organization_name",
    "department_name", "agency_name", "company_name",
)


COMPANY_PATTERNS = [
    re.compile(
        r"\b([A-Z][A-Za-z0-9&.,'’\- ]{2,80}?)\s+"
        r"(?:announced|announces|plans to|has launched|has begun|"
        r"will migrate|is migrating|is moving|will move|is seeking|seeks)\b"
    ),
    re.compile(
        r"\b(?:by|for|from)\s+([A-Z][A-Za-z0-9&.,'’\- ]{2,80}?)\s+"
        r"(?:announced|launches|begins|starts|seeks|plans|issued|published)\b"
    ),
]


class BuyingSignalClassifier:
    """Classify a page as a current, organisation-specific buying signal."""

    def __init__(
        self,
        llm_model=None,
        use_llm: bool = False,
        minimum_signal_strength: float = 0.45,
    ):
        self.llm_model = llm_model
        # Gemini/LLM validation is intentionally disabled in this layer.
        # The argument is retained so existing startup code remains compatible.
        self.use_llm = False
        self.minimum_signal_strength = max(
            0.0, min(float(minimum_signal_strength), 1.0)
        )
        self.llm_disabled_for_run = False

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    @classmethod
    def _lower_text(cls, value: str | None) -> str:
        return cls._normalize_text(value).lower()

    @staticmethod
    def _domain(metadata: dict[str, Any] | None) -> str:
        url = str((metadata or {}).get("source_url") or "")
        return urlsplit(url).netloc.lower().removeprefix("www.")

    @staticmethod
    def _domain_matches(domain: str, known: set[str]) -> bool:
        return any(domain == item or domain.endswith(f".{item}") for item in known)

    @classmethod
    def _clean_company_name(cls, value: str | None) -> str | None:
        if not value:
            return None

        cleaned = re.sub(r"\s+", " ", str(value)).strip(" \t\r\n,.;:-|–—")
        lower = cleaned.lower()

        if not cleaned or len(cleaned) < 2 or len(cleaned) > 100:
            return None

        blocked_exact = {
            "the company", "the organization", "the organisation",
            "the government", "the university", "the bank", "the agency",
            "partner program", "overview", "home", "careers", "jobs",
            "digital transformation", "cloud migration",
        }
        if lower in blocked_exact:
            return None

        if any(lower.startswith(prefix) for prefix in TITLE_LIKE_PREFIXES):
            return None

        if any(token in lower for token in (
            " - jobs", " at jobgether", "find a tender", "research report",
            "accepted paper", "case study", "customer story", "#shorts",
            "posted this", "key insights", "smart strategies",
        )):
            return None

        word_count = len(cleaned.split())
        if word_count > 10:
            return None

        return cleaned

    @classmethod
    def _looks_like_organisation(cls, value: str | None) -> bool:
        cleaned = cls._clean_company_name(value)
        if not cleaned:
            return False

        lower = cleaned.lower()
        words = re.findall(r"[a-z0-9]+", lower)
        if not words:
            return False

        if any(word in ORG_SUFFIXES for word in words):
            return True

        # Brand-style names are allowed when concise and title-cased.
        if len(words) <= 5 and re.search(r"[A-Z]", cleaned):
            return True

        return False

    @classmethod
    def _clean_procurement_company_name(
        cls,
        value: str | None,
    ) -> str | None:
        if not value:
            return None

        cleaned = cls._normalize_text(str(value))
        cleaned = re.sub(
            r"^(?:procurement|tender|solicitation|rfp|rfq|rfi)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" -:|,.;")

        # Prefer the organisation-looking part of composite titles.
        parts = [
            part.strip(" -:|,.;")
            for part in re.split(r"\s*(?:\||\s+-\s+|–|—)\s*", cleaned)
            if part.strip(" -:|,.;")
        ]

        organisation_parts = [
            part
            for part in parts
            if cls._looks_like_organisation(part)
        ]
        if organisation_parts:
            return cls._clean_company_name(organisation_parts[-1])

        # Common compact scrape artefact:
        # "ProcurementFinance System Cloud Migration - Cardiff Metropolitan University"
        suffix_pattern = "|".join(
            sorted((re.escape(x) for x in ORG_SUFFIXES), key=len, reverse=True)
        )
        matches = list(
            re.finditer(
                rf"\b([A-Z][A-Za-z0-9&.,'’()\- ]{{2,90}}?"
                rf"(?:{suffix_pattern}))\b",
                cleaned,
                flags=re.IGNORECASE,
            )
        )
        for match in reversed(matches):
            candidate = cls._clean_company_name(match.group(1))
            if candidate and cls._looks_like_organisation(candidate):
                return candidate

        return cls._clean_company_name(cleaned)

    def _extract_procurement_buyer(
        self,
        *,
        title: str,
        snippet: str,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> str | None:
        metadata = metadata or {}

        for key in PROCUREMENT_FIELD_KEYS:
            candidate = self._clean_procurement_company_name(metadata.get(key))
            if candidate and self._looks_like_organisation(candidate):
                return candidate

        combined = "\n".join(
            part for part in (title, snippet, text[:12000]) if part
        )

        for pattern in PROCUREMENT_BUYER_PATTERNS:
            match = pattern.search(combined)
            if not match:
                continue
            candidate = self._clean_procurement_company_name(match.group(1))
            if candidate and self._looks_like_organisation(candidate):
                return candidate

        # Tender titles often use "... | Buyer" or "... - Buyer".
        for value in (title, snippet):
            candidate = self._clean_procurement_company_name(value)
            if candidate and self._looks_like_organisation(candidate):
                return candidate

        return self._extract_company_name(
            title=title,
            snippet=snippet,
            text=text,
            metadata=metadata,
        )

    @staticmethod
    def _has_procurement_language(
        combined_lower: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        metadata = metadata or {}
        url = str(metadata.get("source_url") or metadata.get("url") or "")
        path = urlsplit(url).path.lower()

        terms = SIGNAL_RULES[BuyingSignalType.OPEN_PROCUREMENT]["terms"]
        if any(term in combined_lower for term in terms):
            return True

        if any(marker in path for marker in PROCUREMENT_PATH_MARKERS):
            return True

        return any(metadata.get(key) for key in (
            "procurement_id", "tender_id", "solicitation_id",
            "contracting_authority", "procuring_entity", "buyer_name",
        ))

    @classmethod
    def _has_award_language(cls, combined_lower: str) -> bool:
        return any(
            term in combined_lower
            for term in SIGNAL_RULES[BuyingSignalType.CONTRACT_AWARD]["terms"]
        )

    @classmethod
    def _has_completed_language(cls, combined_lower: str) -> bool:
        return any(
            term in combined_lower
            for term in SIGNAL_RULES[BuyingSignalType.COMPLETED_PROJECT]["terms"]
        )

    def _extract_company_name(
        self,
        *,
        title: str,
        snippet: str,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> str | None:
        metadata = metadata or {}

        for key in (
            "company_name", "organization_name", "organisation_name",
            "buyer_name", "procuring_entity", "contracting_authority",
        ):
            candidate = self._clean_company_name(metadata.get(key))
            if candidate and self._looks_like_organisation(candidate):
                return candidate

        combined = "\n".join(part for part in (title, snippet, text[:6000]) if part)
        for pattern in COMPANY_PATTERNS:
            match = pattern.search(combined)
            if match:
                candidate = self._clean_company_name(match.group(1))
                if candidate and self._looks_like_organisation(candidate):
                    return candidate

        # Extract names that end in an organisation suffix.
        suffix_pattern = "|".join(sorted((re.escape(x) for x in ORG_SUFFIXES), key=len, reverse=True))
        org_pattern = re.compile(
            rf"\b([A-Z][A-Za-z0-9&.,'’\- ]{{1,65}}?\s+(?:{suffix_pattern}))\b",
            re.IGNORECASE,
        )
        for match in org_pattern.finditer(combined):
            candidate = self._clean_company_name(match.group(1))
            if candidate and self._looks_like_organisation(candidate):
                return candidate

        return None

    @staticmethod
    def _match_terms(text: str, terms: set[str]) -> list[str]:
        return sorted({term for term in terms if term in text})

    @classmethod
    def _extract_evidence(
        cls,
        text: str,
        matched_terms: list[str],
        maximum: int = 3,
    ) -> list[str]:
        if not text or not matched_terms:
            return []

        sentences = re.split(r"(?<=[.!?])\s+|\n+", cls._normalize_text(text))
        evidence: list[str] = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(term in lower for term in matched_terms):
                cleaned = sentence.strip()
                if cleaned and cleaned not in evidence:
                    evidence.append(cleaned[:500])
            if len(evidence) >= maximum:
                break
        return evidence

    def _detect_content_type(
        self,
        *,
        title: str,
        snippet: str,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> ContentType:
        combined = self._lower_text("\n".join((title, snippet, text[:12000])))
        domain = self._domain(metadata)
        path = urlsplit(str((metadata or {}).get("source_url") or "")).path.lower()

        if self._domain_matches(domain, VIDEO_DOMAINS):
            return ContentType.VIDEO
        if self._domain_matches(domain, SOCIAL_DOMAINS):
            return ContentType.SOCIAL_POST
        if self._domain_matches(domain, JOB_AGGREGATOR_DOMAINS):
            return ContentType.RECRUITMENT_AGGREGATOR

        if any(term in combined for term in HISTORICAL_TERMS):
            if any(term in combined for term in SIGNAL_RULES[BuyingSignalType.CONTRACT_AWARD]["terms"]):
                return ContentType.CONTRACT_AWARD
            if any(term in combined for term in CONTENT_MARKERS[ContentType.CASE_STUDY]):
                return ContentType.CASE_STUDY
            return ContentType.COMPLETED_PROJECT

        if self._has_procurement_language(combined, metadata):
            return ContentType.PROCUREMENT

        if any(marker in path for marker in ("/research", "/paper", "/papers", "/journal")) or any(
            term in combined for term in CONTENT_MARKERS[ContentType.RESEARCH]
        ):
            return ContentType.RESEARCH

        if any(marker in path for marker in ("/case-study", "/case-studies", "/customer-stories")) or any(
            term in combined for term in CONTENT_MARKERS[ContentType.CASE_STUDY]
        ):
            return ContentType.CASE_STUDY

        if any(marker in path for marker in ("/webinar", "/event", "/events")) or any(
            term in combined for term in CONTENT_MARKERS[ContentType.WEBINAR_OR_EVENT]
        ):
            return ContentType.WEBINAR_OR_EVENT

        job_words = {"job", "jobs", "career", "careers", "we are hiring", "apply now", "the role"}
        if any(word in combined for word in job_words) or any(marker in path for marker in ("/job", "/jobs", "/career", "/careers")):
            if any(term in combined for term in SIGNAL_RULES[BuyingSignalType.EXECUTIVE_HIRING]["terms"]):
                return ContentType.EXECUTIVE_JOB_POST
            return ContentType.TECHNICAL_JOB_POST

        if any(term in combined for term in CONTENT_MARKERS[ContentType.VENDOR_MARKETING]):
            return ContentType.VENDOR_MARKETING

        if any(term in combined for term in CONTENT_MARKERS[ContentType.EDUCATIONAL_ARTICLE]):
            return ContentType.EDUCATIONAL_ARTICLE

        if any(verb in combined for verb in ACTION_VERBS) and any(term in combined for term in PROJECT_TERMS):
            return ContentType.COMPANY_ANNOUNCEMENT

        if "/news/" in path or "/story/" in path or "press release" in combined:
            return ContentType.NEWS_REPORT

        return ContentType.UNKNOWN

    @classmethod
    def _buyer_event_sentences(cls, combined: str, company_name: str | None) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", cls._normalize_text(combined))
        matches: list[str] = []

        for sentence in sentences:
            lower = sentence.lower()
            has_action = any(verb in lower for verb in ACTION_VERBS)
            has_project = any(term in lower for term in PROJECT_TERMS)
            mentions_company = bool(company_name and company_name.lower() in lower)

            if has_action and has_project and (mentions_company or cls._looks_like_event_subject(sentence)):
                matches.append(sentence[:500])

        return matches[:3]

    @staticmethod
    def _looks_like_event_subject(sentence: str) -> bool:
        # A conservative proxy: an action sentence begins with a capitalised
        # named subject rather than generic phrases such as "businesses".
        match = re.match(r"^([A-Z][A-Za-z0-9&.,'’\- ]{2,70})\s+", sentence.strip())
        if not match:
            return False
        subject = match.group(1).strip().lower()
        generic = {
            "businesses", "companies", "organizations", "organisations",
            "enterprises", "teams", "customers", "users", "this article",
            "the report", "the study", "we", "our team",
        }
        return subject not in generic

    @staticmethod
    def _extract_years(text: str) -> list[int]:
        return [int(year) for year in re.findall(r"\b(20\d{2})\b", text)]

    def _is_stale(self, combined_lower: str, content_type: ContentType) -> bool:
        current_year = datetime.now(timezone.utc).year
        years = self._extract_years(combined_lower)

        # Explicit lifecycle language is stronger than dates.
        if any(term in combined_lower for term in {
            "cancelled", "canceled", "expired",
            "deadline has passed", "no longer accepting",
            "procurement completed", "withdrawn", "archived tender",
        }):
            return True

        # Avoid treating generic navigation words such as "closed" as proof
        # unless they appear in an opportunity lifecycle phrase.
        if any(phrase in combined_lower for phrase in {
            "tender closed", "procurement closed", "solicitation closed",
            "bidding closed", "submissions closed", "opportunity closed",
            "status: closed", "status closed",
        }):
            return True

        if self._has_award_language(combined_lower):
            return True

        if self._has_completed_language(combined_lower):
            return True

        # A source whose newest explicit year is older than the previous year
        # is normally stale for procurement, hiring and announcements.
        if years and max(years) < current_year - 1 and content_type in {
            ContentType.PROCUREMENT,
            ContentType.COMPANY_ANNOUNCEMENT,
            ContentType.EXECUTIVE_JOB_POST,
            ContentType.TECHNICAL_JOB_POST,
            ContentType.NEWS_REPORT,
        }:
            return True

        return False

    def _reject(
        self,
        *,
        signal_type: BuyingSignalType,
        content_type: ContentType,
        company_name: str | None,
        reasons: list[str],
        matched_terms: list[str] | None = None,
        confidence: float = 0.9,
        historical: bool = False,
    ) -> BuyingSignalResult:
        return BuyingSignalResult(
            signal_type=signal_type,
            signal_strength=0.0,
            is_buyer_signal=False,
            company_name=company_name,
            matched_terms=matched_terms or [],
            is_historical=historical,
            is_actionable=False,
            confidence=confidence,
            rejection_reasons=reasons,
            content_type=content_type,
            buyer_event_detected=False,
        )

    def _classify_procurement(
        self,
        *,
        title: str,
        snippet: str,
        text: str,
        metadata: dict[str, Any] | None,
        combined: str,
        combined_lower: str,
        expected_signal_type: str | None,
    ) -> BuyingSignalResult:
        metadata = metadata or {}
        buyer_name = self._extract_procurement_buyer(
            title=title,
            snippet=snippet,
            text=text,
            metadata=metadata,
        )

        procurement_terms = self._match_terms(
            combined_lower,
            SIGNAL_RULES[BuyingSignalType.OPEN_PROCUREMENT]["terms"],
        )
        award_terms = self._match_terms(
            combined_lower,
            SIGNAL_RULES[BuyingSignalType.CONTRACT_AWARD]["terms"],
        )
        completed_terms = self._match_terms(
            combined_lower,
            SIGNAL_RULES[BuyingSignalType.COMPLETED_PROJECT]["terms"],
        )

        if award_terms:
            return self._reject(
                signal_type=BuyingSignalType.CONTRACT_AWARD,
                content_type=ContentType.CONTRACT_AWARD,
                company_name=buyer_name,
                reasons=["The contract has already been awarded."],
                matched_terms=award_terms,
                confidence=0.95,
                historical=True,
            )

        if completed_terms:
            return self._reject(
                signal_type=BuyingSignalType.COMPLETED_PROJECT,
                content_type=ContentType.COMPLETED_PROJECT,
                company_name=buyer_name,
                reasons=["The procurement or implementation appears completed."],
                matched_terms=completed_terms,
                confidence=0.92,
                historical=True,
            )

        if self._is_stale(combined_lower, ContentType.PROCUREMENT):
            return self._reject(
                signal_type=BuyingSignalType.OPEN_PROCUREMENT,
                content_type=ContentType.PROCUREMENT,
                company_name=buyer_name,
                reasons=[
                    "The procurement appears closed, expired, cancelled or too old to be actionable."
                ],
                matched_terms=procurement_terms,
                confidence=0.90,
                historical=True,
            )

        if not buyer_name:
            return self._reject(
                signal_type=BuyingSignalType.OPEN_PROCUREMENT,
                content_type=ContentType.PROCUREMENT,
                company_name=None,
                reasons=[
                    "A procurement signal was found, but the procuring organisation could not be identified."
                ],
                matched_terms=procurement_terms,
                confidence=0.74,
            )

        # A published procurement notice is itself a concrete buyer event.
        strength = 1.0
        if expected_signal_type == BuyingSignalType.OPEN_PROCUREMENT.value:
            strength = 1.0

        evidence = self._extract_evidence(
            combined,
            procurement_terms or [
                "request for proposal", "tender", "solicitation", "procurement"
            ],
            maximum=3,
        )
        if not evidence:
            evidence = [
                self._normalize_text(part)[:500]
                for part in (title, snippet)
                if self._normalize_text(part)
            ][:3]

        return BuyingSignalResult(
            signal_type=BuyingSignalType.OPEN_PROCUREMENT,
            signal_strength=round(strength, 4),
            is_buyer_signal=True,
            company_name=buyer_name,
            evidence=evidence,
            matched_terms=procurement_terms,
            is_historical=False,
            is_actionable=True,
            confidence=0.92,
            rejection_reasons=[],
            content_type=ContentType.PROCUREMENT,
            buyer_event_detected=True,
        )

    def _rule_based_classify(
        self,
        *,
        title: str,
        snippet: str,
        text: str,
        expected_signal_type: str | None,
        metadata: dict[str, Any] | None,
    ) -> BuyingSignalResult:
        metadata = dict(metadata or {})
        metadata.setdefault("source_url", metadata.get("url"))

        combined = "\n".join(part for part in (title, snippet, text) if part)
        combined_lower = self._lower_text(combined)
        content_type = self._detect_content_type(
            title=title, snippet=snippet, text=text, metadata=metadata
        )

        # Lifecycle states always take priority.
        if self._has_award_language(combined_lower):
            buyer_name = self._extract_procurement_buyer(
                title=title, snippet=snippet, text=text, metadata=metadata
            )
            return self._reject(
                signal_type=BuyingSignalType.CONTRACT_AWARD,
                content_type=ContentType.CONTRACT_AWARD,
                company_name=buyer_name,
                reasons=["The contract has already been awarded."],
                matched_terms=self._match_terms(
                    combined_lower,
                    SIGNAL_RULES[BuyingSignalType.CONTRACT_AWARD]["terms"],
                ),
                confidence=0.95,
                historical=True,
            )

        if self._has_completed_language(combined_lower):
            company_name = self._extract_company_name(
                title=title, snippet=snippet, text=text, metadata=metadata
            )
            return self._reject(
                signal_type=BuyingSignalType.COMPLETED_PROJECT,
                content_type=ContentType.COMPLETED_PROJECT,
                company_name=company_name,
                reasons=["The project appears completed or historical."],
                matched_terms=self._match_terms(
                    combined_lower,
                    SIGNAL_RULES[BuyingSignalType.COMPLETED_PROJECT]["terms"],
                ),
                confidence=0.92,
                historical=True,
            )

        # Procurement must be classified before migration/general signals.
        if self._has_procurement_language(combined_lower, metadata):
            return self._classify_procurement(
                title=title,
                snippet=snippet,
                text=text,
                metadata=metadata,
                combined=combined,
                combined_lower=combined_lower,
                expected_signal_type=expected_signal_type,
            )

        company_name = self._extract_company_name(
            title=title, snippet=snippet, text=text, metadata=metadata
        )

        hard_reject_types = {
            ContentType.RESEARCH,
            ContentType.EDUCATIONAL_ARTICLE,
            ContentType.WEBINAR_OR_EVENT,
            ContentType.VIDEO,
            ContentType.SOCIAL_POST,
            ContentType.CASE_STUDY,
            ContentType.VENDOR_MARKETING,
            ContentType.RECRUITMENT_AGGREGATOR,
            ContentType.TECHNICAL_JOB_POST,
        }

        reject_signal_map = {
            ContentType.RESEARCH: BuyingSignalType.EDUCATIONAL_CONTENT,
            ContentType.EDUCATIONAL_ARTICLE: BuyingSignalType.EDUCATIONAL_CONTENT,
            ContentType.WEBINAR_OR_EVENT: BuyingSignalType.EDUCATIONAL_CONTENT,
            ContentType.VIDEO: BuyingSignalType.EDUCATIONAL_CONTENT,
            ContentType.SOCIAL_POST: BuyingSignalType.IRRELEVANT,
            ContentType.CASE_STUDY: BuyingSignalType.COMPLETED_PROJECT,
            ContentType.VENDOR_MARKETING: BuyingSignalType.VENDOR_MARKETING,
            ContentType.RECRUITMENT_AGGREGATOR: BuyingSignalType.TECHNICAL_HIRING,
            ContentType.TECHNICAL_JOB_POST: BuyingSignalType.TECHNICAL_HIRING,
        }

        if content_type in hard_reject_types:
            reason = {
                ContentType.RESEARCH: "Research or academic analysis is not a current buyer event.",
                ContentType.EDUCATIONAL_ARTICLE: "Generic educational or thought-leadership content is not a buyer event.",
                ContentType.WEBINAR_OR_EVENT: "A webinar or event page is not an organisation-specific buying opportunity.",
                ContentType.VIDEO: "Video content is not accepted as a sales-qualified buyer signal.",
                ContentType.SOCIAL_POST: "Social posts are too weak and ambiguous for automatic lead qualification.",
                ContentType.CASE_STUDY: "Case studies describe completed or historical work.",
                ContentType.VENDOR_MARKETING: "Vendor marketing content describes services rather than buyer demand.",
                ContentType.RECRUITMENT_AGGREGATOR: "Anonymous or aggregator job listings are not sales-qualified buyer opportunities.",
                ContentType.TECHNICAL_JOB_POST: "A technical vacancy alone is too weak to qualify as a sales lead.",
            }[content_type]
            return self._reject(
                signal_type=reject_signal_map[content_type],
                content_type=content_type,
                company_name=company_name,
                reasons=[reason],
                historical=content_type in {ContentType.CASE_STUDY},
            )

        if content_type == ContentType.CONTRACT_AWARD:
            return self._reject(
                signal_type=BuyingSignalType.CONTRACT_AWARD,
                content_type=content_type,
                company_name=company_name,
                reasons=["The contract has already been awarded."],
                historical=True,
                matched_terms=self._match_terms(
                    combined_lower,
                    SIGNAL_RULES[BuyingSignalType.CONTRACT_AWARD]["terms"],
                ),
            )

        if content_type == ContentType.COMPLETED_PROJECT:
            return self._reject(
                signal_type=BuyingSignalType.COMPLETED_PROJECT,
                content_type=content_type,
                company_name=company_name,
                reasons=["The project appears completed or historical."],
                historical=True,
            )

        stale = self._is_stale(combined_lower, content_type)
        if stale:
            return self._reject(
                signal_type=(
                    BuyingSignalType.OPEN_PROCUREMENT
                    if content_type == ContentType.PROCUREMENT
                    else BuyingSignalType.IRRELEVANT
                ),
                content_type=content_type,
                company_name=company_name,
                reasons=["The source appears closed, awarded, completed, expired or too old to be actionable."],
                historical=True,
            )

        scores: list[tuple[BuyingSignalType, float, list[str]]] = []
        for signal_type, rule in SIGNAL_RULES.items():
            matched = self._match_terms(combined_lower, rule["terms"])
            if not matched:
                continue
            score = min(1.0, float(rule["weight"]) + min(0.16, 0.04 * (len(matched) - 1)))
            if expected_signal_type and signal_type.value == expected_signal_type:
                score = min(1.0, score + 0.03)
            scores.append((signal_type, score, matched))

        if not scores:
            return self._reject(
                signal_type=BuyingSignalType.IRRELEVANT,
                content_type=content_type,
                company_name=company_name,
                reasons=["No supported buying signal was detected."],
                confidence=0.7,
            )

        selected_type, selected_score, matched_terms = max(scores, key=lambda item: item[1])

        if selected_type in {BuyingSignalType.CONTRACT_AWARD, BuyingSignalType.COMPLETED_PROJECT}:
            return self._reject(
                signal_type=selected_type,
                content_type=content_type,
                company_name=company_name,
                reasons=["The detected event is already awarded or completed."],
                matched_terms=matched_terms,
                historical=True,
            )

        event_sentences = self._buyer_event_sentences(combined, company_name)
        buyer_event_detected = bool(event_sentences)

        # Partnership phrases like "work with implementation partners" must
        # not count unless they explicitly describe vendor/partner selection.
        if selected_type == BuyingSignalType.PARTNERSHIP_SEARCH:
            explicit_partner_search = any(term in combined_lower for term in {
                "seeking implementation partner", "seeking technology partner",
                "seeking strategic partner", "seeking vendor",
                "looking for a partner", "looking for a vendor",
                "partner selection", "vendor selection",
                "inviting technology partners",
            })
            if not explicit_partner_search:
                return self._reject(
                    signal_type=selected_type,
                    content_type=content_type,
                    company_name=company_name,
                    reasons=["The page mentions partners but does not show an active partner or vendor search."],
                    matched_terms=matched_terms,
                )

        if selected_type == BuyingSignalType.EXECUTIVE_HIRING:
            if content_type != ContentType.EXECUTIVE_JOB_POST:
                content_type = ContentType.EXECUTIVE_JOB_POST
            if not company_name:
                return self._reject(
                    signal_type=selected_type,
                    content_type=content_type,
                    company_name=None,
                    reasons=["An executive hiring signal was detected, but the hiring organisation is unknown."],
                    matched_terms=matched_terms,
                )
            buyer_event_detected = True

        # All non-procurement, non-executive signals need a named buyer and a
        # concrete organisation + action + project sentence.
        if selected_type == BuyingSignalType.OPEN_PROCUREMENT:
            return self._reject(
                signal_type=selected_type,
                content_type=content_type,
                company_name=company_name,
                reasons=[
                    "Procurement terms were detected too late in classification; the source could not be validated as an open procurement."
                ],
                matched_terms=matched_terms,
            )

        if selected_type != BuyingSignalType.EXECUTIVE_HIRING:
            if not company_name:
                return self._reject(
                    signal_type=selected_type,
                    content_type=content_type,
                    company_name=None,
                    reasons=["Technology keywords were found, but no identifiable buyer organisation was detected."],
                    matched_terms=matched_terms,
                )
            if not buyer_event_detected:
                return self._reject(
                    signal_type=selected_type,
                    content_type=content_type,
                    company_name=company_name,
                    reasons=["The page discusses the topic but does not contain a concrete buyer action or initiative."],
                    matched_terms=matched_terms,
                )

        evidence = event_sentences or self._extract_evidence(combined, matched_terms)
        confidence = min(0.98, 0.66 + 0.06 * len(matched_terms) + (0.10 if company_name else 0.0))

        return BuyingSignalResult(
            signal_type=selected_type,
            signal_strength=round(selected_score, 4),
            is_buyer_signal=selected_score >= self.minimum_signal_strength,
            company_name=company_name,
            evidence=evidence,
            matched_terms=matched_terms,
            is_historical=False,
            is_actionable=True,
            confidence=round(confidence, 4),
            rejection_reasons=[],
            content_type=content_type,
            buyer_event_detected=buyer_event_detected,
        )

    def classify(
        self,
        *,
        title: str = "",
        snippet: str = "",
        text: str = "",
        expected_signal_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BuyingSignalResult:
        """Return only the deterministic candidate-generation assessment.

        Gemini validation is performed later by a separate batch validator.
        This method must never make an external LLM call.
        """
        return self._rule_based_classify(
            title=title,
            snippet=snippet,
            text=text,
            expected_signal_type=expected_signal_type,
            metadata=metadata,
        )

    def _classify_with_llm(
        self,
        *,
        title: str,
        snippet: str,
        text: str,
        expected_signal_type: str | None,
        fallback: BuyingSignalResult,
    ) -> BuyingSignalResult:
        """Deprecated compatibility stub; no external LLM call is made."""
        logger.debug(
            "Ignoring direct _classify_with_llm call because buying-signal "
            "classification is configured as rule-only."
        )
        return fallback