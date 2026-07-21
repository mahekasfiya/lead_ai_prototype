from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Optional, List
from module_3.intelligence.models import (
    LeadIntelligenceReport,
)

class DocumentType(str, Enum):
    RFP = "rfp"
    RFQ = "rfq"
    EOI = "eoi"
    TENDER = "tender"
    PROCUREMENT_NOTICE = "procurement_notice"
    INVITATION_TO_BID = "invitation_to_bid"
    DIRECT_REQUIREMENT = "direct_requirement"
    PARTNER_REQUEST = "partner_request"
    NEWS_ABOUT_REQUIREMENT = "news_about_requirement"
    VENDOR_SERVICE_PAGE = "vendor_service_page"
    DIRECTORY = "directory"
    JOB_POSTING = "job_posting"
    TRAINING = "training"
    ARTICLE = "article"
    UNKNOWN = "unknown"

class OrganizationRole(str, Enum):
    BUYER = "buyer"
    PROVIDER = "provider"
    AGGREGATOR = "aggregator"
    PUBLISHER = "publisher"
    UNKNOWN = "unknown"

class RequirementStatus(str, Enum):
    OPEN = "open"
    UPCOMING = "upcoming"
    CLOSED = "closed"
    EXPIRED = "expired"
    UNCLEAR = "unclear"

class QualificationResult(BaseModel):
    document_type: DocumentType
    is_service_requirement: bool
    organization_role: OrganizationRole
    requirement_status: RequirementStatus
    buyer_intent_score: float = Field(ge=0.0, le=1.0)
    provider_probability: float = Field(ge=0.0, le=1.0)
    explicit_requirement: bool
    requires_external_supplier: bool
    evidence_quotes: List[str]
    rejection_reasons: List[str]
    confidence: float = Field(ge=0.0, le=1.0)

class LeadProfile(BaseModel):
    """
    Structured lead information received from later modules.

    The teammate handling SerpAPI, scraping, and cleaning can
    populate these fields before sending the lead for analysis.
    """

    company_name: str | None = Field(
        default=None,
        max_length=250,
    )

    industry: str | None = Field(
        default=None,
        max_length=150,
    )

    country: str | None = Field(
        default=None,
        max_length=150,
    )

    source_url: str | None = Field(
        default=None,
        max_length=2000,
    )

    summary: str | None = Field(
        default=None,
        max_length=5000,
    )

    content: str | None = Field(
        default=None,
        max_length=50000,
    )

    technologies: list[str] = Field(
        default_factory=list,
    )

    projects: list[str] = Field(
        default_factory=list,
    )

    signals: list[str] = Field(
        default_factory=list,
    )

    keywords: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "company_name",
        "industry",
        "country",
        "source_url",
        "summary",
        "content",
        mode="before",
    )
    @classmethod
    def clean_optional_text(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if not isinstance(value, str):
            return value

        cleaned = " ".join(value.strip().split())

        return cleaned or None

    @field_validator(
        "technologies",
        "projects",
        "signals",
        "keywords",
        mode="before",
    )
    @classmethod
    def clean_string_lists(
        cls,
        value: Any,
    ) -> list[str]:
        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError(
                "Value must be a list of strings."
            )

        cleaned_values: list[str] = []

        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    "All list items must be strings."
                )

            cleaned = " ".join(item.strip().split())

            if cleaned and cleaned not in cleaned_values:
                cleaned_values.append(cleaned)

        return cleaned_values

    def build_analysis_text(self) -> str:
        """
        Convert the structured lead profile into the text that
        will be sent to the embedding and explainability engines.
        """
        sections: list[str] = []

        if self.company_name:
            sections.append(
                f"Company: {self.company_name}"
            )

        if self.industry:
            sections.append(
                f"Industry: {self.industry}"
            )

        if self.country:
            sections.append(
                f"Country: {self.country}"
            )

        if self.summary:
            sections.append(
                f"Summary:\n{self.summary}"
            )

        if self.content:
            sections.append(
                f"Content:\n{self.content}"
            )

        if self.technologies:
            sections.append(
                "Technologies:\n"
                + "\n".join(
                    f"- {item}"
                    for item in self.technologies
                )
            )

        if self.projects:
            sections.append(
                "Projects:\n"
                + "\n".join(
                    f"- {item}"
                    for item in self.projects
                )
            )

        if self.signals:
            sections.append(
                "Buying signals:\n"
                + "\n".join(
                    f"- {item}"
                    for item in self.signals
                )
            )

        if self.keywords:
            sections.append(
                "Keywords:\n"
                + "\n".join(
                    f"- {item}"
                    for item in self.keywords
                )
            )

        analysis_text = "\n\n".join(sections).strip()

        if not analysis_text:
            raise ValueError(
                "Lead profile must contain at least one "
                "meaningful text field."
            )

        return analysis_text


class AnalyzeLeadRequest(BaseModel):
    """
    Request body for POST /analyze-lead.
    """

    lead: LeadProfile

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    minimum_similarity: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
    )


class EvidenceResponse(BaseModel):
    """
    Explanation evidence grouped by type.
    """

    matched_technologies: list[str] = Field(
        default_factory=list,
    )

    matched_capabilities: list[str] = Field(
        default_factory=list,
    )

    matched_business_problems: list[str] = Field(
        default_factory=list,
    )

    detected_buying_signals: list[str] = Field(
        default_factory=list,
    )

    detected_evidence: list[str] = Field(
        default_factory=list,
    )

    matched_keywords: list[str] = Field(
        default_factory=list,
    )

    matched_industries: list[str] = Field(
        default_factory=list,
    )


class ServiceMatchResponse(BaseModel):
    """
    One matched Triway service.
    """

    rank: int

    service_id: str

    service_name: str

    category: str

    similarity_score: float

    similarity_percentage: float

    confidence: str

    evidence_count: int

    evidence: EvidenceResponse

    explanation: str

    service_match_score: float

    service_match_percentage: float

    service_match_confidence: str

    score_breakdown: dict[str, Any]

class AnalyzeLeadResponse(BaseModel):
    company_name: str | None = None
    industry: str | None = None
    country: str | None = None
    source_url: str | None = None
    matched_services: list[ServiceMatchResponse]
    result_count: int

class DiscoverLeadsRequest(BaseModel):
    """
    Controls one SerpAPI lead-discovery run.
    """

    max_queries: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    results_per_query: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    max_leads: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    minimum_similarity: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )

    selected_service_ids: list[str] = Field(
        default_factory=list,
    )
    


class DiscoveredLeadResponse(BaseModel):
    source_title: str
    source_url: str
    source_snippet: str | None = None
    search_query: str

    company_name: str | None = None
    industry: str | None = None
    country: str | None = None

    matched_services: list[ServiceMatchResponse]

    top_service_id: str | None = None
    top_service_name: str | None = None
    top_service_match_percentage: float | None = None
    qualification: QualificationResult | None = None
    intelligence: LeadIntelligenceReport | None = None


class DiscoverLeadsResponse(BaseModel):
    queries_executed: list[str]
    sources_collected: int
    sources_analyzed: int
    leads_found: int

    leads: list[DiscoveredLeadResponse] = Field(
        default_factory=list
    )