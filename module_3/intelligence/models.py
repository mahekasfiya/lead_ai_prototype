from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OpportunityAssessment(BaseModel):
    """
    Business interpretation of one qualified lead.
    """

    initiative: str | None = None

    business_need: str | None = None

    buying_stage: str = "unknown"

    urgency: str = "unknown"

    opportunity_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    priority: str

    reasons: list[str] = Field(
        default_factory=list,
    )

    risks: list[str] = Field(
        default_factory=list,
    )


class SalesRecommendation(BaseModel):
    """
    Recommended action for the sales or business-development team.
    """

    pursue: bool

    recommended_action: str

    primary_service_id: str | None = None

    primary_service_name: str | None = None

    supporting_service_ids: list[str] = Field(
        default_factory=list,
    )

    talking_points: list[str] = Field(
        default_factory=list,
    )

class ExtractedContact(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    department: str | None = None
    role: str | None = None
    procurement_url: str | None = None
    source: str = "lead_text"
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
    )


class LeadIntelligenceReport(BaseModel):
    """
    Final Person 2 output for one discovered lead.
    """

    opportunity: OpportunityAssessment

    recommendation: SalesRecommendation

    extracted_contacts: list[ExtractedContact] = Field(
    default_factory=list
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )