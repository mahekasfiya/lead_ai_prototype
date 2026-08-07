from __future__ import annotations
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List

class PlannedSearchQuery(BaseModel):

    service_id: str
    service_name: str
    query: str

    source_type: str
    platform: str
    intent_type: str

    strategy: str

    strategy_order: int = Field(
        ge=1,
        description="One-based position of this strategy for the service.",
    )

    priority: int = Field(
        ge=1,
        description="Relative execution priority of the query strategy.",
    )

    target_country: Optional[str] = None

    # -------------------------
    # LLM Query Ranking
    # -------------------------

    rank: int = Field(
        default=999,
        ge=1,
        description="LLM ranking of this query (1 = best).",
    )

    buyer_specificity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Likelihood of finding a specific buyer organisation.",
    )

    current_opportunity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Likelihood of finding an active or upcoming opportunity.",
    )

    service_relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How closely the query matches the supplied service.",
    )

    regional_precision_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Likelihood of returning results from the target region.",
    )

    source_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Likelihood of returning high-quality buyer sources.",
    )

    noise_risk_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Likelihood of returning noisy or irrelevant results.",
    )

    final_query_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite score calculated by Python.",
    )

    ranking_reason: Optional[str] = Field(
        default=None,
        description="Short explanation for the ranking.",
    )

class SearchCandidate(BaseModel):

    source_url: HttpUrl
    source_title: Optional[str] = None
    source_snippet: Optional[str] = None
    source_domain: Optional[str] = None

    search_query: str
    service_id: Optional[str] = None
    service_name: Optional[str] = None

    source_type: Optional[str] = None
    platform: Optional[str] = None
    intent_type: Optional[str] = None
    strategy: Optional[str] = None
    strategy_order: Optional[int] = Field(
        default=None,
        ge=1,
    )
    priority: Optional[int] = Field(
        default=None,
        ge=1,
    )

class FetchedDocument(BaseModel):
    final_url: HttpUrl
    canonical_url: Optional[HttpUrl] = None
    content_type: str
    title: Optional[str] = None
    text: str
    text_chunks: List[str]
    fetch_status: str
    fetch_error: Optional[str] = None