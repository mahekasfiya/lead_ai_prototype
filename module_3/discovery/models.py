from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


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