from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List

class PlannedSearchQuery(BaseModel):
    service_id: str
    service_name: str
    query: str
    intent_type: str
    target_country: Optional[str] = None

class SearchCandidate(BaseModel):
    source_url: HttpUrl
    source_title: Optional[str] = None
    source_snippet: Optional[str] = None
    source_domain: Optional[str] = None
    search_query: str
    service_id: Optional[str] = None

class FetchedDocument(BaseModel):
    final_url: HttpUrl
    canonical_url: Optional[HttpUrl] = None
    content_type: str
    title: Optional[str] = None
    text: str
    text_chunks: List[str]
    fetch_status: str
    fetch_error: Optional[str] = None