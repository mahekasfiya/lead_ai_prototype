from __future__ import annotations

import logging
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)

from app.collector.text_extractor import extract_text
from app.collector.webpage_fetcher import fetch_page
from app.search.serpapi import search

from module_3.discovery.query_generator import (
    QueryGenerator,
)
from module_3.schemas import (
    AnalyzeLeadRequest,
    DiscoverLeadsRequest,
    DiscoverLeadsResponse,
    DiscoveredLeadResponse,
    LeadProfile,
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
}


def normalize_url(url: str) -> str:
    parts = urlsplit(url)

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(
            parts.query,
            keep_blank_values=True,
        )
        if key.lower() not in TRACKING_PARAMETERS
    ]

    normalized_path = (
        parts.path.rstrip("/") or "/"
    )

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            normalized_path,
            urlencode(filtered_query),
            "",
        )
    )


class LeadDiscoveryService:
    """
    Searches the web using buying-intent queries and sends
    each candidate through the existing lead-analysis service.
    """

    def __init__(
        self,
        analysis_service: LeadAnalysisService,
    ) -> None:
        if analysis_service is None:
            raise ValueError(
                "analysis_service cannot be None."
            )
        self.analysis_service = analysis_service
        self.query_generator = QueryGenerator()

    def discover(
        self,
        request: DiscoverLeadsRequest,
    ) -> DiscoverLeadsResponse:
        query_records = self.query_generator.generate(
            max_queries=request.max_queries,
            selected_service_ids=(
                request.selected_service_ids
            ),
        )

        collected_results = []
        seen_urls: set[str] = set()

        for query_record in query_records:
            search_results = search(
                query=query_record["query"],
                num_results=request.results_per_query,
            )

            for result in search_results:
                normalized_url = normalize_url(
                    result.url
                )

                if (
                    not normalized_url
                    or normalized_url in seen_urls
                ):
                    continue

                seen_urls.add(normalized_url)

                collected_results.append(
                    {
                        "query": query_record["query"],
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                    }
                )

        discovered_leads: list[
            DiscoveredLeadResponse
        ] = []

        sources_analyzed = 0

        for source in collected_results:
            try:
                html = fetch_page(source["url"])
                page_text = extract_text(html)

            except Exception as exc:
                logger.warning(
                    "Failed to fetch %s: %s",
                    source["url"],
                    exc,
                )

                page_text = ""

            combined_content = "\n\n".join(
                value
                for value in [
                    source["title"],
                    source["snippet"],
                    page_text,
                ]
                if value
            )

            if not combined_content.strip():
                continue

            lead = LeadProfile(
                company_name=None,
                industry=None,
                country=None,
                source_url=source["url"],
                summary=source["snippet"]
                or source["title"],
                content=combined_content,
                technologies=[],
                projects=[],
                signals=[],
                keywords=[],
                metadata={
                    "search_query": source["query"],
                    "source_title": source["title"],
                },
            )

            analysis_request = AnalyzeLeadRequest(
                lead=lead,
                top_k=3,
                minimum_similarity=(
                    request.minimum_similarity
                ),
            )

            analysis_response = (
                self.analysis_service.analyze(
                    analysis_request
                )
            )

            sources_analyzed += 1

            if not analysis_response.matched_services:
                continue

            top_match = (
                analysis_response.matched_services[0]
            )

            discovered_leads.append(
                DiscoveredLeadResponse(
                    source_title=source["title"],
                    source_url=source["url"],
                    source_snippet=source["snippet"],
                    search_query=source["query"],
                    company_name=(
                        analysis_response.company_name
                    ),
                    industry=(
                        analysis_response.industry
                    ),
                    country=(
                        analysis_response.country
                    ),
                    matched_services=(
                        analysis_response.matched_services
                    ),
                    top_service_id=(
                        top_match.service_id
                    ),
                    top_service_name=(
                        top_match.service_name
                    ),
                    top_service_match_percentage=(
                        top_match
                        .service_match_percentage
                    ),
                )
            )

        discovered_leads.sort(
            key=lambda lead: (
                lead.top_service_match_percentage
                or 0.0
            ),
            reverse=True,
        )

        discovered_leads = discovered_leads[
            :request.max_leads
        ]

        return DiscoverLeadsResponse(
            queries_executed=[
                record["query"]
                for record in query_records
            ],
            sources_collected=len(
                collected_results
            ),
            sources_analyzed=sources_analyzed,
            leads_found=len(discovered_leads),
            leads=discovered_leads,
        )