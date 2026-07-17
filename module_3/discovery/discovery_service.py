from __future__ import annotations

import logging
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import List, Optional

from app.collector.text_extractor import extract_text
from app.search.serpapi import search

from module_3.discovery.query_generator import QueryGenerator
from module_3.discovery.document_fetcher import DocumentFetcher
from module_3.discovery.requirement_classifier import RequirementClassifier
from module_3.discovery.contradiction_checker import ContradictionChecker
from module_3.discovery.qualification_gate import QualificationGate
from module_3.discovery.models import SearchCandidate
from module_3.schemas import (
    AnalyzeLeadRequest,
    DiscoverLeadsRequest,
    DiscoverLeadsResponse,
    DiscoveredLeadResponse,
    LeadProfile,
    QualificationResult,
)
from module_3.service import LeadAnalysisService
from module_3.discovery.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)

TRACKING_PARAMETERS = {
    "utm_source", "utm_medium", "utm_campaign",
    "utm_term", "utm_content", "gclid", "fbclid",
}

def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMETERS
    ]
    normalized_path = parts.path.rstrip("/") or "/"
    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        normalized_path,
        urlencode(filtered_query),
        "",
    ))

class LeadDiscoveryService:
    def __init__(
        self,
        analysis_service: LeadAnalysisService,
        config: dict,
    ):
        llm_model=config.get("llm_model")
        use_gemini=config.get("use_gemini",False)

        self.analysis_service = analysis_service
        # Query generator with LLM option
        self.query_generator = QueryGenerator(
            knowledge_base_path=config.get("knowledge_base_path"),
            use_llm=use_gemini,
            llm_model=llm_model,
            planner_prompt_path=config.get("planner_prompt_path"),
        )
        # Document fetcher
        self.fetcher = DocumentFetcher(
            timeout=config.get("fetch_timeout", 30),
            max_size=config.get("fetch_max_size", 10 * 1024 * 1024),
        )
        # Classifier
        self.classifier = RequirementClassifier(
            llm_model=llm_model,
            use_gemini=use_gemini,
            max_chunks=config.get("max_chunks", 3),
        )
        self.contradiction = ContradictionChecker()
        self.gate = QualificationGate({
            "min_buyer_score": config.get("min_buyer_score", 0.6),
            "max_provider_prob": config.get("max_provider_prob", 0.4),
        })
        self.config = config

    def discover(self, request: DiscoverLeadsRequest) -> DiscoverLeadsResponse:
        # 1. Generate queries
        query_records = self.query_generator.generate(
            max_queries=request.max_queries,
            selected_service_ids=request.selected_service_ids,
        )

        # 2. Collect search results, deduplicate
        collected_candidates: List[SearchCandidate] = []
        seen_urls: set[str] = set()
        for record in query_records:
            results = search(record["query"], num_results=request.results_per_query)
            for result in results:
                norm = normalize_url(result.url)
                if norm in seen_urls:
                    continue
                seen_urls.add(norm)
                candidate = SearchCandidate(
                    source_url=result.url,
                    source_title=result.title,
                    source_snippet=result.snippet,
                    source_domain=urlsplit(result.url).netloc,
                    search_query=record["query"],
                    service_id=record["service_id"],
                )
                collected_candidates.append(candidate)

        # 3. Fetch, classify, qualify
        qualified_pairs = []  # (candidate, doc, qual)
        for candidate in collected_candidates:
            # Fetch
            doc = self.fetcher.fetch(str(candidate.source_url))
            if doc.fetch_status != "success":
                logger.warning(f"Fetch failed for {candidate.source_url}: {doc.fetch_error}")
                continue

            # Classify
            qual = self.classifier.classify(doc.text, doc.text_chunks)

            # Contradiction check
            contradiction_passed, _ = self.contradiction.check(doc.text, qual)

            # Gate
            if self.gate.apply(candidate, doc, qual, contradiction_passed):
                qualified_pairs.append((candidate, doc, qual))
            
            else:
                logger.info(f"❌ REJECTED: {candidate.source_url}")
                logger.info(f"   Document type: {qual.document_type}")
                logger.info(f"   is_service_requirement: {qual.is_service_requirement}")
                logger.info(f"   organization_role: {qual.organization_role}")
                logger.info(f"   buyer_intent_score: {qual.buyer_intent_score}")
                logger.info(f"   provider_probability: {qual.provider_probability}")    
                logger.info(f"   explicit_requirement: {qual.explicit_requirement}")
                logger.info(f"   requires_external_supplier: {qual.requires_external_supplier}")
                logger.info(f"   rejection_reasons: {qual.rejection_reasons}")
        # 4. Run existing analysis on qualified candidates
        discovered_leads: List[DiscoveredLeadResponse] = []
        for candidate, doc, qual in qualified_pairs:
            # Build LeadProfile as before
            combined = "\n\n".join([
                candidate.source_title or "",
                candidate.source_snippet or "",
                doc.text
            ]).strip()

            metadata = MetadataExtractor.extract_all(
                url=str(candidate.source_url),
                title=candidate.source_title or "",
                snippet=candidate.source_snippet or "",
                text=doc.text
            )

            if not combined:
                continue
            lead = LeadProfile(
                company_name=metadata['company_name'],
                industry=metadata['industry'],
                country=metadata['country'],
                source_url=str(candidate.source_url),
                summary=candidate.source_snippet or candidate.source_title,
                content=combined,
                technologies=[],
                projects=[],
                signals=[],
                keywords=[],
                metadata={
                    "search_query": candidate.search_query,
                    "source_title": candidate.source_title,
                    "extracted_emails": metadata['emails'],
                },
            )
            analysis_req = AnalyzeLeadRequest(
                lead=lead,
                top_k=3,
                minimum_similarity=request.minimum_similarity,
            )
            analysis_resp = self.analysis_service.analyze(analysis_req)
            if not analysis_resp.matched_services:
                continue
            top_match = analysis_resp.matched_services[0]
            discovered_leads.append(
                DiscoveredLeadResponse(
                    source_title=candidate.source_title or "",
                    source_url=str(candidate.source_url),
                    source_snippet=candidate.source_snippet,
                    search_query=candidate.search_query,
                    company_name=analysis_resp.company_name,
                    industry=analysis_resp.industry,
                    country=analysis_resp.country,
                    matched_services=analysis_resp.matched_services,
                    top_service_id=top_match.service_id,
                    top_service_name=top_match.service_name,
                    top_service_match_percentage=top_match.service_match_percentage,
                    qualification=qual,  # new
                )
            )

        # Sort and limit
        discovered_leads.sort(key=lambda x: x.top_service_match_percentage or 0.0, reverse=True)
        discovered_leads = discovered_leads[:request.max_leads]

        return DiscoverLeadsResponse(
            queries_executed=[rec["query"] for rec in query_records],
            sources_collected=len(collected_candidates),
            sources_analyzed=len(collected_candidates),
            leads_found=len(discovered_leads),
            leads=discovered_leads,
        )