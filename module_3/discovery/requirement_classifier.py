from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Sequence

from module_3.schemas import (
    QualificationResult,
    DocumentType,
    OrganizationRole,
    RequirementStatus,
)

logger = logging.getLogger(__name__)


SOURCE_PROCUREMENT = "PROCUREMENT"
SOURCE_DIRECT_PROJECT = "DIRECT_PROJECT"
SOURCE_PARTNER_SEARCH = "PARTNER_SEARCH"
SOURCE_HIRING_SIGNAL = "HIRING_SIGNAL"
SOURCE_GENERAL_WEB = "GENERAL_WEB"


class RequirementClassifier:
    def __init__(
        self,
        llm_model=None,
        use_gemini: bool = False,
        max_chunks: int = 3,
    ):
        self.llm_model = llm_model
        self.use_gemini = use_gemini
        self.max_chunks = max(1, max_chunks)

    def classify(
        self,
        text: str,
        text_chunks: List[str],
        source_type: str = SOURCE_GENERAL_WEB,
        platform: str = "general_web",
    ) -> QualificationResult:
        combined = self._combine_text(text, text_chunks)
        normalized_source = self._normalize_source_type(source_type)
        normalized_platform = (platform or "general_web").strip().lower()

        if self.use_gemini and self.llm_model:
            try:
                return self._classify_with_gemini(
                    combined,
                    source_type=normalized_source,
                    platform=normalized_platform,
                )
            except Exception as exc:
                logger.error(
                    "Gemini classification failed: %s; falling back to rules",
                    exc,
                )

        return self._classify_rule_based(
            combined,
            source_type=normalized_source,
            platform=normalized_platform,
        )

    def _combine_text(self, text: str, text_chunks: List[str]) -> str:
        selected_chunks = [
            chunk.strip()
            for chunk in (text_chunks or [])[: self.max_chunks]
            if chunk and chunk.strip()
        ]

        combined = " ".join(selected_chunks)
        if not combined.strip():
            combined = (text or "").strip()

        combined = re.sub(r"\s+", " ", combined).strip()
        return combined[:12000]

    @staticmethod
    def _normalize_source_type(source_type: str) -> str:
        normalized = (source_type or SOURCE_GENERAL_WEB).strip().upper()
        aliases = {
            "MARKETPLACE": SOURCE_DIRECT_PROJECT,
            "FREELANCER": SOURCE_DIRECT_PROJECT,
            "PEOPLEPERHOUR": SOURCE_DIRECT_PROJECT,
            "PARTNER": SOURCE_PARTNER_SEARCH,
            "HIRING": SOURCE_HIRING_SIGNAL,
            "JOB": SOURCE_HIRING_SIGNAL,
            "GENERAL": SOURCE_GENERAL_WEB,
        }
        return aliases.get(normalized, normalized)

    def _classify_with_gemini(
        self,
        text: str,
        *,
        source_type: str,
        platform: str,
    ) -> QualificationResult:
        prompt = f"""
You are a classifier for B2B IT-services lead discovery.

Source context:
- source_type: {source_type}
- platform: {platform}

Interpret the page according to its source type:
- PROCUREMENT: formal RFP, RFQ, tender, EOI, bid, or procurement notice.
- DIRECT_PROJECT: marketplace or project post where a buyer requests work, deliverables, a budget, milestones, bids, or proposals.
- PARTNER_SEARCH: an organization seeking an implementation partner, technology partner, integrator, consultant, vendor, or service provider.
- HIRING_SIGNAL: a job vacancy or recruitment signal. This is not a direct service lead.
- GENERAL_WEB: other web pages that may describe a requirement or initiative.

Important rules:
1. Do not require formal tender terminology for DIRECT_PROJECT.
2. A marketplace project post normally implies an external supplier is required.
3. A PARTNER_SEARCH can be a valid buyer requirement even without formal procurement language.
4. A HIRING_SIGNAL must have is_service_requirement=false and document_type=job_posting unless the page independently contains an explicit external-services request.
5. Provider marketing pages, directories, articles, reports, training pages, and generic news are not direct buyer requirements.

Classify these fields:
- document_type: one of rfp, rfq, eoi, tender, procurement_notice, invitation_to_bid, direct_requirement, partner_request, implementation_announcement, digital_transformation_initiative, modernization_project, news_about_requirement, vendor_service_page, directory, job_posting, training, article, unknown
- is_service_requirement: boolean
- organization_role: buyer, provider, aggregator, publisher, unknown
- requirement_status: open, upcoming, closed, expired, unclear
- buyer_intent_score: float 0.0-1.0
- provider_probability: float 0.0-1.0
- explicit_requirement: boolean
- requires_external_supplier: boolean
- evidence_quotes: list of up to 3 short supporting quotes
- rejection_reasons: list of reasons when not a valid buyer requirement
- confidence: float 0.0-1.0

Page text:
{text}

Return only valid JSON with exactly those fields.
"""

        response = self.llm_model.generate_content(prompt)
        content = (response.text or "").strip()

        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()

        data = json.loads(content)
        return self._result_from_mapping(data)

    def _classify_rule_based(
        self,
        text: str,
        *,
        source_type: str,
        platform: str,
    ) -> QualificationResult:
        lowered = text.casefold()

        provider_patterns = {
            "we provide": 1.0,
            "we offer": 1.0,
            "our services": 1.0,
            "our solutions": 1.0,
            "leading provider": 1.0,
            "we deliver": 0.8,
            "we specialize": 1.0,
            "our expertise": 0.8,
            "contact us": 0.6,
            "get a quote": 0.8,
            "free consultation": 0.8,
            "service offering": 1.0,
            "case study": 0.7,
            "success story": 0.7,
        }

        procurement_patterns = {
            "request for proposal": 2.0,
            "request for quotation": 2.0,
            "invitation to bid": 2.0,
            "expression of interest": 1.8,
            "tender notice": 1.8,
            "procurement notice": 1.8,
            "bid submission": 1.5,
            "submission deadline": 1.5,
            "scope of work": 1.2,
            "evaluation criteria": 0.8,
            "invites proposals": 1.6,
            "invites bids": 1.6,
            "rfp": 1.5,
            "rfq": 1.5,
            "eoi": 1.2,
            "tender": 1.2,
            "vendor": 0.5,
            "proposal": 0.5,
        }

        direct_project_patterns = {
            "looking for": 1.2,
            "we need": 1.3,
            "need a developer": 1.6,
            "need an expert": 1.6,
            "project description": 1.2,
            "project budget": 1.4,
            "fixed price": 0.8,
            "hourly rate": 0.8,
            "deliverables": 1.2,
            "milestone": 1.0,
            "place a bid": 1.5,
            "submit a proposal": 1.5,
            "skills required": 0.8,
            "project deadline": 1.0,
            "required skills": 0.8,
            "hire": 0.3,
        }

        partner_patterns = {
            "implementation partner": 1.8,
            "technology partner": 1.6,
            "migration partner": 1.8,
            "integration partner": 1.7,
            "strategic partner": 1.0,
            "seeking a partner": 1.6,
            "looking for a partner": 1.6,
            "seeking vendor": 1.5,
            "seeking a vendor": 1.5,
            "seeking service provider": 1.6,
            "inviting service providers": 1.6,
            "systems integrator": 1.2,
            "implementation consultant": 1.2,
        }

        hiring_patterns = {
            "job description": 1.4,
            "apply now": 1.2,
            "vacancy": 1.4,
            "careers": 1.0,
            "recruiting": 1.2,
            "we are hiring": 1.6,
            "job opening": 1.5,
            "employment type": 1.2,
            "salary": 0.8,
            "resume": 0.8,
            "curriculum vitae": 0.8,
        }

        closed_patterns = [
            "closed",
            "expired",
            "deadline has passed",
            "no longer accepting",
            "submission period has ended",
        ]
        upcoming_patterns = ["upcoming", "will be issued", "planned tender"]

        provider_score = self._weighted_score(lowered, provider_patterns)
        procurement_score = self._weighted_score(lowered, procurement_patterns)
        project_score = self._weighted_score(lowered, direct_project_patterns)
        partner_score = self._weighted_score(lowered, partner_patterns)
        hiring_score = self._weighted_score(lowered, hiring_patterns)

        source_bonus = {
            SOURCE_PROCUREMENT: (1.5, 0.0, 0.0, 0.0),
            SOURCE_DIRECT_PROJECT: (0.0, 1.5, 0.0, 0.0),
            SOURCE_PARTNER_SEARCH: (0.0, 0.0, 1.5, 0.0),
            SOURCE_HIRING_SIGNAL: (0.0, 0.0, 0.0, 1.5),
        }.get(source_type, (0.0, 0.0, 0.0, 0.0))

        procurement_score += source_bonus[0]
        project_score += source_bonus[1]
        partner_score += source_bonus[2]
        hiring_score += source_bonus[3]

        buyer_signal_score = max(procurement_score, project_score, partner_score)
        buyer_intent_score = self._normalize_score(buyer_signal_score, scale=5.0)
        provider_probability = self._normalize_score(provider_score, scale=4.0)

        evidence = self._evidence_quotes(
            text,
            [
                *procurement_patterns.keys(),
                *direct_project_patterns.keys(),
                *partner_patterns.keys(),
                *hiring_patterns.keys(),
            ],
        )

        status = RequirementStatus.UNCLEAR
        if any(pattern in lowered for pattern in closed_patterns):
            status = RequirementStatus.CLOSED
        elif any(pattern in lowered for pattern in upcoming_patterns):
            status = RequirementStatus.UPCOMING
        elif buyer_signal_score > 0:
            status = RequirementStatus.OPEN

        document_type = self._document_type(
            lowered,
            source_type=source_type,
            procurement_score=procurement_score,
            project_score=project_score,
            partner_score=partner_score,
            hiring_score=hiring_score,
        )

        if source_type == SOURCE_HIRING_SIGNAL or (
            hiring_score >= 2.0 and hiring_score > buyer_signal_score
        ):
            return QualificationResult(
                document_type=DocumentType.JOB_POSTING,
                is_service_requirement=False,
                organization_role=OrganizationRole.BUYER,
                requirement_status=status,
                buyer_intent_score=min(buyer_intent_score, 0.35),
                provider_probability=provider_probability,
                explicit_requirement=False,
                requires_external_supplier=False,
                evidence_quotes=evidence,
                rejection_reasons=[
                    "Hiring activity is an enrichment signal, not a direct external-services requirement."
                ],
                confidence=self._confidence(
                    primary_score=hiring_score,
                    opposing_score=max(buyer_signal_score, provider_score),
                    source_consistent=source_type == SOURCE_HIRING_SIGNAL,
                ),
            )

        is_provider = provider_score > buyer_signal_score + 0.75
        is_buyer = buyer_signal_score >= self._minimum_signal(source_type) and not is_provider

        explicit_requirement = self._explicit_requirement(
            lowered,
            source_type=source_type,
        )
        requires_external_supplier = self._requires_external_supplier(
            lowered,
            source_type=source_type,
            explicit_requirement=explicit_requirement,
        )

        role = OrganizationRole.BUYER if is_buyer else OrganizationRole.PROVIDER
        rejection_reasons: list[str] = []

        if is_provider:
            rejection_reasons.append("Provider-marketing language outweighs buyer requirement signals.")
        elif not is_buyer:
            rejection_reasons.append("No sufficiently strong buyer requirement was detected for this source type.")

        if source_type == SOURCE_DIRECT_PROJECT and is_buyer:
            explicit_requirement = True
            requires_external_supplier = True

        if source_type == SOURCE_PARTNER_SEARCH and is_buyer:
            requires_external_supplier = True

        confidence = self._confidence(
            primary_score=buyer_signal_score if is_buyer else max(provider_score, buyer_signal_score),
            opposing_score=provider_score if is_buyer else buyer_signal_score,
            source_consistent=self._source_consistent(
                source_type,
                procurement_score,
                project_score,
                partner_score,
            ),
        )

        return QualificationResult(
            document_type=document_type,
            is_service_requirement=is_buyer,
            organization_role=role,
            requirement_status=status,
            buyer_intent_score=buyer_intent_score,
            provider_probability=provider_probability,
            explicit_requirement=explicit_requirement,
            requires_external_supplier=requires_external_supplier,
            evidence_quotes=evidence,
            rejection_reasons=rejection_reasons,
            confidence=confidence,
        )

    @staticmethod
    def _weighted_score(text: str, patterns: Dict[str, float]) -> float:
        return sum(weight for phrase, weight in patterns.items() if phrase in text)

    @staticmethod
    def _normalize_score(score: float, *, scale: float) -> float:
        if score <= 0:
            return 0.0
        return round(min(score / scale, 1.0), 4)

    @staticmethod
    def _minimum_signal(source_type: str) -> float:
        return {
            SOURCE_PROCUREMENT: 1.5,
            SOURCE_DIRECT_PROJECT: 1.5,
            SOURCE_PARTNER_SEARCH: 1.4,
            SOURCE_GENERAL_WEB: 2.0,
        }.get(source_type, 2.0)

    @staticmethod
    def _explicit_requirement(text: str, *, source_type: str) -> bool:
        if source_type == SOURCE_DIRECT_PROJECT:
            return any(
                phrase in text
                for phrase in (
                    "looking for",
                    "we need",
                    "project description",
                    "place a bid",
                    "submit a proposal",
                    "deliverables",
                )
            )

        return any(
            phrase in text
            for phrase in (
                "request for proposal",
                "request for quotation",
                "invitation to bid",
                "invites proposals",
                "invites bids",
                "seeking a partner",
                "looking for a partner",
                "seeking vendor",
                "seeking service provider",
                "rfp",
                "rfq",
                "tender",
            )
        )

    @staticmethod
    def _requires_external_supplier(
        text: str,
        *,
        source_type: str,
        explicit_requirement: bool,
    ) -> bool:
        if source_type in {SOURCE_DIRECT_PROJECT, SOURCE_PARTNER_SEARCH}:
            return explicit_requirement

        return any(
            phrase in text
            for phrase in (
                "vendor",
                "supplier",
                "service provider",
                "implementation partner",
                "technology partner",
                "systems integrator",
                "consultant",
                "submit a proposal",
                "submit a bid",
            )
        ) or explicit_requirement

    @staticmethod
    def _source_consistent(
        source_type: str,
        procurement_score: float,
        project_score: float,
        partner_score: float,
    ) -> bool:
        if source_type == SOURCE_PROCUREMENT:
            return procurement_score >= max(project_score, partner_score)
        if source_type == SOURCE_DIRECT_PROJECT:
            return project_score >= max(procurement_score, partner_score)
        if source_type == SOURCE_PARTNER_SEARCH:
            return partner_score >= max(procurement_score, project_score)
        return True

    @staticmethod
    def _confidence(
        *,
        primary_score: float,
        opposing_score: float,
        source_consistent: bool,
    ) -> float:
        margin = max(0.0, primary_score - opposing_score)
        evidence_component = min(primary_score / 5.0, 1.0)
        margin_component = min(margin / 3.0, 1.0)
        source_component = 0.15 if source_consistent else 0.0
        confidence = 0.25 + (0.4 * evidence_component) + (0.2 * margin_component) + source_component
        return round(min(max(confidence, 0.1), 0.98), 4)

    @staticmethod
    def _document_type(
        text: str,
        *,
        source_type: str,
        procurement_score: float,
        project_score: float,
        partner_score: float,
        hiring_score: float,
    ) -> DocumentType:
        if "request for quotation" in text or re.search(r"\brfq\b", text):
            return DocumentType.RFQ
        if "request for proposal" in text or re.search(r"\brfp\b", text):
            return DocumentType.RFP
        if "expression of interest" in text or re.search(r"\beoi\b", text):
            return DocumentType.EOI
        if "invitation to bid" in text:
            return DocumentType.INVITATION_TO_BID
        if "tender" in text:
            return DocumentType.TENDER
        if hiring_score >= max(procurement_score, project_score, partner_score):
            return DocumentType.JOB_POSTING
        if source_type == SOURCE_PARTNER_SEARCH or partner_score >= max(procurement_score, project_score):
            return DocumentType.PARTNER_REQUEST
        if source_type == SOURCE_DIRECT_PROJECT or project_score > procurement_score:
            return DocumentType.DIRECT_REQUIREMENT
        if source_type == SOURCE_PROCUREMENT or procurement_score > 0:
            return DocumentType.PROCUREMENT_NOTICE
        return DocumentType.UNKNOWN

    @staticmethod
    def _evidence_quotes(
        text: str,
        phrases: Sequence[str],
        *,
        limit: int = 3,
    ) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        found: list[str] = []

        for sentence in sentences:
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            lowered = cleaned.casefold()
            if not cleaned:
                continue
            if any(phrase in lowered for phrase in phrases):
                found.append(cleaned[:300])
            if len(found) >= limit:
                break

        return found

    @staticmethod
    def _result_from_mapping(data: dict) -> QualificationResult:
        return QualificationResult(
            document_type=DocumentType(data.get("document_type", "unknown")),
            is_service_requirement=bool(data.get("is_service_requirement", False)),
            organization_role=OrganizationRole(data.get("organization_role", "unknown")),
            requirement_status=RequirementStatus(data.get("requirement_status", "unclear")),
            buyer_intent_score=float(data.get("buyer_intent_score", 0.0)),
            provider_probability=float(data.get("provider_probability", 0.0)),
            explicit_requirement=bool(data.get("explicit_requirement", False)),
            requires_external_supplier=bool(data.get("requires_external_supplier", False)),
            evidence_quotes=list(data.get("evidence_quotes", []))[:3],
            rejection_reasons=list(data.get("rejection_reasons", [])),
            confidence=float(data.get("confidence", 0.5)),
        )