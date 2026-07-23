from __future__ import annotations

import logging
from typing import Any, List

from module_3.intelligence.contact_extractor import ContactExtractor
from module_3.intelligence.models import (
    LeadIntelligenceReport,
    OpportunityAssessment,
    SalesRecommendation,
)
from module_3.schemas import (
    AnalyzeLeadResponse,
    LeadProfile,
    QualificationResult,
    RequirementStatus,
    ServiceMatchResponse,
)

logger = logging.getLogger(__name__)


class LeadIntelligenceService:
    def __init__(self, llm_model: Any = None) -> None:
        self.contact_extractor = ContactExtractor()
        self.llm_model = llm_model

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def build_report(
        self,
        lead: LeadProfile,
        qualification: QualificationResult,
        analysis: AnalyzeLeadResponse,
    ) -> LeadIntelligenceReport:
        top_match = (
            analysis.matched_services[0]
            if analysis.matched_services
            else None
        )

        opportunity_score, score_reasons = self._calculate_opportunity_score(
            qualification=qualification,
            top_service_percentage=(
                top_match.service_match_percentage if top_match else 0.0
            ),
            evidence_count=top_match.evidence_count if top_match else 0,
        )

        priority = self._get_priority(opportunity_score)
        urgency = self._get_urgency(qualification.requirement_status)
        risks = self._build_risks(qualification=qualification, lead=lead)

        initiative = (
            lead.summary
            or lead.metadata.get("source_title")
            or "Qualified service requirement"
        )

        business_need = self._build_business_need(
            lead=lead,
            top_service_name=top_match.service_name if top_match else None,
        )

        supporting_services = [
            match.service_id for match in analysis.matched_services[1:]
        ]

        pursue = opportunity_score >= 60 and top_match is not None

        recommendation = SalesRecommendation(
            pursue=pursue,
            recommended_action=self._recommended_action(priority=priority, pursue=pursue),
            primary_service_id=top_match.service_id if top_match else None,
            primary_service_name=top_match.service_name if top_match else None,
            supporting_service_ids=supporting_services,
            talking_points=self._build_talking_points(lead=lead, analysis=analysis),
        )

        opportunity = OpportunityAssessment(
            initiative=initiative,
            business_need=business_need,
            buying_stage=self._get_buying_stage(qualification),
            urgency=urgency,
            opportunity_score=round(opportunity_score, 2),
            priority=priority,
            reasons=score_reasons,
            risks=risks,
        )

        # Contact extraction
        contact_text_parts = [
            lead.summary,
            lead.metadata.get("source_title"),
            lead.metadata.get("source_snippet"),
            lead.metadata.get("full_text"),
            lead.metadata.get("cleaned_content"),
        ]
        contact_text = " ".join(str(part) for part in contact_text_parts if part)

        extracted_contacts = self.contact_extractor.extract(
            text=contact_text,
            source_url=lead.source_url,
        )

        return LeadIntelligenceReport(
            opportunity=opportunity,
            recommendation=recommendation,
            extracted_contacts=extracted_contacts,
            metadata={
                "source_url": lead.source_url,
                "company_name": lead.company_name,
                "industry": lead.industry,
                "country": lead.country,
            },
        )

    def generate_email_draft(
        self,
        lead: LeadProfile,
        matched_services: List[ServiceMatchResponse],
    ) -> str:
        contact_text = self._build_contact_text(lead)
        contacts = self.contact_extractor.extract(contact_text, lead.source_url)
        emails = [c.email for c in contacts if c.email]
        phones = [c.phone for c in contacts if c.phone]

        top_match = matched_services[0] if matched_services else None
        service_name = top_match.service_name if top_match else "our services"
        evidence = " ".join(top_match.evidence.detected_evidence[:2]) if top_match else ""

        prompt = f"""
You are a business development assistant.
Draft a concise, professional email to the following lead about our services.

Lead: {lead.company_name or 'Organization'} – {lead.summary or 'Technology initiative'}
Matched service: {service_name}
Key evidence: {evidence}

Write a short email (2‑3 paragraphs) with a subject line.
Be polite, direct, and actionable.
Keep it under 150 words.
"""

        if self.llm_model is None:
            draft = self._template_email(lead, service_name)
        else:
            try:
                response = self.llm_model.generate_content(prompt)
                draft = response.text.strip()
            except Exception as e:
                logger.warning("Email generation failed, using template: %s", e)
                draft = self._template_email(lead, service_name)

        if emails:
            draft += f"\n\nContact(s): {', '.join(emails)}"
        if phones:
            draft += f"\nPhone: {', '.join(phones)}"

        return draft

    def _template_email(self, lead: LeadProfile, service_name: str) -> str:
        return f"""
Subject: Opportunity to support your {service_name} project

Dear {lead.company_name or 'Team'},

I came across your recent initiative and believe our expertise in {service_name} could be valuable.

We have successfully helped similar organisations with their technology transformations.

I would welcome a brief conversation to explore how we might support your goals.

Best regards,
Triway Technologies
"""

    def _build_contact_text(self, lead: LeadProfile) -> str:
        parts = [
            lead.summary,
            lead.metadata.get("source_title"),
            lead.metadata.get("source_snippet"),
            lead.metadata.get("full_text"),
            lead.metadata.get("cleaned_content"),
            lead.content,
        ]
        return " ".join(str(p) for p in parts if p)

    # ------------------------------------------------------------------
    # Private helpers (fully implemented)
    # ------------------------------------------------------------------

    def _calculate_opportunity_score(
        self,
        qualification: QualificationResult,
        top_service_percentage: float,
        evidence_count: int,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []

        buyer_component = qualification.buyer_intent_score * 35
        service_component = max(0.0, min(top_service_percentage, 100.0)) * 0.40
        evidence_component = min(evidence_count * 2.5, 15.0)

        status_component = {
            RequirementStatus.OPEN: 10.0,
            RequirementStatus.UPCOMING: 8.0,
            RequirementStatus.UNCLEAR: 3.0,
            RequirementStatus.CLOSED: 0.0,
            RequirementStatus.EXPIRED: 0.0,
        }.get(qualification.requirement_status, 0.0)

        score = buyer_component + service_component + evidence_component + status_component

        if qualification.buyer_intent_score >= 0.8:
            reasons.append("Strong buyer-intent classification.")
        if top_service_percentage >= 70:
            reasons.append("Strong match with a Triway service.")
        if evidence_count >= 3:
            reasons.append("Multiple supporting evidence signals were detected.")
        if qualification.requirement_status == RequirementStatus.OPEN:
            reasons.append("The requirement is currently open.")

        return min(score, 100.0), reasons

    def _get_priority(self, score: float) -> str:
        if score >= 80:
            return "critical"
        if score >= 65:
            return "high"
        if score >= 45:
            return "medium"
        return "low"

    def _get_urgency(self, status: RequirementStatus) -> str:
        if status == RequirementStatus.OPEN:
            return "high"
        if status == RequirementStatus.UPCOMING:
            return "medium"
        if status in {RequirementStatus.CLOSED, RequirementStatus.EXPIRED}:
            return "low"
        return "unknown"

    def _get_buying_stage(self, qualification: QualificationResult) -> str:
        doc_type = qualification.document_type.value
        if doc_type in {"rfp", "rfq", "tender", "invitation_to_bid", "procurement_notice"}:
            return "active_procurement"
        if doc_type in {"eoi", "partner_request"}:
            return "vendor_evaluation"
        if doc_type == "news_about_requirement":
            return "early_signal"
        return "requirement_identified"

    def _build_business_need(self, lead: LeadProfile, top_service_name: str | None) -> str | None:
        if lead.summary and top_service_name:
            return f"{lead.summary} The strongest identified Triway capability is {top_service_name}."
        return lead.summary

    def _build_risks(self, qualification: QualificationResult, lead: LeadProfile) -> list[str]:
        risks: list[str] = []
        if qualification.requirement_status in {RequirementStatus.CLOSED, RequirementStatus.EXPIRED}:
            risks.append("The requirement may no longer be open.")
        if not lead.company_name:
            risks.append("The buyer organization was not confidently extracted.")
        if not lead.country:
            risks.append("The opportunity location is unknown.")
        if qualification.confidence < 0.7:
            risks.append("The requirement classification confidence is limited.")
        return risks

    def _recommended_action(self, priority: str, pursue: bool) -> str:
        if not pursue:
            return "Review manually before assigning the lead to the sales team."
        if priority == "critical":
            return "Assign immediately to business development and begin buyer-contact research."
        if priority == "high":
            return "Validate the procurement details and contact the buyer within one business day."
        return "Add to the opportunity pipeline and perform additional qualification."

    def _build_talking_points(self, lead: LeadProfile, analysis: AnalyzeLeadResponse) -> list[str]:
        points: list[str] = []
        for match in analysis.matched_services[:3]:
            points.append(f"Position {match.service_name}: {match.explanation}")
        if lead.industry:
            points.append(f"Tailor the proposal to the {lead.industry} industry.")
        if lead.country:
            points.append(f"Confirm delivery coverage and references for {lead.country}.")
        return points