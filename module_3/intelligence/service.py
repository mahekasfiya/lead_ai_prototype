from __future__ import annotations

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
)

from module_3.intelligence.contact_extractor import ContactExtractor


class LeadIntelligenceService:
    def __init__(self) -> None:
        self.contact_extractor = ContactExtractor()

    def build_report(
        self,
        lead: LeadProfile,
        qualification: QualificationResult,
        analysis: AnalyzeLeadResponse,
    ) -> LeadIntelligenceReport:

        # Get the highest-ranked Triway service match.
        top_match = (
            analysis.matched_services[0]
            if analysis.matched_services
            else None
        )

        # Calculate the overall opportunity score.
        opportunity_score, score_reasons = (
            self._calculate_opportunity_score(
                qualification=qualification,
                top_service_percentage=(
                    top_match.service_match_percentage
                    if top_match
                    else 0.0
                ),
                evidence_count=(
                    top_match.evidence_count
                    if top_match
                    else 0
                ),
            )
        )

        priority = self._get_priority(
            opportunity_score
        )

        urgency = self._get_urgency(
            qualification.requirement_status
        )

        risks = self._build_risks(
            qualification=qualification,
            lead=lead,
        )

        initiative = (
            lead.summary
            or lead.metadata.get("source_title")
            or "Qualified service requirement"
        )

        business_need = self._build_business_need(
            lead=lead,
            top_service_name=(
                top_match.service_name
                if top_match
                else None
            ),
        )

        supporting_services = [
            match.service_id
            for match in analysis.matched_services[1:]
        ]

        pursue = (
            opportunity_score >= 60
            and top_match is not None
        )

        recommendation = SalesRecommendation(
            pursue=pursue,
            recommended_action=self._recommended_action(
                priority=priority,
                pursue=pursue,
            ),
            primary_service_id=(
                top_match.service_id
                if top_match
                else None
            ),
            primary_service_name=(
                top_match.service_name
                if top_match
                else None
            ),
            supporting_service_ids=supporting_services,
            talking_points=self._build_talking_points(
                lead=lead,
                analysis=analysis,
            ),
        )

        opportunity = OpportunityAssessment(
            initiative=initiative,
            business_need=business_need,
            buying_stage=self._get_buying_stage(
                qualification
            ),
            urgency=urgency,
            opportunity_score=round(
                opportunity_score,
                2,
            ),
            priority=priority,
            reasons=score_reasons,
            risks=risks,
        )

        # --------------------------------------------------
        # Contact extraction
        # --------------------------------------------------

        # Collect all available text supplied by Person 1.
        contact_text_parts = [
            lead.summary,
            lead.metadata.get("source_title"),
            lead.metadata.get("source_snippet"),
            lead.metadata.get("full_text"),
            lead.metadata.get("cleaned_content"),
        ]

        # Remove empty values and combine everything into one string.
        contact_text = " ".join(
            str(part)
            for part in contact_text_parts
            if part
        )

        # Extract names, emails, phone numbers, departments
        # and procurement URLs from the available text.
        extracted_contacts = (
            self.contact_extractor.extract(
                text=contact_text,
                source_url=lead.source_url,
            )
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

    def _calculate_opportunity_score(
        self,
        qualification: QualificationResult,
        top_service_percentage: float,
        evidence_count: int,
    ) -> tuple[float, list[str]]:

        reasons: list[str] = []

        buyer_component = (
            qualification.buyer_intent_score * 35
        )

        service_component = (
            max(
                0.0,
                min(
                    top_service_percentage,
                    100.0,
                ),
            )
            * 0.40
        )

        evidence_component = min(
            evidence_count * 2.5,
            15.0,
        )

        status_component = {
            RequirementStatus.OPEN: 10.0,
            RequirementStatus.UPCOMING: 8.0,
            RequirementStatus.UNCLEAR: 3.0,
            RequirementStatus.CLOSED: 0.0,
            RequirementStatus.EXPIRED: 0.0,
        }.get(
            qualification.requirement_status,
            0.0,
        )

        score = (
            buyer_component
            + service_component
            + evidence_component
            + status_component
        )

        if qualification.buyer_intent_score >= 0.8:
            reasons.append(
                "Strong buyer-intent classification."
            )

        if top_service_percentage >= 70:
            reasons.append(
                "Strong match with a Triway service."
            )

        if evidence_count >= 3:
            reasons.append(
                "Multiple supporting evidence signals "
                "were detected."
            )

        if (
            qualification.requirement_status
            == RequirementStatus.OPEN
        ):
            reasons.append(
                "The requirement is currently open."
            )

        return min(score, 100.0), reasons

    def _get_priority(
        self,
        score: float,
    ) -> str:

        if score >= 80:
            return "critical"

        if score >= 65:
            return "high"

        if score >= 45:
            return "medium"

        return "low"

    def _get_urgency(
        self,
        status: RequirementStatus,
    ) -> str:

        if status == RequirementStatus.OPEN:
            return "high"

        if status == RequirementStatus.UPCOMING:
            return "medium"

        if status in {
            RequirementStatus.CLOSED,
            RequirementStatus.EXPIRED,
        }:
            return "low"

        return "unknown"

    def _get_buying_stage(
        self,
        qualification: QualificationResult,
    ) -> str:

        document_type = (
            qualification.document_type.value
        )

        if document_type in {
            "rfp",
            "rfq",
            "tender",
            "invitation_to_bid",
            "procurement_notice",
        }:
            return "active_procurement"

        if document_type in {
            "eoi",
            "partner_request",
        }:
            return "vendor_evaluation"

        if document_type == "news_about_requirement":
            return "early_signal"

        return "requirement_identified"

    def _build_business_need(
        self,
        lead: LeadProfile,
        top_service_name: str | None,
    ) -> str | None:

        if lead.summary and top_service_name:
            return (
                f"{lead.summary} "
                f"The strongest identified Triway "
                f"capability is {top_service_name}."
            )

        return lead.summary

    def _build_risks(
        self,
        qualification: QualificationResult,
        lead: LeadProfile,
    ) -> list[str]:

        risks: list[str] = []

        if qualification.requirement_status in {
            RequirementStatus.CLOSED,
            RequirementStatus.EXPIRED,
        }:
            risks.append(
                "The requirement may no longer be open."
            )

        if not lead.company_name:
            risks.append(
                "The buyer organization was not "
                "confidently extracted."
            )

        if not lead.country:
            risks.append(
                "The opportunity location is unknown."
            )

        if qualification.confidence < 0.7:
            risks.append(
                "The requirement classification "
                "confidence is limited."
            )

        return risks

    def _recommended_action(
        self,
        priority: str,
        pursue: bool,
    ) -> str:

        if not pursue:
            return (
                "Review manually before assigning the lead "
                "to the sales team."
            )

        if priority == "critical":
            return (
                "Assign immediately to business development "
                "and begin buyer-contact research."
            )

        if priority == "high":
            return (
                "Validate the procurement details and contact "
                "the buyer within one business day."
            )

        return (
            "Add to the opportunity pipeline and perform "
            "additional qualification."
        )

    def _build_talking_points(
        self,
        lead: LeadProfile,
        analysis: AnalyzeLeadResponse,
    ) -> list[str]:

        points: list[str] = []

        for match in analysis.matched_services[:3]:
            points.append(
                f"Position {match.service_name}: "
                f"{match.explanation}"
            )

        if lead.industry:
            points.append(
                f"Tailor the proposal to the "
                f"{lead.industry} industry."
            )

        if lead.country:
            points.append(
                f"Confirm delivery coverage and references "
                f"for {lead.country}."
            )

        return points