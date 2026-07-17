import json
import logging
from typing import List
from module_3.schemas import (
    QualificationResult, DocumentType, OrganizationRole, RequirementStatus
)

logger = logging.getLogger(__name__)

class RequirementClassifier:
    def __init__(self, llm_model=None, use_gemini: bool = True, max_chunks: int = 3):
        self.llm_model = llm_model
        self.use_gemini = use_gemini
        self.max_chunks = max_chunks

    def classify(self, text: str, text_chunks: List[str]) -> QualificationResult:
        combined = " ".join(text_chunks[:self.max_chunks])
        if len(combined) > 12000:
            combined = combined[:12000]

        if self.use_gemini and self.llm_model:
            try:
                return self._classify_with_gemini(combined)
            except Exception as e:
                logger.error(f"Gemini classification failed: {e}, falling back to rule-based")
                return self._classify_rule_based(combined)
        else:
            return self._classify_rule_based(combined)

    def _classify_with_gemini(self, text: str) -> QualificationResult:
        prompt = f"""
You are a classifier for B2B IT services lead discovery.

Analyze this web page text and determine if the organization is a BUYER (seeking external IT services) or a PROVIDER (selling services).

Classify the following attributes:
- document_type: Choose one: rfp, rfq, eoi, tender, procurement_notice, invitation_to_bid, direct_requirement, partner_request, implementation_announcement, digital_transformation_initiative, modernization_project, news_about_requirement, vendor_service_page, directory, job_posting, training, article, unknown
- is_service_requirement: boolean (true if this is a genuine request for services or signal of intent)
- organization_role: buyer, provider, aggregator, publisher, unknown
- requirement_status: open, upcoming, closed, expired, unclear (how active is this opportunity?)
- buyer_intent_score: float 0.0-1.0 (how strong is the buying intent? 0.9+ = explicit RFP/tender, 0.7+ = clear partner request, 0.5+ = implementation announcement, 0.3+ = modernization initiative)
- provider_probability: float 0.0-1.0 (probability it's a provider page)
- explicit_requirement: boolean (does it explicitly ask for external supplier/vendor?)
- requires_external_supplier: boolean (does it require external vendor assistance?)
- evidence_quotes: list of up to 3 short quotes supporting your classification
- rejection_reasons: list of reasons if NOT a buyer requirement
- confidence: float 0.0-1.0 (overall confidence in this classification)

Page text:
{text}

Return ONLY valid JSON with these fields.
"""
        response = self.llm_model.generate_content(prompt)
        content = response.text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        data = json.loads(content)

        return QualificationResult(
            document_type=DocumentType(data.get("document_type", "unknown")),
            is_service_requirement=data.get("is_service_requirement", False),
            organization_role=OrganizationRole(data.get("organization_role", "unknown")),
            requirement_status=RequirementStatus(data.get("requirement_status", "unclear")),
            buyer_intent_score=float(data.get("buyer_intent_score", 0.0)),
            provider_probability=float(data.get("provider_probability", 0.0)),
            explicit_requirement=data.get("explicit_requirement", False),
            requires_external_supplier=data.get("requires_external_supplier", False),
            evidence_quotes=data.get("evidence_quotes", []),
            rejection_reasons=data.get("rejection_reasons", []),
            confidence=float(data.get("confidence", 0.5))
        )

    def _classify_rule_based(self, text: str) -> QualificationResult:
        # Keep the rule-based classifier we wrote earlier as a fallback
        # (same code as before - I'll include it here for completeness)
        import re
        text_lower = text.lower()

        buyer_patterns = [
            r"rfp", r"request for proposal", r"rfq", r"request for quotation",
            r"tender", r"procurement", r"eoi", r"expression of interest",
            r"invites proposals", r"call for bids", r"bid submission",
            r"we are seeking", r"we need a", r"looking for a partner",
            r"we require", r"we are looking for", r"seeking implementation partner",
            r"invitation to bid", r"announces", r"implementation", r"modernization",
            r"digital transformation", r"hiring", r"recruiting"
        ]
        provider_patterns = [
            r"we provide", r"we offer", r"our services", r"our solutions",
            r"as a service provider", r"we are a leading provider",
            r"we deliver", r"we specialize in", r"we have expertise in",
            r"contact us for", r"request a quote", r"get a free consultation",
            r"managed services", r"service offering", r"market research", r"report"
        ]

        buyer_regex = [re.compile(p, re.IGNORECASE) for p in buyer_patterns]
        provider_regex = [re.compile(p, re.IGNORECASE) for p in provider_patterns]

        buyer_count = sum(1 for pat in buyer_regex if pat.search(text_lower))
        provider_count = sum(1 for pat in provider_regex if pat.search(text_lower))

        is_buyer = buyer_count >= 2 and provider_count <= buyer_count
        role = OrganizationRole.BUYER if is_buyer else OrganizationRole.PROVIDER
        is_explicit = any(p in text_lower for p in ["rfp", "tender", "invites", "proposal"])

        buyer_intent = min(buyer_count / 4.0, 1.0)
        provider_prob = min(provider_count / 4.0, 1.0)

        if provider_count > buyer_count * 2:
            role = OrganizationRole.PROVIDER
            is_buyer = False

        doc_type = DocumentType.UNKNOWN
        if "rfp" in text_lower:
            doc_type = DocumentType.RFP
        elif "tender" in text_lower:
            doc_type = DocumentType.TENDER
        elif "hiring" in text_lower or "recruiting" in text_lower:
            doc_type = DocumentType.JOB_POSTING
        elif "announces" in text_lower or "implementation" in text_lower:
            doc_type = DocumentType.DIRECT_REQUIREMENT

        status = RequirementStatus.OPEN
        if "closed" in text_lower or "expired" in text_lower:
            status = RequirementStatus.CLOSED
        elif "upcoming" in text_lower:
            status = RequirementStatus.UPCOMING

        return QualificationResult(
            document_type=doc_type,
            is_service_requirement=is_buyer,
            organization_role=role,
            requirement_status=status,
            buyer_intent_score=buyer_intent,
            provider_probability=provider_prob,
            explicit_requirement=is_explicit,
            requires_external_supplier=is_explicit,
            evidence_quotes=[],
            rejection_reasons=[] if is_buyer else ["No strong buyer intent detected"],
            confidence=buyer_intent
        )