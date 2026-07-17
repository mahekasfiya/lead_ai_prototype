import re
from typing import List, Tuple
from module_3.schemas import QualificationResult, OrganizationRole

class ContradictionChecker:
    def __init__(self):
        self.provider_phrases = [
            r"we provide", r"we offer", r"our services", r"our solutions",
            r"as a service provider", r"we are a leading provider",
            r"we deliver", r"we specialize in", r"we have expertise in",
            r"contact us for", r"request a quote", r"get a free consultation"
        ]
        self.buyer_phrases = [
            r"invites proposals", r"request for proposal", r"request for quotation",
            r"tender", r"procurement", r"eoi", r"expression of interest",
            r"we are seeking", r"we need a", r"looking for a partner",
            r"we require", r"we are looking for", r"call for bids",
            r"bid submission", r"proposal submission"
        ]
        self.provider_regex = [re.compile(p, re.IGNORECASE) for p in self.provider_phrases]
        self.buyer_regex = [re.compile(p, re.IGNORECASE) for p in self.buyer_phrases]

    def check(self, text: str, qual: QualificationResult) -> Tuple[bool, List[str]]:
        text_lower = text.lower()
        provider_count = sum(1 for pat in self.provider_regex if pat.search(text_lower))
        buyer_count = sum(1 for pat in self.buyer_regex if pat.search(text_lower))

        messages = []
        if provider_count > buyer_count * 2 and provider_count >= 3:
            messages.append(f"Provider signals ({provider_count}) dominate buyer signals ({buyer_count})")
        if qual.organization_role == OrganizationRole.BUYER and provider_count > 2:
            messages.append("Classification says buyer but document contains provider-like language")
        if qual.organization_role == OrganizationRole.PROVIDER and buyer_count >= 2:
            messages.append("Classification says provider but document has procurement language")

        passed = len(messages) == 0
        # Override if strong buyer signal
        if qual.is_service_requirement and qual.explicit_requirement and qual.buyer_intent_score >= 0.7:
            if provider_count < buyer_count * 2:
                passed = True
                messages = []
        return passed, messages