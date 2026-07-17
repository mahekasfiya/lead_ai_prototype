import logging
from typing import Optional
from module_3.schemas import QualificationResult, OrganizationRole
from module_3.discovery.models import SearchCandidate, FetchedDocument

logger = logging.getLogger(__name__)

class QualificationGate:
    def __init__(self, config: dict):
        self.min_buyer_score = config.get("min_buyer_score", 0.6)
        self.max_provider_prob = config.get("max_provider_prob", 0.4)

    def apply(self, candidate: SearchCandidate, doc: FetchedDocument, qual: QualificationResult, contradiction_passed: bool) -> bool:
        if not qual.is_service_requirement:
            logger.debug(f"Rejected: is_service_requirement false")
            return False
        if qual.organization_role != OrganizationRole.BUYER:
            logger.debug(f"Rejected: organization_role {qual.organization_role} != buyer")
            return False
        if not qual.requires_external_supplier:
            logger.debug("Rejected: requires_external_supplier false")
            return False
        if not qual.explicit_requirement:
            logger.debug("Rejected: explicit_requirement false")
            return False
        if qual.buyer_intent_score < self.min_buyer_score:
            logger.debug(f"Rejected: buyer_intent_score {qual.buyer_intent_score} < {self.min_buyer_score}")
            return False
        if qual.provider_probability > self.max_provider_prob:
            logger.debug(f"Rejected: provider_probability {qual.provider_probability} > {self.max_provider_prob}")
            return False
        if not contradiction_passed:
            logger.debug("Rejected: contradiction check failed")
            return False
        return True