from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from module_3.schemas import QualificationResult, OrganizationRole
from module_3.discovery.models import SearchCandidate, FetchedDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualificationDecision:
    accepted: bool
    source_type: str
    reason: str | None
    buyer_intent_score: float
    provider_probability: float

    def __bool__(self) -> bool:
        """Preserve compatibility with existing `if decision:` code."""
        return self.accepted


class QualificationGate:
    """Apply source-aware qualification rules to discovered opportunities."""

    SOURCE_ALIASES = {
        "procurement": "PROCUREMENT",
        "tender": "PROCUREMENT",
        "rfp": "PROCUREMENT",
        "rfq": "PROCUREMENT",
        "direct_project": "DIRECT_PROJECT",
        "marketplace": "DIRECT_PROJECT",
        "freelancer": "DIRECT_PROJECT",
        "peopleperhour": "DIRECT_PROJECT",
        "partner_search": "PARTNER_SEARCH",
        "partner": "PARTNER_SEARCH",
        "general_web": "PARTNER_SEARCH",
        "hiring_signal": "HIRING_SIGNAL",
        "hiring": "HIRING_SIGNAL",
        "job_board": "HIRING_SIGNAL",
    }

    def __init__(self, config: dict[str, Any]):
        # Existing keys remain supported as procurement defaults.
        procurement_buyer = float(config.get("min_buyer_score", 0.60))
        procurement_provider = float(config.get("max_provider_prob", 0.40))

        configured_thresholds = config.get("qualification_thresholds", {})

        self.thresholds = {
            "PROCUREMENT": {
                "min_buyer_score": procurement_buyer,
                "max_provider_prob": procurement_provider,
                "require_external_supplier": True,
                "require_explicit_requirement": True,
                "require_contradiction_pass": True,
            },
            "DIRECT_PROJECT": {
                "min_buyer_score": 0.45,
                "max_provider_prob": 0.60,
                # Publishing a project on a marketplace already implies
                # willingness to use an external provider.
                "require_external_supplier": False,
                "require_explicit_requirement": True,
                "require_contradiction_pass": True,
            },
            "PARTNER_SEARCH": {
                "min_buyer_score": 0.40,
                "max_provider_prob": 0.50,
                "require_external_supplier": False,
                # Partner-search language can be explicit without resembling
                # a formal procurement specification.
                "require_explicit_requirement": False,
                "require_contradiction_pass": True,
            },
            "HIRING_SIGNAL": {
                "min_buyer_score": 1.0,
                "max_provider_prob": 0.0,
                "require_external_supplier": False,
                "require_explicit_requirement": False,
                "require_contradiction_pass": True,
            },
        }

        self._merge_configured_thresholds(configured_thresholds)

    def _merge_configured_thresholds(self, configured: Any) -> None:
        if not isinstance(configured, dict):
            return

        for raw_source_type, values in configured.items():
            source_type = self._normalise_source_type(raw_source_type)

            if source_type not in self.thresholds or not isinstance(values, dict):
                continue

            current = self.thresholds[source_type]

            for key in (
                "min_buyer_score",
                "max_provider_prob",
                "require_external_supplier",
                "require_explicit_requirement",
                "require_contradiction_pass",
            ):
                if key in values:
                    current[key] = values[key]

    @classmethod
    def _normalise_source_type(cls, source_type: str | None) -> str:
        if not source_type:
            return "PROCUREMENT"

        cleaned = str(source_type).strip().replace("-", "_").casefold()
        return cls.SOURCE_ALIASES.get(cleaned, cleaned.upper())

    @staticmethod
    def _score(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _reject(
        self,
        *,
        source_type: str,
        qual: QualificationResult,
        reason: str,
    ) -> QualificationDecision:
        decision = QualificationDecision(
            accepted=False,
            source_type=source_type,
            reason=reason,
            buyer_intent_score=self._score(qual.buyer_intent_score),
            provider_probability=self._score(qual.provider_probability),
        )
        logger.debug(
            "Qualification rejected | Source: %s | Reason: %s | "
            "Buyer score: %.3f | Provider probability: %.3f",
            source_type,
            reason,
            decision.buyer_intent_score,
            decision.provider_probability,
        )
        return decision

    def apply(
        self,
        candidate: SearchCandidate,
        doc: FetchedDocument,
        qual: QualificationResult,
        contradiction_passed: bool,
        source_type: str = "PROCUREMENT",
    ) -> QualificationDecision:
        del candidate, doc  # Reserved for future source/domain-specific rules.

        source_type = self._normalise_source_type(source_type)

        # Hiring is a discovery signal, not a qualified sales opportunity.
        if source_type == "HIRING_SIGNAL":
            return self._reject(
                source_type=source_type,
                qual=qual,
                reason="Hiring signals require enrichment and cannot qualify directly.",
            )

        rules = self.thresholds.get(source_type)
        if rules is None:
            logger.warning(
                "Unknown qualification source type '%s'; using PROCUREMENT rules.",
                source_type,
            )
            source_type = "PROCUREMENT"
            rules = self.thresholds[source_type]

        if not qual.is_service_requirement:
            return self._reject(
                source_type=source_type,
                qual=qual,
                reason="The content is not classified as a service requirement.",
            )

        if qual.organization_role != OrganizationRole.BUYER:
            return self._reject(
                source_type=source_type,
                qual=qual,
                reason=f"Organization role is {qual.organization_role}, not BUYER.",
            )

        if rules["require_external_supplier"] and not qual.requires_external_supplier:
            return self._reject(
                source_type=source_type,
                qual=qual,
                reason="No requirement for an external supplier was detected.",
            )

        if rules["require_explicit_requirement"] and not qual.explicit_requirement:
            return self._reject(
                source_type=source_type,
                qual=qual,
                reason="The service requirement is not explicit enough for this source.",
            )

        buyer_score = self._score(qual.buyer_intent_score)
        min_buyer_score = float(rules["min_buyer_score"])
        if buyer_score < min_buyer_score:
            return self._reject(
                source_type=source_type,
                qual=qual,
                reason=(
                    f"Buyer intent score {buyer_score:.3f} is below the "
                    f"{source_type} threshold {min_buyer_score:.3f}."
                ),
            )

        provider_probability = self._score(qual.provider_probability)
        max_provider_probability = float(rules["max_provider_prob"])
        if provider_probability > max_provider_probability:
            return self._reject(
                source_type=source_type,
                qual=qual,
                reason=(
                    f"Provider probability {provider_probability:.3f} exceeds the "
                    f"{source_type} limit {max_provider_probability:.3f}."
                ),
            )

        if rules["require_contradiction_pass"] and not contradiction_passed:
            return self._reject(
                source_type=source_type,
                qual=qual,
                reason="Contradiction check failed.",
            )

        decision = QualificationDecision(
            accepted=True,
            source_type=source_type,
            reason=None,
            buyer_intent_score=buyer_score,
            provider_probability=provider_probability,
        )
        logger.debug(
            "Qualification accepted | Source: %s | Buyer score: %.3f | "
            "Provider probability: %.3f",
            source_type,
            buyer_score,
            provider_probability,
        )
        return decision