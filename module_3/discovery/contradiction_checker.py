from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from module_3.schemas import QualificationResult, OrganizationRole


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContradictionDecision:
    passed: bool
    source_type: str
    platform: str
    messages: List[str]
    buyer_signal_score: float
    provider_signal_score: float


class ContradictionChecker:
    def __init__(self):
        self.provider_patterns: Dict[str, float] = {
            r"\bwe provide\b": 1.5,
            r"\bwe offer\b": 1.5,
            r"\bour services\b": 1.0,
            r"\bour solutions\b": 1.0,
            r"\bas a service provider\b": 2.0,
            r"\bwe are a leading provider\b": 2.0,
            r"\bwe deliver\b": 1.0,
            r"\bwe specialize in\b": 1.5,
            r"\bwe have expertise in\b": 1.5,
            r"\bcontact us for\b": 0.75,
            r"\brequest a quote\b": 1.0,
            r"\bget a free consultation\b": 2.0,
            r"\bour portfolio\b": 1.5,
            r"\bhire me\b": 2.0,
            r"\bfreelancer profile\b": 2.5,
            r"\bagency profile\b": 2.0,
        }

        self.source_buyer_patterns: Dict[str, Dict[str, float]] = {
            "PROCUREMENT": {
                r"\brequest for proposal\b": 3.0,
                r"\brfp\b": 2.5,
                r"\brequest for quotation\b": 3.0,
                r"\brfq\b": 2.5,
                r"\binvitation to bid\b": 3.0,
                r"\bcall for bids\b": 2.5,
                r"\btender\b": 2.0,
                r"\bprocurement notice\b": 2.5,
                r"\bsubmission deadline\b": 2.5,
                r"\bproposal submission\b": 2.0,
                r"\bbid submission\b": 2.0,
                r"\bscope of work\b": 2.0,
                r"\btender number\b": 2.0,
                r"\binvites proposals\b": 2.5,
                r"\bexpression of interest\b": 2.5,
            },
            "DIRECT_PROJECT": {
                r"\bproject description\b": 2.0,
                r"\bwe need\b": 2.0,
                r"\bneed an? experienced\b": 2.0,
                r"\blooking for an? freelancer\b": 2.5,
                r"\blooking for an? developer\b": 2.5,
                r"\brequired skills\b": 1.5,
                r"\bdeliverables\b": 2.0,
                r"\bbudget\b": 1.5,
                r"\bfixed price\b": 1.5,
                r"\bhourly rate\b": 1.25,
                r"\bmilestone\b": 1.25,
                r"\bsubmit (?:a )?proposal\b": 2.0,
                r"\bplace a bid\b": 2.5,
                r"\bproject owner\b": 1.5,
            },
            "PARTNER_SEARCH": {
                r"\bseeking implementation partner\b": 3.0,
                r"\blooking for (?:an? )?technology partner\b": 3.0,
                r"\blooking for (?:an? )?implementation partner\b": 3.0,
                r"\bseeking system integrator\b": 3.0,
                r"\bseeking vendor\b": 2.5,
                r"\binviting service providers\b": 2.5,
                r"\bvendor selection\b": 2.0,
                r"\bpartner onboarding\b": 1.5,
                r"\brequest for partnership\b": 2.0,
                r"\bexternal consultant\b": 1.5,
            },
            "HIRING_SIGNAL": {
                r"\bwe are hiring\b": 2.0,
                r"\bjob opening\b": 2.0,
                r"\bvacancy\b": 1.5,
                r"\bcareers\b": 1.0,
                r"\brecruiting\b": 1.5,
            },
            "GENERAL_WEB": {
                r"\bwe are seeking\b": 2.0,
                r"\bwe require\b": 2.0,
                r"\bwe are looking for\b": 2.0,
                r"\blooking for a partner\b": 2.5,
                r"\bseeking implementation partner\b": 3.0,
                r"\brequest for proposal\b": 3.0,
                r"\btender\b": 2.0,
            },
        }

        self.source_rules = {
            "PROCUREMENT": {
                "provider_dominance_ratio": 1.75,
                "minimum_provider_score": 3.0,
                "strong_buyer_override": 5.0,
            },
            "DIRECT_PROJECT": {
                "provider_dominance_ratio": 2.5,
                "minimum_provider_score": 4.0,
                "strong_buyer_override": 4.0,
            },
            "PARTNER_SEARCH": {
                "provider_dominance_ratio": 2.0,
                "minimum_provider_score": 3.5,
                "strong_buyer_override": 4.0,
            },
            "HIRING_SIGNAL": {
                "provider_dominance_ratio": 99.0,
                "minimum_provider_score": 99.0,
                "strong_buyer_override": 99.0,
            },
            "GENERAL_WEB": {
                "provider_dominance_ratio": 1.75,
                "minimum_provider_score": 3.0,
                "strong_buyer_override": 5.0,
            },
        }

        self.compiled_provider_patterns = {
            pattern: (re.compile(pattern, re.IGNORECASE), weight)
            for pattern, weight in self.provider_patterns.items()
        }
        self.compiled_buyer_patterns = {
            source_type: {
                pattern: (re.compile(pattern, re.IGNORECASE), weight)
                for pattern, weight in patterns.items()
            }
            for source_type, patterns in self.source_buyer_patterns.items()
        }

    @staticmethod
    def _normalise_source_type(source_type: str | None) -> str:
        value = (source_type or "GENERAL_WEB").strip().upper()
        return value if value in {
            "PROCUREMENT",
            "DIRECT_PROJECT",
            "PARTNER_SEARCH",
            "HIRING_SIGNAL",
            "GENERAL_WEB",
        } else "GENERAL_WEB"

    @staticmethod
    def _normalise_text(value: str | None) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    @staticmethod
    def _score_text(
        text: str,
        compiled_patterns: Dict[str, Tuple[re.Pattern[str], float]],
        multiplier: float,
    ) -> Tuple[float, List[str]]:
        score = 0.0
        matched: List[str] = []

        for raw_pattern, (pattern, weight) in compiled_patterns.items():
            matches = pattern.findall(text)
            if not matches:
                continue

            occurrence_count = len(matches)
            occurrence_weight = min(occurrence_count, 3)
            score += weight * occurrence_weight * multiplier
            matched.append(raw_pattern)

        return score, matched

    def check(
        self,
        text: str,
        qual: QualificationResult,
        source_type: str = "GENERAL_WEB",
        platform: str = "web",
        title: str = "",
        snippet: str = "",
    ) -> ContradictionDecision:
        normalized_source = self._normalise_source_type(source_type)
        normalized_platform = (platform or "web").strip().lower()

        if normalized_source == "HIRING_SIGNAL":
            return ContradictionDecision(
                passed=True,
                source_type=normalized_source,
                platform=normalized_platform,
                messages=[
                    "Hiring signals are enrichment inputs, not direct leads."
                ],
                buyer_signal_score=0.0,
                provider_signal_score=0.0,
            )

        body_text = self._normalise_text(text)
        title_text = self._normalise_text(title)
        snippet_text = self._normalise_text(snippet)

        buyer_patterns = self.compiled_buyer_patterns[normalized_source]

        buyer_title_score, _ = self._score_text(
            title_text,
            buyer_patterns,
            multiplier=3.0,
        )
        buyer_snippet_score, _ = self._score_text(
            snippet_text,
            buyer_patterns,
            multiplier=2.0,
        )
        buyer_body_score, buyer_matches = self._score_text(
            body_text,
            buyer_patterns,
            multiplier=1.0,
        )

        provider_title_score, _ = self._score_text(
            title_text,
            self.compiled_provider_patterns,
            multiplier=3.0,
        )
        provider_snippet_score, _ = self._score_text(
            snippet_text,
            self.compiled_provider_patterns,
            multiplier=2.0,
        )
        provider_body_score, provider_matches = self._score_text(
            body_text,
            self.compiled_provider_patterns,
            multiplier=1.0,
        )

        buyer_score = (
            buyer_title_score
            + buyer_snippet_score
            + buyer_body_score
        )
        provider_score = (
            provider_title_score
            + provider_snippet_score
            + provider_body_score
        )

        rules = self.source_rules[normalized_source]
        messages: List[str] = []

        provider_dominates = (
            provider_score >= rules["minimum_provider_score"]
            and provider_score
            > buyer_score * rules["provider_dominance_ratio"]
        )

        if provider_dominates:
            messages.append(
                "Provider signals dominate buyer signals "
                f"({provider_score:.2f} vs {buyer_score:.2f})."
            )

        if (
            qual.organization_role == OrganizationRole.BUYER
            and provider_dominates
        ):
            messages.append(
                "The classifier marked the page as a buyer, but the page "
                "contains materially stronger provider-language signals."
            )

        if (
            qual.organization_role == OrganizationRole.PROVIDER
            and buyer_score >= rules["strong_buyer_override"]
        ):
            messages.append(
                "The classifier marked the page as a provider, but the page "
                "contains strong source-consistent buyer signals."
            )

        # Marketplace pages often contain platform boilerplate. If the project
        # itself is explicit and buyer-classified, tolerate incidental provider
        # language unless that language strongly dominates title/snippet too.
        if normalized_source == "DIRECT_PROJECT":
            explicit_marketplace_request = (
                qual.is_service_requirement
                and qual.organization_role == OrganizationRole.BUYER
                and qual.explicit_requirement
                and buyer_score >= 2.0
            )

            if explicit_marketplace_request and not (
                provider_title_score > buyer_title_score * 2.0
                or provider_snippet_score > buyer_snippet_score * 2.0
            ):
                messages = [
                    message for message in messages
                    if not message.startswith("Provider signals dominate")
                    and not message.startswith(
                        "The classifier marked the page as a buyer"
                    )
                ]

        strong_buyer_override = (
            qual.is_service_requirement
            and qual.organization_role == OrganizationRole.BUYER
            and qual.explicit_requirement
            and qual.buyer_intent_score >= 0.7
            and buyer_score >= rules["strong_buyer_override"]
            and not provider_dominates
        )

        if strong_buyer_override:
            messages = []

        passed = len(messages) == 0

        logger.debug(
            "Contradiction decision | Source: %s | Platform: %s | "
            "Buyer score: %.2f | Provider score: %.2f | "
            "Passed: %s | Buyer matches: %s | Provider matches: %s",
            normalized_source,
            normalized_platform,
            buyer_score,
            provider_score,
            passed,
            buyer_matches,
            provider_matches,
        )

        return ContradictionDecision(
            passed=passed,
            source_type=normalized_source,
            platform=normalized_platform,
            messages=messages,
            buyer_signal_score=round(buyer_score, 4),
            provider_signal_score=round(provider_score, 4),
        )