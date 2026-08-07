from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeadlineAssessment:
    status: str
    deadline: date | None
    matched_text: str | None
    reason: str
    confidence: float

    @property
    def is_expired(self) -> bool:
        return self.status == "expired"

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def requires_manual_review(self) -> bool:
        return self.status == "unknown"


STRONG_DEADLINE_LABEL_PATTERN = re.compile(
    r"""
    (?:
        submission\s+deadline
        |proposal\s+deadline
        |bid\s+deadline
        |tender\s+deadline
        |application\s+deadline
        |response\s+deadline
        |closing\s+date
        |closing\s+time
        |last\s+date(?:\s+and\s+time)?
            (?:\s+for\s+(?:bid\s+)?submission)?
        |deadline\s+for\s+submission
        |bid\s+closing(?:\s+date|\s+time)?
        |tender\s+closing(?:\s+date|\s+time)?
        |proposals?\s+due
        |bids?\s+due
        |submit\s+(?:by|before)
        |due\s+date
        |submission\s+date
        |proposal\s+submission
            (?:\s+date|\s+deadline)?
        |bid\s+submission
            (?:\s+end)?
            (?:\s+date|\s+deadline)?
        |submission\s+end\s+date
        |response\s+due(?:\s+date)?
        |offer\s+due(?:\s+date)?
        |quotation\s+due(?:\s+date)?
        |rfp\s+due(?:\s+date)?
        |rfq\s+due(?:\s+date)?
        |proposals?\s+shall\s+be\s+received
        |responses?\s+must\s+be\s+received
        |submission\s+closes
        |closing\s+of\s+bids
        |closing\s+of\s+proposals
        |proposal\s+receipt
        |bid\s+receipt
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


AMBIGUOUS_DEADLINE_LABEL_PATTERN = re.compile(
    r"""
    (?:
        proposal\s+opening
        |bid\s+opening
        |sealed\s+proposals
        |sealed\s+bids
        |electronic\s+proposals
        |electronic\s+bids
        |question\s+deadline
        |clarification\s+deadline
        |pre[-\s]?bid\s+meeting
        |award\s+date
        |contract\s+start
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

HISTORICAL_DEADLINE_PATTERN = re.compile(
    r"""
    (?:
        original\s+deadline
        |previous\s+deadline
        |former\s+deadline
        |initial\s+deadline
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # ISO / year-first formats: 2026-07-23, 2026/7/23, 2026.07.23
    re.compile(
        r"\b(?P<year>20\d{2})[-/.](?P<month>0?[1-9]|1[0-2])[-/.]"
        r"(?P<day>0?[1-9]|[12]\d|3[01])\b"
    ),

    # Day-first numeric formats: 23-07-2026, 23/7/2026, 23.07.2026
    re.compile(
        r"\b(?P<day>0?[1-9]|[12]\d|3[01])[-/.]"
        r"(?P<month>0?[1-9]|1[0-2])[-/.](?P<year>20\d{2})\b"
    ),

    # Day month-name year: 23 July 2026, 23-Jul-2026, 23 JUL 2026
    re.compile(
        r"\b(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
        r"(?:\s+|[-/.])"
        r"(?P<month_name>January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
        r"Sep|Sept|Oct|Nov|Dec)"
        r"(?:\s+|[-/.])(?P<year>20\d{2})\b",
        re.IGNORECASE,
    ),

    # Month-name day year: July 23, 2026 / Jul-23-2026
    re.compile(
        r"\b(?P<month_name>January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
        r"Sep|Sept|Oct|Nov|Dec)"
        r"(?:\s+|[-/.])(?P<day>0?[1-9]|[12]\d|3[01])"
        r"(?:st|nd|rd|th)?(?:,?\s+|[-/.])(?P<year>20\d{2})\b",
        re.IGNORECASE,
    ),

    # Year month-name day: 2026 July 23
    re.compile(
        r"\b(?P<year>20\d{2})\s+"
        r"(?P<month_name>January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
        r"Sep|Sept|Oct|Nov|Dec)\s+"
        r"(?P<day>0?[1-9]|[12]\d|3[01])\b",
        re.IGNORECASE,
    ),
)


MONTHS = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
    'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
    'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}

EXPIRED_LANGUAGE = {
    'closed', 'expired', 'deadline has passed', 'submission period has ended',
    'no longer accepting', 'tender closed','tender is closed', 'bidding closed','bidding is closed',
    'applications closed','applications are closed', 'archived tender', 'solicitation closed','opportunity is closed',
    'opportunity closed', 'responses are no longer being accepted',
    'bid submission closed', 'proposal submission closed',
    'contract awarded', 'award notice', 'cancelled solicitation',
    'canceled solicitation',
}

ACTIVE_LANGUAGE = {
    'open for submission', 'accepting proposals', 'accepting bids',
    'currently open', 'open tender', 'inviting proposals', 'inviting bids',
    'responses are being accepted', 'solicitation is open',
    'bid submission is open', 'proposal submission is open',
}


class DeadlineChecker:
    def __init__(
            self,
            *,
            today: date | None = None,
            grace_days: int = 0,
            context_window: int = 500,
    ):
        self.today = today or date.today()
        self.grace_days = max(0, grace_days)
        self.context_window = max(80, context_window)

    @staticmethod
    def _normalise_text(text: str | None) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _parse_date_match(match: re.Match[str]) -> date | None:
        groups = match.groupdict()
        try:
            year = int(groups['year'])
            day = int(groups['day'])
            if groups.get('month'):
                month = int(groups['month'])
            else:
                month = MONTHS[groups.get('month_name', '').casefold()]
            return date(year, month, day)
        except (KeyError, TypeError, ValueError):
            return None

    def _extract_labeled_dates(
            self,
            text: str,
    ) -> list[tuple[date, str, int, bool]]:
        """
        Extract dates near deadline-related labels.
        The final boolean indicates whether the label is a strong
        submission-deadline label.
        """
        candidates: list[tuple[date, str, int, bool]] = []
        label_groups = (
            (STRONG_DEADLINE_LABEL_PATTERN, True),
            (AMBIGUOUS_DEADLINE_LABEL_PATTERN, False),
        )
        
        for label_pattern, is_strong in label_groups:
            for label_match in label_pattern.finditer(text):
                start = max(0, label_match.start() - 120)
                end = min(
                    len(text),
                    label_match.end() + self.context_window,
                )
                context = text[start:end]
                label_start = label_match.start() - start
                label_end = label_match.end() - start

                historical_context = text[
                     max(0, label_match.start() - 30):
                     label_match.end() + 30
                ]
                if HISTORICAL_DEADLINE_PATTERN.search(
                    historical_context
                ):
                    continue

                for pattern in DATE_PATTERNS:
                    for date_match in pattern.finditer(context):
                        parsed = self._parse_date_match(
                            date_match
                        )

                        if parsed is None:
                            continue

                        if date_match.start() >= label_end:
                            distance = (
                                date_match.start() - label_end
                            )
                        else:
                            distance = (
                                label_start
                                - date_match.end()
                                + self.context_window
                            )

                        excerpt_start = min(
                            label_start,
                            date_match.start(),
                        )
                        excerpt_end = max(
                            label_end,
                            date_match.end(),
                        )

                        matched_text = context[
                            excerpt_start:excerpt_end
                        ].strip(" :-–—")

                        candidates.append(
                            (
                                parsed,
                                matched_text,
                                distance,
                                is_strong,
                            )
                        )

        return candidates

    def assess(self, *, title: str = '', snippet: str = '', text: str = '') -> DeadlineAssessment:
        combined = self._normalise_text('\n'.join(part for part in [title, snippet, text] if part))
        lowered = combined.casefold()

        if not combined:
            return DeadlineAssessment('unknown', None, None, 'No content was available for deadline analysis.', 0.0)

        expired_phrases = sorted(phrase for phrase in EXPIRED_LANGUAGE if phrase in lowered)
        active_phrases = sorted(phrase for phrase in ACTIVE_LANGUAGE if phrase in lowered)
        labeled_dates = self._extract_labeled_dates(combined)

        if expired_phrases and not active_phrases:
                    return DeadlineAssessment(
                        'expired', None, expired_phrases[0],
                        'The page explicitly states that the opportunity is closed or expired.',
                        0.82,
                    )

        if labeled_dates:
            strong_dates = [
                item
                for item in labeled_dates
                if item[3] is True
            ]
            candidate_pool = (
                strong_dates
                if strong_dates
                else labeled_dates
            )
            candidate_pool.sort(
                key=lambda item: (
                    item[2],
                    -item[0].toordinal(),
                )
            )
            best_distance = candidate_pool[0][2]
            nearest = [
                item
                for item in candidate_pool
                if item[2] == best_distance
            ]
            selected_date, matched_text, _, is_strong = max(
                nearest,
                key=lambda item: item[0],
            )
            expiry_cutoff = (
                selected_date.toordinal()
                + self.grace_days
            )
            confidence = 0.96 if is_strong else 0.55
            if self.today.toordinal() > expiry_cutoff:
                return DeadlineAssessment(
                    "expired",
                    selected_date,
                    matched_text,
                    (
                        "Detected submission deadline "
                        f"{selected_date.isoformat()}, which is before "
                        f"{self.today.isoformat()}."
                    ),
                    confidence,
                )
            return DeadlineAssessment(
                "active",
                selected_date,
                matched_text,
                (
                    "Detected submission deadline "
                    f"{selected_date.isoformat()}, which has not passed "
                    f"as of {self.today.isoformat()}."
                ),
                confidence,
            )

        if active_phrases and not expired_phrases:
            return DeadlineAssessment(
                'active', None, active_phrases[0],
                'The page explicitly states that submissions or bids are open.',
                0.72,
            )

        return DeadlineAssessment(
            'unknown', None, None,
            'No reliable labeled deadline or explicit active/expired status was found.',
            0.25,
        )