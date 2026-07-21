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


DEADLINE_LABEL_PATTERN = re.compile(
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
        |last\s+date\s+for\s+submission
        |deadline\s+for\s+submission
        |bid\s+closing
        |tender\s+closing
        |proposals?\s+due
        |bids?\s+due
        |submit\s+(?:by|before)
        |due\s+date
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?P<year>20\d{2})[-/.](?P<month>0?[1-9]|1[0-2])[-/.]"
        r"(?P<day>0?[1-9]|[12]\d|3[01])\b"
    ),
    re.compile(
        r"\b(?P<day>0?[1-9]|[12]\d|3[01])[-/.]"
        r"(?P<month>0?[1-9]|1[0-2])[-/.](?P<year>20\d{2})\b"
    ),
    re.compile(
        r"\b(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+"
        r"(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s*,?\s+"
        r"(?P<year>20\d{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+"
        r"(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+"
        r"(?P<year>20\d{2})\b",
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
    'no longer accepting', 'tender closed', 'bidding closed',
    'applications closed', 'archived tender',
}

ACTIVE_LANGUAGE = {
    'open for submission', 'accepting proposals', 'accepting bids',
    'currently open', 'open tender', 'inviting proposals', 'inviting bids',
}


class DeadlineChecker:
    def __init__(self, *, today: date | None = None, grace_days: int = 0, context_window: int = 180):
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

    def _extract_labeled_dates(self, text: str) -> list[tuple[date, str, int]]:
        candidates: list[tuple[date, str, int]] = []
        for label_match in DEADLINE_LABEL_PATTERN.finditer(text):
            start = max(0, label_match.start() - 30)
            end = min(len(text), label_match.end() + self.context_window)
            context = text[start:end]
            for pattern in DATE_PATTERNS:
                for date_match in pattern.finditer(context):
                    parsed = self._parse_date_match(date_match)
                    if parsed is None:
                        continue
                    label_start = label_match.start() - start
                    matched_text = context[label_start:date_match.end()].strip(' :-–—')
                    distance = abs(date_match.start() - (label_match.end() - start))
                    candidates.append((parsed, matched_text, distance))
        return candidates

    def assess(self, *, title: str = '', snippet: str = '', text: str = '') -> DeadlineAssessment:
        combined = self._normalise_text('\n'.join(part for part in [title, snippet, text] if part))
        lowered = combined.casefold()

        if not combined:
            return DeadlineAssessment('unknown', None, None, 'No content was available for deadline analysis.', 0.0)

        expired_phrases = sorted(phrase for phrase in EXPIRED_LANGUAGE if phrase in lowered)
        active_phrases = sorted(phrase for phrase in ACTIVE_LANGUAGE if phrase in lowered)
        labeled_dates = self._extract_labeled_dates(combined)

        if labeled_dates:
            labeled_dates.sort(key=lambda item: (item[2], -item[0].toordinal()))
            best_distance = labeled_dates[0][2]
            nearest = [item for item in labeled_dates if item[2] == best_distance]
            selected_date, matched_text, _ = max(nearest, key=lambda item: item[0])
            expiry_cutoff = selected_date.toordinal() + self.grace_days

            if self.today.toordinal() > expiry_cutoff:
                return DeadlineAssessment(
                    'expired', selected_date, matched_text,
                    f'Detected submission deadline {selected_date.isoformat()}, which is before {self.today.isoformat()}.',
                    0.96,
                )
            return DeadlineAssessment(
                'active', selected_date, matched_text,
                f'Detected submission deadline {selected_date.isoformat()}, which has not passed as of {self.today.isoformat()}.',
                0.96,
            )

        if expired_phrases and not active_phrases:
            return DeadlineAssessment(
                'expired', None, expired_phrases[0],
                'The page explicitly states that the opportunity is closed or expired.',
                0.82,
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