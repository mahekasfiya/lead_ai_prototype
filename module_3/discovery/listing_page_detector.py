from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class ListingPageAssessment:
    is_listing_page: bool
    confidence: float
    reason: str
    matched_indicators: list[str]


class ListingPageDetector:
    TITLE_PATTERNS = (
        r"\bglobal\s+.+\s+tenders?\b",
        r"\blatest\s+.+\s+tenders?\b",
        r"\b.+\s+bids?,\s*rfp\s*&\s*government\s+contracts\b",
        r"\bgovernment\s+contracts\b",
        r"\btender\s+listings?\b",
        r"\bbrowse\s+tenders?\b",
        r"\bsearch\s+results?\b",
        r"\ball\s+tenders?\b",
        r"\bpopular\s+tenders?\b",
        r"\bbid\s+opportunities\b",
    )

    URL_PATH_PATTERNS = (
        r"/popular-tenders?/",
        r"/search/",
        r"/category/",
        r"/categories/",
        r"/tag/",
        r"/tags/",
        r"/tenders/$",
        r"/bids/$",
        r"/contracts/$",
        r"/tender-list",
        r"/bid-opportunities",
    )

    BODY_PATTERNS = (
        r"\bshowing\s+\d+\s*(?:-|to)\s*\d+\b",
        r"\bpage\s+\d+\s+of\s+\d+\b",
        r"\bnext\s+page\b",
        r"\bprevious\s+page\b",
        r"\bload\s+more\b",
        r"\bsort\s+by\b",
        r"\bfilter\s+by\b",
        r"\bview\s+all\s+tenders\b",
        r"\bsearch\s+tenders\b",
        r"\bdisplaying\s+\d+\s+results\b",
    )

    SPECIFIC_OPPORTUNITY_PATTERNS = (
        r"\bsolicitation\s+(?:number|id)\b",
        r"\btender\s+(?:reference|number|id)\b",
        r"\brfp\s+(?:number|no\.?|id)\b",
        r"\brfq\s+(?:number|no\.?|id)\b",
        r"\bsubmission\s+deadline\b",
        r"\bclosing\s+date\b",
        r"\bcontracting\s+authority\b",
        r"\bscope\s+of\s+work\b",
    )

    def assess(
        self,
        *,
        title: str = "",
        url: str = "",
        text: str = "",
    ) -> ListingPageAssessment:
        title_text = re.sub(r"\s+", " ", title or "").strip()
        body_text = re.sub(r"\s+", " ", text or "").strip()
        parsed_url = urlparse(url or "")
        path = parsed_url.path.casefold()
        query = parse_qs(parsed_url.query)

        listing_indicators: list[str] = []
        specific_indicators: list[str] = []

        for pattern in self.TITLE_PATTERNS:
            if re.search(pattern, title_text, re.IGNORECASE):
                listing_indicators.append(f"title:{pattern}")

        for pattern in self.URL_PATH_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                listing_indicators.append(f"url:{pattern}")

        for pattern in self.BODY_PATTERNS:
            if re.search(pattern, body_text, re.IGNORECASE):
                listing_indicators.append(f"body:{pattern}")

        for pattern in self.SPECIFIC_OPPORTUNITY_PATTERNS:
            if re.search(
                pattern,
                f"{title_text} {body_text}",
                re.IGNORECASE,
            ):
                specific_indicators.append(pattern)

        # Common directory pagination and search parameters.
        if any(key in query for key in {"page", "pg", "offset", "start"}):
            listing_indicators.append("url:pagination-query")

        # URLs such as /q/cloud_migration are normally search/listing pages.
        if re.search(r"/q/[^/]+/?$", path):
            listing_indicators.append("url:query-listing-path")

        listing_score = len(listing_indicators)
        specific_score = len(specific_indicators)

        is_listing = (
            listing_score >= 2
            and specific_score < 3
        )

        if is_listing:
            return ListingPageAssessment(
                is_listing_page=True,
                confidence=min(0.98, 0.65 + listing_score * 0.07),
                reason=(
                    "The page appears to list multiple opportunities rather "
                    "than describe one specific procurement."
                ),
                matched_indicators=listing_indicators,
            )

        return ListingPageAssessment(
            is_listing_page=False,
            confidence=0.75,
            reason="The page was not confidently classified as a listing page.",
            matched_indicators=specific_indicators,
        )