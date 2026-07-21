import re
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any

class MetadataExtractor:
    """Rule-based extraction of company name, country, industry, and emails."""

    # Common patterns to find the company name in text
    COMPANY_PATTERNS = [
        r'(?i)(?:^|\.\s)([A-Z][a-z]+(?: [A-Z][a-z]+)*) (?:invites|seeks|issues|announces|launches|plans|is seeking)',
        r'(?i)(?:^|\.\s)([A-Z][a-z]+(?: [A-Z][a-z]+)*) (?:has issued|has published|releases)',
        r'(?i)(?:^|\.\s)([A-Z][a-z]+(?: [A-Z][a-z]+)*) (?:tender|RFP|RFQ|procurement)',
    ]

    # TLD to country mapping (simplified)
    TLD_COUNTRY = {
        'ng': 'Nigeria',
        'uk': 'United Kingdom',
        'us': 'United States',
        'ca': 'Canada',
        'in': 'India',
        'ae': 'United Arab Emirates',
        'sa': 'Saudi Arabia',
        'sg': 'Singapore',
        'my': 'Malaysia',
        'au': 'Australia',
        'de': 'Germany',
        'fr': 'France',
        'jp': 'Japan',
        'kr': 'South Korea',
        'za': 'South Africa',
        'br': 'Brazil',
        'mx': 'Mexico',
        'it': 'Italy',
        'es': 'Spain',
        'nl': 'Netherlands',
        'se': 'Sweden',
        'ch': 'Switzerland',
        'gov': 'United States',  # often US government
        'edu': 'United States',  # often US
    }

    # Industry keywords (simple mapping)
    INDUSTRY_KEYWORDS = {
        'banking': ['bank', 'financial', 'credit', 'insurance', 'investment'],
        'government': ['government', 'public sector', 'ministry', 'agency', 'federal'],
        'healthcare': ['hospital', 'clinic', 'health', 'medical', 'pharma'],
        'corporate': ['corporation', 'inc.', 'llc', 'ltd', 'holdings', 'group'],
        'manufacturing': ['manufacturing', 'factory', 'production', 'industrial'],
        'retail': ['retail', 'store', 'e-commerce', 'commerce'],
        'education': ['university', 'college', 'school', 'education'],
        'technology': ['tech', 'software', 'it', 'solutions', 'digital'],
    }

    @classmethod
    def extract_all(cls, url: str, title: str, snippet: str, text: str) -> Dict[str, Any]:
        """Return dict with company_name, country, industry, emails."""
        return {
            'company_name': cls.extract_company_name(text, title),
            'country': cls.extract_country(url, text),
            'industry': cls.extract_industry(text, title, snippet),
            'emails': cls.extract_emails(text),
        }

    @classmethod
    def extract_company_name(cls, text: str, title: str) -> Optional[str]:
        # Try from title first (often contains company name)
        if title:
            # remove common suffixes like - RFP, - Tender, etc.
            clean_title = re.sub(r'(?i)\s*[-–]\s*(RFP|RFQ|tender|procurement|invitation|notice).*$', '', title).strip()
            if len(clean_title) > 3 and len(clean_title) < 100:
                return clean_title

        # Try patterns in the first few paragraphs
        first_paragraphs = text[:1500]  # look at beginning
        for pattern in cls.COMPANY_PATTERNS:
            match = re.search(pattern, first_paragraphs)
            if match:
                return match.group(1).strip()

        # Fallback: try to find a capitalized phrase before "invites" etc.
        match = re.search(r'(?i)([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+(?:invites|seeks|issues|publishes)', first_paragraphs)
        if match:
            return match.group(1).strip()

        return None

    @classmethod
    def extract_country(cls, url: str, text: str) -> Optional[str]:
        # First, try from URL TLD
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Extract TLD
        parts = domain.split('.')
        if len(parts) > 1:
            tld = parts[-1]
            if tld in cls.TLD_COUNTRY:
                return cls.TLD_COUNTRY[tld]

        # If .com or .org, try to find country mentions in text
        if tld in ('com', 'org', 'net', 'int'):
            country_patterns = [
                r'(?i)in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[,.]',  # "in Nigeria,"
                r'(?i)based\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?i)located\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?i)(?:^|\s)([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)\s+(?:government|ministry|central bank)',
            ]
            for pat in country_patterns:
                match = re.search(pat, text[:2000])
                if match:
                    country = match.group(1).strip()
                    # map common aliases (e.g., "Nigeria" -> "Nigeria")
                    if country in cls.TLD_COUNTRY.values():
                        return country
                    # try to find in known countries list (simple check)
                    for known in cls.TLD_COUNTRY.values():
                        if known.lower() in country.lower() or country.lower() in known.lower():
                            return known
        return None

    @classmethod
    def extract_industry(cls, text: str, title: str, snippet: str) -> Optional[str]:
        combined = (title or '') + ' ' + (snippet or '') + ' ' + text[:2000]
        combined_lower = combined.lower()
        # Score each industry
        scores = {}
        for industry, keywords in cls.INDUSTRY_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in combined_lower:
                    score += 1
            if score > 0:
                scores[industry] = score
        if scores:
            # return the highest scored
            return max(scores, key=scores.get)
        return None

    @classmethod
    def extract_emails(cls, text: str) -> List[str]:
        # Simple email regex (common pattern)
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(email_pattern, text)))