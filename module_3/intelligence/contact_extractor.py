from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from module_3.intelligence.models import ExtractedContact


class ContactExtractor:
    """
    Extracts publicly available business contact details from lead text.

    This component does not perform web scraping. It analyzes text and URLs
    already collected by the discovery pipeline.
    """

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    PHONE_PATTERN = re.compile(
        r"""
        (?:
            \+?\d{1,3}[\s.\-()]*
        )?
        (?:
            \d{2,4}[\s.\-()]*
        ){2,4}
        \d{2,4}
        """,
        re.VERBOSE,
    )

    CONTACT_NAME_PATTERNS = [
        re.compile(
            r"(?:contact person|contact|procurement officer|tender officer)"
            r"\s*[:\-]\s*"
            r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,3})",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:attention|attn\.?)\s*[:\-]\s*"
            r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,3})",
            re.IGNORECASE,
        ),
    ]

    DEPARTMENT_KEYWORDS = {
        "procurement": "Procurement",
        "purchasing": "Purchasing",
        "sourcing": "Strategic Sourcing",
        "tender": "Tender Department",
        "information technology": "Information Technology",
        "it department": "Information Technology",
        "digital transformation": "Digital Transformation",
        "finance department": "Finance",
        "contracts department": "Contracts",
        "vendor management": "Vendor Management",
    }

    PROCUREMENT_LINK_TERMS = (
        "tender",
        "procurement",
        "vendor",
        "supplier",
        "bid",
        "rfp",
        "registration",
    )

    def extract(
        self,
        text: str | None,
        source_url: str | None = None,
    ) -> list[ExtractedContact]:
        clean_text = self._normalize_text(text)

        emails = self._extract_emails(clean_text)
        phones = self._extract_phones(clean_text)
        names = self._extract_names(clean_text)
        departments = self._extract_departments(clean_text)
        procurement_links = self._extract_urls(
            clean_text,
            source_url=source_url,
        )

        contacts: list[ExtractedContact] = []

        maximum_contacts = max(
            len(emails),
            len(phones),
            len(names),
            1 if departments or procurement_links else 0,
        )

        for index in range(maximum_contacts):
            contact = ExtractedContact(
                name=names[index] if index < len(names) else None,
                email=emails[index] if index < len(emails) else None,
                phone=phones[index] if index < len(phones) else None,
                department=departments[0] if departments else None,
                procurement_url=(
                    procurement_links[index]
                    if index < len(procurement_links)
                    else None
                ),
                source="lead_text",
            )

            if self._has_contact_data(contact):
                contacts.append(contact)

        return contacts

    def _normalize_text(self, text: str | None) -> str:
        if not text:
            return ""

        cleaned_lines = [
            re.sub(r"[ \t]+", " ", line).strip()
            for line in text.splitlines()
        ]
        return "\n".join(
            line
            for line in cleaned_lines
            if line
        )

    def _extract_emails(self, text: str) -> list[str]:
        emails = {
            email.lower().strip(".,;:")
            for email in self.EMAIL_PATTERN.findall(text)
        }

        ignored_domains = {
            "example.com",
            "email.com",
            "domain.com",
        }

        return sorted(
            email
            for email in emails
            if email.split("@")[-1] not in ignored_domains
        )

    def _extract_phones(self, text: str) -> list[str]:
        phones: set[str] = set()

        for match in self.PHONE_PATTERN.findall(text):
            normalized = re.sub(r"\s+", " ", match).strip(" .,-")

            digit_count = len(re.sub(r"\D", "", normalized))

            if 7 <= digit_count <= 15:
                phones.add(normalized)

        return sorted(phones)

    def _extract_names(self, text: str) -> list[str]:
        names: list[str] = []
        patterns = [
            re.compile(
                r"^(?:contact person|contact|procurement officer|"
                r"tender officer)[ \t]*[:\-][ \t]*"
                r"([A-Z][A-Za-z.'\-]+"
                r"(?:[ \t]+[A-Z][A-Za-z.'\-]+){1,3})"
                r"[ \t]*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            re.compile(
                r"^(?:attention|attn\.?)[ \t]*[:\-][ \t]*"
                r"([A-Z][A-Za-z.'\-]+"
                r"(?:[ \t]+[A-Z][A-Za-z.'\-]+){1,3})"
                r"[ \t]*$",
                re.IGNORECASE | re.MULTILINE,
            ),
        ]
        for pattern in patterns:
            for match in pattern.findall(text):
                clean_name = match.strip(" .,-")
                if clean_name and clean_name not in names:
                    names.append(clean_name)
        return names

    def _extract_departments(self, text: str) -> list[str]:
        lowered_text = text.lower()
        departments: list[str] = []

        for keyword, normalized_name in self.DEPARTMENT_KEYWORDS.items():
            if keyword in lowered_text and normalized_name not in departments:
                departments.append(normalized_name)

        return departments

    def _extract_urls(
        self,
        text: str,
        source_url: str | None,
    ) -> list[str]:
        raw_urls = re.findall(
            r"https?://[^\s<>'\"\]\)]+",
            text,
            flags=re.IGNORECASE,
        )

        procurement_urls: list[str] = []

        for raw_url in raw_urls:
            clean_url = raw_url.rstrip(".,;:")

            if self._is_procurement_url(clean_url):
                procurement_urls.append(clean_url)

        if source_url and self._is_procurement_url(source_url):
            procurement_urls.append(source_url)

        return list(dict.fromkeys(procurement_urls))

    def _is_procurement_url(self, url: str) -> bool:
        lowered_url = url.lower()

        return any(
            term in lowered_url
            for term in self.PROCUREMENT_LINK_TERMS
        )

    def _has_contact_data(self, contact: ExtractedContact) -> bool:
        return any(
            [
                contact.name,
                contact.email,
                contact.phone,
                contact.department,
                contact.procurement_url,
            ]
        )