from __future__ import annotations

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit

logger = logging.getLogger(__name__)


class CompanyDiscovery:
    """
    Discover companies listed on specified websites (primary) and, if needed,
    fall back to general web search (secondary).
    Uses the shared LLM adapter and search provider.
    """

    DEFAULT_WEBSITES = [
        "dubaichamber.com",
        "dubai-businessdirectory.com",
        "yellowpages-uae.com",
        "ded.ae",
        "ae.opensooq.com",
        "ae.kompass.com",
        "bayt.com",
        "digitaldubai.ae",
        "dubaipulse.gov.ae",
        "dsc.gov.ae",
        "moec.gov.ae",
        "dubaicustoms.gov.ae",
        "atozservicesuae.com",
        "zoominfo.com",
        "scribd.com",
        "issuu.com",
        "slideshare.net",
        "academia.edu",
        "calameo.com",
        "bookboon.com",
        "openlibrary.org",
        "gutenberg.org",
        "docdroid.net",
    ]

    # Minimum number of leads to consider primary search successful
    MIN_PRIMARY_LEADS = 3

    def __init__(
        self,
        *,
        llm_model: Any,
        search_provider: Any,
        max_query_attempts: int = 3,
        max_enrichment_attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        if llm_model is None:
            raise ValueError("LLM model adapter is required.")
        if search_provider is None:
            raise ValueError("Search provider is required.")

        self.llm_model = llm_model
        self.search_provider = search_provider
        self.max_query_attempts = max(1, max_query_attempts)
        self.max_enrichment_attempts = max(1, max_enrichment_attempts)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)

        self.skip_domains = {
            "facebook.com", "instagram.com", "youtube.com", "wikipedia.org",
            "linkedin.com",  # separate pipeline for LinkedIn
        }

        self.skip_title_patterns = [
            r"(?i)\b(?:list|top|ranking|directory)\b",
        ]

    def discover(
        self,
        criteria: dict[str, Any],
        websites: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Discover companies:
        1. Primary: search on specified websites (region‑specific).
        2. If primary yields < MIN_PRIMARY_LEADS, fall back to secondary: general web search.
        """
        target_websites = websites or self.DEFAULT_WEBSITES
        # Filter out obviously unsuitable ones
        filtered = [
            w for w in target_websites
            if w not in {"wattpad.com", "facebook.com", "instagram.com", "youtube.com"}
        ]

        # ---- Primary search ----
        all_companies = []
        for site in filtered:
            logger.info(f"Primary: Searching site: {site}")
            companies = self._discover_on_site(site, criteria)
            all_companies.extend(companies)

        # Remove duplicates (by name and domain)
        unique = {}
        for c in all_companies:
            key = (c.get("name", "").casefold(), c.get("domain", "").casefold())
            if key not in unique:
                unique[key] = c

        result = list(unique.values())

        # Country filter (primary)
        target_countries = criteria.get("countries", [])
        if target_countries:
            target_lower = [c.casefold() for c in target_countries]
            original_count = len(result)
            result = [
                c for c in result
                if c.get("country") and c["country"].casefold() in target_lower
            ]
            logger.info(
                "Primary country filter: kept %s of %s companies (targets: %s)",
                len(result),
                original_count,
                target_countries,
            )

        # If we have enough leads, return them
        if len(result) >= self.MIN_PRIMARY_LEADS:
            logger.info("Primary search successful: %s leads", len(result))
            return result

        # ---- Secondary search (fallback) ----
        logger.info(
            "Primary search insufficient (%s leads). Running secondary general web search.",
            len(result)
        )
        secondary_results = self._secondary_search(criteria)
        if secondary_results:
            logger.info("Secondary search found %s leads", len(secondary_results))
            # Combine and deduplicate (primary results are already filtered)
            combined = result + secondary_results
            # Deduplicate again (some may overlap)
            unique = {}
            for c in combined:
                key = (c.get("name", "").casefold(), c.get("domain", "").casefold())
                if key not in unique:
                    unique[key] = c
            return list(unique.values())

        return result

    def _discover_on_site(self, site: str, criteria: dict) -> list[dict]:
        queries = self._generate_queries_for_site(site, criteria)
        if not queries:
            return []

        results = self._execute_queries(queries)
        candidates = self._extract_company_candidates(results, site)
        enriched = []
        for cand in candidates:
            enriched_c = self._enrich_company(cand, criteria)
            if enriched_c and enriched_c.get("is_company", False):
                enriched.append(enriched_c)
        return enriched

    def _generate_queries_for_site(self, site: str, criteria: dict) -> list[str]:
        target_countries = criteria.get("countries", ["UAE", "Saudi Arabia", "Oman"])
        country_terms = " OR ".join(f'"{c}"' for c in target_countries)

        prompt = f"""
You are a lead generation assistant. Given:
- Website domain: {site}
- Target countries: {target_countries}

Generate 5 search queries (in JSON list) that would find companies listed on this website in these countries.
The queries should use "site:{site}" and include country names.
Examples:
- site:yellowpages-uae.com "UAE" "company"
- site:bayt.com "Saudi Arabia" "business"
- site:dubaichamber.com "member" "Dubai"
- site:zoominfo.com "UAE" "company profile"

Return only JSON list of strings, no extra text.
"""
        for attempt in range(1, self.max_query_attempts + 1):
            try:
                raw = self._call_llm(prompt, max_tokens=500)
                parsed = self._parse_json_response(raw)
                if isinstance(parsed, list):
                    queries = [str(q).strip() for q in parsed if q]
                    return queries
            except Exception:
                logger.warning(f"Query generation attempt {attempt} failed for {site}")
                time.sleep(self.retry_delay_seconds)
        return []

    def _secondary_search(self, criteria: dict) -> list[dict]:
        """
        Fallback: general web search (no site restriction) for companies in target countries.
        """
        target_countries = criteria.get("countries", ["UAE", "Saudi Arabia", "Oman"])
        industries = criteria.get("industries", [])

        # Build generic queries
        queries = []
        for country in target_countries:
            for industry in industries[:2]:  # limit to first two to save tokens
                queries.append(f'"{industry}" companies in "{country}"')
            # Also add generic "top companies" queries if no industry provided
            if not industries:
                queries.append(f'"top companies" in "{country}"')
                queries.append(f'"business directory" "{country}"')

        # Limit to 5 queries
        queries = queries[:5]

        if not queries:
            logger.warning("No secondary queries generated.")
            return []

        logger.info("Secondary queries: %s", queries)

        all_results = []
        for q in queries:
            try:
                res = self.search_provider.search(q, num_results=10)
                if isinstance(res, list):
                    all_results.extend(res)
            except Exception:
                logger.exception(f"Secondary search failed for query: {q}")

        # Extract candidates (no site restriction)
        candidates = []
        seen = set()
        for item in all_results:
            title, snippet, link = self._read_search_result(item)
            if not title or not link:
                continue
            domain = urlparse(link).netloc
            if domain in self.skip_domains:
                continue
            name = title.split(" - ")[0].strip()
            if not name or len(name) < 2:
                continue
            if name.casefold() in seen:
                continue
            seen.add(name.casefold())
            candidates.append({
                "name": name,
                "domain": domain,
                "url": link,
                "snippet": snippet,
                "source_site": "web_search",
                "industry": None,
                "country": None,
                "buying_signals": [],
                "requires_it_services": False,
                "contacts": [],
                "score": 0,
                "is_company": False,
            })

        # Enrich candidates
        enriched = []
        for cand in candidates:
            enriched_c = self._enrich_company(cand, criteria)
            if enriched_c and enriched_c.get("is_company", False):
                enriched.append(enriched_c)

        # Filter by country again
        target_lower = [c.casefold() for c in target_countries]
        filtered = [
            c for c in enriched
            if c.get("country") and c["country"].casefold() in target_lower
        ]
        logger.info("Secondary country filter: kept %s of %s", len(filtered), len(enriched))
        return filtered

    def _execute_queries(self, queries: list[str]) -> list[Any]:
        results = []
        for q in queries:
            try:
                res = self.search_provider.search(q, num_results=10)
                if isinstance(res, list):
                    results.extend(res)
            except Exception:
                logger.exception(f"Search failed for query: {q}")
        return results

    def _extract_company_candidates(self, results: list, site: str) -> list[dict]:
        candidates = []
        seen = set()
        for item in results:
            title, snippet, link = self._read_search_result(item)
            if not title or not link:
                continue
            domain = urlparse(link).netloc
            if domain in self.skip_domains:
                continue
            name = title.split(" - ")[0].strip()
            if not name or len(name) < 2:
                continue
            if name.casefold() in seen:
                continue
            seen.add(name.casefold())
            candidates.append({
                "name": name,
                "domain": domain,
                "url": link,
                "snippet": snippet,
                "source_site": site,
                "industry": None,
                "country": None,
                "buying_signals": [],
                "requires_it_services": False,
                "contacts": [],
                "score": 0,
                "is_company": False,
            })
        return candidates

    def _enrich_company(self, company: dict, criteria: dict) -> dict | None:
        prompt = f"""
You are a company enrichment assistant. Given:

Company name: {company['name']}
Domain: {company['domain']}
Snippet: {company['snippet']}

Provide the following in JSON:
- industry: string (industry of the company, if evident)
- country: string (country the company is based in, if evident)
- is_company: boolean (true if this is a specific company, not a list or directory)
- buying_signals: list of strings (e.g., ["cloud migration", "cybersecurity", "AI automation"])
- requires_it_services: boolean (true if the company might need IT services like cloud, security, or automation)
- contacts: array of objects with keys: title (e.g., CEO, CIO), email (if found), phone (if found), linkedin (if found)
Only include contacts if explicitly mentioned in the snippet.

Return only JSON.
"""
        for attempt in range(1, self.max_enrichment_attempts + 1):
            try:
                raw = self._call_llm(prompt, max_tokens=600)
                data = self._parse_json_response(raw)
                if isinstance(data, dict):
                    company["industry"] = self._nullable_string(data.get("industry"))
                    company["country"] = self._nullable_string(data.get("country"))
                    company["is_company"] = bool(data.get("is_company", False))
                    company["buying_signals"] = data.get("buying_signals", [])
                    company["requires_it_services"] = bool(data.get("requires_it_services", False))
                    company["contacts"] = data.get("contacts", [])
                    company["score"] = self._calculate_company_score(company, criteria)
                    return company
            except Exception:
                logger.warning(f"Enrichment attempt {attempt} failed for {company['name']}")
                time.sleep(self.retry_delay_seconds)
        return None

    def _calculate_company_score(self, company: dict, criteria: dict) -> int:
        score = 0

        target_industries = [i.casefold() for i in criteria.get("industries", [])]
        industry = str(company.get("industry") or "").casefold()
        if target_industries and any(t in industry or industry in t for t in target_industries):
            score += 30

        target_countries = [c.casefold() for c in criteria.get("countries", [])]
        country = str(company.get("country") or "").casefold()
        if target_countries and any(t in country or country in t for t in target_countries):
            score += 20

        signals = company.get("buying_signals", [])
        if signals:
            score += min(len(signals) * 10, 30)

        if company.get("requires_it_services", False):
            score += 20

        return min(score, 100)

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        if hasattr(self.llm_model, "generate_content"):
            try:
                response = self.llm_model.generate_content(prompt, max_tokens=max_tokens)
            except TypeError:
                response = self.llm_model.generate_content(prompt)
        else:
            raise TypeError("Unsupported LLM adapter")
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise ValueError("LLM response not text")
        return text.strip()

    def _parse_json_response(self, raw: str) -> Any:
        cleaned = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                return json.loads(cleaned[start:end+1])
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end > start:
                return json.loads(cleaned[start:end+1])
            raise ValueError("Could not parse JSON")

    def _read_search_result(self, item: Any) -> tuple[str, str, str]:
        if hasattr(item, "title"):
            return str(getattr(item, "title", "") or ""), str(getattr(item, "snippet", "") or ""), str(getattr(item, "url", "") or "")
        if isinstance(item, dict):
            return str(item.get("title", "")), str(item.get("snippet", "")), str(item.get("link", item.get("url", "")))
        return "", "", ""

    @staticmethod
    def _nullable_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in ("none", "null", "unknown"):
            return None
        return text

    def export_csv(self, leads: list[dict], filepath: str | Path):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "domain", "industry", "country", "score", "contacts", "source_site"])
            writer.writeheader()
            for lead in leads:
                writer.writerow({
                    "name": lead.get("name"),
                    "domain": lead.get("domain"),
                    "industry": lead.get("industry"),
                    "country": lead.get("country"),
                    "score": lead.get("score", 0),
                    "contacts": json.dumps(lead.get("contacts", [])),
                    "source_site": lead.get("source_site"),
                })