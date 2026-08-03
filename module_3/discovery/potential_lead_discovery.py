from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

import anthropic
from app.search import serpapi

logger = logging.getLogger(__name__)


class PotentialLeadDiscovery:
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.serpapi = serpapi

        # Domains to skip (list pages, directories, social media, etc.)
        self.skip_domains = {
            "crunchbase.com", "zoominfo.com", "glassdoor.com", "aeroleads.com",
            "facebook.com", "instagram.com", "youtube.com",
            "wikipedia.org", "projectstartups.com", "dealroom.co", "kenresearch.com",
            "tracxn.com", "reveliolabs.com", "consultancy-me.com", "magnitt.com",
            "growthlist.co", "fintechnews.ae"
        }

        # Title patterns that indicate list/directory pages (skip them)
        self.skip_title_patterns = [
            r'(?i)list\s+of', r'(?i)top\s+\d+', r'(?i)ranking',
            r'(?i)hub\s+of', r'(?i)directory', r'(?i)trends',
            r'(?i)report\s+on', r'(?i)post\s+by', r'(?i)news\s+article'
        ]

        # Default job titles to search for (can be overridden in criteria)
        self.default_titles = ["CIO", "CTO", "IT Director", "VP Engineering", "Procurement Manager"]

    def discover(self, criteria: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Main entry point: generate LinkedIn-focused queries, execute them,
        extract people, and enrich with Claude.
        """
        try:
            queries = self._generate_queries(criteria)
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return []

        results = self._execute_queries(queries)
        people = self._extract_people(results)
        enriched = []
        for person in people:
            enriched_person = self._enrich_person(person, criteria)
            if enriched_person and enriched_person.get("is_relevant", False):
                enriched.append(enriched_person)

        # Sort by score (highest first)
        enriched.sort(key=lambda x: x.get("score", 0), reverse=True)
        return enriched

    def _generate_queries(self, criteria: dict) -> list[str]:
        """
        Generate LinkedIn-focused search queries using Claude.
        """
        prompt = self._build_query_prompt(criteria)
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-5",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text
                raw = re.sub(r"```json\s*", "", raw)
                raw = re.sub(r"```", "", raw)
                queries = json.loads(raw)
                if isinstance(queries, list) and len(queries) > 0:
                    return queries
            except Exception as e:
                logger.warning(f"Query generation attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return []

    def _build_query_prompt(self, criteria: dict) -> str:
        """
        Build a prompt that asks Claude to generate LinkedIn search queries.
        """
        industries = criteria.get("industries", [])
        countries = criteria.get("countries", [])
        titles = criteria.get("titles", self.default_titles)

        prompt = f"""You are a lead generation assistant. Given these criteria:
- Industries: {industries or "Any"}
- Countries: {countries or "Any"}
- Job titles: {titles}

Generate 5 search queries (in JSON list) that would find professionals on LinkedIn matching these criteria.
Use site:linkedin.com/in in the queries to target LinkedIn profiles.
Examples:
- site:linkedin.com/in "CIO" "UAE" "banking"
- site:linkedin.com/in "CTO" "Saudi Arabia" "fintech"
- site:linkedin.com/in "IT Director" "Dubai" "financial services"

Return only JSON list of strings, no extra text.
"""
        return prompt

    def _execute_queries(self, queries: list[str]) -> list[dict]:
        """
        Execute each query using SerpAPI and collect organic results.
        """
        all_results = []
        for q in queries:
            try:
                res = serpapi.search(q, num_results=10)
                if isinstance(res, list):
                    all_results.extend(res)
            except Exception as e:
                logger.error(f"SerpAPI failed for query '{q}': {e}")
        return all_results

    def _extract_people(self, results: list) -> list[dict]:
        """
        From search results, extract people with LinkedIn URLs.
        Also extract company and job title from snippet when possible.
        """
        people = []
        seen = set()

        for item in results:
            # Handle both dict and SearchResult object
            if hasattr(item, 'title'):
                title = item.title or ""
                snippet = item.snippet or ""
                link = item.url or ""
            else:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "") or item.get("url", "")

            if not title or not link:
                continue

            # Only keep LinkedIn profile URLs
            if "linkedin.com/in" not in link:
                continue

            # Extract name from title (usually "Name - Title at Company | LinkedIn")
            name_match = re.match(r"^(.*?)\s*[-|]", title)
            name = name_match.group(1).strip() if name_match else title.split(" - ")[0].strip()

            # Extract company and job title from snippet
            # Snippet often contains "... at Company · Title"
            company = None
            job_title = None

            if " at " in snippet:
                parts = snippet.split(" at ")
                if len(parts) > 1:
                    job_title = parts[0].strip()
                    company_part = parts[1].split(" · ")[0].strip()
                    company = company_part

            # If no company in snippet, try to infer from title
            if not company:
                match = re.search(r"\| (.*?) on LinkedIn", title)
                if match:
                    company = match.group(1).strip()

            if name and name not in seen:
                seen.add(name)
                people.append({
                    "name": name,
                    "linkedin_url": link,
                    "company": company,
                    "job_title": job_title,
                    "industry": None,
                    "country": None,
                    "score": 0,
                    "is_relevant": False,
                })

        return people

    def _enrich_person(self, person: dict, criteria: dict) -> dict | None:
        """
        Use Claude to enrich a person with industry, country, and relevance.
        """
        industries = criteria.get("industries", [])
        countries = criteria.get("countries", [])

        prompt = f"""
You are a lead enrichment assistant. Given a person:
Name: {person['name']}
LinkedIn: {person['linkedin_url']}
Company: {person.get('company', 'Unknown')}
Job Title: {person.get('job_title', 'Unknown')}

Provide the following in JSON format:
- industry: string (the industry of their company, if evident, else null)
- country: string (the country they are based in, if evident, else null)
- is_relevant: boolean (true if they are a decision-maker in target industries/countries)

Target industries: {industries or "Any"}
Target countries: {countries or "Any"}
Decision-maker roles include: CIO, CTO, IT Director, VP Engineering, Procurement Manager, Head of IT.

Return only JSON, no extra text.
"""
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-5",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text
                raw = re.sub(r"```json\s*", "", raw)
                raw = re.sub(r"```", "", raw)
                data = json.loads(raw)

                person["industry"] = data.get("industry")
                person["country"] = data.get("country")
                person["is_relevant"] = data.get("is_relevant", False)

                # Calculate score
                score = 0
                if person.get("industry") and any(i.lower() in person["industry"].lower() for i in (industries or [])):
                    score += 30
                if person.get("country") and any(c.lower() in person["country"].lower() for c in (countries or [])):
                    score += 30
                if person.get("company"):
                    score += 20
                if person.get("job_title"):
                    score += 20
                person["score"] = score

                return person

            except Exception as e:
                logger.warning(f"Enrichment attempt {attempt+1} failed for {person['name']}: {e}")
                time.sleep(2)

        return None

    def export_csv(self, leads: list[dict], filepath: str):
        """
        Export leads to CSV file.
        """
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "name", "linkedin_url", "company", "job_title",
                "industry", "country", "score", "is_relevant"
            ])
            writer.writeheader()
            for lead in leads:
                writer.writerow({
                    "name": lead.get("name"),
                    "linkedin_url": lead.get("linkedin_url"),
                    "company": lead.get("company"),
                    "job_title": lead.get("job_title"),
                    "industry": lead.get("industry"),
                    "country": lead.get("country"),
                    "score": lead.get("score", 0),
                    "is_relevant": lead.get("is_relevant", False),
                })

    def generate_linkedin_message(self, person: dict) -> str:
        """
        Generate a personalized LinkedIn InMail message using Claude.
        """
        prompt = f"""
    You are a business development assistant. Write a short, professional LinkedIn InMail message to the following person, introducing Triway Technologies' IT services.

    Person: {person.get('name', '')}
    Company: {person.get('company', 'their organization')}
    Job Title: {person.get('job_title', '')}
    Industry: {person.get('industry', '')}
    Country: {person.get('country', '')}

    Triway Technologies offers cloud migration, cybersecurity, AI automation, and enterprise software implementation services.
    The message should be concise, personalized, and mention that you found their profile on LinkedIn. Keep it under 150 words and include a call to action (e.g., "I'd love to connect and learn more about your current projects").
    Return only the message text, no extra formatting.
    """
        response = self.client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()