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


class PotentialLeadDiscovery:
    """
    Discover potential decision-makers using LLM-generated search queries.

    The class does not create its own Anthropic client. Instead, it receives
    the shared provider-neutral LLM adapter created during application startup.

    Expected LLM adapter interface:

        response = llm_model.generate_content(
            prompt,
            max_tokens=500,
        )

        text = response.text

    Expected search-provider interface:

        results = search_provider.search(
            query,
            num_results=10,
        )
    """

    DEFAULT_TITLES = [
        "CIO",
        "CTO",
        "IT Director",
        "Head of IT",
        "VP Engineering",
        "Procurement Manager",
    ]

    DEFAULT_QUERY_COUNT = 5
    DEFAULT_RESULTS_PER_QUERY = 10

    def __init__(
        self,
        *,
        llm_model: Any,
        search_provider: Any,
        max_query_attempts: int = 3,
        max_enrichment_attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ) -> None:
        if llm_model is None:
            raise ValueError(
                "An LLM model adapter is required."
            )

        if search_provider is None:
            raise ValueError(
                "A search provider is required."
            )

        self.llm_model = llm_model
        self.search_provider = search_provider
        self.max_query_attempts = max(1, max_query_attempts)
        self.max_enrichment_attempts = max(
            1,
            max_enrichment_attempts,
        )
        self.retry_delay_seconds = max(
            0.0,
            retry_delay_seconds,
        )

        self.skip_domains = {
            "crunchbase.com",
            "zoominfo.com",
            "glassdoor.com",
            "aeroleads.com",
            "facebook.com",
            "instagram.com",
            "youtube.com",
            "wikipedia.org",
            "projectstartups.com",
            "dealroom.co",
            "kenresearch.com",
            "tracxn.com",
            "reveliolabs.com",
            "consultancy-me.com",
            "magnitt.com",
            "growthlist.co",
            "fintechnews.ae",
        }

        self.skip_title_patterns = [
            r"(?i)\blist\s+of\b",
            r"(?i)\btop\s+\d+\b",
            r"(?i)\branking\b",
            r"(?i)\bdirectory\b",
            r"(?i)\bmarket\s+report\b",
            r"(?i)\bindustry\s+report\b",
            r"(?i)\btrends\b",
        ]

    def discover(
        self,
        criteria: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Generate search queries, execute them, extract professionals,
        and enrich candidates using the shared LLM adapter.
        """

        normalized_criteria = self._normalize_criteria(criteria)

        try:
            queries = self._generate_queries(
                normalized_criteria
            )
        except Exception:
            logger.exception(
                "Potential-lead query generation failed."
            )
            return []

        if not queries:
            logger.warning(
                "Potential-lead query generation returned no queries."
            )
            return []

        search_results = self._execute_queries(queries)
        people = self._extract_people(search_results)

        enriched: list[dict[str, Any]] = []

        for person in people:
            enriched_person = self._enrich_person(
                person=person,
                criteria=normalized_criteria,
            )

            if (
                enriched_person
                and enriched_person.get("is_relevant") is True
            ):
                enriched.append(enriched_person)

        enriched.sort(
            key=lambda item: item.get("score", 0),
            reverse=True,
        )

        return enriched

    def _normalize_criteria(
        self,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        industries = self._clean_string_list(
            criteria.get("industries")
        )
        countries = self._clean_string_list(
            criteria.get("countries")
        )
        titles = self._clean_string_list(
            criteria.get("titles")
        )
        technologies = self._clean_string_list(
            criteria.get("technologies")
        )

        if not titles:
            titles = list(self.DEFAULT_TITLES)

        min_employees = self._to_positive_int(
            criteria.get("min_employees")
        )
        max_employees = self._to_positive_int(
            criteria.get("max_employees")
        )

        if (
            min_employees is not None
            and max_employees is not None
            and min_employees > max_employees
        ):
            min_employees, max_employees = (
                max_employees,
                min_employees,
            )

        revenue = str(
            criteria.get("revenue") or ""
        ).strip() or None

        recent_funding = bool(
            criteria.get("recent_funding", False)
        )

        return {
            "industries": industries,
            "countries": countries,
            "titles": titles,
            "technologies": technologies,
            "min_employees": min_employees,
            "max_employees": max_employees,
            "revenue": revenue,
            "recent_funding": recent_funding,
        }

    def _generate_queries(
        self,
        criteria: dict[str, Any],
    ) -> list[str]:
        """
        Generate LinkedIn-focused search queries using the shared LLM.
        """

        prompt = self._build_query_prompt(criteria)

        for attempt in range(
            1,
            self.max_query_attempts + 1,
        ):
            try:
                raw = self._call_llm(
                    prompt,
                    max_tokens=900,
                )

                parsed = self._parse_json_response(raw)

                queries = self._normalize_queries(parsed)

                if queries:
                    logger.info(
                        "Potential-lead planner generated %s queries.",
                        len(queries),
                    )
                    return queries

                raise ValueError(
                    "LLM returned no usable search queries."
                )

            except Exception as exc:
                logger.warning(
                    "Potential-lead query generation attempt "
                    "%s/%s failed: %s",
                    attempt,
                    self.max_query_attempts,
                    exc,
                )

                if attempt < self.max_query_attempts:
                    time.sleep(self.retry_delay_seconds)

        return self._fallback_queries(criteria)

    def _build_query_prompt(
        self,
        criteria: dict[str, Any],
    ) -> str:
        industries = criteria["industries"]
        countries = criteria["countries"]
        titles = criteria["titles"]
        technologies = criteria["technologies"]
        min_employees = criteria["min_employees"]
        max_employees = criteria["max_employees"]
        revenue = criteria["revenue"]
        recent_funding = criteria["recent_funding"]

        employee_text = "Not specified"

        if (
            min_employees is not None
            and max_employees is not None
        ):
            employee_text = (
                f"{min_employees} to {max_employees}"
            )
        elif min_employees is not None:
            employee_text = f"At least {min_employees}"
        elif max_employees is not None:
            employee_text = f"Up to {max_employees}"

        return f"""
You are a lead generation assistant. Given these criteria:
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
""".strip()

    def _fallback_queries(
        self,
        criteria: dict[str, Any],
    ) -> list[str]:
        """
        Generate deterministic queries if the LLM planner is unavailable.
        """

        titles = criteria["titles"] or self.DEFAULT_TITLES
        industries = criteria["industries"] or [""]
        countries = criteria["countries"] or [""]
        technologies = criteria["technologies"] or [""]

        queries: list[str] = []

        for title in titles:
            for industry in industries:
                for country in countries:
                    terms = [
                        "site:linkedin.com/in",
                        f'"{title}"',
                    ]

                    if country:
                        terms.append(f'"{country}"')

                    if industry:
                        terms.append(f'"{industry}"')

                    if technologies and technologies[0]:
                        terms.append(
                            f'"{technologies[0]}"'
                        )

                    query = " ".join(terms)

                    if query not in queries:
                        queries.append(query)

                    if (
                        len(queries)
                        >= self.DEFAULT_QUERY_COUNT
                    ):
                        return queries

        return queries

    def _execute_queries(
        self,
        queries: list[str],
    ) -> list[Any]:
        """
        Execute queries through the injected search provider.
        """

        all_results: list[Any] = []

        for query in queries:
            try:
                results = self.search_provider.search(
                    query,
                    num_results=self.DEFAULT_RESULTS_PER_QUERY,
                )

                if isinstance(results, list):
                    all_results.extend(results)
                else:
                    logger.warning(
                        "Search provider returned a non-list "
                        "response for query: %s",
                        query,
                    )

            except Exception:
                logger.exception(
                    "Search-provider failure for query: %s",
                    query,
                )

        return all_results

    def _extract_people(
        self,
        results: list[Any],
    ) -> list[dict[str, Any]]:
        """
        Extract LinkedIn profile candidates from search results.

        Search-result data is treated as unverified public evidence.
        """

        people: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for item in results:
            title, snippet, link = self._read_search_result(
                item
            )

            if not title or not link:
                continue

            normalized_url = self._normalize_linkedin_url(link)

            if not normalized_url:
                continue

            if normalized_url in seen_urls:
                continue

            domain = self._extract_domain(normalized_url)

            if domain in self.skip_domains:
                continue

            if self._matches_skip_title(title):
                continue

            name = self._extract_name(title)

            if not name:
                continue

            job_title, company = (
                self._extract_role_and_company(
                    title=title,
                    snippet=snippet,
                )
            )

            seen_urls.add(normalized_url)

            people.append(
                {
                    "name": name,
                    "linkedin_url": normalized_url,
                    "company": company,
                    "job_title": job_title,
                    "industry": None,
                    "country": None,
                    "source_title": title,
                    "source_snippet": snippet,
                    "score": 0,
                    "is_relevant": False,
                    "verification_status": (
                        "unverified_search_result"
                    ),
                    "relevance_reason": None,
                }
            )

        return people

    def _enrich_person(
        self,
        *,
        person: dict[str, Any],
        criteria: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Use the shared LLM adapter to classify a discovered person.

        The model must use only the supplied title, snippet, company,
        job title and profile URL. Unknown values must remain null.
        """

        prompt = self._build_enrichment_prompt(
            person=person,
            criteria=criteria,
        )

        for attempt in range(
            1,
            self.max_enrichment_attempts + 1,
        ):
            try:
                raw = self._call_llm(
                    prompt,
                    max_tokens=1200,
                )

                parsed = self._parse_json_response(raw)

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "Enrichment response must be a JSON object."
                    )

                enriched = dict(person)

                enriched["industry"] = (
                    self._nullable_string(
                        parsed.get("industry")
                    )
                )
                enriched["country"] = (
                    self._nullable_string(
                        parsed.get("country")
                    )
                )
                enriched["job_title"] = (
                    self._nullable_string(
                        parsed.get("job_title")
                    )
                    or person.get("job_title")
                )
                enriched["company"] = (
                    self._nullable_string(
                        parsed.get("company")
                    )
                    or person.get("company")
                )
                enriched["is_relevant"] = bool(
                    parsed.get("is_relevant", False)
                )
                enriched["relevance_reason"] = (
                    self._nullable_string(
                        parsed.get("reason")
                    )
                )
                enriched["evidence"] = (
                    self._clean_string_list(
                        parsed.get("evidence")
                    )
                )

                enriched["score"] = self._calculate_score(
                    person=enriched,
                    criteria=criteria,
                )

                return enriched

            except Exception as exc:
                logger.warning(
                    "Potential-lead enrichment attempt "
                    "%s/%s failed for %s: %s",
                    attempt,
                    self.max_enrichment_attempts,
                    person.get("name", "unknown"),
                    exc,
                )

                if attempt < self.max_enrichment_attempts:
                    time.sleep(self.retry_delay_seconds)

        return None

    def _build_enrichment_prompt(
        self,
        *,
        person: dict[str, Any],
        criteria: dict[str, Any],
    ) -> str:
        return f"""
You are a lead enrichment assistant. Given a person:
Name: {person['name']}
LinkedIn: {person['linkedin_url']}
Company: {person.get('company', 'Unknown')}
Job Title: {person.get('job_title', 'Unknown')}

Provide the following in JSON format:
- industry: string (the industry of their company, if evident, else null)
- country: string (the country they are based in, if evident, else null)
- is_relevant: boolean (true if they are a decision-maker in target industries/countries)

Target industries: {criteria.get('industries',[])}
Target countries: {criteria.get('countries',[]) }
Decision-maker roles include: CIO, CTO, IT Director, VP Engineering, Procurement Manager, Head of IT.

Return only JSON, no extra text.
""".strip()

    def _calculate_score(
        self,
        *,
        person: dict[str, Any],
        criteria: dict[str, Any],
    ) -> int:
        score = 0

        target_titles = [
            item.casefold()
            for item in criteria["titles"]
        ]
        target_industries = [
            item.casefold()
            for item in criteria["industries"]
        ]
        target_countries = [
            item.casefold()
            for item in criteria["countries"]
        ]

        job_title = str(
            person.get("job_title") or ""
        ).casefold()
        industry = str(
            person.get("industry") or ""
        ).casefold()
        country = str(
            person.get("country") or ""
        ).casefold()

        if job_title and any(
            title in job_title or job_title in title
            for title in target_titles
        ):
            score += 35

        if not target_industries:
            score += 10
        elif industry and any(
            target in industry or industry in target
            for target in target_industries
        ):
            score += 25

        if not target_countries:
            score += 10
        elif country and any(
            target in country or country in target
            for target in target_countries
        ):
            score += 20

        if person.get("company"):
            score += 10

        if person.get("evidence"):
            score += 10

        return min(score, 100)

    def generate_linkedin_message(
        self,
        person: dict[str, Any],
    ) -> str:
        """
        Generate a concise LinkedIn outreach draft using the shared LLM.
        """

        prompt = f"""
You are a professional B2B business-development assistant.

Write a short LinkedIn InMail message for this person:

- Name: {person.get("name") or ""}
- Company: {person.get("company") or "their organization"}
- Job title: {person.get("job_title") or ""}
- Industry: {person.get("industry") or ""}
- Country: {person.get("country") or ""}
- Evidence: {person.get("evidence") or []}

Triway Technologies provides:

- cloud migration
- cybersecurity
- AI automation
- enterprise software implementation

Requirements:

- Keep the message under 150 words.
- Be professional and natural.
- Do not claim knowledge that is not in the supplied evidence.
- Do not say that private LinkedIn data was accessed.
- Include one concise call to action.
- Return only the message text.
""".strip()

        return self._call_llm(
            prompt,
            max_tokens=350,
        )

    def export_csv(
        self,
        leads: list[dict[str, Any]],
        filepath: str | Path,
    ) -> None:
        """
        Export discovered potential leads to CSV.
        """

        output_path = Path(filepath)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "name",
            "linkedin_url",
            "company",
            "job_title",
            "industry",
            "country",
            "score",
            "is_relevant",
            "verification_status",
            "relevance_reason",
        ]

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for lead in leads:
                writer.writerow(
                    {
                        field: lead.get(field)
                        for field in fieldnames
                    }
                )

    def _call_llm(
        self,
        prompt: str,
        *,
        max_tokens: int,
    ) -> str:
        """
        Call the shared provider-neutral LLM adapter.
        """

        if hasattr(self.llm_model, "generate_content"):
            try:
                response = self.llm_model.generate_content(
                    prompt,
                    max_tokens=max_tokens,
                )
            except TypeError:
                response = self.llm_model.generate_content(
                    prompt
                )

        elif hasattr(self.llm_model, "generate"):
            response = self.llm_model.generate(prompt)

        elif callable(self.llm_model):
            response = self.llm_model(prompt)

        else:
            raise TypeError(
                "Unsupported LLM adapter. Expected "
                "generate_content(), generate(), or callable."
            )

        if isinstance(response, str):
            text = response
        else:
            text = getattr(response, "text", None)

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "LLM response did not contain usable text."
            )

        return text.strip()

    @staticmethod
    def _parse_json_response(
        raw: str,
    ) -> Any:
        cleaned = raw.strip()

        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced_match:
            cleaned = fenced_match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            object_start = cleaned.find("{")
            object_end = cleaned.rfind("}")

            if (
                object_start != -1
                and object_end > object_start
            ):
                try:
                    return json.loads(
                        cleaned[
                            object_start : object_end + 1
                        ]
                    )
                except json.JSONDecodeError:
                    pass

            list_start = cleaned.find("[")
            list_end = cleaned.rfind("]")

            if (
                list_start != -1
                and list_end > list_start
            ):
                return json.loads(
                    cleaned[
                        list_start : list_end + 1
                    ]
                )

        raise ValueError(
            "LLM response was not valid JSON."
        )

    @staticmethod
    def _normalize_queries(
        parsed: Any,
    ) -> list[str]:
        if isinstance(parsed, dict):
            parsed = parsed.get("queries")

        if not isinstance(parsed, list):
            raise ValueError(
                "Query-planner output must contain a list."
            )

        queries: list[str] = []
        seen: set[str] = set()

        for item in parsed:
            query = str(item or "").strip()

            if not query:
                continue

            if "site:linkedin.com/in" not in query.casefold():
                continue

            normalized = " ".join(
                query.casefold().split()
            )

            if normalized in seen:
                continue

            seen.add(normalized)
            queries.append(query)

            if (
                len(queries)
                >= PotentialLeadDiscovery.DEFAULT_QUERY_COUNT
            ):
                break

        return queries

    @staticmethod
    def _read_search_result(
        item: Any,
    ) -> tuple[str, str, str]:
        if hasattr(item, "title"):
            title = str(
                getattr(item, "title", "") or ""
            ).strip()
            snippet = str(
                getattr(item, "snippet", "") or ""
            ).strip()
            link = str(
                getattr(item, "url", "") or ""
            ).strip()

            return title, snippet, link

        if isinstance(item, dict):
            title = str(
                item.get("title") or ""
            ).strip()
            snippet = str(
                item.get("snippet") or ""
            ).strip()
            link = str(
                item.get("link")
                or item.get("url")
                or ""
            ).strip()

            return title, snippet, link

        return "", "", ""

    @staticmethod
    def _normalize_linkedin_url(
        url: str,
    ) -> str | None:
        try:
            parsed = urlsplit(url)
        except ValueError:
            return None

        host = parsed.netloc.casefold()

        if not (
            host == "linkedin.com"
            or host.endswith(".linkedin.com")
        ):
            return None

        path = parsed.path.rstrip("/")

        if not path.casefold().startswith("/in/"):
            return None

        return urlunsplit(
            (
                "https",
                parsed.netloc,
                path,
                "",
                "",
            )
        )

    @staticmethod
    def _extract_domain(
        url: str,
    ) -> str:
        domain = urlparse(url).netloc.casefold()
        return domain.removeprefix("www.")

    def _matches_skip_title(
        self,
        title: str,
    ) -> bool:
        return any(
            re.search(pattern, title)
            for pattern in self.skip_title_patterns
        )

    @staticmethod
    def _extract_name(
        title: str,
    ) -> str | None:
        cleaned = re.sub(
            r"\s*\|\s*LinkedIn.*$",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()

        match = re.match(
            r"^(.*?)\s+(?:-|–|—|\|)\s+",
            cleaned,
        )

        if match:
            name = match.group(1).strip()
        else:
            name = cleaned.split(" - ", 1)[0].strip()

        if not name or len(name) > 120:
            return None

        return name

    @staticmethod
    def _extract_role_and_company(
        *,
        title: str,
        snippet: str,
    ) -> tuple[str | None, str | None]:
        job_title: str | None = None
        company: str | None = None

        title_without_linkedin = re.sub(
            r"\s*\|\s*LinkedIn.*$",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()

        title_parts = re.split(
            r"\s+(?:-|–|—|\|)\s+",
            title_without_linkedin,
            maxsplit=1,
        )

        if len(title_parts) == 2:
            role_company = title_parts[1].strip()

            at_match = re.match(
                r"(.+?)\s+at\s+(.+)",
                role_company,
                flags=re.IGNORECASE,
            )

            if at_match:
                job_title = at_match.group(1).strip()
                company = at_match.group(2).strip()
            else:
                job_title = role_company

        snippet_at_match = re.search(
            r"(?P<role>[^.·|]{2,100}?)\s+at\s+"
            r"(?P<company>[^.·|]{2,100})",
            snippet,
            flags=re.IGNORECASE,
        )

        if snippet_at_match:
            job_title = (
                job_title
                or snippet_at_match.group("role").strip()
            )
            company = (
                company
                or snippet_at_match.group(
                    "company"
                ).strip()
            )

        return (
            PotentialLeadDiscovery._nullable_string(
                job_title
            ),
            PotentialLeadDiscovery._nullable_string(
                company
            ),
        )

    @staticmethod
    def _clean_string_list(
        value: Any,
    ) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, (list, tuple, set)):
            return []

        cleaned: list[str] = []
        seen: set[str] = set()

        for item in value:
            text = str(item or "").strip()

            if not text:
                continue

            normalized = text.casefold()

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(text)

        return cleaned

    @staticmethod
    def _nullable_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        text = str(value).strip()

        if not text or text.casefold() in {
            "none",
            "null",
            "unknown",
            "n/a",
        }:
            return None

        return text

    @staticmethod
    def _to_positive_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None

        return parsed if parsed > 0 else None