from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from module_2.config import settings
from module_2.similarity_engine import SimilarityEngine
from module_2.embedding_provider import EmbeddingProvider

class ExplainabilityEngine:

    def __init__(
        self,
        knowledge_base_path: Path | None = None,
        similarity_engine: SimilarityEngine | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.knowledge_base_path = (
            knowledge_base_path
            or settings.knowledge_base_path
        )

        self.similarity_engine = (
            similarity_engine
            or SimilarityEngine(provider=provider)
        )

        self.service_lookup = self._load_service_lookup()

    def _load_service_lookup(self) -> dict[str, dict[str, Any]]:
        """
        Load services from the knowledge base and index them by service_id.
        """
        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: "
                f"{self.knowledge_base_path}"
            )

        try:
            with self.knowledge_base_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid knowledge-base JSON: {exc}"
            ) from exc

        services = data.get("services")

        if not isinstance(services, list):
            raise ValueError(
                "Knowledge base must contain a 'services' list."
            )

        lookup: dict[str, dict[str, Any]] = {}

        for service in services:
            service_id = str(
                service.get("service_id", "")
            ).strip()

            if not service_id:
                raise ValueError(
                    "A service record is missing service_id."
                )

            lookup[service_id] = service

        return lookup

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for matching.

        Example:
        'TAFC-to-TAFJ Migration'
        becomes
        'tafc to tafj migration'
        """
        text = text.lower()

        text = re.sub(
            r"[-_/]",
            " ",
            text,
        )

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _clean_values(values: Any) -> list[str]:
        """
        Return a cleaned list of strings.
        """
        if not isinstance(values, list):
            return []

        cleaned: list[str] = []

        for value in values:
            text = str(value).strip()

            if text:
                cleaned.append(text)

        return cleaned

    def _phrase_matches(
        self,
        query_text: str,
        candidates: list[str],
    ) -> list[str]:
        """
        Find candidate phrases that appear directly in the query.

        Both query and candidate are normalized before matching.
        """
        normalized_query = self.normalize_text(query_text)

        matches: list[str] = []

        for candidate in candidates:
            normalized_candidate = self.normalize_text(candidate)

            if not normalized_candidate:
                continue

            if normalized_candidate in normalized_query:
                matches.append(candidate)

        return matches

    def _token_overlap_matches(
        self,
        query_text: str,
        candidates: list[str],
        minimum_overlap: float = 0.6,
    ) -> list[str]:
        """
        Find partial matches using token overlap.

        This helps match phrases such as:

        Candidate:
        'core banking modernization'

        Query:
        'modernizing the bank's core platform'
        """
        query_tokens = set(
            self.normalize_text(query_text).split()
        )

        matches: list[str] = []

        for candidate in candidates:
            candidate_tokens = set(
                self.normalize_text(candidate).split()
            )

            if not candidate_tokens:
                continue

            overlap = len(
                query_tokens.intersection(candidate_tokens)
            )

            overlap_ratio = overlap / len(candidate_tokens)

            if overlap_ratio >= minimum_overlap:
                matches.append(candidate)

        return matches

    def find_matches(
        self,
        query_text: str,
        values: list[str],
    ) -> list[str]:
        """
        Combine direct phrase matching and token-overlap matching.
        """
        direct_matches = self._phrase_matches(
            query_text,
            values,
        )

        partial_matches = self._token_overlap_matches(
            query_text,
            values,
        )

        combined: list[str] = []

        for match in direct_matches + partial_matches:
            if match not in combined:
                combined.append(match)

        return combined

    @staticmethod
    def confidence_label(score: float) -> str:
        """
        Convert semantic similarity into a readable confidence label.

        These thresholds are provisional and should later be calibrated
        using real lead examples.
        """
        if score >= 0.70:
            return "High"

        if score >= 0.50:
            return "Medium"

        return "Low"

    def explain_service_match(
        self,
        query_text: str,
        similarity_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate an explanation for one matched service.
        """
        service_id = similarity_result["service_id"]

        service = self.service_lookup.get(service_id)

        if service is None:
            raise KeyError(
                f"Service {service_id} not found "
                "in knowledge base."
            )

        field_map = {
            "technologies": "Matched technologies",
            "capabilities": "Matched capabilities",
            "problems_solved": "Matched business problems",
            "buying_signals": "Detected buying signals",
            "evidence_phrases": "Detected evidence",
            "search_keywords": "Matched keywords",
            "target_industries": "Matched industries",
        }

        evidence: dict[str, list[str]] = {}

        for field_name, display_name in field_map.items():
            values = self._clean_values(
                service.get(field_name)
            )

            matches = self.find_matches(
                query_text,
                values,
            )

            if matches:
                evidence[display_name] = matches

        evidence_count = sum(
            len(matches)
            for matches in evidence.values()
        )

        similarity_score = float(
            similarity_result["similarity_score"]
        )

        return {
            "service_id": service_id,
            "service_name": similarity_result[
                "service_name"
            ],
            "category": similarity_result["category"],
            "similarity_score": similarity_score,
            "similarity_percentage": similarity_result[
                "similarity_percentage"
            ],
            "confidence": self.confidence_label(
                similarity_score
            ),
            "evidence_count": evidence_count,
            "evidence": evidence,
            "explanation": self._build_explanation(
                service_name=similarity_result[
                    "service_name"
                ],
                evidence=evidence,
                similarity_percentage=similarity_result[
                    "similarity_percentage"
                ],
            ),
        }

    @staticmethod
    def _build_explanation(
        service_name: str,
        evidence: dict[str, list[str]],
        similarity_percentage: float,
    ) -> str:
        """
        Build a short readable explanation.
        """
        if not evidence:
            return (
                f"The input is semantically similar to "
                f"{service_name} with a score of "
                f"{similarity_percentage}%, but no direct "
                "knowledge-base evidence phrase was found."
            )

        strongest_items: list[str] = []

        preferred_sections = [
            "Matched technologies",
            "Detected buying signals",
            "Detected evidence",
            "Matched capabilities",
            "Matched business problems",
            "Matched keywords",
            "Matched industries",
        ]

        for section in preferred_sections:
            for item in evidence.get(section, []):
                if item not in strongest_items:
                    strongest_items.append(item)

                if len(strongest_items) == 3:
                    break

            if len(strongest_items) == 3:
                break

        evidence_text = ", ".join(strongest_items)

        return (
            f"The input matched {service_name} because it "
            f"contains evidence related to {evidence_text}. "
            f"The semantic similarity score is "
            f"{similarity_percentage}%."
        )

    def analyze(
        self,
        query_text: str,
        top_k: int = 5,
        minimum_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Run semantic retrieval and generate explanations
        for the returned services.
        """
        query_text = query_text.strip()

        if not query_text:
            raise ValueError("Query text cannot be empty.")

        similarity_results = self.similarity_engine.search(
            query_text=query_text,
            top_k=top_k,
            minimum_score=minimum_score,
        )

        explained_results = [
            self.explain_service_match(
                query_text,
                result,
            )
            for result in similarity_results
        ]

        return explained_results


def print_explanations(
    query_text: str,
    results: list[dict[str, Any]],
) -> None:
    """
    Print explainable semantic-search results.
    """
    print("\nInput text:")
    print(query_text)

    print("\nExplainable matches:\n")

    if not results:
        print("No matching services found.")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['service_name']}\n"
            f"   Category: {result['category']}\n"
            f"   Similarity: "
            f"{result['similarity_percentage']}%\n"
            f"   Confidence: {result['confidence']}\n"
            f"   Evidence count: "
            f"{result['evidence_count']}\n"
        )

        if result["evidence"]:
            for section, matches in result[
                "evidence"
            ].items():
                print(f"   {section}:")

                for match in matches:
                    print(f"      - {match}")

        else:
            print(
                "   No direct phrases were detected."
            )

        print(
            f"   Explanation: "
            f"{result['explanation']}\n"
        )


def main() -> None:
    """
    Run an interactive explainability test.
    """
    engine = ExplainabilityEngine()

    query = input(
        "Enter a company signal, tender, or lead text:\n> "
    ).strip()

    results = engine.analyze(
        query_text=query,
        top_k=5,
    )

    print_explanations(
        query,
        results,
    )


if __name__ == "__main__":
    main()