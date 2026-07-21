from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from module_2.config import settings
from module_2.embedding_provider import EmbeddingProvider
from module_2.local_provider import LocalEmbeddingProvider


DEFAULT_MATCHING_WEIGHTS: dict[str, float] = {
    "embedding": 0.50,
    "technology": 0.20,
    "capability": 0.20,
    "industry": 0.10,
}

SOURCE_RELIABILITY: dict[str, float] = {
    "PROCUREMENT": 0.95,
    "DIRECT_PROJECT": 0.88,
    "PARTNER_SEARCH": 0.85,
    "HIRING_SIGNAL": 0.45,
    "GENERAL_WEB": 0.65,
}

PLATFORM_RELIABILITY: dict[str, float] = {
    "government": 1.00,
    "official": 0.97,
    "procurement": 0.95,
    "serpapi": 0.75,
    "freelancer": 0.90,
    "upwork": 0.92,
    "peopleperhour": 0.88,
    "guru": 0.86,
    "fiverr": 0.78,
    "linkedin": 0.72,
    "web": 0.65,
}


@dataclass(frozen=True)
class MatchContext:
    buyer_intent_score: float = 0.0
    qualification_confidence: float = 0.5
    source_type: str = "GENERAL_WEB"
    platform: str = "web"
    detected_technologies: tuple[str, ...] = ()
    detected_capabilities: tuple[str, ...] = ()
    detected_industries: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoreBreakdown:
    embedding_score: float
    technology_score: float
    capability_score: float
    industry_score: float
    final_score: float
    matched_technologies: tuple[str, ...] = field(default_factory=tuple)
    matched_capabilities: tuple[str, ...] = field(default_factory=tuple)
    matched_industries: tuple[str, ...] = field(default_factory=tuple)


class SimilarityEngine:
    """
    Hybrid matching engine for Triway services.

    The final ranking combines:
    - semantic embedding similarity
    - explicit technology overlap
    - capability overlap
    - industry overlap
    - buyer-intent strength
    - qualification confidence
    - source/platform reliability

    The existing search(query_text, top_k, minimum_score) usage remains valid.
    Additional context can be passed for source-aware ranking.
    """

    def __init__(
        self,
        embeddings_path: Path | None = None,
        provider: EmbeddingProvider | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.embeddings_path = (
            embeddings_path
            or settings.output_directory / "triway_service_embeddings.json"
        )
        self.provider = provider or LocalEmbeddingProvider()
        self.weights = self._validate_weights(
            weights
            or getattr(settings, "matching_weights", None)
            or DEFAULT_MATCHING_WEIGHTS
        )
        self.services = self._load_embeddings()

    @staticmethod
    def _validate_weights(
        weights: Mapping[str, float],
    ) -> dict[str, float]:
        required = set(DEFAULT_MATCHING_WEIGHTS)
        supplied = set(weights)

        missing = required - supplied
        if missing:
            raise ValueError(
                "Missing matching weights: "
                + ", ".join(sorted(missing))
            )

        normalized: dict[str, float] = {}
        for key in required:
            value = float(weights[key])
            if value < 0:
                raise ValueError(
                    f"Matching weight '{key}' cannot be negative."
                )
            normalized[key] = value

        total = sum(normalized.values())
        if total <= 0:
            raise ValueError("Matching weights must sum to more than zero.")

        return {
            key: value / total
            for key, value in normalized.items()
        }

    def _load_embeddings(self) -> list[dict[str, Any]]:
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {self.embeddings_path}"
            )

        try:
            with self.embeddings_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid embeddings JSON: {exc}"
            ) from exc

        services = data.get("services")
        if not isinstance(services, list):
            raise ValueError(
                "Embeddings file must contain a 'services' list."
            )
        if not services:
            raise ValueError(
                "Embeddings file contains no service records."
            )

        validated: list[dict[str, Any]] = []

        for index, service in enumerate(services):
            if not isinstance(service, dict):
                raise ValueError(
                    f"Service at position {index} must be an object."
                )

            vector = service.get("embedding")
            if not isinstance(vector, list):
                raise ValueError(
                    f"Service at position {index} has no valid embedding."
                )

            if len(vector) != self.provider.embedding_dimension:
                raise ValueError(
                    "Embedding dimension mismatch for "
                    f"{service.get('service_id', index)}."
                )

            clean = dict(service)
            clean["_technologies"] = self._extract_terms(
                service,
                (
                    "technologies",
                    "technology",
                    "tools",
                    "platforms",
                    "products",
                    "keywords",
                ),
            )
            clean["_capabilities"] = self._extract_terms(
                service,
                (
                    "capabilities",
                    "capability",
                    "services",
                    "service_capabilities",
                    "offerings",
                    "features",
                ),
            )
            clean["_industries"] = self._extract_terms(
                service,
                (
                    "industries",
                    "industry",
                    "verticals",
                    "sectors",
                ),
            )
            validated.append(clean)

        return validated

    @staticmethod
    def _flatten_values(value: Any) -> Iterable[str]:
        if value is None:
            return []

        if isinstance(value, str):
            parts = re.split(r"[,;/|]\s*|\n+", value)
            return [part.strip() for part in parts if part.strip()]

        if isinstance(value, Mapping):
            flattened: list[str] = []
            for nested_value in value.values():
                flattened.extend(
                    SimilarityEngine._flatten_values(nested_value)
                )
            return flattened

        if isinstance(value, (list, tuple, set)):
            flattened = []
            for item in value:
                flattened.extend(
                    SimilarityEngine._flatten_values(item)
                )
            return flattened

        return [str(value).strip()]

    @classmethod
    def _extract_terms(
        cls,
        service: Mapping[str, Any],
        keys: tuple[str, ...],
    ) -> tuple[str, ...]:
        terms: list[str] = []

        for key in keys:
            if key in service:
                terms.extend(cls._flatten_values(service[key]))

        metadata = service.get("metadata")
        if isinstance(metadata, Mapping):
            for key in keys:
                if key in metadata:
                    terms.extend(cls._flatten_values(metadata[key]))

        deduplicated: list[str] = []
        seen: set[str] = set()

        for term in terms:
            normalized = cls._normalize_term(term)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduplicated.append(term.strip())

        return tuple(deduplicated)

    @staticmethod
    def _normalize_term(value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"[^a-z0-9+#.\- ]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        a = np.asarray(vector_a, dtype=np.float32)
        b = np.asarray(vector_b, dtype=np.float32)

        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0.0

        score = float(np.dot(a, b) / denominator)
        return max(-1.0, min(score, 1.0))

    @classmethod
    def _term_overlap(
        cls,
        query_text: str,
        explicit_terms: Iterable[str],
        service_terms: Iterable[str],
    ) -> tuple[float, tuple[str, ...]]:
        normalized_query = cls._normalize_term(query_text)

        requested = {
            cls._normalize_term(term)
            for term in explicit_terms
            if cls._normalize_term(term)
        }

        service_map = {
            cls._normalize_term(term): term
            for term in service_terms
            if cls._normalize_term(term)
        }

        if not service_map:
            return 0.0, ()

        matched: list[str] = []

        for normalized_service_term, original_term in service_map.items():
            explicitly_detected = normalized_service_term in requested
            present_in_text = cls._contains_term(
                normalized_query,
                normalized_service_term,
            )

            if explicitly_detected or present_in_text:
                matched.append(original_term)

        denominator = max(
            1,
            min(
                len(service_map),
                len(requested) if requested else len(service_map),
            ),
        )
        score = min(len(matched) / denominator, 1.0)

        return score, tuple(sorted(set(matched)))

    @staticmethod
    def _contains_term(
        normalized_text: str,
        normalized_term: str,
    ) -> bool:
        if not normalized_term:
            return False

        if len(normalized_term) <= 2:
            return bool(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(normalized_term)}"
                    rf"(?![a-z0-9])",
                    normalized_text,
                )
            )

        return normalized_term in normalized_text

    @staticmethod
    def _source_score(
        source_type: str,
        platform: str,
    ) -> float:
        source_key = (source_type or "GENERAL_WEB").strip().upper()
        platform_key = (platform or "web").strip().lower()

        source_score = SOURCE_RELIABILITY.get(
            source_key,
            SOURCE_RELIABILITY["GENERAL_WEB"],
        )

        platform_score = PLATFORM_RELIABILITY.get(
            platform_key,
            PLATFORM_RELIABILITY["web"],
        )

        return (source_score * 0.65) + (platform_score * 0.35)

    def _score_service(
            self,
            query_text: str,
            query_vector: list[float],
            service: Mapping[str, Any],
            context: MatchContext,
        ) -> ScoreBreakdown:
        raw_embedding_score = self.cosine_similarity(
            query_vector,
            service["embedding"],
        )
        # Cosine similarity can be negative, while ranking uses 0..1.
        embedding_score = self._clamp(raw_embedding_score)
        technology_score, matched_technologies = self._term_overlap(
            query_text,
            context.detected_technologies,
            service.get("_technologies", ()),
        )
        capability_score, matched_capabilities = self._term_overlap(
            query_text,
            context.detected_capabilities,
            service.get("_capabilities", ()),
        )
        industry_score, matched_industries = self._term_overlap(
            query_text,
            context.detected_industries,
            service.get("_industries", ()),
        )
        component_scores = {
            "embedding": embedding_score,
            "technology": technology_score,
            "capability": capability_score,
            "industry": industry_score,
        }
        final_score = sum(
            component_scores[name] * self.weights[name]
            for name in component_scores
        )
        return ScoreBreakdown(
            embedding_score=round(embedding_score, 4),
            technology_score=round(technology_score, 4),
            capability_score=round(capability_score, 4),
            industry_score=round(industry_score, 4),
            final_score=round(self._clamp(final_score), 4),
            matched_technologies=matched_technologies,
            matched_capabilities=matched_capabilities,
            matched_industries=matched_industries,
        )

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        minimum_score: float = 0.0,
        *,
        buyer_intent_score: float = 0.0,
        qualification_confidence: float = 0.5,
        source_type: str = "GENERAL_WEB",
        platform: str = "web",
        detected_technologies: Iterable[str] | None = None,
        detected_capabilities: Iterable[str] | None = None,
        detected_industries: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Match lead text against Triway services.

        Existing calls remain valid:

            search(query_text, top_k=5, minimum_score=0.0)

        For hybrid ranking, pass the optional qualification and source context.
        minimum_score now applies to final_score rather than raw cosine score.
        """
        query_text = query_text.strip()

        if not query_text:
            raise ValueError("Query text cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between 0.0 and 1.0."
            )

        context = MatchContext(
            buyer_intent_score=self._clamp(buyer_intent_score),
            qualification_confidence=self._clamp(
                qualification_confidence
            ),
            source_type=source_type,
            platform=platform,
            detected_technologies=tuple(
                detected_technologies or ()
            ),
            detected_capabilities=tuple(
                detected_capabilities or ()
            ),
            detected_industries=tuple(
                detected_industries or ()
            ),
        )

        query_vector = self.provider.embed_document(query_text)
        results: list[dict[str, Any]] = []

        for service in self.services:
            breakdown = self._score_service(
                query_text,
                query_vector,
                service,
                context,
            )

            if breakdown.final_score < minimum_score:
                continue

            results.append(
                {
                    "service_id": service["service_id"],
                    "service_name": service["service_name"],
                    "category": service.get("category", "Unknown"),
                    "similarity_score": breakdown.embedding_score,
                    "similarity_percentage": round(
                        breakdown.embedding_score * 100,
                        2,
                    ),
                    "embedding_score": breakdown.embedding_score,
                    "technology_score": breakdown.technology_score,
                    "capability_score": breakdown.capability_score,
                    "industry_score": breakdown.industry_score,
                    "final_score": breakdown.final_score,
                    "final_percentage": round(
                        breakdown.final_score * 100,
                        2,
                    ),
                    "matched_technologies": list(
                        breakdown.matched_technologies
                    ),
                    "matched_capabilities": list(
                        breakdown.matched_capabilities
                    ),
                    "matched_industries": list(
                        breakdown.matched_industries
                    ),
                    "score_weights": dict(self.weights),
                }
            )

        results.sort(
            key=lambda item: (
                item["final_score"],
                item["embedding_score"],
            ),
            reverse=True,
        )

        return results[:top_k]


def print_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    print("\nQuery:")
    print(query)
    print("\nTop matching Triway services:\n")

    if not results:
        print("No matching services found.")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['service_name']}\n"
            f"   Service ID: {result['service_id']}\n"
            f"   Category: {result['category']}\n"
            f"   Final score: {result['final_percentage']}%\n"
            f"   Embedding: {result['similarity_percentage']}%\n"
            f"   Technology: "
            f"{result['technology_score'] * 100:.2f}%\n"
            f"   Capability: "
            f"{result['capability_score'] * 100:.2f}%\n"
            f"   Industry: "
            f"{result['industry_score'] * 100:.2f}%\n"
        )


def main() -> None:
    engine = SimilarityEngine()

    query = input(
        "Enter a company signal, tender description, or lead text:\n> "
    ).strip()

    results = engine.search(
        query_text=query,
        top_k=5,
    )

    print_results(query, results)


if __name__ == "__main__":
    main()