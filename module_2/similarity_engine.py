from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from module_2.config import settings
from module_2.local_provider import LocalEmbeddingProvider
from module_2.embedding_provider import EmbeddingProvider

class SimilarityEngine:
    """
    Semantic similarity engine for matching input text
    against Triway service embeddings.
    """

    def __init__(
        self,
        embeddings_path: Path | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.embeddings_path = (
            embeddings_path
            or settings.output_directory
            / "triway_service_embeddings.json"
        )

        self.provider = provider or LocalEmbeddingProvider()

        self.services = self._load_embeddings()

    def _load_embeddings(self) -> list[dict[str, Any]]:
        """
        Load previously generated service embeddings.
        """
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

        for index, service in enumerate(services):
            vector = service.get("embedding")

            if not isinstance(vector, list):
                raise ValueError(
                    f"Service at position {index} has no valid embedding."
                )

            if len(vector) != self.provider.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch for "
                    f"{service.get('service_id')}."
                )

        return services

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        a = np.asarray(vector_a, dtype=np.float32)
        b = np.asarray(vector_b, dtype=np.float32)

        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return 0.0

        return float(np.dot(a, b) / denominator)

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        minimum_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Match input text against Triway services.

        Args:
            query_text:
                Company description, tender text, news signal,
                or any text to classify.

            top_k:
                Maximum number of results returned.

            minimum_score:
                Exclude results below this similarity threshold.
        """
        query_text = query_text.strip()

        if not query_text:
            raise ValueError("Query text cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.provider.embed_document(query_text)

        results: list[dict[str, Any]] = []

        for service in self.services:
            score = self.cosine_similarity(
                query_vector,
                service["embedding"],
            )

            if score < minimum_score:
                continue

            results.append(
                {
                    "service_id": service["service_id"],
                    "service_name": service["service_name"],
                    "category": service["category"],
                    "similarity_score": round(score, 4),
                    "similarity_percentage": round(score * 100, 2),
                }
            )

        results.sort(
            key=lambda item: item["similarity_score"],
            reverse=True,
        )

        return results[:top_k]


def print_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    """
    Print search results in a readable format.
    """
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
            f"   Similarity: "
            f"{result['similarity_percentage']}%\n"
        )


def main() -> None:
    """
    Run an interactive semantic search test.
    """
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