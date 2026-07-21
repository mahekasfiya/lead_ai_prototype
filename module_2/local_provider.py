from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from module_2.config import settings
from module_2.embedding_provider import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using SentenceTransformers.

    This implementation is used during development.

    Since the rest of the project depends only on the
    EmbeddingProvider interface, replacing this with an
    OpenAI implementation later requires almost no changes.
    """

    def __init__(self) -> None:

        print(
            f"\nLoading embedding model:\n"
            f"{settings.local_embedding_model}\n"
        )

        self._model = SentenceTransformer(
            settings.local_embedding_model
        )

        # Determine embedding dimension automatically
        test_vector = self._model.encode(
            "Triway Technologies",
            normalize_embeddings=settings.normalize_embeddings,
            convert_to_numpy=True,
        )

        self._embedding_dimension = len(test_vector)

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return settings.local_embedding_model

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:

        cleaned_texts = self.validate_texts(texts)

        embeddings = self._model.encode(
            cleaned_texts,
            batch_size=settings.local_embedding_batch_size,
            normalize_embeddings=settings.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype(np.float32)

        validated = self.validate_embeddings(
            embeddings.tolist(),
            expected_count=len(cleaned_texts),
        )

        return validated

    def embed_document(
        self,
        text: str,
    ) -> list[float]:

        return super().embed_document(text)

    def similarity(
        self,
        vector1: Sequence[float],
        vector2: Sequence[float],
    ) -> float:
        """
        Compute cosine similarity.

        Useful later for testing before we move to pgvector.
        """

        a = np.array(vector1, dtype=np.float32)
        b = np.array(vector2, dtype=np.float32)

        similarity = np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

        return float(similarity)

    def information(self) -> dict:

        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "batch_size": settings.local_embedding_batch_size,
            "normalized": settings.normalize_embeddings,
        }