from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    """
    Base interface for all embedding providers.

    The rest of the application will depend on this interface,
    not on a specific model or API provider.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Human-readable provider name.

        Examples:
        - local
        - openai
        - azure_openai
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Name of the embedding model being used.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Number of values in each embedding vector.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts:
                A sequence of non-empty strings.

        Returns:
            A list of embedding vectors in the same order
            as the supplied texts.
        """
        raise NotImplementedError

    def embed_document(self, text: str) -> list[float]:
        """
        Generate an embedding for one document.
        """
        vectors = self.embed_documents([text])

        if len(vectors) != 1:
            raise RuntimeError(
                "Embedding provider returned an unexpected "
                "number of vectors."
            )

        return vectors[0]

    def validate_texts(
        self,
        texts: Sequence[str],
    ) -> list[str]:
        """
        Validate and clean text input before embedding.
        """
        if isinstance(texts, str):
            raise TypeError(
                "embed_documents expects a sequence of strings, "
                "not one string."
            )

        cleaned_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Text at position {index} must be a string."
                )

            cleaned = text.strip()

            if not cleaned:
                raise ValueError(
                    f"Text at position {index} cannot be empty."
                )

            cleaned_texts.append(cleaned)

        if not cleaned_texts:
            raise ValueError(
                "At least one text must be provided."
            )

        return cleaned_texts

    def validate_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
        expected_count: int,
    ) -> list[list[float]]:
        """
        Validate vectors returned by an embedding implementation.
        """
        if len(embeddings) != expected_count:
            raise ValueError(
                "Embedding count mismatch. "
                f"Expected {expected_count}, received {len(embeddings)}."
            )

        validated: list[list[float]] = []

        for index, vector in enumerate(embeddings):
            if len(vector) != self.embedding_dimension:
                raise ValueError(
                    f"Embedding at position {index} has dimension "
                    f"{len(vector)}, expected "
                    f"{self.embedding_dimension}."
                )

            try:
                numeric_vector = [
                    float(value)
                    for value in vector
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Embedding at position {index} contains "
                    "non-numeric values."
                ) from exc

            validated.append(numeric_vector)

        return validated