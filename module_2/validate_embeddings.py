from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from module_2.config import settings
from module_2.embedding_provider import EmbeddingProvider


class EmbeddingValidationError(Exception):
    pass


def load_embedding_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise EmbeddingValidationError(
            f"Embeddings file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise EmbeddingValidationError(
            f"Invalid embeddings JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise EmbeddingValidationError(
            "Embeddings file root must be a JSON object."
        )

    return data


def validate_embedding_file(
    provider: EmbeddingProvider,
    embeddings_path: Path | None = None,
) -> dict[str, Any]:
    path = (
        embeddings_path
        or settings.output_directory
        / "triway_service_embeddings.json"
    )

    data = load_embedding_file(path)

    metadata = data.get("metadata")
    services = data.get("services")

    if not isinstance(metadata, dict):
        raise EmbeddingValidationError(
            "Missing or invalid metadata section."
        )

    if not isinstance(services, list) or not services:
        raise EmbeddingValidationError(
            "Missing or empty services list."
        )

    stored_model = metadata.get("model")
    stored_dimension = metadata.get("dimension")
    stored_normalized = metadata.get("normalized")

    if stored_model != provider.model_name:
        raise EmbeddingValidationError(
            f"Model mismatch. Stored model: {stored_model}. "
            f"Current model: {provider.model_name}."
        )

    if stored_dimension != provider.embedding_dimension:
        raise EmbeddingValidationError(
            f"Dimension mismatch. Stored dimension: "
            f"{stored_dimension}. Current dimension: "
            f"{provider.embedding_dimension}."
        )

    if stored_normalized != settings.normalize_embeddings:
        raise EmbeddingValidationError(
            "Normalization setting does not match the "
            "current configuration."
        )

    seen_service_ids: set[str] = set()

    for index, service in enumerate(services, start=1):
        if not isinstance(service, dict):
            raise EmbeddingValidationError(
                f"Service at position {index} must be an object."
            )

        service_id = str(
            service.get("service_id", "")
        ).strip()

        if not service_id:
            raise EmbeddingValidationError(
                f"Service at position {index} is missing service_id."
            )

        if service_id in seen_service_ids:
            raise EmbeddingValidationError(
                f"Duplicate service_id found: {service_id}"
            )

        seen_service_ids.add(service_id)

        vector = service.get("embedding")

        if not isinstance(vector, list):
            raise EmbeddingValidationError(
                f"{service_id} has no valid embedding list."
            )

        if len(vector) != provider.embedding_dimension:
            raise EmbeddingValidationError(
                f"{service_id} has dimension {len(vector)}, "
                f"expected {provider.embedding_dimension}."
            )

        for value in vector:
            if not isinstance(value, (int, float)):
                raise EmbeddingValidationError(
                    f"{service_id} contains a non-numeric value."
                )

            if not math.isfinite(float(value)):
                raise EmbeddingValidationError(
                    f"{service_id} contains NaN or infinity."
                )

        if settings.normalize_embeddings:
            magnitude = float(
                np.linalg.norm(
                    np.asarray(vector, dtype=np.float32)
                )
            )

            if not 0.98 <= magnitude <= 1.02:
                raise EmbeddingValidationError(
                    f"{service_id} is expected to be normalized, "
                    f"but magnitude is {magnitude:.4f}."
                )

    return {
        "status": "valid",
        "path": str(path),
        "provider": provider.provider_name,
        "model": provider.model_name,
        "dimension": provider.embedding_dimension,
        "service_count": len(services),
        "normalized": settings.normalize_embeddings,
        "embedding_version": metadata.get(
            "embedding_version"
        ),
    }

def main() -> None:
    from module_2.local_provider import (
        LocalEmbeddingProvider,
    )

    provider = LocalEmbeddingProvider()

    result = validate_embedding_file(provider)

    print("\nEmbedding validation passed.\n")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()