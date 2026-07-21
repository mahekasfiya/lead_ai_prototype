from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from module_2.config import settings
from module_2.local_provider import LocalEmbeddingProvider
from module_2.text_builder import prepare_all_services


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def create_provider() -> LocalEmbeddingProvider:
    """
    Create the configured embedding provider.

    For now, only the local provider is enabled.
    Later, this function can return OpenAI or Azure providers.
    """
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider()

    raise ValueError(
        f"Unsupported embedding provider: {settings.embedding_provider}"
    )


def build_embedding_records(
    prepared_services: list[dict[str, Any]],
    provider: LocalEmbeddingProvider,
) -> list[dict[str, Any]]:
    """
    Generate embeddings and attach them to service metadata.
    """
    texts = [
        service["embedding_text"]
        for service in prepared_services
    ]

    logger.info(
        "Generating embeddings for %s services.",
        len(texts),
    )

    vectors = provider.embed_documents(texts)

    if len(vectors) != len(prepared_services):
        raise RuntimeError(
            "Number of generated vectors does not match "
            "the number of prepared services."
        )

    generated_at = datetime.now(timezone.utc).isoformat()

    embedding_records: list[dict[str, Any]] = []

    for service, vector in zip(
        prepared_services,
        vectors,
        strict=True,
    ):
        embedding_records.append(
            {
                "service_id": service["service_id"],
                "service_name": service["service_name"],
                "category": service["category"],
                "embedding_text": service["embedding_text"],
                "content_hash": service["content_hash"],
                "embedding_provider": provider.provider_name,
                "embedding_model": provider.model_name,
                "embedding_dimension": provider.embedding_dimension,
                "embedding_version": settings.embedding_version,
                "normalized": settings.normalize_embeddings,
                "generated_at": generated_at,
                "embedding": vector,
            }
        )

    return embedding_records


def save_embeddings(
    records: list[dict[str, Any]],
    output_directory: Path,
) -> Path:
    """
    Save embedding records as JSON.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "triway_service_embeddings.json"
    )

    output_data = {
        "metadata": {
            "provider": records[0]["embedding_provider"],
            "model": records[0]["embedding_model"],
            "dimension": records[0]["embedding_dimension"],
            "embedding_version": records[0]["embedding_version"],
            "normalized": records[0]["normalized"],
            "service_count": len(records),
            "generated_at": records[0]["generated_at"],
        },
        "services": records,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def main() -> None:
    """
    Run the complete service embedding generation process.
    """
    logger.info(
        "Loading knowledge base from: %s",
        settings.knowledge_base_path,
    )

    prepared_services = prepare_all_services(
        settings.knowledge_base_path
    )

    logger.info(
        "Prepared %s service records.",
        len(prepared_services),
    )

    provider = create_provider()

    logger.info(
        "Provider: %s | Model: %s | Dimension: %s",
        provider.provider_name,
        provider.model_name,
        provider.embedding_dimension,
    )

    records = build_embedding_records(
        prepared_services,
        provider,
    )

    output_path = save_embeddings(
        records,
        settings.output_directory,
    )

    logger.info(
        "Embeddings generated successfully."
    )

    logger.info(
        "Output saved to: %s",
        output_path,
    )

    logger.info(
        "Generated %s vectors with dimension %s.",
        len(records),
        provider.embedding_dimension,
    )


if __name__ == "__main__":
    main()