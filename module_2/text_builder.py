from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EMBEDDING_FIELDS = (
    "service_name",
    "category",
    "description",
    "problems_solved",
    "capabilities",
    "technologies",
    "target_industries",
    "buying_signals",
    "search_keywords",
)


def _clean_text(value: Any) -> str:
    """
    Convert a value into clean text.

    Removes unnecessary whitespace and safely handles None values.
    """
    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def _clean_list(values: Any) -> list[str]:
    """
    Convert a JSON list into a cleaned list of non-empty strings.
    """
    if not isinstance(values, list):
        return []

    cleaned: list[str] = []

    for value in values:
        text = _clean_text(value)

        if text:
            cleaned.append(text)

    return cleaned


def _format_list_section(title: str, values: list[str]) -> str:
    """
    Format list values into a stable embedding-text section.
    """
    if not values:
        return ""

    content = "\n".join(f"- {value}" for value in values)

    return f"{title}:\n{content}"


def build_embedding_text(service: dict[str, Any]) -> str:
    """
    Build stable, embedding-friendly text for one Triway service.

    Metadata such as service_id, region priority, country code,
    and region policy are intentionally excluded.
    """
    sections: list[str] = []

    service_name = _clean_text(service.get("service_name"))
    category = _clean_text(service.get("category"))
    description = _clean_text(service.get("description"))

    if not service_name:
        raise ValueError("Service record is missing service_name.")

    sections.append(f"Service: {service_name}")

    if category:
        sections.append(f"Category: {category}")

    if description:
        sections.append(f"Description:\n{description}")

    list_sections = (
        ("Problems solved", "problems_solved"),
        ("Capabilities", "capabilities"),
        ("Technologies", "technologies"),
        ("Target industries", "target_industries"),
        ("Buying signals", "buying_signals"),
        ("Search keywords", "search_keywords"),
    )

    for title, field_name in list_sections:
        values = _clean_list(service.get(field_name))
        section = _format_list_section(title, values)

        if section:
            sections.append(section)

    return "\n\n".join(sections).strip()


def generate_content_hash(embedding_text: str) -> str:
    """
    Generate a SHA-256 hash for the embedding text.

    This allows the system to detect whether a service definition
    has changed and therefore requires re-embedding.
    """
    normalized_text = _clean_text(embedding_text)

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def prepare_service_record(service: dict[str, Any]) -> dict[str, Any]:
    """
    Create the record that will later be passed to the embedding provider.
    """
    service_id = _clean_text(service.get("service_id"))
    service_name = _clean_text(service.get("service_name"))
    category = _clean_text(service.get("category"))

    if not service_id:
        raise ValueError("Service record is missing service_id.")

    embedding_text = build_embedding_text(service)

    return {
        "service_id": service_id,
        "service_name": service_name,
        "category": category,
        "embedding_text": embedding_text,
        "content_hash": generate_content_hash(embedding_text),
    }


def load_knowledge_base(path: Path) -> dict[str, Any]:
    """
    Load and validate the Triway knowledge-base JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge-base file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in knowledge-base file: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Knowledge-base root must be a JSON object."
        )

    services = data.get("services")

    if not isinstance(services, list):
        raise ValueError(
            "Knowledge base must contain a 'services' list."
        )

    if not services:
        raise ValueError(
            "Knowledge base contains no service records."
        )

    return data


def prepare_all_services(
    knowledge_base_path: Path,
) -> list[dict[str, Any]]:
    """
    Load the knowledge base and prepare all services for embedding.
    """
    knowledge_base = load_knowledge_base(
        knowledge_base_path
    )

    prepared_records: list[dict[str, Any]] = []
    seen_service_ids: set[str] = set()

    for index, service in enumerate(
        knowledge_base["services"],
        start=1,
    ):
        if not isinstance(service, dict):
            raise ValueError(
                f"Service record at position {index} "
                "must be a JSON object."
            )

        prepared = prepare_service_record(service)

        service_id = prepared["service_id"]

        if service_id in seen_service_ids:
            raise ValueError(
                f"Duplicate service_id found: {service_id}"
            )

        seen_service_ids.add(service_id)
        prepared_records.append(prepared)

    return prepared_records