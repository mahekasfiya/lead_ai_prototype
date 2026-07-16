from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from module_2.config import settings


BUYING_INTENT_SUFFIXES = [
    "RFP",
    "tender",
    "procurement",
    "seeking implementation partner",
    "implementation project",
]


class QueryGenerator:
    """
    Generates buying-intent search queries from the Triway
    service knowledge base.
    """

    def __init__(
        self,
        knowledge_base_path: Path | None = None,
    ) -> None:
        self.knowledge_base_path = (
            knowledge_base_path
            or settings.knowledge_base_path
        )

        self.services = self._load_services()

    def _load_services(self) -> list[dict[str, Any]]:
        with self.knowledge_base_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        services = data.get("services", [])

        if not isinstance(services, list):
            raise ValueError(
                "Knowledge base must contain a services list."
            )

        return services

    @staticmethod
    def _first_non_empty(
        service: dict[str, Any],
        field_name: str,
    ) -> str | None:
        values = service.get(field_name, [])

        if not isinstance(values, list):
            return None

        for value in values:
            text = str(value).strip()

            if text:
                return text

        return None

    def generate(
        self,
        max_queries: int,
        selected_service_ids: list[str] | None = None,
    ) -> list[dict[str, str]]:
        selected_ids = set(
            selected_service_ids or []
        )

        candidates: list[dict[str, str]] = []

        for service in self.services:
            service_id = str(
                service.get("service_id", "")
            ).strip()

            if (
                selected_ids
                and service_id not in selected_ids
            ):
                continue

            service_name = str(
                service.get("service_name", "")
            ).strip()

            phrase = (
                self._first_non_empty(
                    service,
                    "evidence_phrases",
                )
                or self._first_non_empty(
                    service,
                    "search_keywords",
                )
                or service_name
            )

            if not phrase:
                continue

            suffix = BUYING_INTENT_SUFFIXES[
                len(candidates)
                % len(BUYING_INTENT_SUFFIXES)
            ]

            query = f'"{phrase}" {suffix}'

            candidates.append(
                {
                    "service_id": service_id,
                    "service_name": service_name,
                    "query": query,
                }
            )

            if len(candidates) >= max_queries:
                break

        return candidates