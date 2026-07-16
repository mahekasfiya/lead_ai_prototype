from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from module_2.config import settings


class ServiceMatchScorer:
    """
    Calculates a transparent service-match score.

    This is not the final business lead score. It measures how
    strongly a lead matches a Triway service using the information
    currently available to Modules 1–3.
    """

    SEMANTIC_WEIGHT = 0.45
    EVIDENCE_WEIGHT = 0.40
    REGION_WEIGHT = 0.10
    INDUSTRY_WEIGHT = 0.05

    def __init__(
        self,
        knowledge_base_path: Path | None = None,
    ) -> None:
        self.knowledge_base_path = (
            knowledge_base_path
            or settings.knowledge_base_path
        )

        self.knowledge_base = self._load_knowledge_base()
        self.service_lookup = self._build_service_lookup()
        self.region_lookup = self._build_region_lookup()

    def _load_knowledge_base(self) -> dict[str, Any]:
        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: "
                f"{self.knowledge_base_path}"
            )

        with self.knowledge_base_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Knowledge-base root must be a JSON object."
            )

        return data

    def _build_service_lookup(
        self,
    ) -> dict[str, dict[str, Any]]:
        services = self.knowledge_base.get("services", [])

        lookup: dict[str, dict[str, Any]] = {}

        for service in services:
            service_id = str(
                service.get("service_id", "")
            ).strip()

            if service_id:
                lookup[service_id] = service

        return lookup

    @staticmethod
    def _normalize(value: str | None) -> str:
        if not value:
            return ""

        return " ".join(
            value.lower().strip().split()
        )

    def _build_region_lookup(
        self,
    ) -> dict[str, dict[str, Any]]:
        regions = self.knowledge_base.get(
            "service_regions",
            [],
        )

        lookup: dict[str, dict[str, Any]] = {}

        for region in regions:
            region_name = self._normalize(
                region.get("region")
            )

            country_code = self._normalize(
                region.get("country_code")
            )

            if region_name:
                lookup[region_name] = region

            if country_code:
                lookup[country_code] = region

        aliases = {
            "uae": "united arab emirates",
            "uk": "united kingdom",
            "usa": "united states",
            "us": "united states",
        }

        for alias, official_name in aliases.items():
            official_region = lookup.get(official_name)

            if official_region:
                lookup[alias] = official_region

        return lookup

    @staticmethod
    def calculate_evidence_strength(
        evidence_count: int,
    ) -> float:
        """
        Convert evidence count to a value between 0 and 1.

        Five or more evidence matches receive full evidence strength.
        """
        if evidence_count <= 0:
            return 0.0

        return min(evidence_count / 5.0, 1.0)

    def calculate_region_score(
        self,
        country: str | None,
    ) -> tuple[float, dict[str, Any]]:
        """
        Return region score and region explanation.

        Priority 1 receives 1.0.
        Priority 2 receives 0.7.
        Unknown countries receive 0.0.
        """
        normalized_country = self._normalize(country)

        if not normalized_country:
            return 0.0, {
                "status": "unknown",
                "priority": None,
                "coverage_type": None,
            }

        region = self.region_lookup.get(
            normalized_country
        )

        if region is None:
            return 0.0, {
                "status": "unsupported",
                "priority": None,
                "coverage_type": None,
            }

        priority = region.get("priority")

        if priority == 1:
            score = 1.0
        elif priority == 2:
            score = 0.7
        else:
            score = 0.4

        return score, {
            "status": "supported",
            "priority": priority,
            "coverage_type": region.get(
                "coverage_type"
            ),
            "matched_region": region.get("region"),
        }

    def calculate_industry_score(
        self,
        service_id: str,
        lead_industry: str | None,
    ) -> float:
        if not lead_industry:
            return 0.0

        service = self.service_lookup.get(service_id)

        if service is None:
            return 0.0

        normalized_lead_industry = self._normalize(
            lead_industry
        )

        target_industries = service.get(
            "target_industries",
            [],
        )

        for industry in target_industries:
            normalized_target = self._normalize(
                str(industry)
            )

            if (
                normalized_lead_industry
                in normalized_target
                or normalized_target
                in normalized_lead_industry
            ):
                return 1.0

        return 0.0

    @staticmethod
    def confidence_label(
        score: float,
    ) -> str:
        if score >= 0.75:
            return "High"

        if score >= 0.55:
            return "Medium"

        return "Low"

    def score_match(
        self,
        service_id: str,
        similarity_score: float,
        evidence_count: int,
        lead_country: str | None,
        lead_industry: str | None,
    ) -> dict[str, Any]:
        semantic_component = float(similarity_score)
        
        if semantic_component > 1:
            semantic_component = semantic_component / 100
        
        semantic_component = max(
            0.0,
            min(semantic_component, 1.0),
        )

        evidence_component = (
            self.calculate_evidence_strength(
                evidence_count
            )
        )

        region_component, region_details = (
            self.calculate_region_score(
                lead_country
            )
        )

        industry_component = (
            self.calculate_industry_score(
                service_id,
                lead_industry,
            )
        )

        weighted_semantic = (
            semantic_component
            * self.SEMANTIC_WEIGHT
        )

        weighted_evidence = (
            evidence_component
            * self.EVIDENCE_WEIGHT
        )

        weighted_region = (
            region_component
            * self.REGION_WEIGHT
        )

        weighted_industry = (
            industry_component
            * self.INDUSTRY_WEIGHT
        )

        

        final_score = (
            weighted_semantic
            + weighted_evidence
            + weighted_region
            + weighted_industry
        )

        final_score = round(
            min(final_score, 1.0),
            4,
        )

        return {
            "service_match_score": final_score,
            "service_match_percentage": round(
                final_score * 100,
                2,
            ),
            "service_match_confidence": (
                self.confidence_label(final_score)
            ),
            "score_breakdown": {
                "semantic_similarity": {
                    "raw_score": round(
                        semantic_component,
                        4,
                    ),
                    "weight": self.SEMANTIC_WEIGHT,
                    "weighted_score": round(
                        weighted_semantic,
                        4,
                    ),
                },
                "evidence_strength": {
                    "raw_score": round(
                        evidence_component,
                        4,
                    ),
                    "weight": self.EVIDENCE_WEIGHT,
                    "weighted_score": round(
                        weighted_evidence,
                        4,
                    ),
                    "evidence_count": evidence_count,
                },
                "region_match": {
                    "raw_score": round(
                        region_component,
                        4,
                    ),
                    "weight": self.REGION_WEIGHT,
                    "weighted_score": round(
                        weighted_region,
                        4,
                    ),
                    **region_details,
                },
                "industry_match": {
                    "raw_score": round(
                        industry_component,
                        4,
                    ),
                    "weight": self.INDUSTRY_WEIGHT,
                    "weighted_score": round(
                        weighted_industry,
                        4,
                    ),
                },
            },
        }