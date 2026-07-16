from __future__ import annotations


from module_2.explainability_engine import (
    ExplainabilityEngine,
)

from module_2.embedding_provider import (
    EmbeddingProvider,
)

from module_2.similarity_engine import (
    SimilarityEngine,
)

from module_3.scoring import (
    ServiceMatchScorer,
)

from module_3.schemas import (
    AnalyzeLeadRequest,
    AnalyzeLeadResponse,
    EvidenceResponse,
    ServiceMatchResponse,
)


class LeadAnalysisService:
    """
    Service layer connecting:

    Lead profile
        |
        v
    Semantic retrieval
        |
        v
    Explainability
        |
        v
    Business scoring
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
    ) -> None:

        similarity_engine = SimilarityEngine(
            provider=provider
        )

        self.engine = ExplainabilityEngine(
            similarity_engine=similarity_engine
        )

        self.scorer = ServiceMatchScorer()


    def calculate_total_evidence(
        self,
        evidence: dict,
    ) -> int:

        count = 0

        fields = [
            "Matched technologies",
            "Matched capabilities",
            "Matched business problems",
            "Detected buying signals",
            "Detected evidence",
            "Matched keywords",
            "Matched industries",
        ]

        for field in fields:

            values = evidence.get(
                field,
                [],
            )

            count += len(values)

        return count



    def analyze(
        self,
        request: AnalyzeLeadRequest,
    ) -> AnalyzeLeadResponse:

        analysis_text = (
            request.lead.build_analysis_text()
        )


        # Retrieve more candidates first
        # Then rank using business scoring
        results = self.engine.analyze(
            query_text=analysis_text,
            top_k=20,
            minimum_score=request.minimum_similarity,
        )


        matched_services: list[
            ServiceMatchResponse
        ] = []


        for rank, result in enumerate(
            results,
            start=1,
        ):

            evidence = result.get(
                "evidence",
                {},
            )


            evidence_response = EvidenceResponse(

                matched_technologies=evidence.get(
                    "Matched technologies",
                    [],
                ),

                matched_capabilities=evidence.get(
                    "Matched capabilities",
                    [],
                ),

                matched_business_problems=evidence.get(
                    "Matched business problems",
                    [],
                ),

                detected_buying_signals=evidence.get(
                    "Detected buying signals",
                    [],
                ),

                detected_evidence=evidence.get(
                    "Detected evidence",
                    [],
                ),

                matched_keywords=evidence.get(
                    "Matched keywords",
                    [],
                ),

                matched_industries=evidence.get(
                    "Matched industries",
                    [],
                ),
            )


            # Calculate stronger evidence signal
            evidence_count = (
                self.calculate_total_evidence(
                    evidence
                )
            )


            score_result = self.scorer.score_match(

                service_id=result["service_id"],

                similarity_score=result[
                    "similarity_score"
                ],

                evidence_count=evidence_count,

                lead_country=request.lead.country,

                lead_industry=request.lead.industry,
            )


            matched_services.append(

                ServiceMatchResponse(

                    rank=rank,

                    service_id=result[
                        "service_id"
                    ],

                    service_name=result[
                        "service_name"
                    ],

                    category=result[
                        "category"
                    ],


                    similarity_score=result[
                        "similarity_score"
                    ],

                    similarity_percentage=result[
                        "similarity_percentage"
                    ],


                    confidence=result[
                        "confidence"
                    ],


                    evidence_count=evidence_count,


                    evidence=evidence_response,


                    explanation=result[
                        "explanation"
                    ],


                    service_match_score=score_result[
                        "service_match_score"
                    ],

                    service_match_percentage=score_result[
                        "service_match_percentage"
                    ],

                    service_match_confidence=score_result[
                        "service_match_confidence"
                    ],

                    score_breakdown=score_result[
                        "score_breakdown"
                    ],
                )
            )


        # Rank using business score
        matched_services.sort(
            key=lambda item:
                item.service_match_score,
            reverse=True,
        )


        # Keep only requested number
        matched_services = matched_services[
            :request.top_k
        ]


        # Reassign ranks
        for index, service in enumerate(
            matched_services,
            start=1,
        ):
            service.rank = index



        return AnalyzeLeadResponse(

            company_name=request.lead.company_name,

            industry=request.lead.industry,

            country=request.lead.country,

            source_url=request.lead.source_url,

            matched_services=matched_services,

            result_count=len(
                matched_services
            ),
        )