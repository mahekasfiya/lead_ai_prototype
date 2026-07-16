from __future__ import annotations

import logging

from app.search.serpapi import search
from app.collector.webpage_fetcher import fetch_page
from app.collector.text_extractor import extract_text

from module_2.local_provider import LocalEmbeddingProvider
from module_2.validate_embeddings import (
    validate_embedding_file,
    EmbeddingValidationError,
)

from module_3.schemas import (
    LeadProfile,
    AnalyzeLeadRequest,
)

from module_3.service import LeadAnalysisService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def create_analysis_service():

    logger.info(
        "Loading embedding provider..."
    )

    provider = LocalEmbeddingProvider()

    logger.info(
        "Validating embeddings..."
    )

    try:
        validation = validate_embedding_file(
            provider=provider
        )

    except EmbeddingValidationError as exc:
        logger.error(
            "Embedding validation failed:"
        )
        logger.error(exc)

        raise

    logger.info(
        "Embedding validation successful."
    )

    logger.info(
        "Services loaded: %s",
        validation["service_count"],
    )

    service = LeadAnalysisService(
        provider=provider
    )

    return service



def build_lead_profile(
    title: str,
    url: str,
    text: str,
) -> LeadProfile:

    return LeadProfile(

        company_name=None,

        source_url=url,

        summary=title,

        content=text,

        keywords=[
            "CyberArk",
            "PAM",
            "privileged access management",
        ],

    )



def main():

    logger.info(
        "Starting Triway Lead Intelligence Pipeline"
    )


    # -----------------------------
    # Module 1
    # Search
    # -----------------------------

    results = search(
        "CyberArk implementation services"
    )


    # -----------------------------
    # Module 2 + Module 3
    # Initialize once
    # -----------------------------

    analysis_service = (
        create_analysis_service()
    )


    # -----------------------------
    # Process search results
    # -----------------------------

    for result in results:

        print("=" * 80)

        print(
            "TITLE:"
        )
        print(
            result.title
        )

        print(
            "\nURL:"
        )
        print(
            result.url
        )


        try:

            page = fetch_page(
                result.url
            )

            text = extract_text(
                page
            )


        except Exception as exc:

            print(
                f"Failed fetching page: {exc}"
            )

            continue



        lead = build_lead_profile(
            title=result.title,
            url=result.url,
            text=text,
        )


        request = AnalyzeLeadRequest(

            lead=lead,

            top_k=3,

            minimum_similarity=0.25,

        )


        response = analysis_service.analyze(
            request
        )


        print(
            "\nMATCHED SERVICES:\n"
        )


        for match in response.matched_services:

            print(
                "-" * 20
            )

            print(
                f"Rank: {match.rank}"
            )

            print(
                f"Service: "
                f"{match.service_name}"
            )

            print(
                f"Similarity: "
                f"{match.similarity_percentage}%"
            )

            print(
                f"Confidence: "
                f"{match.confidence}"
            )


            print(
                "\nEvidence:"
            )


            print(
                match.evidence.model_dump()
            )


            print(
                "\nExplanation:"
            )

            print(
                match.explanation
            )

            print()



if __name__ == "__main__":
    main()