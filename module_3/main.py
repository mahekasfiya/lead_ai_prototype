from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from module_2.local_provider import LocalEmbeddingProvider
from module_2.validate_embeddings import (
    EmbeddingValidationError,
    validate_embedding_file,
)
from module_3.schemas import (
    AnalyzeLeadRequest,
    AnalyzeLeadResponse,
)
from module_3.service import LeadAnalysisService
from module_3.discovery.discovery_service import (
    LeadDiscoveryService,
)
from module_3.schemas import (
    AnalyzeLeadRequest,
    AnalyzeLeadResponse,
    DiscoverLeadsRequest,
    DiscoverLeadsResponse,
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


lead_analysis_service: LeadAnalysisService | None = None
lead_discovery_service: LeadDiscoveryService | None = None
embedding_validation_result: dict | None = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    global lead_analysis_service
    global lead_discovery_service
    global embedding_validation_result

    logger.info("Starting Triway Lead Intelligence API.")
    logger.info("Loading embedding model.")

    provider = LocalEmbeddingProvider()

    logger.info(
        "Embedding model loaded: %s",
        provider.model_name,
    )

    logger.info("Validating stored service embeddings.")

    embedding_validation_result = validate_embedding_file(
        provider=provider,
    )

    logger.info(
        "Embedding validation passed. "
        "Services: %s | Dimension: %s | Version: %s",
        embedding_validation_result["service_count"],
        embedding_validation_result["dimension"],
        embedding_validation_result["embedding_version"],
    )

    # 1. Create analysis service first
    lead_analysis_service = LeadAnalysisService(
        provider=provider
    )

    logger.info(
        "Lead analysis service loaded successfully."
    )

    # 2. Only then create discovery service
    lead_discovery_service = LeadDiscoveryService(
        analysis_service=lead_analysis_service
    )

    logger.info(
        "Lead discovery service loaded successfully."
    )

    yield

    logger.info(
        "Shutting down Triway Lead Intelligence API."
    )

    lead_discovery_service = None
    lead_analysis_service = None
    embedding_validation_result = None

app = FastAPI(
    title="Triway Lead Intelligence API",
    description=(
        "Semantic lead analysis API that matches company profiles, "
        "tenders, news, and business signals against Triway services."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, str]:
    return {
        "service": "Triway Lead Intelligence API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get(
    "/health",
    tags=["System"],
)
def health_check() -> dict:
    """
    Lightweight process health check.
    """
    return {
        "status": "healthy",
        "model_loaded": lead_analysis_service is not None,
    }


@app.get(
    "/readiness",
    tags=["System"],
)
def readiness_check() -> dict:
    """
    Confirm that the model, embeddings, and analysis service
    are fully initialized and ready to receive requests.
    """
    if (
        lead_analysis_service is None
        or embedding_validation_result is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lead analysis service is not ready.",
        )

    return {
        "status": "ready",
        "provider": embedding_validation_result["provider"],
        "model": embedding_validation_result["model"],
        "dimension": embedding_validation_result["dimension"],
        "service_count": embedding_validation_result[
            "service_count"
        ],
        "normalized": embedding_validation_result[
            "normalized"
        ],
        "embedding_version": embedding_validation_result[
            "embedding_version"
        ],
    }


@app.post(
    "/analyze-lead",
    response_model=AnalyzeLeadResponse,
    status_code=status.HTTP_200_OK,
    tags=["Lead Analysis"],
)
def analyze_lead(
    request: AnalyzeLeadRequest,
) -> AnalyzeLeadResponse:
    if lead_analysis_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lead analysis service is not ready.",
        )

    try:
        return lead_analysis_service.analyze(request)

    except ValueError as exc:
        logger.warning(
            "Invalid lead analysis request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error while analyzing lead."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lead analysis failed due to an internal error.",
        ) from exc
    
@app.post(
    "/discover-leads",
    response_model=DiscoverLeadsResponse,
    tags=["Lead Discovery"],
)
def discover_leads(
    request: DiscoverLeadsRequest,
) -> DiscoverLeadsResponse:
    if lead_discovery_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lead discovery service is not ready.",
        )

    try:
        return lead_discovery_service.discover(
            request
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error during lead discovery."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lead discovery failed.",
        ) from exc