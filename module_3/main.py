from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from anthropic import Anthropic
from pydantic import BaseModel
from app.search import serpapi

from module_2.local_provider import LocalEmbeddingProvider
from module_2.validate_embeddings import (
    validate_embedding_file,
)

from module_3.discovery.discovery_service import LeadDiscoveryService
from module_3.schemas import (
    AnalyzeLeadRequest,
    AnalyzeLeadResponse,
    DiscoverLeadsRequest,
    DiscoverLeadsResponse,
    LeadProfile,
    ServiceMatchResponse,
)
from module_3.service import LeadAnalysisService
from module_3.intelligence.service import LeadIntelligenceService
from module_3.discovery.potential_lead_discovery import PotentialLeadDiscovery

# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------
# Global service instances
# ------------------------------
lead_analysis_service: LeadAnalysisService | None = None
lead_discovery_service: LeadDiscoveryService | None = None
lead_intelligence_service: LeadIntelligenceService | None = None
potential_lead_discovery: PotentialLeadDiscovery | None = None
embedding_validation_result: dict | None = None

# ------------------------------
# Lifespan for service initialisation
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global lead_analysis_service
    global lead_discovery_service
    global lead_intelligence_service
    global embedding_validation_result
    global potential_lead_discovery

    logger.info("Starting Triway Lead Intelligence API.")
    logger.info("Loading embedding model.")

    provider = LocalEmbeddingProvider()

    logger.info(
        "Embedding model loaded: %s",
        provider.model_name,
    )

    logger.info("Validating stored service embeddings.")
    embedding_validation_result = validate_embedding_file(provider=provider)

    logger.info(
        "Embedding validation passed. "
        "Services: %s | Dimension: %s | Version: %s",
        embedding_validation_result["service_count"],
        embedding_validation_result["dimension"],
        embedding_validation_result["embedding_version"],
    )

    # Create analysis service
    lead_analysis_service = LeadAnalysisService(provider=provider)
    logger.info("Lead analysis service loaded successfully.")

    # ------------ Claude Setup ------------
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    use_claude = os.getenv("USE_CLAUDE", "false").lower() == "true"
    claude_model_name = os.getenv(
        "CLAUDE_MODEL",
        "claude-sonnet-5",
    )
    llm_model = None
    if use_claude and anthropic_api_key:
        try:
            client = Anthropic(api_key=anthropic_api_key)
            class ClaudeWrapper:
                """
                Temporary compatibility wrapper.
                Existing project components currently expect:
                response = llm_model.generate_content(prompt)
                text = response.text
                This wrapper exposes the same interface while internally
                calling Anthropic's Messages API.
                """
                def __init__(
                        self,
                        client: Anthropic,
                        model_name: str,
                ) -> None:
                    self.client = client
                    self.model_name = model_name
                class Response:
                    def __init__(self, text: str) -> None:
                        self.text = text
                def generate_content(
                        self,
                        prompt: str,
                        *,
                        max_tokens: int = 4096,
                ) -> "ClaudeWrapper.Response":
                    message = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_tokens,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )
                    block_types = [
                        getattr(block, "type", "unknown")
                        for block in message.content
                    ]
                    logger.debug(
                        "Claude response | stop_reason=%s | block_types=%s | "
                        "output_tokens=%s",
                        message.stop_reason,
                        block_types,
                        getattr(message.usage, "output_tokens", None),
                    )
                    if message.stop_reason == "max_tokens":
                        raise RuntimeError(
                            "Claude response was truncated because max_tokens "
                            f"was reached ({max_tokens})."
                        )
                    if message.stop_reason == "refusal":
                        raise RuntimeError(
                            "Claude refused to process the request."
                        )
                    text_parts = [
                        block.text
                        for block in message.content
                        if getattr(block, "type", None) == "text"
                        and getattr(block, "text", None)
                    ]
                    text = "\n".join(text_parts).strip()
                    if not text:
                        raise RuntimeError(
                            "Claude returned no text content. "
                            f"stop_reason={message.stop_reason}, "
                            f"block_types={block_types}"
                        )
                    return self.Response(text=text)
            llm_model = ClaudeWrapper(
                client=client,
                model_name=claude_model_name,
            )
            logger.info(
                "Claude model loaded successfully: %s",
                claude_model_name,
            )
        except Exception:
            logger.exception("Failed to initialize Claude.")
            llm_model = None
    else:
        logger.info(
            "Claude disabled or ANTHROPIC_API_KEY is missing. "
            "Using rule-based processing."
        )

    # Build the discovery config
    discovery_config = {
        "knowledge_base_path": Path(
            os.getenv(
                "KNOWLEDGE_BASE_PATH",
                "data/triway_knowledge_base_v0_2_extended.json",
            )
        ),
        "use_llm": use_claude and llm_model is not None,
        "llm_model": llm_model,
        "fetch_timeout": int(os.getenv("FETCH_TIMEOUT", 30)),
        "fetch_max_size": int(
            os.getenv("FETCH_MAX_SIZE", 10485760)
        ),
        "min_buyer_score": float(
            os.getenv("MIN_BUYER_SCORE", 0.6)
        ),
        "max_provider_prob": float(
            os.getenv("MAX_PROVIDER_PROB", 0.4)
        ),
        "max_chunks": 3,
        "default_queries_per_service": int(
            os.getenv("DEFAULT_QUERIES_PER_SERVICE", 2)
        ),
        "default_max_total_queries": int(
            os.getenv("DEFAULT_MAX_TOTAL_QUERIES", 50)
        ),
        "llm_batch_size": int(
            os.getenv("LLM_BATCH_SIZE", 8)
        ),
        "llm_max_candidates": int(
            os.getenv("LLM_MAX_CANDIDATES", 20)
        ),
        "llm_max_excerpt_chars": int(
            os.getenv("LLM_MAX_EXCERPT_CHARS", 3000)
        ),
        "planner_prompt_path": None,
    }

    # Create discovery service with the config
    lead_discovery_service = LeadDiscoveryService(
        analysis_service=lead_analysis_service,
        config=discovery_config,
    )
    logger.info("Lead discovery service loaded successfully.")

    # ---- NEW: Create intelligence service for email drafting ----
    lead_intelligence_service = LeadIntelligenceService(llm_model=llm_model)
    logger.info("Lead intelligence service (with email drafting) loaded.")
    potential_lead_discovery = None
    if llm_model is not None:
        potential_lead_discovery = PotentialLeadDiscovery(
            llm_model=llm_model,
            search_provider=serpapi,
        )
        logger.info("Potential lead discovery service loaded successfully.")
    else:
        logger.warning(
            "Potential lead discovery service disabled because Claude is unavailable."
        )
    yield

    # Cleanup
    logger.info("Shutting down Triway Lead Intelligence API.")
    lead_discovery_service = None
    lead_analysis_service = None
    lead_intelligence_service = None
    embedding_validation_result = None
    potential_lead_discovery = None

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(
    title="Triway Lead Intelligence API",
    description=(
        "Semantic lead analysis API that matches company profiles, "
        "tenders, news, and business signals against Triway services."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------- Existing endpoints (unchanged) ----------
@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {
        "service": "Triway Lead Intelligence API",
        "status": "running",
        "version": "1.0.0",
    }

@app.get("/health", tags=["System"])
def health_check() -> dict:
    return {
        "status": "healthy",
        "model_loaded": lead_analysis_service is not None,
    }

@app.get("/readiness", tags=["System"])
def readiness_check() -> dict:
    if lead_analysis_service is None or embedding_validation_result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lead analysis service is not ready.",
        )
    return {
        "status": "ready",
        "provider": embedding_validation_result["provider"],
        "model": embedding_validation_result["model"],
        "dimension": embedding_validation_result["dimension"],
        "service_count": embedding_validation_result["service_count"],
        "normalized": embedding_validation_result["normalized"],
        "embedding_version": embedding_validation_result["embedding_version"],
    }

@app.post("/analyze-lead", response_model=AnalyzeLeadResponse, tags=["Lead Analysis"])
def analyze_lead(request: AnalyzeLeadRequest) -> AnalyzeLeadResponse:
    if lead_analysis_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lead analysis service is not ready.",
        )
    try:
        return lead_analysis_service.analyze(request)
    except ValueError as exc:
        logger.warning("Invalid lead analysis request: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while analyzing lead.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lead analysis failed due to an internal error.",
        ) from exc

@app.post("/discover-leads", response_model=DiscoverLeadsResponse, tags=["Lead Discovery"])
def discover_leads(request: DiscoverLeadsRequest) -> DiscoverLeadsResponse:
    if lead_discovery_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lead discovery service is not ready.",
        )
    try:
        logger.info(
            "Lead discovery request | "
            "Queries/service: %s | "
            "Max total queries: %s | "
            "Results/query: %s | "
            "Minimum similarity: %.2f | "
            "Selected services: %s",
            request.queries_per_service,
            request.max_total_queries,
            request.results_per_query,
            request.minimum_similarity,
            len(request.selected_service_ids),
            )
        
        return lead_discovery_service.discover(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during lead discovery.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lead discovery failed.",
        ) from exc

# ---------- NEW ENDPOINT for email drafting ----------
class EmailDraftRequest(BaseModel):
    lead: LeadProfile
    matched_services: list[ServiceMatchResponse]

@app.post("/generate-email", tags=["Lead Intelligence"])
def generate_email_draft(request: EmailDraftRequest) -> dict:
    if lead_intelligence_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lead intelligence service is not ready.",
        )
    try:
        draft = lead_intelligence_service.generate_email_draft(
            lead=request.lead,
            matched_services=request.matched_services,
        )
        return {"email_draft": draft}
    except Exception as exc:
        logger.exception("Email draft generation failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email generation failed: {exc}",
        )

class PotentialLeadRequest(BaseModel):
    industries: list[str] | None = None
    countries: list[str] | None = None
    titles: list[str] | None = None
    min_employees: int | None = None
    max_employees: int | None = None
    revenue: str | None = None
    technologies: list[str] | None = None
    recent_funding: bool | None = None

@app.post("/discover-potential-leads")
def discover_potential_leads(request: PotentialLeadRequest):
    if potential_lead_discovery is None:
        raise HTTPException(
            status_code=503,
            detail="Potential lead discovery is unavailable.",
        )

    try:
        criteria = request.model_dump(
            exclude_none=True
        )

        results = potential_lead_discovery.discover(
            criteria
        )

        return {
            "leads": results,
            "count": len(results),
        }

    except Exception:
        logger.exception(
            "Potential lead discovery failed."
        )
        raise HTTPException(
            status_code=500,
            detail="Potential lead discovery failed.",
        )

class LinkedInMessageRequest(BaseModel):
    lead: dict
@app.post(
    "/generate-linkedin-message",
    tags=["Lead Intelligence"],
)
def generate_linkedin_message(
    request: LinkedInMessageRequest,
) -> dict:
    if potential_lead_discovery is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Potential lead discovery service is not ready.",
        )
    try:
        message = (
            potential_lead_discovery.generate_linkedin_message(
                request.lead
            )
        )
        return {"message": message}
    except Exception as exc:
        logger.exception(
            "LinkedIn message generation failed."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc