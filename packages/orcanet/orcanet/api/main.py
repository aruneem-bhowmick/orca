"""OrcaNet FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from sqlalchemy.ext.asyncio import async_sessionmaker

from orca_shared.clients.orcalab_client import OrcaLabClient
from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.registry.models import get_engine
from orcanet.api.middleware import add_middleware
from orcanet.embeddings.cross_domain import CrossDomainEmbedder
from orcanet.reasoning.agent import OrcaNetAgent
from orcanet.retrieval.query_expander import QueryExpander
from orcanet.retrieval.ranker import LLMRanker
from orcanet.retrieval.retriever import HybridRetriever
from orcanet.transfer.architecture_transfer import ArchitectureTransfer
from orcanet.transfer.feature_transfer import FeatureTransfer
from orcanet.transfer.weight_transfer import WeightTransfer

logger = logging.getLogger("orcanet.api")

_DEFAULT_DB_URL = "postgresql+asyncpg://localhost:5432/orca_registry"
_DEFAULT_ORCAMIND_URL = "http://localhost:8000"
_DEFAULT_ORCALAB_URL = "http://localhost:8001"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_url = os.environ.get("DATABASE_URL", _DEFAULT_DB_URL)
    engine = get_engine(db_url)
    app.state.db_engine = engine
    app.state.db_sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    app.state.embedder = CrossDomainEmbedder()

    from orcamind.embedders.similarity import FaissIndex

    faiss_path = os.environ.get("FAISS_INDEX_PATH", "data/faiss_index")
    idx = FaissIndex(dim=app.state.embedder.output_dim, metric="cosine")
    try:
        idx.load(faiss_path)
        logger.info("FAISS index loaded from %s (%d vectors)", faiss_path, len(idx))
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.warning("FAISS index not loaded (%s): %s", faiss_path, exc)
        idx = None

    llm_provider = os.environ.get("ORCANET_LLM_PROVIDER", "openai")
    llm_api_key = os.environ.get("ORCANET_LLM_API_KEY")

    transfer_strategies = {
        "feature": FeatureTransfer(),
        "weight": WeightTransfer(),
        "architecture": ArchitectureTransfer(),
    }

    app.state.agent = OrcaNetAgent(
        llm_provider=llm_provider,
        api_key=llm_api_key,
        transfer_strategies=transfer_strategies,
    )
    app.state.transfer_strategies = transfer_strategies

    expander = QueryExpander(app.state.agent.llm)
    ranker = LLMRanker(app.state.agent.llm)

    if idx is not None:
        from orca_shared.registry.repository import TaskRepository

        app.state.retriever = HybridRetriever(
            faiss_index=idx,
            task_repository=None,
            embedder=app.state.embedder,
            query_expander=expander,
            llm_ranker=ranker,
        )
    else:
        app.state.retriever = None

    orcamind_url = os.environ.get("ORCAMIND_URL", _DEFAULT_ORCAMIND_URL)
    orcalab_url = os.environ.get("ORCALAB_URL", _DEFAULT_ORCALAB_URL)
    app.state.orcamind_client = OrcaMindClient(orcamind_url)
    app.state.orcalab_client = OrcaLabClient(orcalab_url)
    app.state.orcamind_url = orcamind_url
    app.state.orcalab_url = orcalab_url

    yield

    results = await asyncio.gather(
        engine.dispose(),
        app.state.orcamind_client.aclose(),
        app.state.orcalab_client.aclose(),
        return_exceptions=True,
    )
    for exc in results:
        if isinstance(exc, Exception):
            logger.warning("Error during shutdown cleanup: %s", exc)


async def _check_http(url: str, timeout: float) -> bool:
    """Return True if the service at *url*/health responds 200 within *timeout* seconds."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception as exc:
        logger.warning("Health check failed for %s: %s", url, exc)
        return False


async def _check_llm(agent: OrcaNetAgent, timeout: float) -> bool:
    """Return True if the LLM backend responds within *timeout* seconds."""
    try:
        await asyncio.wait_for(agent.llm.ainvoke("ping"), timeout=timeout)
        return True
    except Exception as exc:
        logger.warning("LLM health check failed: %s", exc)
        return False


def create_app() -> FastAPI:
    from orcanet.api.routers.embed import router as embed_router
    from orcanet.api.routers.explain import router as explain_router
    from orcanet.api.routers.retrieve import router as retrieve_router
    from orcanet.api.routers.transfer import router as transfer_router

    app = FastAPI(
        title="OrcaNet",
        version="1.0.0",
        description="Cross-domain knowledge transfer agent",
        docs_url="/docs",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Service health"},
            {"name": "transfer", "description": "Transfer scoring and recommendations"},
            {"name": "retrieve", "description": "Similar task retrieval"},
            {"name": "explain", "description": "LLM-powered transfer explanations"},
            {"name": "embed", "description": "Cross-domain embeddings"},
        ],
    )

    add_middleware(app)

    app.include_router(transfer_router, prefix="/api/v1")
    app.include_router(retrieve_router, prefix="/api/v1")
    app.include_router(explain_router, prefix="/api/v1")
    app.include_router(embed_router, prefix="/api/v1")

    @app.get("/", tags=["health"])
    async def root() -> dict:
        return {"name": "OrcaNet", "version": "1.0.0", "status": "ok"}

    @app.get("/health", tags=["health"])
    async def health(request: Request) -> dict:
        orcamind_ok, orcalab_ok, llm_ok = await asyncio.gather(
            _check_http(request.app.state.orcamind_url, timeout=3.0),
            _check_http(request.app.state.orcalab_url, timeout=3.0),
            _check_llm(request.app.state.agent, timeout=5.0),
        )
        overall_ok = orcamind_ok and orcalab_ok and llm_ok
        return {
            "status": "healthy" if overall_ok else "degraded",
            "orcamind": orcamind_ok,
            "orcalab": orcalab_ok,
            "llm": llm_ok,
        }

    return app
