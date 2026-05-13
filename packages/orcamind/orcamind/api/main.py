"""OrcaMind FastAPI application factory."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from orca_shared.registry.models import get_engine
from orcamind.embedders.similarity import FaissIndex
from orcamind.embedders.statistical import StatisticalEmbedder
from orcamind.selectors.nearest_neighbor import NearestNeighborSelector
from orcamind.selectors.predictor import PerformancePredictor

from .middleware import add_middleware

logger = logging.getLogger("orcamind.api")

_DEFAULT_DB_URL = "postgresql+asyncpg://localhost:5432/orca_registry"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_url = os.environ.get("DATABASE_URL", _DEFAULT_DB_URL)
    engine = get_engine(db_url)
    app.state.db_engine = engine
    app.state.db_sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    app.state.stat_embedder = StatisticalEmbedder()
    app.state.nn_selector = NearestNeighborSelector()
    app.state.predictor = PerformancePredictor()

    faiss_path = os.environ.get("FAISS_INDEX_PATH", "data/faiss_index")
    idx = FaissIndex(dim=app.state.stat_embedder.embedding_dim, metric="cosine")
    try:
        idx.load(faiss_path)
        app.state.faiss_index = idx
        logger.info("FAISS index loaded from %s (%d vectors)", faiss_path, len(idx))
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.warning("FAISS index not loaded (%s): %s", faiss_path, exc)
        app.state.faiss_index = None

    yield

    await engine.dispose()


def create_app() -> FastAPI:
    from .routers.adapt import router as adapt_router
    from .routers.feedback import router as feedback_router
    from .routers.models import router as models_router
    from .routers.recommend import router as recommend_router
    from .routers.tasks import router as tasks_router

    app = FastAPI(
        title="OrcaMind",
        version="1.0.0",
        description="Meta-learning model selection and warm-start service",
        docs_url="/docs",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Service health"},
            {"name": "tasks", "description": "Task registry"},
            {"name": "recommend", "description": "Model recommendation"},
            {"name": "feedback", "description": "Experiment feedback"},
            {"name": "models", "description": "Available model architectures"},
            {"name": "adapt", "description": "Meta-adaptation jobs"},
        ],
    )

    add_middleware(app)

    app.include_router(tasks_router, prefix="/api/v1")
    app.include_router(recommend_router, prefix="/api/v1")
    app.include_router(feedback_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(adapt_router, prefix="/api/v1")

    @app.get("/", tags=["health"])
    async def root() -> dict:
        return {"name": "OrcaMind", "version": "1.0.0", "status": "ok"}

    @app.get("/health", tags=["health"])
    async def health(request: Request) -> dict:
        db_ok = False
        try:
            async with request.app.state.db_sessionmaker() as session:
                await session.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            pass

        faiss_ok = request.app.state.faiss_index is not None

        mlflow_ok = False
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{mlflow_uri}/health", timeout=2.0)
                mlflow_ok = resp.status_code == 200
            except Exception:
                pass

        overall_ok = db_ok and faiss_ok and (not mlflow_uri or mlflow_ok)
        return {
            "status": "healthy" if overall_ok else "degraded",
            "db": db_ok,
            "faiss": faiss_ok,
            "mlflow": mlflow_ok,
        }

    return app
