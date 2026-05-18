"""OrcaLab FastAPI application factory."""

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

from .middleware import add_middleware

logger = logging.getLogger("orcalab.api")

_DEFAULT_DB_URL = "postgresql+asyncpg://localhost:5432/orca_registry"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_url = os.environ.get("DATABASE_URL", _DEFAULT_DB_URL)
    engine = get_engine(db_url)
    app.state.db_engine = engine
    app.state.db_sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    app.state.sweeps: dict = {}

    yield

    await engine.dispose()


def create_app() -> FastAPI:
    from .routers.experiments import router as experiments_router
    from .routers.search_spaces import router as search_spaces_router
    from .routers.sweeps import router as sweeps_router

    app = FastAPI(
        title="OrcaLab",
        version="0.1.0",
        description="Hyperparameter search and experiment orchestration",
        docs_url="/docs",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Service health"},
            {"name": "experiments", "description": "Experiment lifecycle"},
            {"name": "sweeps", "description": "Hyperparameter sweeps"},
            {"name": "search-spaces", "description": "Search space definitions"},
        ],
    )

    add_middleware(app)

    app.include_router(experiments_router, prefix="/api/v1")
    app.include_router(sweeps_router, prefix="/api/v1")
    app.include_router(search_spaces_router, prefix="/api/v1")

    @app.get("/", tags=["health"])
    async def root() -> dict:
        return {"name": "OrcaLab", "version": "0.1.0", "status": "ok"}

    @app.get("/health", tags=["health"])
    async def health(request: Request) -> dict:
        db_ok = False
        try:
            async with request.app.state.db_sessionmaker() as session:
                await session.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:
            logger.warning("Database health check failed: %s", exc)

        prefect_ok = False
        prefect_url = os.environ.get("PREFECT_API_URL")
        if prefect_url:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{prefect_url}/health", timeout=2.0)
                prefect_ok = resp.status_code == 200
            except Exception as exc:
                logger.warning("Prefect health check failed (%s): %s", prefect_url, exc)

        overall_ok = db_ok and (not prefect_url or prefect_ok)
        return {
            "status": "healthy" if overall_ok else "degraded",
            "db": db_ok,
            "prefect": prefect_ok,
        }

    return app


app = create_app()
