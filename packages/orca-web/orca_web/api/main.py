"""Orca Web BFF – FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from orca_shared.registry.models import get_engine

from ..config import settings
from .middleware import add_middleware

logger = logging.getLogger("orca_web.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown resources for the BFF."""
    engine = get_engine(settings.database_url)
    app.state.db_engine = engine
    app.state.db_sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    app.state.http_client = httpx.AsyncClient()
    logger.info("Orca Web BFF started")

    yield

    await app.state.db_engine.dispose()
    await app.state.http_client.aclose()
    logger.info("Orca Web BFF stopped")


def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""
    from .routers.auth import router as auth_router
    from .routers.dashboard import router as dashboard_router
    from .routers.users import router as users_router

    app = FastAPI(
        title="Orca Web",
        version="0.1.0",
        root_path="/api/v1",
        lifespan=lifespan,
    )

    add_middleware(app)

    app.include_router(auth_router)
    app.include_router(dashboard_router)
    app.include_router(users_router)

    @app.get("/health", tags=["health"])
    async def health(request: Request) -> Response:
        """Check connectivity to all backing services."""
        import redis.asyncio as aioredis

        async def _check_postgres() -> bool:
            try:
                async with request.app.state.db_sessionmaker() as session:
                    await session.execute(text("SELECT 1"))
                return True
            except Exception as exc:
                logger.warning("Postgres health check failed: %s", exc)
                return False

        async def _check_redis() -> bool:
            try:
                r = aioredis.from_url(settings.redis_url)
                try:
                    await r.ping()
                finally:
                    await r.aclose()
                return True
            except Exception as exc:
                logger.warning("Redis health check failed: %s", exc)
                return False

        async def _check_upstream(name: str, url: str) -> bool:
            try:
                resp = await request.app.state.http_client.get(
                    f"{url}/health", timeout=2.0,
                )
                return resp.status_code == 200
            except Exception as exc:
                logger.warning("%s health check failed: %s", name, exc)
                return False

        pg, rd, mind, lab, net = await asyncio.gather(
            _check_postgres(),
            _check_redis(),
            _check_upstream("orcamind", settings.orcamind_api_url),
            _check_upstream("orcalab", settings.orcalab_api_url),
            _check_upstream("orcanet", settings.orcanet_api_url),
        )

        services = {
            "postgres": pg,
            "redis": rd,
            "orcamind": mind,
            "orcalab": lab,
            "orcanet": net,
        }
        all_ok = all(services.values())
        import json

        body = json.dumps({
            "status": "healthy" if all_ok else "degraded",
            "services": services,
        })
        return Response(
            content=body,
            media_type="application/json",
            status_code=200 if all_ok else 503,
        )

    return app


app = create_app()
