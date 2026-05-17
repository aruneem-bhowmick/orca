"""CORS and request-logging middleware for the OrcaLab FastAPI app."""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("orcalab.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def add_middleware(app: FastAPI) -> None:
    cors_origins_env = os.environ.get("CORS_ORIGINS", "")
    if cors_origins_env:
        allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
        allow_credentials = True
    else:
        allow_origins = ["*"]
        allow_credentials = False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
