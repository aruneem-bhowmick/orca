"""Shared proxy utilities for forwarding requests to upstream services.

Provides a generic request-forwarding function and activity-logging helper
used by the OrcaMind, OrcaLab, and OrcaNet proxy routers.  The proxy
helper copies query parameters, request body, and content-type from the
incoming browser request, injects an ``X-Orca-User-ID`` header, and
forwards the call to the upstream service via the shared httpx client.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from orca_web.models.user import User
from orca_web.repository.history_repo import HistoryRepository

logger = logging.getLogger("orca_web.proxy")


async def proxy_request(
    *,
    request: Request,
    method: str,
    target_url: str,
    user: User,
) -> Response:
    """Forward an authenticated request to an upstream service.

    Copies query parameters, request body, and content-type from the
    incoming request.  Injects an ``X-Orca-User-ID`` header so upstream
    services can optionally track the caller.

    Returns a :class:`~fastapi.Response` mirroring the upstream status
    code, body, and content-type.  On connection errors returns 502; on
    timeout (10 s) returns 504.

    Parameters
    ----------
    request:
        The incoming FastAPI request.
    method:
        HTTP method to use for the upstream call (``GET``, ``POST``, etc.).
    target_url:
        Fully-qualified URL of the upstream endpoint.
    user:
        The authenticated user making the request.
    """
    http_client: httpx.AsyncClient = request.app.state.http_client

    headers: dict[str, str] = {"X-Orca-User-ID": str(user.user_id)}
    content_type = request.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    body = await request.body() if method in ("POST", "PUT", "DELETE", "PATCH") else None

    try:
        upstream = await http_client.request(
            method=method,
            url=target_url,
            params=dict(request.query_params),
            content=body,
            headers=headers,
            timeout=10.0,
        )
    except httpx.TimeoutException:
        logger.warning("Upstream timeout: %s %s", method, target_url)
        return JSONResponse(status_code=504, content={"detail": "Service timeout"})
    except httpx.NetworkError:
        logger.warning("Upstream connection error: %s %s", method, target_url)
        return JSONResponse(status_code=502, content={"detail": "Service unavailable"})

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )


def _parse_resource_id(response_body: bytes) -> str | None:
    """Extract a resource ID from an upstream JSON response.

    Looks for common ID field names (``task_id``, ``experiment_id``,
    ``sweep_id``, ``mapping_id``) in the top-level JSON object.

    Returns ``None`` when the body is not valid JSON or no known ID
    field is present.
    """
    try:
        data = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("task_id", "experiment_id", "sweep_id", "mapping_id"):
        if key in data:
            return str(data[key])
    return None


async def log_proxy_activity(
    *,
    history_repo: HistoryRepository,
    user_id: uuid.UUID,
    action: str,
    resource_type: str,
    service: str,
    response: Response,
) -> None:
    """Log a mutating proxy call to the activity log.

    Parses the upstream response body to extract a resource ID when
    available.  Logging failures are caught and logged rather than
    propagated to avoid masking the actual upstream response.

    Parameters
    ----------
    history_repo:
        Repository for persisting activity log entries.
    user_id:
        UUID of the user who initiated the request.
    action:
        Action label (e.g. ``task_created``, ``experiment_started``).
    resource_type:
        Type of the affected resource (e.g. ``task``, ``experiment``).
    service:
        Name of the upstream service (``orcamind``, ``orcalab``, ``orcanet``).
    response:
        The proxy response whose body may contain a resource ID.
    """
    resource_id = _parse_resource_id(response.body)
    try:
        await history_repo.log_activity(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            service=service,
        )
    except Exception:
        logger.exception("Failed to log proxy activity: %s", action)
