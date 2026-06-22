"""Authenticated WebSocket proxy for live OrcaLab experiment metrics.

Relays real-time experiment metric updates between the browser and the
upstream OrcaLab WebSocket endpoint.  The browser connects to the BFF,
which validates the JWT from a ``token`` query parameter, opens an
upstream WebSocket to OrcaLab, and forwards messages in both directions
using concurrent asyncio tasks.  A 30-second heartbeat ping detects stale
upstream connections.

Typical flow
------------
1. Browser opens ``WS /orcalab/ws/experiments/{id}/live?token=<jwt>``
2. BFF validates the JWT and extracts the user ID
3. BFF opens ``ws://{ORCALAB}/api/v1/experiments/{id}/live``
4. Messages are relayed bidirectionally until either side disconnects
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import quote

import websockets
import websockets.exceptions
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from orca_web.auth.jwt import decode_token
from orca_web.config import settings

logger = logging.getLogger("orca_web.websocket")

router = APIRouter(tags=["websocket"])

_HEARTBEAT_INTERVAL: int = 30
"""Seconds between upstream heartbeat pings."""

_UPSTREAM_CONNECT_TIMEOUT: int = 10
"""Maximum seconds to wait for the upstream WebSocket handshake."""

_PONG_TIMEOUT: int = 10
"""Maximum seconds to wait for a pong reply before treating the upstream as stale."""


def _build_upstream_ws_url(experiment_id: str) -> str:
    """Construct the OrcaLab WebSocket URL for a live experiment stream.

    Converts the HTTP-scheme ``orcalab_api_url`` setting to a WebSocket
    scheme (``ws://`` or ``wss://``) and appends the experiment live-metrics
    path.

    Parameters
    ----------
    experiment_id:
        Identifier of the experiment to stream metrics for.

    Returns
    -------
    str
        Fully-qualified WebSocket URL targeting OrcaLab's live endpoint.
    """
    base = settings.orcalab_api_url
    if base.startswith("https://"):
        ws_base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        ws_base = "ws://" + base[len("http://"):]
    else:
        ws_base = "ws://" + base
    ws_base = ws_base.rstrip("/")
    return f"{ws_base}/api/v1/experiments/{quote(experiment_id, safe='')}/live"


def _validate_token(token: str | None) -> str | None:
    """Validate a JWT access token and return the user ID.

    Returns the ``sub`` claim (user ID) from the JWT if the token is a
    valid, unexpired access token.  Returns ``None`` when the token is
    missing, malformed, expired, or carries a non-access token type.

    Parameters
    ----------
    token:
        Raw JWT string extracted from the ``token`` query parameter.

    Returns
    -------
    str or None
        The user ID encoded in the token, or ``None`` on any failure.
    """
    if not token:
        return None
    try:
        payload = decode_token(token)
    except Exception:  # noqa: BLE001 – intentionally broad to ensure safety
        return None
    if payload.get("type") != "access":
        return None
    return payload.get("sub")


@router.websocket("/orcalab/ws/experiments/{experiment_id}/live")
async def experiment_live_proxy(
    websocket: WebSocket,
    experiment_id: str,
) -> None:
    """Proxy live experiment metrics between the browser and OrcaLab.

    Authenticates the browser connection via a ``token`` query parameter
    containing a valid JWT access token.  On success, opens an upstream
    WebSocket to OrcaLab and relays messages bidirectionally:

    - **OrcaLab -> Browser**: metric update JSON frames
    - **Browser -> OrcaLab**: control messages (pause, resume, cancel)

    Close codes
    -----------
    - **4001** -- Invalid or missing authentication token.
    - **1000** -- Normal closure when either side disconnects.

    Parameters
    ----------
    websocket:
        Browser-side WebSocket connection managed by FastAPI/Starlette.
    experiment_id:
        UUID of the experiment whose live metrics to stream.
    """
    # ── Authentication via query-parameter JWT ──────────────────────────
    token = websocket.query_params.get("token")
    user_id = _validate_token(token)
    if user_id is None:
        await websocket.accept()
        await websocket.close(code=4001, reason="Invalid or missing token")
        return

    # ── Connect to upstream OrcaLab WebSocket ──────────────────────────
    upstream_url = _build_upstream_ws_url(experiment_id)
    try:
        upstream = await asyncio.wait_for(
            websockets.connect(upstream_url, ping_interval=None),
            timeout=_UPSTREAM_CONNECT_TIMEOUT,
        )
    except Exception:  # noqa: BLE001 – covers OSError, timeout, DNS, etc.
        logger.warning(
            "Failed to connect to upstream WebSocket: %s", upstream_url,
        )
        await websocket.send_json({"error": "upstream_unavailable"})
        await websocket.close()
        return

    # ── Concurrent relay tasks ─────────────────────────────────────────
    async def _relay_upstream_to_browser() -> None:
        """Forward metric update messages from OrcaLab to the browser."""
        try:
            async for message in upstream:
                await websocket.send_text(
                    message if isinstance(message, str) else message.decode()
                )
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _relay_browser_to_upstream() -> None:
        """Forward control messages from the browser to OrcaLab."""
        try:
            while True:
                data = await websocket.receive_text()
                await upstream.send(data)
        except WebSocketDisconnect:
            pass

    async def _heartbeat() -> None:
        """Ping upstream every 30 s and await the pong to detect stale connections."""
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
                pong = await upstream.ping()
                await asyncio.wait_for(pong, timeout=_PONG_TIMEOUT)
        except Exception:  # noqa: BLE001
            pass

    tasks = [
        asyncio.create_task(_relay_upstream_to_browser()),
        asyncio.create_task(_relay_browser_to_upstream()),
        asyncio.create_task(_heartbeat()),
    ]

    try:
        # Wait for any task to finish (one side disconnected or stale)
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # Cancel remaining tasks and suppress CancelledError
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Close both connections gracefully
        try:
            await upstream.close()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to close upstream connection", exc_info=True)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to close browser connection", exc_info=True)

    logger.info(
        "WebSocket relay ended for experiment %s (user %s)",
        experiment_id,
        user_id,
    )
