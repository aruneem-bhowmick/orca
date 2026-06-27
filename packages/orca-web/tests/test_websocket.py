"""Tests for orca_web.api.websocket — WebSocket proxy for live metrics.

Verifies JWT authentication from query parameters, upstream URL
construction, bidirectional message relay, upstream connection failure
handling, disconnect cleanup, and heartbeat behavior.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from orca_web.api.websocket import (
    _build_upstream_ws_url,
    _validate_token,
    experiment_live_proxy,
)
from orca_web.auth.jwt import create_access_token, create_refresh_token


# ── Helpers ────────────────────────────────────────────────────────────


def _resolved_future():
    """Return an already-resolved ``asyncio.Future`` mimicking a pong waiter."""
    fut = asyncio.get_event_loop().create_future()
    fut.set_result(None)
    return fut


class _MockUpstream:
    """Simulates an upstream ``websockets`` connection.

    Yields *messages* via ``async for``, then stops iteration.  Tracks
    calls to ``close``, ``ping``, and ``send`` via ``AsyncMock`` stubs.
    ``ping`` returns a resolved future so that ``await pong`` succeeds
    immediately, matching the ``websockets`` library behaviour.
    """

    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self._idx = 0
        self.close = AsyncMock()
        self.ping = AsyncMock(side_effect=lambda: _resolved_future())
        self.send = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx < len(self._messages):
            msg = self._messages[self._idx]
            self._idx += 1
            return msg
        raise StopAsyncIteration


class _BlockingUpstream(_MockUpstream):
    """Upstream that blocks on iteration until explicitly released.

    Uses an ``asyncio.Event`` rather than a fixed sleep so that the
    upstream stays alive exactly as long as the test needs it to,
    avoiding timing-sensitive failures.  Call ``release()`` or set
    ``_done`` from the test (or rely on task cancellation) to unblock.
    """

    def __init__(self):
        super().__init__()
        self._done = asyncio.Event()

    def release(self):
        """Signal the upstream to stop iteration."""
        self._done.set()

    async def __anext__(self):
        await self._done.wait()
        raise StopAsyncIteration


class _ReceiveSequence:
    """Async callable that returns *messages* in order, then disconnects.

    Meant to replace ``ws.receive_text`` on a mock browser WebSocket.
    """

    def __init__(self, *messages: str):
        self._messages = list(messages)
        self._idx = 0

    async def __call__(self):
        if self._idx < len(self._messages):
            msg = self._messages[self._idx]
            self._idx += 1
            return msg
        raise WebSocketDisconnect()


class _DelayedDisconnect:
    """Async callable that blocks for *delay* seconds, then disconnects.

    Keeps the browser-to-upstream relay alive long enough for other
    concurrent tasks to execute.
    """

    def __init__(self, delay: float = 0.15):
        self._delay = delay

    async def __call__(self):
        await asyncio.sleep(self._delay)
        raise WebSocketDisconnect()


def _make_browser_ws(*, token: str | None = None) -> MagicMock:
    """Build a mock browser WebSocket.

    By default ``receive_text`` raises ``WebSocketDisconnect`` immediately.
    Callers may replace ``ws.receive_text`` after creation to customise
    the browser's behaviour.
    """
    ws = MagicMock()
    ws.query_params = {"token": token} if token else {}
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect())
    return ws


def _valid_token(mock_settings, user_id: str | None = None) -> str:
    """Create a valid JWT access token using the test secret key."""
    return create_access_token(user_id or str(uuid.uuid4()))


def _fake_connect(upstream):
    """Return an async function suitable for patching ``websockets.connect``."""

    async def _connect(*_args, **_kwargs):
        return upstream

    return _connect


# ── URL construction ───────────────────────────────────────────────────


class TestBuildUpstreamWsUrl:
    """Tests for the upstream WebSocket URL construction helper."""

    def test_converts_http_to_ws(self, mock_settings):
        """HTTP-scheme base URL becomes ws://."""
        mock_settings.orcalab_api_url = "http://orcalab:8001"
        url = _build_upstream_ws_url("exp-42")
        assert url == "ws://orcalab:8001/api/v1/experiments/exp-42/live"

    def test_converts_https_to_wss(self, mock_settings):
        """HTTPS-scheme base URL becomes wss://."""
        mock_settings.orcalab_api_url = "https://orcalab.prod:443"
        url = _build_upstream_ws_url("exp-42")
        assert url == "wss://orcalab.prod:443/api/v1/experiments/exp-42/live"

    def test_handles_bare_host(self, mock_settings):
        """Base URL without a scheme gets ws:// prepended."""
        mock_settings.orcalab_api_url = "orcalab:8001"
        url = _build_upstream_ws_url("exp-42")
        assert url == "ws://orcalab:8001/api/v1/experiments/exp-42/live"

    def test_strips_trailing_slash_from_base_url(self, mock_settings):
        """Trailing slashes on the base URL do not produce double slashes."""
        mock_settings.orcalab_api_url = "http://orcalab:8001/"
        url = _build_upstream_ws_url("exp-42")
        assert url == "ws://orcalab:8001/api/v1/experiments/exp-42/live"

    def test_encodes_special_characters_in_experiment_id(self, mock_settings):
        """Characters unsafe in URL path segments are percent-encoded."""
        url = _build_upstream_ws_url("exp 42/foo")
        assert "/experiments/exp%2042%2Ffoo/live" in url

    def test_includes_experiment_id_in_path(self, mock_settings):
        """The experiment identifier is embedded in the URL path."""
        url = _build_upstream_ws_url("abc-123-def")
        assert "/experiments/abc-123-def/live" in url


# ── Token validation ───────────────────────────────────────────────────


class TestValidateToken:
    """Tests for the JWT validation helper."""

    def test_returns_none_for_none(self, mock_settings):
        """``None`` input yields ``None``."""
        assert _validate_token(None) is None

    def test_returns_none_for_empty_string(self, mock_settings):
        """Empty string yields ``None``."""
        assert _validate_token("") is None

    def test_returns_none_for_garbage_jwt(self, mock_settings):
        """A malformed JWT yields ``None``."""
        assert _validate_token("not.a.valid.jwt") is None

    def test_returns_user_id_for_valid_access_token(self, mock_settings):
        """A properly signed access token returns the user ID."""
        uid = str(uuid.uuid4())
        token = create_access_token(uid)
        assert _validate_token(token) == uid

    def test_returns_none_for_refresh_token(self, mock_settings):
        """Refresh tokens (``type`` != ``access``) are rejected."""
        token, _, _ = create_refresh_token(str(uuid.uuid4()))
        assert _validate_token(token) is None


# ── Authentication rejection ───────────────────────────────────────────


class TestAuthRejection:
    """Tests for authentication failure on the WebSocket endpoint."""

    async def test_closes_4001_when_no_token(self, mock_settings):
        """Missing ``token`` query parameter produces close code 4001."""
        ws = _make_browser_ws()

        await experiment_live_proxy(ws, "exp-1")

        ws.close.assert_awaited_once_with(
            code=4001, reason="Invalid or missing token",
        )

    async def test_closes_4001_when_invalid_token(self, mock_settings):
        """Invalid JWT produces close code 4001."""
        ws = _make_browser_ws(token="garbage-token")

        await experiment_live_proxy(ws, "exp-1")

        ws.close.assert_awaited_once_with(
            code=4001, reason="Invalid or missing token",
        )

    async def test_accepts_before_closing_on_auth_failure(self, mock_settings):
        """The connection is accepted before close so the 4001 code is delivered."""
        ws = _make_browser_ws()

        await experiment_live_proxy(ws, "exp-1")

        ws.accept.assert_awaited_once()

    async def test_authenticates_via_sec_websocket_protocol(self, mock_settings):
        """Authentication succeeds when JWT is provided in Sec-WebSocket-Protocol header."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws()
        ws.headers = {"sec-websocket-protocol": token}

        upstream = _MockUpstream()
        with patch(
            "orca_web.api.websocket.websockets.connect",
            _fake_connect(upstream),
        ):
            await experiment_live_proxy(ws, "exp-1")

        ws.accept.assert_awaited_with(subprotocol=token)
        for call in ws.close.call_args_list:
            assert call.kwargs.get("code") != 4001


# ── Upstream connection failure ────────────────────────────────────────


class TestUpstreamConnectionFailure:
    """Tests for upstream WebSocket connection errors."""

    async def test_sends_error_on_connect_refused(self, mock_settings):
        """Network-level connection failure sends an error payload."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)

        with patch(
            "orca_web.api.websocket.websockets.connect",
            side_effect=OSError("Connection refused"),
        ):
            await experiment_live_proxy(ws, "exp-1")

        ws.send_json.assert_awaited_once_with({"error": "upstream_unavailable"})

    async def test_sends_error_on_connect_timeout(self, mock_settings):
        """Connection timeout sends an error payload."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)

        with patch(
            "orca_web.api.websocket.websockets.connect",
            side_effect=asyncio.TimeoutError(),
        ):
            await experiment_live_proxy(ws, "exp-1")

        ws.send_json.assert_awaited_once_with({"error": "upstream_unavailable"})

    async def test_closes_browser_after_upstream_error(self, mock_settings):
        """Browser connection is closed after sending the error frame."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)

        with patch(
            "orca_web.api.websocket.websockets.connect",
            side_effect=OSError("Connection refused"),
        ):
            await experiment_live_proxy(ws, "exp-1")

        assert ws.close.await_count >= 1


# ── Upstream → Browser relay ───────────────────────────────────────────


class TestUpstreamToBrowserRelay:
    """Tests for forwarding messages from OrcaLab to the browser."""

    async def test_forwards_single_message(self, mock_settings):
        """A single upstream message is forwarded to the browser."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)
        # Keep browser side alive long enough for the upstream message
        ws.receive_text = _DelayedDisconnect(delay=0.15)

        msg = json.dumps(
            {"experiment_id": "exp-1", "status": "running", "epoch": 3},
        )
        upstream = _MockUpstream(messages=[msg])

        with patch(
            "orca_web.api.websocket.websockets.connect",
            _fake_connect(upstream),
        ):
            await experiment_live_proxy(ws, "exp-1")

        ws.send_text.assert_any_call(msg)

    async def test_forwards_multiple_messages(self, mock_settings):
        """Multiple upstream messages are each forwarded to the browser."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)
        ws.receive_text = _DelayedDisconnect(delay=0.15)

        msgs = [
            json.dumps({"epoch": 1, "loss": 0.8}),
            json.dumps({"epoch": 2, "loss": 0.5}),
            json.dumps({"epoch": 3, "loss": 0.3}),
        ]
        upstream = _MockUpstream(messages=msgs)

        with patch(
            "orca_web.api.websocket.websockets.connect",
            _fake_connect(upstream),
        ):
            await experiment_live_proxy(ws, "exp-1")

        for msg in msgs:
            ws.send_text.assert_any_call(msg)

    async def test_decodes_bytes_from_upstream(self, mock_settings):
        """Binary upstream messages are decoded to text for the browser."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)
        ws.receive_text = _DelayedDisconnect(delay=0.15)

        raw = b'{"epoch": 1}'
        upstream = _MockUpstream(messages=[raw])

        with patch(
            "orca_web.api.websocket.websockets.connect",
            _fake_connect(upstream),
        ):
            await experiment_live_proxy(ws, "exp-1")

        ws.send_text.assert_any_call('{"epoch": 1}')


# ── Browser → Upstream relay ───────────────────────────────────────────


class TestBrowserToUpstreamRelay:
    """Tests for forwarding control messages from the browser to OrcaLab."""

    async def test_forwards_control_message(self, mock_settings):
        """A browser control message is forwarded to upstream."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)
        ws.receive_text = _ReceiveSequence('{"action": "pause"}')

        upstream = _BlockingUpstream()

        with patch(
            "orca_web.api.websocket.websockets.connect",
            _fake_connect(upstream),
        ):
            await experiment_live_proxy(ws, "exp-1")

        upstream.send.assert_awaited_once_with('{"action": "pause"}')

    async def test_forwards_multiple_control_messages(self, mock_settings):
        """Multiple browser messages are each forwarded to upstream."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)
        ws.receive_text = _ReceiveSequence(
            '{"action": "pause"}', '{"action": "resume"}',
        )

        upstream = _BlockingUpstream()

        with patch(
            "orca_web.api.websocket.websockets.connect",
            _fake_connect(upstream),
        ):
            await experiment_live_proxy(ws, "exp-1")

        assert upstream.send.await_count == 2
        upstream.send.assert_any_await('{"action": "pause"}')
        upstream.send.assert_any_await('{"action": "resume"}')


# ── Disconnect handling ────────────────────────────────────────────────


class TestDisconnectHandling:
    """Tests for clean shutdown when either side disconnects."""

    async def test_closes_upstream_when_browser_disconnects(self, mock_settings):
        """Browser disconnect triggers upstream connection close."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)  # receive_text raises immediately

        upstream = _BlockingUpstream()

        with patch(
            "orca_web.api.websocket.websockets.connect",
            _fake_connect(upstream),
        ):
            await experiment_live_proxy(ws, "exp-1")

        upstream.close.assert_awaited()

    async def test_closes_browser_when_upstream_disconnects(self, mock_settings):
        """Upstream disconnect triggers browser connection close."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)
        # Browser blocks; upstream finishes immediately (empty messages)
        ws.receive_text = _DelayedDisconnect(delay=0.2)
        upstream = _MockUpstream(messages=[])

        with patch(
            "orca_web.api.websocket.websockets.connect",
            _fake_connect(upstream),
        ):
            await experiment_live_proxy(ws, "exp-1")

        assert ws.close.await_count >= 1


# ── Heartbeat ──────────────────────────────────────────────────────────


class TestHeartbeat:
    """Tests for the periodic upstream heartbeat ping."""

    async def test_pings_upstream_periodically(self, mock_settings):
        """The heartbeat task sends pings to the upstream connection."""
        token = _valid_token(mock_settings)
        ws = _make_browser_ws(token=token)
        # Both sides block long enough for the heartbeat to fire
        ws.receive_text = _DelayedDisconnect(delay=0.2)
        upstream = _BlockingUpstream()

        with (
            patch(
                "orca_web.api.websocket.websockets.connect",
                _fake_connect(upstream),
            ),
            patch("orca_web.api.websocket._HEARTBEAT_INTERVAL", 0.01),
        ):
            await experiment_live_proxy(ws, "exp-1")

        assert upstream.ping.await_count >= 1
