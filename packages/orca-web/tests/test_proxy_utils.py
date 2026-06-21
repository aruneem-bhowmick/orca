"""Tests for orca_web.api.proxy_utils — shared proxy forwarding and logging."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from orca_web.api.proxy_utils import (
    _parse_resource_id,
    log_proxy_activity,
    proxy_request,
)
from orca_web.repository.history_repo import HistoryRepository


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_request(http_client, *, body=b"", query_params=None, content_type=None):
    """Build a mock Request with the given httpx client on app.state."""
    request = MagicMock()
    request.app.state.http_client = http_client
    request.body = AsyncMock(return_value=body)
    request.query_params = query_params or {}
    headers = {}
    if content_type:
        headers["content-type"] = content_type
    request.headers = headers
    return request


def _make_upstream(*, status_code=200, content=b'{}', content_type="application/json"):
    """Build a mock httpx response object."""
    upstream = MagicMock()
    upstream.status_code = status_code
    upstream.content = content
    upstream.headers = {"content-type": content_type}
    return upstream


def _make_user(user_id=None):
    """Build a minimal user mock."""
    user = MagicMock()
    user.user_id = user_id or uuid.uuid4()
    return user


# ── proxy_request ────────────────────────────────────────────────────────


class TestProxyRequest:
    """Tests for the generic proxy_request function."""

    async def test_forwards_get_with_query_params(self):
        """GET requests copy query parameters to the upstream call."""
        upstream = _make_upstream(content=b'[{"task_id":"abc"}]')
        client = AsyncMock()
        client.request = AsyncMock(return_value=upstream)
        user = _make_user()
        request = _make_request(client, query_params={"limit": "5"})

        resp = await proxy_request(
            request=request, method="GET", target_url="http://svc/tasks", user=user,
        )

        client.request.assert_awaited_once()
        call_kwargs = client.request.call_args
        assert call_kwargs.kwargs["method"] == "GET"
        assert call_kwargs.kwargs["url"] == "http://svc/tasks"
        assert call_kwargs.kwargs["params"] == {"limit": "5"}
        assert resp.status_code == 200

    async def test_forwards_post_with_body_and_content_type(self):
        """POST requests copy body and content-type to the upstream call."""
        upstream = _make_upstream(content=b'{"task_id":"new"}', status_code=201)
        client = AsyncMock()
        client.request = AsyncMock(return_value=upstream)
        user = _make_user()
        body = b'{"name":"iris"}'
        request = _make_request(client, body=body, content_type="application/json")

        resp = await proxy_request(
            request=request, method="POST", target_url="http://svc/embed", user=user,
        )

        call_kwargs = client.request.call_args.kwargs
        assert call_kwargs["content"] == body
        assert call_kwargs["headers"]["content-type"] == "application/json"
        assert resp.status_code == 201

    async def test_injects_user_id_header(self):
        """The X-Orca-User-ID header is injected with the user's UUID."""
        upstream = _make_upstream()
        client = AsyncMock()
        client.request = AsyncMock(return_value=upstream)
        user = _make_user()
        request = _make_request(client)

        await proxy_request(
            request=request, method="GET", target_url="http://svc/x", user=user,
        )

        call_headers = client.request.call_args.kwargs["headers"]
        assert call_headers["X-Orca-User-ID"] == str(user.user_id)

    async def test_returns_502_on_network_error(self):
        """Connection failures produce a 502 JSON error response."""
        client = AsyncMock()
        client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))
        user = _make_user()
        request = _make_request(client)

        resp = await proxy_request(
            request=request, method="GET", target_url="http://svc/x", user=user,
        )

        assert resp.status_code == 502
        assert json.loads(resp.body) == {"detail": "Service unavailable"}

    async def test_returns_504_on_timeout(self):
        """Timeout exceptions produce a 504 JSON error response."""
        client = AsyncMock()
        client.request = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        user = _make_user()
        request = _make_request(client)

        resp = await proxy_request(
            request=request, method="GET", target_url="http://svc/x", user=user,
        )

        assert resp.status_code == 504
        assert json.loads(resp.body) == {"detail": "Service timeout"}

    async def test_get_does_not_send_body(self):
        """GET requests must not send a request body to the upstream."""
        upstream = _make_upstream()
        client = AsyncMock()
        client.request = AsyncMock(return_value=upstream)
        user = _make_user()
        request = _make_request(client, body=b"should be ignored")

        await proxy_request(
            request=request, method="GET", target_url="http://svc/x", user=user,
        )

        call_kwargs = client.request.call_args.kwargs
        assert call_kwargs["content"] is None

    async def test_mirrors_upstream_status_code(self):
        """The response mirrors whatever status the upstream returns."""
        upstream = _make_upstream(status_code=404, content=b'{"detail":"not found"}')
        client = AsyncMock()
        client.request = AsyncMock(return_value=upstream)
        user = _make_user()
        request = _make_request(client)

        resp = await proxy_request(
            request=request, method="GET", target_url="http://svc/x", user=user,
        )

        assert resp.status_code == 404

    async def test_uses_10_second_timeout(self):
        """The upstream call uses a 10-second timeout."""
        upstream = _make_upstream()
        client = AsyncMock()
        client.request = AsyncMock(return_value=upstream)
        user = _make_user()
        request = _make_request(client)

        await proxy_request(
            request=request, method="GET", target_url="http://svc/x", user=user,
        )

        assert client.request.call_args.kwargs["timeout"] == 10.0


# ── _parse_resource_id ───────────────────────────────────────────────────


class TestParseResourceId:
    """Tests for extracting resource IDs from upstream response bodies."""

    def test_extracts_task_id(self):
        body = json.dumps({"task_id": "abc-123"}).encode()
        assert _parse_resource_id(body) == "abc-123"

    def test_extracts_experiment_id(self):
        body = json.dumps({"experiment_id": "exp-456"}).encode()
        assert _parse_resource_id(body) == "exp-456"

    def test_extracts_sweep_id(self):
        body = json.dumps({"sweep_id": "sw-789"}).encode()
        assert _parse_resource_id(body) == "sw-789"

    def test_extracts_mapping_id(self):
        body = json.dumps({"mapping_id": "map-000"}).encode()
        assert _parse_resource_id(body) == "map-000"

    def test_returns_none_for_missing_id(self):
        body = json.dumps({"score": 0.85}).encode()
        assert _parse_resource_id(body) is None

    def test_returns_none_for_invalid_json(self):
        assert _parse_resource_id(b"not json") is None

    def test_returns_none_for_non_object(self):
        body = json.dumps([1, 2, 3]).encode()
        assert _parse_resource_id(body) is None

    def test_returns_none_for_empty_body(self):
        assert _parse_resource_id(b"") is None

    def test_prefers_first_matching_key(self):
        """When multiple ID keys are present, the first match wins."""
        body = json.dumps({"task_id": "t1", "experiment_id": "e1"}).encode()
        assert _parse_resource_id(body) == "t1"


# ── log_proxy_activity ───────────────────────────────────────────────────


class TestLogProxyActivity:
    """Tests for the activity-logging helper."""

    async def test_logs_activity_with_resource_id(self):
        """Extracts the resource ID from the response and logs it."""
        repo = AsyncMock(spec=HistoryRepository)
        response = MagicMock()
        response.body = json.dumps({"task_id": "t-abc"}).encode()
        uid = uuid.uuid4()

        await log_proxy_activity(
            history_repo=repo,
            user_id=uid,
            action="task_created",
            resource_type="task",
            service="orcamind",
            response=response,
        )

        repo.log_activity.assert_awaited_once_with(
            user_id=uid,
            action="task_created",
            resource_type="task",
            resource_id="t-abc",
            service="orcamind",
        )

    async def test_logs_activity_without_resource_id(self):
        """When the response has no recognisable ID, resource_id is None."""
        repo = AsyncMock(spec=HistoryRepository)
        response = MagicMock()
        response.body = json.dumps({"score": 0.7}).encode()
        uid = uuid.uuid4()

        await log_proxy_activity(
            history_repo=repo,
            user_id=uid,
            action="transfer_scored",
            resource_type="transfer",
            service="orcanet",
            response=response,
        )

        repo.log_activity.assert_awaited_once()
        assert repo.log_activity.call_args.kwargs["resource_id"] is None

    async def test_suppresses_logging_errors(self):
        """Logging failures are swallowed so the response is not lost."""
        repo = AsyncMock(spec=HistoryRepository)
        repo.log_activity = AsyncMock(side_effect=RuntimeError("db down"))
        response = MagicMock()
        response.body = b"{}"
        uid = uuid.uuid4()

        # Must not raise
        await log_proxy_activity(
            history_repo=repo,
            user_id=uid,
            action="task_created",
            resource_type="task",
            service="orcamind",
            response=response,
        )
