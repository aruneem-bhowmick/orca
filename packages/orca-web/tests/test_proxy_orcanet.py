"""Tests for orca_web.api.routers.orcanet — OrcaNet proxy endpoints.

Verifies correct URL construction, user-ID header injection, error
handling (502 on connection error, 504 on timeout), and activity logging
for all four OrcaNet proxy endpoints (transfer scoring, transfer
recommendation, task retrieval, and transfer explanation).
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from starlette.datastructures import QueryParams

from orca_web.api.routers.orcanet import (
    explain_transfer,
    recommend_transfer,
    retrieve_tasks,
    score_transfer,
)
from orca_web.repository.history_repo import HistoryRepository


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_upstream(*, status_code=200, content=b'{}', content_type="application/json"):
    """Build a mock httpx response."""
    up = MagicMock()
    up.status_code = status_code
    up.content = content
    up.headers = {"content-type": content_type}
    return up


def _make_client(upstream=None, *, side_effect=None):
    """Build a mock httpx.AsyncClient."""
    client = AsyncMock()
    if side_effect:
        client.request = AsyncMock(side_effect=side_effect)
    else:
        client.request = AsyncMock(return_value=upstream or _make_upstream())
    return client


def _make_request(client, *, body=b"", query_params=None, content_type=None):
    """Build a mock FastAPI Request."""
    req = MagicMock()
    req.app.state.http_client = client
    req.body = AsyncMock(return_value=body)
    req.query_params = QueryParams(query_params or [])
    headers = {}
    if content_type:
        headers["content-type"] = content_type
    req.headers = headers
    return req


def _make_user(uid=None):
    """Build a mock User."""
    user = MagicMock()
    user.user_id = uid or uuid.uuid4()
    return user


# ── POST /orcanet/transfer/score ─────────────────────────────────────────


class TestScoreTransfer:
    """Tests for the score_transfer proxy endpoint."""

    async def test_forwards_post_to_transfer_score(self, mock_settings):
        """Maps to POST {ORCANET}/api/v1/transfer/score."""
        content = json.dumps({"overall": 0.85}).encode()
        upstream = _make_upstream(content=content)
        client = _make_client(upstream)
        user = _make_user()
        body = b'{"source_task_id":"t1","target_task_id":"t2"}'
        request = _make_request(client, body=body, content_type="application/json")
        repo = AsyncMock(spec=HistoryRepository)

        resp = await score_transfer(request=request, user=user, history_repo=repo)

        kw = client.request.call_args.kwargs
        assert kw["url"] == "http://localhost:8002/api/v1/transfer/score"
        assert kw["method"] == "POST"
        assert kw["content"] == body
        assert resp.status_code == 200

    async def test_injects_user_id_header(self, mock_settings):
        """X-Orca-User-ID header is present on the upstream call."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await score_transfer(request=request, user=user, history_repo=repo)

        headers = client.request.call_args.kwargs["headers"]
        assert headers["X-Orca-User-ID"] == str(user.user_id)

    async def test_logs_transfer_scored_activity(self, mock_settings):
        """A transfer_scored activity is logged."""
        client = _make_client()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await score_transfer(request=request, user=_make_user(), history_repo=repo)

        kw = repo.log_activity.call_args.kwargs
        assert kw["action"] == "transfer_scored"
        assert kw["resource_type"] == "transfer"
        assert kw["service"] == "orcanet"

    async def test_returns_502_on_connection_error(self, mock_settings):
        """Network failures produce a 502 and activity is still logged."""
        client = _make_client(side_effect=httpx.ConnectError("refused"))
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        resp = await score_transfer(request=request, user=_make_user(), history_repo=repo)

        assert resp.status_code == 502
        repo.log_activity.assert_awaited_once()

    async def test_returns_504_on_timeout(self, mock_settings):
        """Timeouts produce a 504 and activity is still logged."""
        client = _make_client(side_effect=httpx.ReadTimeout("timed out"))
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        resp = await score_transfer(request=request, user=_make_user(), history_repo=repo)

        assert resp.status_code == 504
        repo.log_activity.assert_awaited_once()


# ── POST /orcanet/transfer/recommend ─────────────────────────────────────


class TestRecommendTransfer:
    """Tests for the recommend_transfer proxy endpoint."""

    async def test_forwards_post_to_transfer_recommend(self, mock_settings):
        """Maps to POST {ORCANET}/api/v1/transfer/recommend."""
        client = _make_client()
        request = _make_request(client, body=b'{"target_task_id":"t2"}')
        repo = AsyncMock(spec=HistoryRepository)

        await recommend_transfer(request=request, user=_make_user(), history_repo=repo)

        url = client.request.call_args.kwargs["url"]
        assert url == "http://localhost:8002/api/v1/transfer/recommend"

    async def test_logs_transfer_recommended_activity(self, mock_settings):
        """A transfer_recommended activity is logged."""
        client = _make_client()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await recommend_transfer(request=request, user=_make_user(), history_repo=repo)

        kw = repo.log_activity.call_args.kwargs
        assert kw["action"] == "transfer_recommended"
        assert kw["resource_type"] == "transfer"
        assert kw["service"] == "orcanet"

    async def test_returns_502_on_connection_error(self, mock_settings):
        """Network failures produce a 502."""
        client = _make_client(side_effect=httpx.ConnectError("refused"))
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        resp = await recommend_transfer(request=request, user=_make_user(), history_repo=repo)

        assert resp.status_code == 502


# ── POST /orcanet/retrieve ───────────────────────────────────────────────


class TestRetrieveTasks:
    """Tests for the retrieve_tasks proxy endpoint."""

    async def test_forwards_post_to_retrieve(self, mock_settings):
        """Maps to POST {ORCANET}/api/v1/retrieve."""
        client = _make_client()
        body = b'{"query":"image classification","top_k":5}'
        request = _make_request(client, body=body, content_type="application/json")
        repo = AsyncMock(spec=HistoryRepository)

        await retrieve_tasks(request=request, user=_make_user(), history_repo=repo)

        kw = client.request.call_args.kwargs
        assert kw["url"] == "http://localhost:8002/api/v1/retrieve"
        assert kw["content"] == body

    async def test_logs_tasks_retrieved_activity(self, mock_settings):
        """A tasks_retrieved activity is logged."""
        client = _make_client()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await retrieve_tasks(request=request, user=_make_user(), history_repo=repo)

        kw = repo.log_activity.call_args.kwargs
        assert kw["action"] == "tasks_retrieved"
        assert kw["resource_type"] == "task"
        assert kw["service"] == "orcanet"

    async def test_returns_504_on_timeout(self, mock_settings):
        """Timeouts produce a 504."""
        client = _make_client(side_effect=httpx.ReadTimeout("timed out"))
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        resp = await retrieve_tasks(request=request, user=_make_user(), history_repo=repo)

        assert resp.status_code == 504


# ── POST /orcanet/explain ────────────────────────────────────────────────


class TestExplainTransfer:
    """Tests for the explain_transfer proxy endpoint."""

    async def test_forwards_post_to_explain(self, mock_settings):
        """Maps to POST {ORCANET}/api/v1/explain."""
        client = _make_client()
        body = b'{"source_task_id":"t1","target_task_id":"t2"}'
        request = _make_request(client, body=body, content_type="application/json")
        repo = AsyncMock(spec=HistoryRepository)

        await explain_transfer(request=request, user=_make_user(), history_repo=repo)

        kw = client.request.call_args.kwargs
        assert kw["url"] == "http://localhost:8002/api/v1/explain"
        assert kw["content"] == body

    async def test_logs_transfer_explained_activity(self, mock_settings):
        """A transfer_explained activity is logged."""
        client = _make_client()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await explain_transfer(request=request, user=_make_user(), history_repo=repo)

        kw = repo.log_activity.call_args.kwargs
        assert kw["action"] == "transfer_explained"
        assert kw["resource_type"] == "transfer"
        assert kw["service"] == "orcanet"

    async def test_injects_user_id_header(self, mock_settings):
        """X-Orca-User-ID header is present on the upstream call."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await explain_transfer(request=request, user=user, history_repo=repo)

        headers = client.request.call_args.kwargs["headers"]
        assert headers["X-Orca-User-ID"] == str(user.user_id)

    async def test_returns_502_on_connection_error(self, mock_settings):
        """Network failures produce a 502."""
        client = _make_client(side_effect=httpx.ConnectError("refused"))
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        resp = await explain_transfer(request=request, user=_make_user(), history_repo=repo)

        assert resp.status_code == 502
