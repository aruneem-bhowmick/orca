"""Tests for orca_web.api.routers.orcamind — OrcaMind proxy endpoints.

Verifies correct URL construction, user-ID header injection, error
handling (502 on connection error, 504 on timeout), and activity logging
on mutating (POST) calls.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from orca_web.api.routers.orcamind import (
    create_task,
    find_similar_tasks,
    get_task,
    list_tasks,
    predict_performance,
    recommend_model,
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
    req.query_params = query_params or {}
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


# ── GET /orcamind/tasks ──────────────────────────────────────────────────


class TestListTasks:
    """Tests for the list_tasks proxy endpoint."""

    async def test_forwards_get_to_upstream(self, mock_settings):
        """Builds the correct upstream URL for task listing."""
        upstream = _make_upstream(content=b'[{"task_id":"t1"}]')
        client = _make_client(upstream)
        user = _make_user()
        request = _make_request(client)

        resp = await list_tasks(request=request, user=user)

        call_kwargs = client.request.call_args.kwargs
        assert call_kwargs["url"] == "http://localhost:8000/api/v1/tasks"
        assert call_kwargs["method"] == "GET"
        assert resp.status_code == 200

    async def test_passes_query_params(self, mock_settings):
        """Query parameters from the browser are forwarded to OrcaMind."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client, query_params={"limit": "10", "domain": "vision"})

        await list_tasks(request=request, user=user)

        assert client.request.call_args.kwargs["params"] == {"limit": "10", "domain": "vision"}

    async def test_injects_user_id_header(self, mock_settings):
        """X-Orca-User-ID header is set to the current user's UUID."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)

        await list_tasks(request=request, user=user)

        headers = client.request.call_args.kwargs["headers"]
        assert headers["X-Orca-User-ID"] == str(user.user_id)

    async def test_returns_502_on_connection_error(self, mock_settings):
        """Network failures produce a 502 response."""
        client = _make_client(side_effect=httpx.ConnectError("refused"))
        request = _make_request(client)

        resp = await list_tasks(request=request, user=_make_user())

        assert resp.status_code == 502
        assert json.loads(resp.body)["detail"] == "Service unavailable"

    async def test_returns_504_on_timeout(self, mock_settings):
        """Timeouts produce a 504 response."""
        client = _make_client(side_effect=httpx.ReadTimeout("timed out"))
        request = _make_request(client)

        resp = await list_tasks(request=request, user=_make_user())

        assert resp.status_code == 504
        assert json.loads(resp.body)["detail"] == "Service timeout"


# ── GET /orcamind/tasks/{task_id} ────────────────────────────────────────


class TestGetTask:
    """Tests for the get_task proxy endpoint."""

    async def test_forwards_get_with_task_id(self, mock_settings):
        """The task ID path parameter is included in the upstream URL."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)

        await get_task(task_id="abc-123", request=request, user=user)

        url = client.request.call_args.kwargs["url"]
        assert url == "http://localhost:8000/api/v1/tasks/abc-123"

    async def test_injects_user_id_header(self, mock_settings):
        """X-Orca-User-ID header is set correctly."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)

        await get_task(task_id="x", request=request, user=user)

        headers = client.request.call_args.kwargs["headers"]
        assert headers["X-Orca-User-ID"] == str(user.user_id)


# ── POST /orcamind/tasks ─────────────────────────────────────────────────


class TestCreateTask:
    """Tests for the create_task proxy endpoint."""

    async def test_forwards_post_to_embed_endpoint(self, mock_settings):
        """POST /orcamind/tasks maps to POST {ORCAMIND}/api/v1/tasks/embed."""
        upstream = _make_upstream(content=b'{"task_id":"new-t"}', status_code=201)
        client = _make_client(upstream)
        user = _make_user()
        body = b'{"name":"iris"}'
        request = _make_request(client, body=body, content_type="application/json")
        repo = AsyncMock(spec=HistoryRepository)

        resp = await create_task(request=request, user=user, history_repo=repo)

        call_kwargs = client.request.call_args.kwargs
        assert call_kwargs["url"] == "http://localhost:8000/api/v1/tasks/embed"
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["content"] == body
        assert resp.status_code == 201

    async def test_logs_activity_on_success(self, mock_settings):
        """A task_created activity is logged after the proxy call."""
        upstream = _make_upstream(content=b'{"task_id":"new-t"}')
        client = _make_client(upstream)
        user = _make_user()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await create_task(request=request, user=user, history_repo=repo)

        repo.log_activity.assert_awaited_once()
        call_kwargs = repo.log_activity.call_args.kwargs
        assert call_kwargs["action"] == "task_created"
        assert call_kwargs["resource_type"] == "task"
        assert call_kwargs["service"] == "orcamind"
        assert call_kwargs["resource_id"] == "new-t"

    async def test_returns_502_on_connection_error(self, mock_settings):
        """Network failures produce a 502 and activity is still logged."""
        client = _make_client(side_effect=httpx.ConnectError("refused"))
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        resp = await create_task(request=request, user=_make_user(), history_repo=repo)

        assert resp.status_code == 502
        repo.log_activity.assert_awaited_once()

    async def test_returns_504_on_timeout(self, mock_settings):
        """Timeouts produce a 504 and activity is still logged."""
        client = _make_client(side_effect=httpx.ReadTimeout("timed out"))
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        resp = await create_task(request=request, user=_make_user(), history_repo=repo)

        assert resp.status_code == 504
        repo.log_activity.assert_awaited_once()


# ── POST /orcamind/recommend ─────────────────────────────────────────────


class TestRecommendModel:
    """Tests for the recommend_model proxy endpoint."""

    async def test_forwards_post_to_recommend(self, mock_settings):
        """Maps to POST {ORCAMIND}/api/v1/recommend-model."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client, body=b'{"task_embedding":[0.1]}')
        repo = AsyncMock(spec=HistoryRepository)

        await recommend_model(request=request, user=user, history_repo=repo)

        url = client.request.call_args.kwargs["url"]
        assert url == "http://localhost:8000/api/v1/recommend-model"

    async def test_logs_model_recommended_activity(self, mock_settings):
        """A model_recommended activity is logged."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await recommend_model(request=request, user=user, history_repo=repo)

        call_kwargs = repo.log_activity.call_args.kwargs
        assert call_kwargs["action"] == "model_recommended"
        assert call_kwargs["service"] == "orcamind"


# ── POST /orcamind/similar-tasks ─────────────────────────────────────────


class TestFindSimilarTasks:
    """Tests for the find_similar_tasks proxy endpoint."""

    async def test_forwards_post(self, mock_settings):
        """Maps to POST {ORCAMIND}/api/v1/similar-tasks."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await find_similar_tasks(request=request, user=user, history_repo=repo)

        url = client.request.call_args.kwargs["url"]
        assert url == "http://localhost:8000/api/v1/similar-tasks"

    async def test_logs_similar_tasks_activity(self, mock_settings):
        """A similar_tasks_searched activity is logged."""
        client = _make_client()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await find_similar_tasks(request=request, user=_make_user(), history_repo=repo)

        call_kwargs = repo.log_activity.call_args.kwargs
        assert call_kwargs["action"] == "similar_tasks_searched"
        assert call_kwargs["resource_type"] == "task"


# ── POST /orcamind/predict-performance ───────────────────────────────────


class TestPredictPerformance:
    """Tests for the predict_performance proxy endpoint."""

    async def test_forwards_post(self, mock_settings):
        """Maps to POST {ORCAMIND}/api/v1/predict-performance."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await predict_performance(request=request, user=user, history_repo=repo)

        url = client.request.call_args.kwargs["url"]
        assert url == "http://localhost:8000/api/v1/predict-performance"

    async def test_logs_performance_predicted_activity(self, mock_settings):
        """A performance_predicted activity is logged."""
        client = _make_client()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await predict_performance(request=request, user=_make_user(), history_repo=repo)

        call_kwargs = repo.log_activity.call_args.kwargs
        assert call_kwargs["action"] == "performance_predicted"
        assert call_kwargs["resource_type"] == "prediction"
        assert call_kwargs["service"] == "orcamind"
