"""Tests for orca_web.api.routers.orcalab — OrcaLab proxy endpoints.

Verifies correct URL construction, user-ID header injection, error
handling (502 on connection error, 504 on timeout), and activity logging
on mutating (POST) calls for experiment and sweep endpoints.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from starlette.datastructures import QueryParams

from orca_web.api.routers.orcalab import (
    create_experiment,
    create_sweep,
    get_experiment,
    get_sweep,
    list_experiments,
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


# ── GET /orcalab/experiments ─────────────────────────────────────────────


class TestListExperiments:
    """Tests for the list_experiments proxy endpoint."""

    async def test_forwards_get_to_upstream(self, mock_settings):
        """Constructs the correct upstream URL for experiment listing."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)

        resp = await list_experiments(request=request, user=user)

        call_kwargs = client.request.call_args.kwargs
        assert call_kwargs["url"] == "http://localhost:8001/api/v1/experiments"
        assert call_kwargs["method"] == "GET"
        assert resp.status_code == 200

    async def test_injects_user_id_header(self, mock_settings):
        """X-Orca-User-ID header is set correctly."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)

        await list_experiments(request=request, user=user)

        headers = client.request.call_args.kwargs["headers"]
        assert headers["X-Orca-User-ID"] == str(user.user_id)

    async def test_passes_query_params(self, mock_settings):
        """Query parameters are forwarded to OrcaLab."""
        client = _make_client()
        request = _make_request(client, query_params={"limit": "20", "offset": "5"})

        await list_experiments(request=request, user=_make_user())

        assert client.request.call_args.kwargs["params"] == [("limit", "20"), ("offset", "5")]

    async def test_returns_502_on_connection_error(self, mock_settings):
        """Network failures produce a 502 response."""
        client = _make_client(side_effect=httpx.ConnectError("refused"))
        request = _make_request(client)

        resp = await list_experiments(request=request, user=_make_user())

        assert resp.status_code == 502


# ── GET /orcalab/experiments/{experiment_id} ─────────────────────────────


class TestGetExperiment:
    """Tests for the get_experiment proxy endpoint."""

    async def test_forwards_get_with_experiment_id(self, mock_settings):
        """The experiment ID path parameter is included in the upstream URL."""
        client = _make_client()
        request = _make_request(client)

        await get_experiment(experiment_id="exp-42", request=request, user=_make_user())

        url = client.request.call_args.kwargs["url"]
        assert url == "http://localhost:8001/api/v1/experiments/exp-42"

    async def test_returns_504_on_timeout(self, mock_settings):
        """Timeouts produce a 504 response."""
        client = _make_client(side_effect=httpx.ReadTimeout("timed out"))
        request = _make_request(client)

        resp = await get_experiment(experiment_id="x", request=request, user=_make_user())

        assert resp.status_code == 504


# ── POST /orcalab/experiments ────────────────────────────────────────────


class TestCreateExperiment:
    """Tests for the create_experiment proxy endpoint."""

    async def test_forwards_post_to_experiments(self, mock_settings):
        """POST maps to the OrcaLab experiments endpoint."""
        content = json.dumps({"experiment_id": "new-exp"}).encode()
        upstream = _make_upstream(content=content, status_code=201)
        client = _make_client(upstream)
        user = _make_user()
        body = b'{"task_id":"t1","model_id":"m1"}'
        request = _make_request(client, body=body, content_type="application/json")
        repo = AsyncMock(spec=HistoryRepository)

        resp = await create_experiment(request=request, user=user, history_repo=repo)

        call_kwargs = client.request.call_args.kwargs
        assert call_kwargs["url"] == "http://localhost:8001/api/v1/experiments"
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["content"] == body
        assert resp.status_code == 201

    async def test_logs_experiment_started_activity(self, mock_settings):
        """An experiment_started activity is logged with the experiment ID."""
        content = json.dumps({"experiment_id": "new-exp"}).encode()
        upstream = _make_upstream(content=content)
        client = _make_client(upstream)
        user = _make_user()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await create_experiment(request=request, user=user, history_repo=repo)

        repo.log_activity.assert_awaited_once()
        kw = repo.log_activity.call_args.kwargs
        assert kw["action"] == "experiment_started"
        assert kw["resource_type"] == "experiment"
        assert kw["service"] == "orcalab"
        assert kw["resource_id"] == "new-exp"

    async def test_returns_502_on_connection_error(self, mock_settings):
        """Network failures produce a 502 and activity is still logged."""
        client = _make_client(side_effect=httpx.ConnectError("refused"))
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        resp = await create_experiment(request=request, user=_make_user(), history_repo=repo)

        assert resp.status_code == 502
        repo.log_activity.assert_awaited_once()

    async def test_returns_504_on_timeout(self, mock_settings):
        """Timeouts produce a 504 and activity is still logged."""
        client = _make_client(side_effect=httpx.ReadTimeout("timed out"))
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        resp = await create_experiment(request=request, user=_make_user(), history_repo=repo)

        assert resp.status_code == 504
        repo.log_activity.assert_awaited_once()

    async def test_injects_user_id_header(self, mock_settings):
        """X-Orca-User-ID header is present on the upstream call."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await create_experiment(request=request, user=user, history_repo=repo)

        headers = client.request.call_args.kwargs["headers"]
        assert headers["X-Orca-User-ID"] == str(user.user_id)


# ── POST /orcalab/sweeps ────────────────────────────────────────────────


class TestCreateSweep:
    """Tests for the create_sweep proxy endpoint."""

    async def test_forwards_post_to_sweeps(self, mock_settings):
        """POST maps to the OrcaLab sweeps endpoint."""
        content = json.dumps({"sweep_id": "sw-1"}).encode()
        upstream = _make_upstream(content=content, status_code=202)
        client = _make_client(upstream)
        request = _make_request(client, body=b'{"task_id":"t1","n_trials":50}')
        repo = AsyncMock(spec=HistoryRepository)

        resp = await create_sweep(request=request, user=_make_user(), history_repo=repo)

        assert client.request.call_args.kwargs["url"] == "http://localhost:8001/api/v1/sweeps"
        assert resp.status_code == 202

    async def test_logs_sweep_started_activity(self, mock_settings):
        """A sweep_started activity is logged with the sweep ID."""
        content = json.dumps({"sweep_id": "sw-1"}).encode()
        upstream = _make_upstream(content=content)
        client = _make_client(upstream)
        request = _make_request(client)
        repo = AsyncMock(spec=HistoryRepository)

        await create_sweep(request=request, user=_make_user(), history_repo=repo)

        kw = repo.log_activity.call_args.kwargs
        assert kw["action"] == "sweep_started"
        assert kw["resource_type"] == "sweep"
        assert kw["service"] == "orcalab"
        assert kw["resource_id"] == "sw-1"


# ── GET /orcalab/sweeps/{sweep_id} ──────────────────────────────────────


class TestGetSweep:
    """Tests for the get_sweep proxy endpoint."""

    async def test_forwards_get_with_sweep_id(self, mock_settings):
        """The sweep ID path parameter is included in the upstream URL."""
        client = _make_client()
        request = _make_request(client)

        await get_sweep(sweep_id="sw-99", request=request, user=_make_user())

        url = client.request.call_args.kwargs["url"]
        assert url == "http://localhost:8001/api/v1/sweeps/sw-99"

    async def test_injects_user_id_header(self, mock_settings):
        """X-Orca-User-ID header is present on the upstream call."""
        client = _make_client()
        user = _make_user()
        request = _make_request(client)

        await get_sweep(sweep_id="sw-99", request=request, user=user)

        headers = client.request.call_args.kwargs["headers"]
        assert headers["X-Orca-User-ID"] == str(user.user_id)
