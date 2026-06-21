"""Tests for the BFF application factory and health endpoint."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App factory tests
# ---------------------------------------------------------------------------


def test_create_app_returns_fastapi_instance(mock_settings):
    """create_app() should produce a FastAPI with correct title and version."""
    from orca_web.api.main import create_app

    app = create_app()
    assert isinstance(app, FastAPI)
    assert app.title == "Orca Web"
    assert app.version == "0.1.0"


def test_create_app_includes_routers(mock_settings):
    """All expected route paths should be registered on the app."""
    from orca_web.api.main import create_app

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    schema = app.openapi()
    paths = set(schema["paths"].keys())
    assert "/auth/login" in paths
    assert "/auth/register" in paths
    assert "/dashboard/overview" in paths
    assert "/dashboard/stats" in paths
    assert "/users/{user_id}" in paths
    assert "/health" in paths


# ---------------------------------------------------------------------------
# Lifespan tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_creates_app_state(mock_settings):
    """Lifespan should set db_engine, db_sessionmaker, and http_client on app.state."""
    from orca_web.api.main import create_app, lifespan

    app = create_app()

    mock_engine = AsyncMock()
    mock_engine.dispose = AsyncMock()

    with patch("orca_web.api.main.get_engine", return_value=mock_engine):
        async with lifespan(app):
            assert app.state.db_engine is mock_engine
            assert app.state.db_sessionmaker is not None
            assert isinstance(app.state.http_client, httpx.AsyncClient)


@pytest.mark.asyncio
async def test_lifespan_cleanup(mock_settings):
    """Lifespan should call engine.dispose() and http_client.aclose() on shutdown."""
    from orca_web.api.main import create_app, lifespan

    app = create_app()

    mock_engine = AsyncMock()
    mock_engine.dispose = AsyncMock()

    with patch("orca_web.api.main.get_engine", return_value=mock_engine):
        async with lifespan(app):
            # capture the http_client created during startup
            http_client = app.state.http_client

        mock_engine.dispose.assert_awaited_once()
        assert http_client.is_closed


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


def _make_health_client(mock_settings) -> TestClient:
    """Create a TestClient with all app.state attributes mocked."""
    from orca_web.api.main import create_app

    app = create_app()

    # Provide the state that lifespan would normally create
    mock_sessionmaker = MagicMock()
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_engine = AsyncMock()

    app.state.db_sessionmaker = mock_sessionmaker
    app.state.http_client = mock_http_client
    app.state.db_engine = mock_engine

    return TestClient(app, raise_server_exceptions=False), mock_sessionmaker, mock_http_client


def test_health_all_healthy(mock_settings):
    """When all services respond OK the endpoint returns 200 / healthy."""
    client, mock_sm, mock_http = _make_health_client(mock_settings)

    # Postgres: sessionmaker context manager chain succeeds
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_sm.return_value = mock_ctx

    # Redis: patch from_url to return a mock that pings OK
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()

    # Upstream services: httpx returns 200
    ok_resp = httpx.Response(200)
    mock_http.get = AsyncMock(return_value=ok_resp)

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert all(data["services"].values())


def test_health_postgres_down(mock_settings):
    """When Postgres is unreachable the endpoint returns 503 / degraded."""
    client, mock_sm, mock_http = _make_health_client(mock_settings)

    # Postgres: raise on enter
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(side_effect=Exception("pg down"))
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_sm.return_value = mock_ctx

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()

    ok_resp = httpx.Response(200)
    mock_http.get = AsyncMock(return_value=ok_resp)

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        resp = client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["services"]["postgres"] is False
    assert data["services"]["redis"] is True


def test_health_redis_down(mock_settings):
    """When Redis is unreachable the endpoint returns 503 / degraded."""
    client, mock_sm, mock_http = _make_health_client(mock_settings)

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_sm.return_value = mock_ctx

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=Exception("redis down"))
    mock_redis.aclose = AsyncMock()

    ok_resp = httpx.Response(200)
    mock_http.get = AsyncMock(return_value=ok_resp)

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        resp = client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["services"]["redis"] is False
    assert data["services"]["postgres"] is True


def test_health_upstream_orcamind_down(mock_settings):
    """When OrcaMind is unreachable the endpoint returns 503 / degraded."""
    client, mock_sm, mock_http = _make_health_client(mock_settings)

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_sm.return_value = mock_ctx

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()

    ok_resp = httpx.Response(200)

    async def _selective_get(url, **kwargs):
        if "localhost:8000" in url:
            raise httpx.ConnectError("orcamind down")
        return ok_resp

    mock_http.get = AsyncMock(side_effect=_selective_get)

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        resp = client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["services"]["orcamind"] is False
    assert data["services"]["orcalab"] is True
    assert data["services"]["orcanet"] is True


def test_health_upstream_orcalab_down(mock_settings):
    """When OrcaLab is unreachable the endpoint returns 503 / degraded."""
    client, mock_sm, mock_http = _make_health_client(mock_settings)

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_sm.return_value = mock_ctx

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()

    ok_resp = httpx.Response(200)

    async def _selective_get(url, **kwargs):
        if "localhost:8001" in url:
            raise httpx.ConnectError("orcalab down")
        return ok_resp

    mock_http.get = AsyncMock(side_effect=_selective_get)

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        resp = client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["services"]["orcalab"] is False
    assert data["services"]["orcamind"] is True
    assert data["services"]["orcanet"] is True


def test_health_upstream_orcanet_down(mock_settings):
    """When OrcaNet is unreachable the endpoint returns 503 / degraded."""
    client, mock_sm, mock_http = _make_health_client(mock_settings)

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_sm.return_value = mock_ctx

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()

    ok_resp = httpx.Response(200)

    async def _selective_get(url, **kwargs):
        if "localhost:8002" in url:
            raise httpx.ConnectError("orcanet down")
        return ok_resp

    mock_http.get = AsyncMock(side_effect=_selective_get)

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        resp = client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["services"]["orcanet"] is False
    assert data["services"]["orcamind"] is True
    assert data["services"]["orcalab"] is True


def test_health_all_down(mock_settings):
    """When every backing service is down the endpoint returns 503 with all false."""
    client, mock_sm, mock_http = _make_health_client(mock_settings)

    # Postgres down
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(side_effect=Exception("pg down"))
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_sm.return_value = mock_ctx

    # Redis down
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=Exception("redis down"))
    mock_redis.aclose = AsyncMock()

    # All upstreams down
    mock_http.get = AsyncMock(side_effect=httpx.ConnectError("all down"))

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        resp = client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "degraded"
    assert not any(data["services"].values())
