"""Integration tests for GET / and GET /health."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient


class TestRootEndpoint:
    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/")
        assert resp.status_code == 200

    async def test_response_body(self, client: AsyncClient) -> None:
        body = (await client.get("/")).json()
        assert body["name"] == "OrcaMind"
        assert body["version"] == "1.0.0"
        assert body["status"] == "ok"


class TestHealthEndpoint:
    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_response_has_required_fields(self, client: AsyncClient) -> None:
        body = (await client.get("/health")).json()
        assert "status" in body
        assert "db" in body
        assert "faiss" in body
        assert "mlflow" in body

    async def test_status_is_degraded_when_faiss_not_loaded(
        self, client: AsyncClient
    ) -> None:
        # The conftest sets faiss_index=None, so db=True but faiss=False → "degraded".
        body = (await client.get("/health")).json()
        assert body["status"] == "degraded"

    async def test_status_is_healthy_when_all_deps_ok(
        self, client: AsyncClient
    ) -> None:
        # Temporarily set a non-None faiss_index so overall_ok becomes True.
        app = client._transport.app  # type: ignore[attr-defined]
        app.state.faiss_index = MagicMock()
        try:
            body = (await client.get("/health")).json()
            assert body["status"] == "healthy"
        finally:
            app.state.faiss_index = None

    async def test_faiss_false_when_index_not_loaded(
        self, client: AsyncClient
    ) -> None:
        # The conftest pre-populates app.state.faiss_index = None, so faiss must be False.
        body = (await client.get("/health")).json()
        assert body["faiss"] is False

    async def test_db_true_when_sessionmaker_succeeds(
        self, client: AsyncClient
    ) -> None:
        # The conftest fake sessionmaker returns a successful mock, so db must be True.
        body = (await client.get("/health")).json()
        assert body["db"] is True

    async def test_mlflow_false_when_uri_not_set(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        body = (await client.get("/health")).json()
        assert body["mlflow"] is False
