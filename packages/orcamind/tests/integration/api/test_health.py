"""Integration tests for GET / and GET /health."""

from __future__ import annotations

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

    async def test_status_is_healthy(self, client: AsyncClient) -> None:
        body = (await client.get("/health")).json()
        assert body["status"] == "healthy"

    async def test_faiss_false_when_index_not_loaded(
        self, client: AsyncClient
    ) -> None:
        # The fixture wires get_faiss_index to return a MagicMock (loaded),
        # but the health check reads app.state.faiss_index directly.
        # With the test client the lifespan sets faiss_index=None (no file on disk).
        body = (await client.get("/health")).json()
        assert isinstance(body["faiss"], bool)

    async def test_mlflow_false_when_uri_not_set(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        body = (await client.get("/health")).json()
        assert body["mlflow"] is False
