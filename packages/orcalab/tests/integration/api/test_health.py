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
        assert body["name"] == "OrcaLab"
        assert body["version"] == "0.1.0"
        assert body["status"] == "ok"


class TestHealthEndpoint:
    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_response_has_required_fields(self, client: AsyncClient) -> None:
        body = (await client.get("/health")).json()
        assert "status" in body
        assert "db" in body
        assert "prefect" in body

    async def test_status_is_healthy_when_db_ok_and_no_prefect_url(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PREFECT_API_URL", raising=False)
        body = (await client.get("/health")).json()
        assert body["status"] == "healthy"
        assert body["db"] is True

    async def test_prefect_false_when_url_not_set(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PREFECT_API_URL", raising=False)
        body = (await client.get("/health")).json()
        assert body["prefect"] is False
