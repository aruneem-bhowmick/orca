"""Integration tests for GET / and GET /health."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import respx
from httpx import AsyncClient, Response


class TestRoot:
    async def test_returns_service_info(self, client: AsyncClient) -> None:
        response = await client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "OrcaNet"
        assert body["version"] == "1.0.0"
        assert body["status"] == "ok"


class TestHealth:
    @respx.mock
    async def test_all_healthy(
        self,
        client: AsyncClient,
        mock_agent: AsyncMock,
    ) -> None:
        respx.get("http://orcamind-test/health").mock(return_value=Response(200))
        respx.get("http://orcalab-test/health").mock(return_value=Response(200))
        mock_agent.llm.ainvoke = AsyncMock(return_value="pong")

        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["orcamind"] is True
        assert body["orcalab"] is True
        assert body["llm"] is True

    @respx.mock
    async def test_orcamind_down_reports_degraded(
        self,
        client: AsyncClient,
        mock_agent: AsyncMock,
    ) -> None:
        respx.get("http://orcamind-test/health").mock(return_value=Response(503))
        respx.get("http://orcalab-test/health").mock(return_value=Response(200))
        mock_agent.llm.ainvoke = AsyncMock(return_value="pong")

        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["orcamind"] is False
        assert body["orcalab"] is True

    @respx.mock
    async def test_llm_failure_reports_degraded(
        self,
        client: AsyncClient,
        mock_agent: AsyncMock,
    ) -> None:
        respx.get("http://orcamind-test/health").mock(return_value=Response(200))
        respx.get("http://orcalab-test/health").mock(return_value=Response(200))
        mock_agent.llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unreachable"))

        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["llm"] is False

    @respx.mock
    async def test_orcalab_down_reports_degraded(
        self,
        client: AsyncClient,
        mock_agent: AsyncMock,
    ) -> None:
        respx.get("http://orcamind-test/health").mock(return_value=Response(200))
        respx.get("http://orcalab-test/health").mock(return_value=Response(503))
        mock_agent.llm.ainvoke = AsyncMock(return_value="pong")

        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["orcalab"] is False
