"""Integration tests for GET / and GET /health."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
import respx
from httpx import AsyncClient, Response

from orcanet.api.main import create_app
from httpx import ASGITransport

from tests.integration.api.conftest import _build_app


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
        mock_session,
        mock_task_repo,
        mock_agent,
        mock_retriever,
        mock_embedder,
        mock_transfer_strategies,
        mock_orcamind_client,
        mock_orcalab_client,
    ) -> None:
        respx.get("http://orcamind-test/health").mock(return_value=Response(200))
        respx.get("http://orcalab-test/health").mock(return_value=Response(200))
        mock_agent.llm.ainvoke = AsyncMock(return_value="pong")

        app = _build_app(
            mock_session, mock_task_repo, mock_agent, mock_retriever,
            mock_embedder, mock_transfer_strategies, mock_orcamind_client,
            mock_orcalab_client,
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["orcamind"] is True
        assert body["orcalab"] is True
        assert body["llm"] is True

    @respx.mock
    async def test_orcamind_down_reports_degraded(
        self,
        mock_session,
        mock_task_repo,
        mock_agent,
        mock_retriever,
        mock_embedder,
        mock_transfer_strategies,
        mock_orcamind_client,
        mock_orcalab_client,
    ) -> None:
        respx.get("http://orcamind-test/health").mock(return_value=Response(503))
        respx.get("http://orcalab-test/health").mock(return_value=Response(200))
        mock_agent.llm.ainvoke = AsyncMock(return_value="pong")

        app = _build_app(
            mock_session, mock_task_repo, mock_agent, mock_retriever,
            mock_embedder, mock_transfer_strategies, mock_orcamind_client,
            mock_orcalab_client,
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["orcamind"] is False
        assert body["orcalab"] is True

    @respx.mock
    async def test_llm_failure_reports_degraded(
        self,
        mock_session,
        mock_task_repo,
        mock_agent,
        mock_retriever,
        mock_embedder,
        mock_transfer_strategies,
        mock_orcamind_client,
        mock_orcalab_client,
    ) -> None:
        respx.get("http://orcamind-test/health").mock(return_value=Response(200))
        respx.get("http://orcalab-test/health").mock(return_value=Response(200))
        mock_agent.llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unreachable"))

        app = _build_app(
            mock_session, mock_task_repo, mock_agent, mock_retriever,
            mock_embedder, mock_transfer_strategies, mock_orcamind_client,
            mock_orcalab_client,
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["llm"] is False
