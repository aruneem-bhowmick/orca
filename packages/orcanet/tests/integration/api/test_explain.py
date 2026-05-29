"""Integration tests for POST /api/v1/explain."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import AsyncClient

from orcanet.reasoning.validators import LLMParsingError


class TestExplain:
    async def test_returns_explanation(
        self,
        client: AsyncClient,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        response = await client.post(
            "/api/v1/explain",
            json={
                "source_task_id": str(source_task_id),
                "target_task_id": str(target_task_id),
                "strategy": "feature",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "explanation" in body
        assert isinstance(body["explanation"], str)
        assert len(body["explanation"]) > 0

    async def test_uses_default_feature_strategy(
        self,
        client: AsyncClient,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        response = await client.post(
            "/api/v1/explain",
            json={
                "source_task_id": str(source_task_id),
                "target_task_id": str(target_task_id),
            },
        )
        assert response.status_code == 200

    async def test_llm_parsing_error_returns_502(
        self,
        client: AsyncClient,
        mock_agent: AsyncMock,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        mock_agent.recommend_transfer.side_effect = LLMParsingError("parse failed")
        response = await client.post(
            "/api/v1/explain",
            json={
                "source_task_id": str(source_task_id),
                "target_task_id": str(target_task_id),
            },
        )
        assert response.status_code == 502

    async def test_generic_agent_error_returns_502(
        self,
        client: AsyncClient,
        mock_agent: AsyncMock,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        mock_agent.recommend_transfer.side_effect = RuntimeError("connection refused")
        response = await client.post(
            "/api/v1/explain",
            json={
                "source_task_id": str(source_task_id),
                "target_task_id": str(target_task_id),
            },
        )
        assert response.status_code == 502
