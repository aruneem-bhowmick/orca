"""Integration tests for transfer scoring, recommendation, and mapping lookup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from orca_shared.schemas.transfer import TransferMapping
from orcanet.reasoning.validators import LLMParsingError
from orcanet.transfer.types import TransferScore


class TestTransferScore:
    async def test_returns_score_for_valid_tasks(
        self,
        client: AsyncClient,
        source_task_id: UUID,
        target_task_id: UUID,
        mock_transfer_score: TransferScore,
    ) -> None:
        response = await client.post(
            "/api/v1/transfer/score",
            json={
                "source_task_id": str(source_task_id),
                "target_task_id": str(target_task_id),
                "strategy": "feature",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["overall"] == pytest.approx(0.72)
        assert body["strategy"] == "feature"
        assert "layer_scores" in body
        assert "recommended_layers" in body

    async def test_unknown_strategy_returns_400(
        self,
        client: AsyncClient,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        response = await client.post(
            "/api/v1/transfer/score",
            json={
                "source_task_id": str(source_task_id),
                "target_task_id": str(target_task_id),
                "strategy": "nonexistent",
            },
        )
        assert response.status_code == 400
        assert "nonexistent" in response.json()["detail"]

    async def test_missing_source_task_returns_404(
        self,
        client: AsyncClient,
        target_task_id: UUID,
    ) -> None:
        response = await client.post(
            "/api/v1/transfer/score",
            json={
                "source_task_id": str(uuid4()),
                "target_task_id": str(target_task_id),
                "strategy": "feature",
            },
        )
        assert response.status_code == 404

    async def test_missing_target_task_returns_404(
        self,
        client: AsyncClient,
        source_task_id: UUID,
    ) -> None:
        response = await client.post(
            "/api/v1/transfer/score",
            json={
                "source_task_id": str(source_task_id),
                "target_task_id": str(uuid4()),
                "strategy": "feature",
            },
        )
        assert response.status_code == 404


class TestTransferRecommend:
    async def test_returns_recommendation(
        self,
        client: AsyncClient,
        target_task_id: UUID,
        source_task_id: UUID,
    ) -> None:
        response = await client.post(
            "/api/v1/transfer/recommend",
            json={
                "target_task_id": str(target_task_id),
                "query_description": "Image classification on small dataset",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "top_sources" in body
        assert body["recommended_strategy"] == "feature"
        assert 0.0 <= body["expected_improvement"] <= 1.0
        assert "explanation" in body
        assert body["top_sources"][0]["task_id"] == str(source_task_id)

    async def test_agent_error_returns_502(
        self,
        client: AsyncClient,
        mock_agent: AsyncMock,
        target_task_id: UUID,
    ) -> None:
        mock_agent.recommend_transfer.side_effect = RuntimeError("LLM down")
        response = await client.post(
            "/api/v1/transfer/recommend",
            json={
                "target_task_id": str(target_task_id),
                "query_description": "test",
            },
        )
        assert response.status_code == 502


class TestGetTransferMapping:
    async def test_returns_mapping_for_known_id(
        self,
        client: AsyncClient,
        mapping_id: UUID,
        source_task_id: UUID,
        target_task_id: UUID,
        now,
        mock_session: AsyncMock,
    ) -> None:
        from orca_shared.registry.models import TransferMapping as TransferMappingORM

        orm_row = MagicMock(spec=TransferMappingORM)
        orm_row.mapping_id = mapping_id
        orm_row.source_task_id = source_task_id
        orm_row.target_task_id = target_task_id
        orm_row.transfer_score = 0.65
        orm_row.transfer_type = "feature"
        orm_row.mapping_metadata = None
        orm_row.created_at = now

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = orm_row
        mock_session.execute = AsyncMock(return_value=result_mock)

        response = await client.get(f"/api/v1/transfer/{mapping_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["mapping_id"] == str(mapping_id)
        assert body["transfer_score"] == pytest.approx(0.65)

    async def test_returns_404_for_unknown_id(
        self,
        client: AsyncClient,
        mock_session: AsyncMock,
    ) -> None:
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=result_mock)

        response = await client.get(f"/api/v1/transfer/{uuid4()}")
        assert response.status_code == 404

    async def test_invalid_uuid_returns_422(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/transfer/not-a-uuid")
        assert response.status_code == 422
