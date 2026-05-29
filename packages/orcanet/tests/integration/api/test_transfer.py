"""Integration tests for transfer scoring, recommendation, validation, and mapping lookup."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from orca_shared.schemas.training import ExperimentResult
from orca_shared.schemas.transfer import TransferMapping
from orcanet.integration.pipeline import ServiceUnavailableError, TransferValidationResult
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


class TestValidateTransfer:
    """Tests for POST /api/v1/transfer/validate."""

    def _make_pipeline_result(
        self,
        source_task_id: UUID,
        target_task_id: UUID,
        score: float = 0.78,
        with_experiment: bool = True,
    ) -> TransferValidationResult:
        now = datetime.now(timezone.utc)
        ts = TransferScore(
            overall=score,
            layer_scores={"layer0": score},
            recommended_layers=["layer0"],
            reasoning="test",
        )
        exp = (
            ExperimentResult(
                experiment_id=uuid4(),
                task_id=target_task_id,
                model_id=None,
                status="COMPLETED",
                metrics={"accuracy": 0.9, "baseline_accuracy": 0.78},
            )
            if with_experiment
            else None
        )
        mapping = TransferMapping(
            mapping_id=uuid4(),
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            transfer_score=score,
            transfer_type="feature",
            metadata=None,
            created_at=now,
        )
        return TransferValidationResult(
            score=ts,
            experiment_result=exp,
            mapping=mapping,
            improvement_over_baseline=0.12 if with_experiment else None,
        )

    @pytest.mark.asyncio
    async def test_returns_200_with_full_result(
        self,
        client: AsyncClient,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        from orcanet.api.deps import get_transfer_pipeline
        from tests.integration.api.conftest import _build_app
        from unittest.mock import AsyncMock, MagicMock
        from httpx import ASGITransport, AsyncClient as HXClient

        pipeline_mock = AsyncMock()
        pipeline_mock.recommend_and_validate = AsyncMock(
            return_value=self._make_pipeline_result(source_task_id, target_task_id)
        )

        app = client.transport.app  # type: ignore[attr-defined]
        app.dependency_overrides[get_transfer_pipeline] = lambda: pipeline_mock

        try:
            response = await client.post(
                "/api/v1/transfer/validate",
                json={
                    "source_task_id": str(source_task_id),
                    "target_task_id": str(target_task_id),
                    "strategy": "feature",
                    "validate": True,
                },
            )
        finally:
            app.dependency_overrides.pop(get_transfer_pipeline, None)

        assert response.status_code == 200
        body = response.json()
        assert "score" in body
        assert body["score"]["overall"] == pytest.approx(0.78)
        assert "mapping" in body
        assert "experiment_result" in body
        assert body["experiment_result"] is not None

    @pytest.mark.asyncio
    async def test_returns_200_without_experiment_when_validate_false(
        self,
        client: AsyncClient,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        from orcanet.api.deps import get_transfer_pipeline

        pipeline_mock = AsyncMock()
        pipeline_mock.recommend_and_validate = AsyncMock(
            return_value=self._make_pipeline_result(
                source_task_id, target_task_id, with_experiment=False
            )
        )

        app = client.transport.app  # type: ignore[attr-defined]
        app.dependency_overrides[get_transfer_pipeline] = lambda: pipeline_mock

        try:
            response = await client.post(
                "/api/v1/transfer/validate",
                json={
                    "source_task_id": str(source_task_id),
                    "target_task_id": str(target_task_id),
                    "validate": False,
                },
            )
        finally:
            app.dependency_overrides.pop(get_transfer_pipeline, None)

        assert response.status_code == 200
        body = response.json()
        assert body["experiment_result"] is None
        assert body["improvement_over_baseline"] is None

    @pytest.mark.asyncio
    async def test_service_unavailable_returns_503(
        self,
        client: AsyncClient,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        from orcanet.api.deps import get_transfer_pipeline

        pipeline_mock = AsyncMock()
        pipeline_mock.recommend_and_validate = AsyncMock(
            side_effect=ServiceUnavailableError("OrcaMind is unreachable")
        )

        app = client.transport.app  # type: ignore[attr-defined]
        app.dependency_overrides[get_transfer_pipeline] = lambda: pipeline_mock

        try:
            response = await client.post(
                "/api/v1/transfer/validate",
                json={
                    "source_task_id": str(source_task_id),
                    "target_task_id": str(target_task_id),
                },
            )
        finally:
            app.dependency_overrides.pop(get_transfer_pipeline, None)

        assert response.status_code == 503
        assert "OrcaMind" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_task_returns_404(
        self,
        client: AsyncClient,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        from orcanet.api.deps import get_transfer_pipeline

        pipeline_mock = AsyncMock()
        pipeline_mock.recommend_and_validate = AsyncMock(
            side_effect=ValueError("Source task not found")
        )

        app = client.transport.app  # type: ignore[attr-defined]
        app.dependency_overrides[get_transfer_pipeline] = lambda: pipeline_mock

        try:
            response = await client.post(
                "/api/v1/transfer/validate",
                json={
                    "source_task_id": str(source_task_id),
                    "target_task_id": str(target_task_id),
                },
            )
        finally:
            app.dependency_overrides.pop(get_transfer_pipeline, None)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_strategy_returns_400(
        self,
        client: AsyncClient,
        source_task_id: UUID,
        target_task_id: UUID,
    ) -> None:
        from orcanet.api.deps import get_transfer_pipeline

        pipeline_mock = AsyncMock()
        pipeline_mock.recommend_and_validate = AsyncMock(
            side_effect=KeyError("nonexistent")
        )

        app = client.transport.app  # type: ignore[attr-defined]
        app.dependency_overrides[get_transfer_pipeline] = lambda: pipeline_mock

        try:
            response = await client.post(
                "/api/v1/transfer/validate",
                json={
                    "source_task_id": str(source_task_id),
                    "target_task_id": str(target_task_id),
                    "strategy": "nonexistent",
                },
            )
        finally:
            app.dependency_overrides.pop(get_transfer_pipeline, None)

        assert response.status_code == 400
