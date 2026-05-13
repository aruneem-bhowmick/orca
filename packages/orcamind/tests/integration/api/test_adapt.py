"""Integration tests for POST /api/v1/adapt."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from orca_shared.registry.models import Model as ModelORM


def _make_model_row(model_id: UUID) -> MagicMock:
    row = MagicMock(spec=ModelORM)
    row.model_id = model_id
    row.name = "mlp"
    row.architecture = "mlp"
    row.config = {}
    row.parameter_count = None
    row.flops = None
    return row


class TestStartAdaptation:
    def _body(self, task_id: UUID, model_id: UUID) -> dict:
        return {"task_id": str(task_id), "model_id": str(model_id)}

    async def test_returns_200(
        self,
        client: AsyncClient,
        task_id: UUID,
        model_id: UUID,
        mock_session: AsyncMock,
    ) -> None:
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = _make_model_row(model_id)
        mock_session.execute.return_value = execute_result

        with patch("orcamind.api.routers.adapt._run_adaptation"):
            resp = await client.post("/api/v1/adapt", json=self._body(task_id, model_id))
        assert resp.status_code == 200

    async def test_response_contains_job_id(
        self,
        client: AsyncClient,
        task_id: UUID,
        model_id: UUID,
        experiment_id: UUID,
        mock_session: AsyncMock,
        mock_experiment_repo: AsyncMock,
    ) -> None:
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = _make_model_row(model_id)
        mock_session.execute.return_value = execute_result

        with patch("orcamind.api.routers.adapt._run_adaptation"):
            body = (
                await client.post("/api/v1/adapt", json=self._body(task_id, model_id))
            ).json()
        assert "job_id" in body
        # job_id is the experiment_id returned by exp_repo.create
        assert UUID(body["job_id"]) == experiment_id

    async def test_returns_404_when_task_not_found(
        self,
        client: AsyncClient,
        model_id: UUID,
        mock_task_repo: AsyncMock,
    ) -> None:
        mock_task_repo.get_by_id.return_value = None
        resp = await client.post(
            "/api/v1/adapt", json=self._body(uuid4(), model_id)
        )
        assert resp.status_code == 404

    async def test_returns_404_when_model_not_found(
        self,
        client: AsyncClient,
        task_id: UUID,
        mock_session: AsyncMock,
    ) -> None:
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = execute_result

        resp = await client.post(
            "/api/v1/adapt", json=self._body(task_id, uuid4())
        )
        assert resp.status_code == 404

    async def test_returns_422_on_missing_fields(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post("/api/v1/adapt", json={})
        assert resp.status_code == 422

    async def test_optional_training_config_accepted(
        self,
        client: AsyncClient,
        task_id: UUID,
        model_id: UUID,
        mock_session: AsyncMock,
    ) -> None:
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = _make_model_row(model_id)
        mock_session.execute.return_value = execute_result

        body_with_config = {
            "task_id": str(task_id),
            "model_id": str(model_id),
            "training_config": {"lr": 0.001, "epochs": 5},
        }
        with patch("orcamind.api.routers.adapt._run_adaptation"):
            resp = await client.post("/api/v1/adapt", json=body_with_config)
        assert resp.status_code == 200
