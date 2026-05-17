"""Integration tests for experiment CRUD endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from orca_shared.schemas.training import ExperimentResult


def _experiment(experiment_id: UUID, status: str = "pending", **kw) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        status=status,
        started_at=datetime.now(timezone.utc),
        **kw,
    )


class TestCreateExperiment:
    async def test_returns_201(self, client: AsyncClient, experiment_id: UUID) -> None:
        resp = await client.post("/api/v1/experiments", json={})
        assert resp.status_code == 201

    async def test_response_has_pending_status(
        self, client: AsyncClient, experiment_id: UUID
    ) -> None:
        body = (await client.post("/api/v1/experiments", json={})).json()
        assert body["status"] == "pending"

    async def test_response_contains_experiment_id(
        self, client: AsyncClient, experiment_id: UUID
    ) -> None:
        body = (await client.post("/api/v1/experiments", json={})).json()
        assert UUID(body["experiment_id"]) == experiment_id

    async def test_calls_repo_create(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        await client.post(
            "/api/v1/experiments",
            json={"training_config": {"lr": 1e-3, "epochs": 10}},
        )
        mock_experiment_repo.create.assert_awaited_once()

    async def test_passes_training_config(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        await client.post(
            "/api/v1/experiments",
            json={"training_config": {"lr": 0.01}},
        )
        kwargs = mock_experiment_repo.create.call_args[1]
        assert kwargs["training_config"] == {"lr": 0.01}


class TestListExperiments:
    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/experiments")
        assert resp.status_code == 200

    async def test_returns_list(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        mock_experiment_repo.list_all.return_value = []
        body = (await client.get("/api/v1/experiments")).json()
        assert isinstance(body, list)

    async def test_default_limit_and_offset_forwarded(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        mock_experiment_repo.list_all.return_value = []
        await client.get("/api/v1/experiments")
        kwargs = mock_experiment_repo.list_all.call_args[1]
        assert kwargs["limit"] == 50
        assert kwargs["offset"] == 0

    async def test_custom_limit_and_offset_forwarded(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        mock_experiment_repo.list_all.return_value = []
        await client.get("/api/v1/experiments?limit=10&offset=10")
        kwargs = mock_experiment_repo.list_all.call_args[1]
        assert kwargs["limit"] == 10
        assert kwargs["offset"] == 10

    async def test_pagination_returns_correct_slice(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        ids = [uuid4() for _ in range(25)]
        page = [_experiment(ids[i]) for i in range(10, 20)]
        mock_experiment_repo.list_all.return_value = page
        body = (await client.get("/api/v1/experiments?limit=10&offset=10")).json()
        assert len(body) == 10
        assert UUID(body[0]["experiment_id"]) == ids[10]

    async def test_calls_list_all(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        mock_experiment_repo.list_all.return_value = []
        await client.get("/api/v1/experiments")
        mock_experiment_repo.list_all.assert_awaited_once()


class TestGetExperiment:
    async def test_returns_200_when_found(
        self, client: AsyncClient, experiment_id: UUID
    ) -> None:
        resp = await client.get(f"/api/v1/experiments/{experiment_id}")
        assert resp.status_code == 200

    async def test_response_contains_experiment_id(
        self, client: AsyncClient, experiment_id: UUID
    ) -> None:
        body = (await client.get(f"/api/v1/experiments/{experiment_id}")).json()
        assert UUID(body["experiment_id"]) == experiment_id

    async def test_returns_404_when_not_found(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        mock_experiment_repo.get_by_id.return_value = None
        resp = await client.get(f"/api/v1/experiments/{uuid4()}")
        assert resp.status_code == 404

    async def test_returns_422_on_non_uuid(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/experiments/not-a-uuid")
        assert resp.status_code == 422


class TestDeleteExperiment:
    @pytest.mark.parametrize("cancellable_status", ["pending", "queued", "running"])
    async def test_cancels_experiment_in_cancellable_state(
        self,
        client: AsyncClient,
        mock_experiment_repo: AsyncMock,
        experiment_id: UUID,
        cancellable_status: str,
    ) -> None:
        mock_experiment_repo.get_by_id.side_effect = [
            _experiment(experiment_id, status=cancellable_status),
            _experiment(experiment_id, status="cancelled"),
        ]
        resp = await client.delete(f"/api/v1/experiments/{experiment_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "cancelled"

    @pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
    async def test_returns_409_for_terminal_status(
        self,
        client: AsyncClient,
        mock_experiment_repo: AsyncMock,
        experiment_id: UUID,
        terminal_status: str,
    ) -> None:
        mock_experiment_repo.get_by_id.return_value = _experiment(
            experiment_id, status=terminal_status
        )
        resp = await client.delete(f"/api/v1/experiments/{experiment_id}")
        assert resp.status_code == 409

    async def test_returns_404_when_not_found(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        mock_experiment_repo.get_by_id.return_value = None
        resp = await client.delete(f"/api/v1/experiments/{uuid4()}")
        assert resp.status_code == 404

    async def test_calls_update_status_if_current_on_cancel(
        self,
        client: AsyncClient,
        mock_experiment_repo: AsyncMock,
        experiment_id: UUID,
    ) -> None:
        mock_experiment_repo.get_by_id.side_effect = [
            _experiment(experiment_id, status="pending"),
            _experiment(experiment_id, status="cancelled"),
        ]
        await client.delete(f"/api/v1/experiments/{experiment_id}")
        mock_experiment_repo.update_status_if_current.assert_awaited_once_with(
            experiment_id, "pending", "cancelled"
        )
