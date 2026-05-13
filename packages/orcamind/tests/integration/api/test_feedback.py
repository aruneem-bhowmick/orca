"""Integration tests for POST /api/v1/feedback."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient


class TestSubmitFeedback:
    def _body(self, experiment_id: UUID) -> dict:
        return {
            "experiment_id": str(experiment_id),
            "metric_name": "accuracy",
            "actual_metric": 0.91,
        }

    async def test_returns_200(
        self, client: AsyncClient, experiment_id: UUID
    ) -> None:
        resp = await client.post("/api/v1/feedback", json=self._body(experiment_id))
        assert resp.status_code == 200

    async def test_response_accepted_true(
        self, client: AsyncClient, experiment_id: UUID
    ) -> None:
        body = (
            await client.post("/api/v1/feedback", json=self._body(experiment_id))
        ).json()
        assert body == {"accepted": True}

    async def test_calls_log_metric_with_is_final(
        self,
        client: AsyncClient,
        experiment_id: UUID,
        mock_perf_repo: AsyncMock,
    ) -> None:
        await client.post("/api/v1/feedback", json=self._body(experiment_id))
        mock_perf_repo.log_metric.assert_awaited_once()
        kwargs = mock_perf_repo.log_metric.call_args[1]
        assert kwargs["is_final"] is True
        assert kwargs["name"] == "accuracy"
        assert kwargs["value"] == pytest.approx(0.91)

    async def test_returns_404_when_experiment_not_found(
        self, client: AsyncClient, mock_experiment_repo: AsyncMock
    ) -> None:
        mock_experiment_repo.get_by_id.return_value = None
        resp = await client.post(
            "/api/v1/feedback", json=self._body(uuid4())
        )
        assert resp.status_code == 404

    async def test_returns_422_on_missing_fields(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post("/api/v1/feedback", json={})
        assert resp.status_code == 422

    async def test_returns_422_without_metric_name(
        self, client: AsyncClient, experiment_id: UUID
    ) -> None:
        body = {
            "experiment_id": str(experiment_id),
            "actual_metric": 0.91,
        }
        resp = await client.post("/api/v1/feedback", json=body)
        assert resp.status_code == 422
