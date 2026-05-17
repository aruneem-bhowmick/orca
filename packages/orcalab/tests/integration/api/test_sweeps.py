"""Integration tests for sweep management endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient


def _seed_sweep(sweeps: dict, n_results: int = 0) -> str:
    sweep_id = str(uuid4())
    results = [
        {"trial_id": str(uuid4()), "objective": float(i), "params": {"lr": 0.01 * i}}
        for i in range(n_results)
    ]
    sweeps[sweep_id] = {
        "sweep_id": sweep_id,
        "task_id": "task-abc",
        "n_trials_total": 10,
        "n_completed": n_results,
        "n_failed": 0,
        "flow_run_id": None,
        "results": results,
    }
    return sweep_id


class TestStartSweep:
    async def test_returns_202(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/sweeps", json={"task_id": "task-1", "n_trials": 5}
        )
        assert resp.status_code == 202

    async def test_response_contains_sweep_id(self, client: AsyncClient) -> None:
        body = (
            await client.post("/api/v1/sweeps", json={"task_id": "task-1"})
        ).json()
        assert "sweep_id" in body
        assert len(body["sweep_id"]) > 0

    async def test_stores_sweep_in_state(
        self, client: AsyncClient, sweeps_store: dict
    ) -> None:
        body = (
            await client.post(
                "/api/v1/sweeps", json={"task_id": "task-1", "n_trials": 20}
            )
        ).json()
        sweep_id = body["sweep_id"]
        assert sweep_id in sweeps_store
        assert sweeps_store[sweep_id]["n_trials_total"] == 20
        assert sweeps_store[sweep_id]["task_id"] == "task-1"

    async def test_no_prefect_call_when_url_not_set(
        self,
        client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("PREFECT_API_URL", raising=False)
        mock_post = AsyncMock()
        with patch("orcalab.api.routers.sweeps.httpx.AsyncClient") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.post = mock_post
            mock_client_cls.return_value = mock_client_instance
            resp = await client.post("/api/v1/sweeps", json={"task_id": "task-1"})
        assert resp.status_code == 202
        mock_post.assert_not_awaited()

    async def test_missing_task_id_returns_422(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/sweeps", json={})
        assert resp.status_code == 422

    async def test_zero_n_trials_returns_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/sweeps", json={"task_id": "task-1", "n_trials": 0}
        )
        assert resp.status_code == 422

    async def test_negative_n_trials_returns_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/sweeps", json={"task_id": "task-1", "n_trials": -5}
        )
        assert resp.status_code == 422

    async def test_search_space_stored_in_sweep_state(
        self, client: AsyncClient, sweeps_store: dict
    ) -> None:
        space = {"name": "lr_search", "parameters": [{"type": "float", "name": "lr"}]}
        body = (
            await client.post(
                "/api/v1/sweeps",
                json={"task_id": "task-1", "search_space": space},
            )
        ).json()
        sweep_id = body["sweep_id"]
        assert sweeps_store[sweep_id]["search_space"] == space


class TestGetSweepStatus:
    async def test_returns_200_for_known_sweep(
        self, client: AsyncClient, sweeps_store: dict
    ) -> None:
        sweep_id = _seed_sweep(sweeps_store)
        resp = await client.get(f"/api/v1/sweeps/{sweep_id}")
        assert resp.status_code == 200

    async def test_returns_correct_trial_counts(
        self, client: AsyncClient, sweeps_store: dict
    ) -> None:
        sweep_id = _seed_sweep(sweeps_store, n_results=3)
        body = (await client.get(f"/api/v1/sweeps/{sweep_id}")).json()
        assert body["n_trials_total"] == 10
        assert body["n_completed"] == 3
        assert body["n_failed"] == 0

    async def test_returns_best_result_when_results_exist(
        self, client: AsyncClient, sweeps_store: dict
    ) -> None:
        sweep_id = _seed_sweep(sweeps_store, n_results=3)
        body = (await client.get(f"/api/v1/sweeps/{sweep_id}")).json()
        assert body["best_result"] is not None
        assert body["best_result"]["objective"] == 2.0

    async def test_returns_404_for_unknown_sweep(self, client: AsyncClient) -> None:
        resp = await client.get(f"/api/v1/sweeps/{uuid4()}")
        assert resp.status_code == 404


class TestGetSweepResults:
    async def test_returns_200_for_known_sweep(
        self, client: AsyncClient, sweeps_store: dict
    ) -> None:
        sweep_id = _seed_sweep(sweeps_store, n_results=3)
        resp = await client.get(f"/api/v1/sweeps/{sweep_id}/results")
        assert resp.status_code == 200

    async def test_results_sorted_by_objective_descending(
        self, client: AsyncClient, sweeps_store: dict
    ) -> None:
        sweep_id = _seed_sweep(sweeps_store, n_results=3)
        body = (await client.get(f"/api/v1/sweeps/{sweep_id}/results")).json()
        objectives = [r["objective"] for r in body]
        assert objectives == sorted(objectives, reverse=True)

    async def test_returns_empty_list_when_no_results(
        self, client: AsyncClient, sweeps_store: dict
    ) -> None:
        sweep_id = _seed_sweep(sweeps_store, n_results=0)
        body = (await client.get(f"/api/v1/sweeps/{sweep_id}/results")).json()
        assert body == []

    async def test_returns_404_for_unknown_sweep(self, client: AsyncClient) -> None:
        resp = await client.get(f"/api/v1/sweeps/{uuid4()}/results")
        assert resp.status_code == 404
