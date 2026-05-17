"""Integration tests for the WebSocket live metric stream endpoint."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from starlette.testclient import TestClient

from orca_shared.registry.repository import ExperimentRepository, SearchSpaceRepository
from orca_shared.schemas.training import ExperimentResult
from orcalab.api.deps import get_db, get_experiment_repo, get_search_space_repo, get_sweeps_store
from orcalab.api.main import create_app


def _make_experiment(experiment_id: UUID, status: str = "running") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        status=status,
        started_at=datetime.now(timezone.utc),
        metrics={"loss": 0.42, "accuracy": 0.88},
    )


def _build_sync_app(
    experiment_states: list[ExperimentResult | None],
) -> object:
    """Build a test app whose DB sessionmaker returns experiments in sequence."""
    app = create_app()

    call_count = {"n": 0}

    @asynccontextmanager
    async def _fake_sessionmaker():
        m = AsyncMock()
        m.__aenter__ = AsyncMock(return_value=m)
        m.__aexit__ = AsyncMock(return_value=False)
        repo_mock = AsyncMock(spec=ExperimentRepository)
        idx = call_count["n"]
        if idx < len(experiment_states):
            repo_mock.get_by_id.return_value = experiment_states[idx]
            call_count["n"] += 1
        else:
            repo_mock.get_by_id.return_value = experiment_states[-1]

        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository",
            return_value=repo_mock,
        ):
            yield m

    app.state.db_sessionmaker = _fake_sessionmaker
    app.state.sweeps = {}

    mock_experiment_repo = AsyncMock(spec=ExperimentRepository)
    mock_search_space_repo = AsyncMock(spec=SearchSpaceRepository)

    app.dependency_overrides[get_db] = lambda: AsyncMock()
    app.dependency_overrides[get_experiment_repo] = lambda: mock_experiment_repo
    app.dependency_overrides[get_search_space_repo] = lambda: mock_search_space_repo
    app.dependency_overrides[get_sweeps_store] = lambda: {}

    return app


class TestWebSocketLiveStream:
    def test_streams_metrics_and_closes_on_completed(self) -> None:
        exp_id = uuid4()
        states = [
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="completed"),
        ]
        app = _build_sync_app(states)

        with TestClient(app) as client:
            with client.websocket_connect(
                f"/api/v1/experiments/{exp_id}/live"
            ) as ws:
                first = ws.receive_json()
                assert first["status"] == "running"
                assert "metrics" in first
                assert first["metrics"]["accuracy"] == pytest.approx(0.88)

                second = ws.receive_json()
                assert second["status"] == "completed"

    def test_streams_metrics_and_closes_on_failed(self) -> None:
        exp_id = uuid4()
        states = [
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="failed"),
        ]
        app = _build_sync_app(states)

        with TestClient(app) as client:
            with client.websocket_connect(
                f"/api/v1/experiments/{exp_id}/live"
            ) as ws:
                ws.receive_json()
                terminal = ws.receive_json()
                assert terminal["status"] == "failed"

    def test_sends_error_and_closes_when_experiment_not_found(self) -> None:
        exp_id = uuid4()
        app = _build_sync_app([None])

        with TestClient(app) as client:
            with client.websocket_connect(
                f"/api/v1/experiments/{exp_id}/live"
            ) as ws:
                msg = ws.receive_json()
                assert "error" in msg

    def test_experiment_id_in_each_message(self) -> None:
        exp_id = uuid4()
        states = [
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="completed"),
        ]
        app = _build_sync_app(states)

        with TestClient(app) as client:
            with client.websocket_connect(
                f"/api/v1/experiments/{exp_id}/live"
            ) as ws:
                msg = ws.receive_json()
                assert UUID(msg["experiment_id"]) == exp_id
