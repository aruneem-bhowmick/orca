"""Integration tests for the WebSocket live metric stream endpoint.

Calls the handler directly with a mocked WebSocket to avoid the starlette
TestClient / httpx 0.28 incompatibility and to eliminate real asyncio.sleep
delays.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import WebSocket

from orca_shared.registry.repository import ExperimentRepository
from orca_shared.schemas.training import ExperimentResult
from orcalab.api.routers.experiments import experiment_live


def _make_experiment(
    experiment_id: UUID,
    status: str = "running",
    metrics: dict | None = None,
) -> ExperimentResult:
    if metrics is None:
        metrics = {"loss": 0.42, "epoch": 3, "accuracy": 0.88}
    return ExperimentResult(
        experiment_id=experiment_id,
        status=status,
        started_at=datetime.now(timezone.utc),
        metrics=metrics,
    )


def _build_websocket(
    experiment_states: list[ExperimentResult | None],
) -> tuple[AsyncMock, AsyncMock]:
    """Build a mock WebSocket whose sessionmaker returns experiments in sequence.

    Returns (websocket_mock, repo_mock).  Callers should patch
    ``orcalab.api.routers.experiments.ExperimentRepository`` to return
    repo_mock so that the handler creates the right mock inside the session.
    """
    call_count = {"n": 0}
    repo_mock = AsyncMock(spec=ExperimentRepository)

    async def _get_by_id(eid: UUID) -> ExperimentResult | None:
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(experiment_states):
            return experiment_states[idx]
        return experiment_states[-1]

    repo_mock.get_by_id.side_effect = _get_by_id

    @asynccontextmanager
    async def _sessionmaker():
        session = AsyncMock()
        yield session

    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.app = MagicMock()
    ws.app.state.db_sessionmaker = _sessionmaker

    return ws, repo_mock


class TestWebSocketLiveStream:
    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_accepts_connection(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        ws.accept.assert_awaited_once()

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_sends_metrics_while_running(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        assert ws.send_json.call_count == 2
        first_arg = ws.send_json.call_args_list[0][0][0]
        assert first_arg["status"] == "running"
        assert first_arg["metrics"]["accuracy"] == pytest.approx(0.88)

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_closes_after_completed_status(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        last_arg = ws.send_json.call_args_list[-1][0][0]
        assert last_arg["status"] == "completed"
        assert ws.send_json.call_count == 2

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_closes_after_failed_status(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="failed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        last_arg = ws.send_json.call_args_list[-1][0][0]
        assert last_arg["status"] == "failed"

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_sends_error_when_experiment_not_found(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([None])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        msg = ws.send_json.call_args_list[0][0][0]
        assert "error" in msg

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_includes_experiment_id_in_each_message(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        for call in ws.send_json.call_args_list:
            msg = call[0][0]
            if "experiment_id" in msg:
                assert UUID(msg["experiment_id"]) == exp_id

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_handles_websocket_disconnect_gracefully(self, mock_sleep) -> None:
        from fastapi import WebSocketDisconnect

        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running"),
        ])
        ws.send_json.side_effect = WebSocketDisconnect()
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)


class TestWebSocketSpecAssertions:
    """Verify the exact assertions required by ORCA-LAB-INSIGHTS spec (Prompt 11/13).

    The spec integration test requires:
        data = ws.receive_json()
        assert "epoch" in data
        assert "loss" in data
    """

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_top_level_epoch_field_present(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        first_msg = ws.send_json.call_args_list[0][0][0]
        assert "epoch" in first_msg

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_top_level_loss_field_present(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        first_msg = ws.send_json.call_args_list[0][0][0]
        assert "loss" in first_msg

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_epoch_reflects_stored_value(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running", metrics={"loss": 0.25, "epoch": 7}),
            _make_experiment(exp_id, status="completed", metrics={"loss": 0.1, "epoch": 10}),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        first_msg = ws.send_json.call_args_list[0][0][0]
        assert first_msg["epoch"] == 7

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_loss_reflects_stored_value(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running", metrics={"loss": 0.25, "epoch": 7}),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        first_msg = ws.send_json.call_args_list[0][0][0]
        assert first_msg["loss"] == pytest.approx(0.25)

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_metrics_dict_still_present_for_backward_compat(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running"),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        first_msg = ws.send_json.call_args_list[0][0][0]
        assert "metrics" in first_msg
        assert isinstance(first_msg["metrics"], dict)

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_epoch_is_none_when_no_metrics_written_yet(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running", metrics={}),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        first_msg = ws.send_json.call_args_list[0][0][0]
        assert "epoch" in first_msg
        assert first_msg["epoch"] is None

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_loss_is_none_when_no_metrics_written_yet(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running", metrics={}),
            _make_experiment(exp_id, status="completed"),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        first_msg = ws.send_json.call_args_list[0][0][0]
        assert "loss" in first_msg
        assert first_msg["loss"] is None

    @patch("orcalab.api.routers.experiments.asyncio.sleep", new_callable=AsyncMock)
    async def test_epoch_advances_across_successive_messages(self, mock_sleep) -> None:
        exp_id = uuid4()
        ws, repo = _build_websocket([
            _make_experiment(exp_id, status="running", metrics={"loss": 0.8, "epoch": 1}),
            _make_experiment(exp_id, status="running", metrics={"loss": 0.6, "epoch": 2}),
            _make_experiment(exp_id, status="completed", metrics={"loss": 0.4, "epoch": 3}),
        ])
        with patch(
            "orcalab.api.routers.experiments.ExperimentRepository", return_value=repo
        ):
            await experiment_live(ws, exp_id)
        epochs = [c[0][0]["epoch"] for c in ws.send_json.call_args_list]
        assert epochs == [1, 2, 3]
