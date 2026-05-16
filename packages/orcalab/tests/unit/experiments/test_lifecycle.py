"""Unit tests for ExperimentLifecycle state machine."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from orcalab.experiments.experiment import Experiment
from orcalab.experiments.lifecycle import (
    ExperimentLifecycle,
    ExperimentStatus,
    InvalidTransitionError,
)


def _make_experiment(status: str = "pending") -> Experiment:
    return Experiment(
        experiment_id=uuid.uuid4(),
        status=status,
    )


def _make_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.update_status = AsyncMock(return_value=None)
    return repo


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------


class TestValidTransitions:
    @pytest.mark.parametrize(
        ("initial", "target"),
        [
            ("pending", ExperimentStatus.QUEUED),
            ("pending", ExperimentStatus.CANCELLED),
            ("queued", ExperimentStatus.RUNNING),
            ("running", ExperimentStatus.COMPLETED),
            ("running", ExperimentStatus.FAILED),
            ("running", ExperimentStatus.CANCELLED),
        ],
    )
    async def test_valid_transition_succeeds(
        self, initial: str, target: ExperimentStatus
    ) -> None:
        exp = _make_experiment(initial)
        lc = ExperimentLifecycle(exp, _make_repo())
        await lc.transition(target)
        assert exp.status == target.value

    async def test_pending_to_queued_updates_status(self) -> None:
        exp = _make_experiment("pending")
        repo = _make_repo()
        lc = ExperimentLifecycle(exp, repo)
        await lc.transition(ExperimentStatus.QUEUED)
        repo.update_status.assert_awaited_once_with(exp.experiment_id, "queued")

    async def test_running_to_completed_updates_status(self) -> None:
        exp = _make_experiment("running")
        repo = _make_repo()
        lc = ExperimentLifecycle(exp, repo)
        await lc.transition(ExperimentStatus.COMPLETED)
        repo.update_status.assert_awaited_once_with(exp.experiment_id, "completed")


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------


class TestInvalidTransitions:
    @pytest.mark.parametrize(
        ("initial", "target"),
        [
            ("running", ExperimentStatus.QUEUED),
            ("running", ExperimentStatus.PENDING),
            ("completed", ExperimentStatus.RUNNING),
            ("completed", ExperimentStatus.FAILED),
            ("failed", ExperimentStatus.RUNNING),
            ("cancelled", ExperimentStatus.RUNNING),
            ("queued", ExperimentStatus.COMPLETED),
            ("pending", ExperimentStatus.RUNNING),
            ("pending", ExperimentStatus.FAILED),
        ],
    )
    async def test_invalid_transition_raises(
        self, initial: str, target: ExperimentStatus
    ) -> None:
        exp = _make_experiment(initial)
        lc = ExperimentLifecycle(exp, _make_repo())
        with pytest.raises(InvalidTransitionError):
            await lc.transition(target)

    async def test_invalid_transition_does_not_update_status(self) -> None:
        exp = _make_experiment("running")
        repo = _make_repo()
        lc = ExperimentLifecycle(exp, repo)
        with pytest.raises(InvalidTransitionError):
            await lc.transition(ExperimentStatus.QUEUED)
        repo.update_status.assert_not_awaited()
        assert exp.status == "running"

    async def test_invalid_transition_error_message_contains_states(self) -> None:
        exp = _make_experiment("completed")
        lc = ExperimentLifecycle(exp, _make_repo())
        with pytest.raises(InvalidTransitionError, match="completed"):
            await lc.transition(ExperimentStatus.RUNNING)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    async def test_audit_log_empty_before_any_transition(self) -> None:
        lc = ExperimentLifecycle(_make_experiment(), _make_repo())
        assert lc.audit_log == []

    async def test_audit_log_records_single_transition(self) -> None:
        exp = _make_experiment("pending")
        lc = ExperimentLifecycle(exp, _make_repo())
        await lc.transition(ExperimentStatus.QUEUED, reason="submitted")
        log = lc.audit_log
        assert len(log) == 1
        entry = log[0]
        assert entry["from"] == "pending"
        assert entry["to"] == "queued"
        assert entry["reason"] == "submitted"
        assert "timestamp" in entry

    async def test_audit_log_records_multiple_transitions(self) -> None:
        exp = _make_experiment("pending")
        lc = ExperimentLifecycle(exp, _make_repo())
        await lc.transition(ExperimentStatus.QUEUED)
        await lc.transition(ExperimentStatus.RUNNING)
        await lc.transition(ExperimentStatus.COMPLETED)
        log = lc.audit_log
        assert len(log) == 3
        assert [e["to"] for e in log] == ["queued", "running", "completed"]

    async def test_audit_log_reason_defaults_to_empty_string(self) -> None:
        exp = _make_experiment("pending")
        lc = ExperimentLifecycle(exp, _make_repo())
        await lc.transition(ExperimentStatus.QUEUED)
        assert lc.audit_log[0]["reason"] == ""

    async def test_audit_log_returns_copy(self) -> None:
        exp = _make_experiment("pending")
        lc = ExperimentLifecycle(exp, _make_repo())
        await lc.transition(ExperimentStatus.QUEUED)
        log = lc.audit_log
        log.clear()
        assert len(lc.audit_log) == 1

    async def test_failed_transition_not_added_to_audit_log(self) -> None:
        exp = _make_experiment("completed")
        lc = ExperimentLifecycle(exp, _make_repo())
        with pytest.raises(InvalidTransitionError):
            await lc.transition(ExperimentStatus.RUNNING)
        assert lc.audit_log == []
