"""Experiment state machine with transition validation and audit logging."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orca_shared.registry.repository import ExperimentRepository

    from orcalab.experiments.experiment import Experiment


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_VALID_TRANSITIONS: frozenset[tuple[ExperimentStatus, ExperimentStatus]] = frozenset(
    {
        (ExperimentStatus.PENDING, ExperimentStatus.QUEUED),
        (ExperimentStatus.PENDING, ExperimentStatus.CANCELLED),
        (ExperimentStatus.QUEUED, ExperimentStatus.RUNNING),
        (ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED),
        (ExperimentStatus.RUNNING, ExperimentStatus.FAILED),
        (ExperimentStatus.RUNNING, ExperimentStatus.CANCELLED),
    }
)


class InvalidTransitionError(Exception):
    """Raised when an experiment status transition is not permitted."""


class ExperimentLifecycle:
    """Manages valid state transitions for a single experiment.

    Validates each transition against the allowed state machine, appends an
    audit log entry, updates the experiment's in-memory status, and persists
    the new status to the database via the injected repository.
    """

    def __init__(self, experiment: Experiment, repository: ExperimentRepository) -> None:
        self._experiment = experiment
        self._repository = repository
        self._audit_log: list[dict] = []

    async def transition(self, new_status: ExperimentStatus, reason: str = "") -> None:
        """Transition to *new_status*, or raise InvalidTransitionError if not allowed."""
        current = ExperimentStatus(self._experiment.status)
        if (current, new_status) not in _VALID_TRANSITIONS:
            raise InvalidTransitionError(
                f"Cannot transition from {current.value!r} to {new_status.value!r}"
            )
        self._audit_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "from": current.value,
                "to": new_status.value,
                "reason": reason,
            }
        )
        self._experiment.status = new_status.value
        await self._repository.update_status(
            self._experiment.experiment_id, new_status.value
        )

    @property
    def audit_log(self) -> list[dict]:
        return list(self._audit_log)
