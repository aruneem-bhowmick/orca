from __future__ import annotations

from typing import Any
from uuid import UUID

from orca_shared.clients._base import _BaseAsyncClient
from orca_shared.schemas.training import ExperimentResult, TrainingConfig


class OrcaLabClient(_BaseAsyncClient):
    """Async httpx client for the OrcaLab experiment orchestration service."""

    async def create_experiment(
        self, task_id: UUID, model_id: UUID, config: TrainingConfig
    ) -> ExperimentResult:
        raise NotImplementedError("OrcaLabClient.create_experiment is not yet implemented")

    async def start_sweep(
        self, experiment_id: UUID, search_space: dict[str, Any]
    ) -> str:
        raise NotImplementedError("OrcaLabClient.start_sweep is not yet implemented")

    async def get_sweep_status(self, sweep_id: str) -> str:
        raise NotImplementedError("OrcaLabClient.get_sweep_status is not yet implemented")

    async def wait_for_completion(
        self, sweep_id: str, poll_interval: float = 5.0
    ) -> ExperimentResult:
        raise NotImplementedError("OrcaLabClient.wait_for_completion is not yet implemented")
