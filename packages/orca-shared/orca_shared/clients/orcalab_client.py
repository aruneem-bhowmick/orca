from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx

from orca_shared.schemas.training import ExperimentResult, TrainingConfig


class OrcaLabClient:
    """Async httpx client for the OrcaLab experiment orchestration service."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=20),
        )

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

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OrcaLabClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
