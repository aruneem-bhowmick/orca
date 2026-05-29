from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from orca_shared.clients._base import _BaseAsyncClient
from orca_shared.schemas.training import ExperimentResult, TrainingConfig

logger = logging.getLogger("orca_shared.clients.orcalab")

_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "completed", "failed", "cancelled", "CANCELLED"})


class OrcaLabClient(_BaseAsyncClient):
    """Async httpx client for the OrcaLab experiment orchestration service."""

    async def create_experiment(
        self,
        task_id: str,
        model_config: Any,
        tags: list[str] | None = None,
    ) -> str:
        """Submit a new experiment to OrcaLab and return the experiment ID.

        Args:
            task_id: UUID string of the target task for this experiment.
            model_config: Model configuration — accepts a Pydantic model with
                ``model_dump()`` or any object serialisable as a dict.  The
                ``model_id`` field is extracted and forwarded if present.
            tags: Optional list of string tags stored as experiment metadata.

        Returns:
            The ``experiment_id`` of the newly created experiment as a string.
        """
        if hasattr(model_config, "model_dump"):
            config_dict: dict[str, Any] = model_config.model_dump(mode="json")
        elif hasattr(model_config, "__dict__"):
            config_dict = dict(vars(model_config))
        else:
            config_dict = dict(model_config) if model_config is not None else {}

        model_id = config_dict.get("model_id")

        payload: dict[str, Any] = {
            "task_id": str(task_id),
            "training_config": {
                "extra": {
                    "model_config": config_dict,
                    "tags": tags or [],
                }
            },
        }
        if model_id is not None:
            payload["model_id"] = str(model_id)

        response = await self._client.post("/api/v1/experiments", json=payload)
        response.raise_for_status()
        return str(response.json()["experiment_id"])

    async def wait_for_completion(
        self,
        experiment_id: str,
        timeout: int = 3600,
        poll_interval: int = 30,
    ) -> ExperimentResult:
        """Poll OrcaLab until the experiment reaches a terminal status.

        Args:
            experiment_id: ID of the experiment to watch.
            timeout: Maximum seconds to wait before raising ``TimeoutError``.
            poll_interval: Seconds between status polls.

        Returns:
            The final :class:`~orca_shared.schemas.training.ExperimentResult`.

        Raises:
            TimeoutError: When the experiment does not complete within *timeout*.
            httpx.HTTPStatusError: On non-2xx responses from OrcaLab.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            response = await self._client.get(f"/api/v1/experiments/{experiment_id}")
            response.raise_for_status()
            result = ExperimentResult.model_validate(response.json())

            if result.status in _TERMINAL_STATUSES:
                return result

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Experiment {experiment_id!r} did not reach a terminal status "
                    f"within {timeout}s (last status: {result.status!r})"
                )

            await asyncio.sleep(min(poll_interval, remaining))

    async def start_sweep(
        self, experiment_id: UUID, search_space: dict[str, Any]
    ) -> str:
        """Launch a hyperparameter sweep for an existing experiment.

        Not yet implemented — raises :exc:`NotImplementedError`.
        """
        raise NotImplementedError("OrcaLabClient.start_sweep is not yet implemented")

    async def get_sweep_status(self, sweep_id: str) -> str:
        """Return the current status string for a running sweep.

        Not yet implemented — raises :exc:`NotImplementedError`.
        """
        raise NotImplementedError("OrcaLabClient.get_sweep_status is not yet implemented")
