"""Prefect flow: continuous learning loop across multiple tasks."""

from __future__ import annotations

import asyncio

from prefect import flow

from orca_shared.clients.orcamind_client import OrcaMindClient
from orcalab.experiments.runner import ExperimentRunner
from orcalab.orchestration.flows.meta_sweep import meta_informed_sweep


@flow(name="continuous_learning")
async def continuous_learning_loop(
    task_ids: list[str],
    iterations: int = 10,
    trials_per_iteration: int = 20,
    iteration_sleep_seconds: float = 60.0,
    *,
    runner: ExperimentRunner,
    orcamind_client: OrcaMindClient | None = None,
) -> None:
    for iteration in range(iterations):
        for task_id in task_ids:
            await meta_informed_sweep(
                task_id,
                n_trials=trials_per_iteration,
                use_orcamind=orcamind_client is not None,
                runner=runner,
                orcamind_client=orcamind_client,
            )
        if iteration < iterations - 1:
            await asyncio.sleep(iteration_sleep_seconds)
