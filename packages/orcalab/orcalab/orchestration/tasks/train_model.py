"""Prefect task: run a single training experiment."""

from __future__ import annotations

from prefect import task

from orca_shared.schemas.training import ExperimentResult
from orcalab.experiments.experiment import Experiment
from orcalab.experiments.runner import ExperimentRunner
from orcalab.pruning.base import Pruner


@task(name="train_model", timeout_seconds=3600)
async def train_model(
    experiment: Experiment,
    pruner: Pruner | None,
    runner: ExperimentRunner,
) -> ExperimentResult:
    return await runner.run(experiment, pruner=pruner)
