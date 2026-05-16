"""Prefect flow: run a single end-to-end experiment."""

from __future__ import annotations

from uuid import UUID, uuid4

import pandas as pd
from prefect import flow

from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.schemas.training import ExperimentResult, TrainingConfig
from orca_shared.storage.base import StorageBackend
from orcalab.experiments.experiment import Experiment
from orcalab.experiments.runner import ExperimentRunner
from orcalab.orchestration.tasks.evaluate import evaluate
from orcalab.orchestration.tasks.log_results import log_results
from orcalab.orchestration.tasks.prepare_data import prepare_data
from orcalab.orchestration.tasks.train_model import train_model
from orcalab.pruning.base import Pruner


@flow(name="single_experiment")
async def run_single_experiment(
    task_id: str,
    model_config: dict,
    training_config: dict,
    *,
    storage: StorageBackend | None = None,
    pruner: Pruner | None = None,
    runner: ExperimentRunner,
    orcamind_client: OrcaMindClient | None = None,
) -> ExperimentResult:
    data: pd.DataFrame
    if storage is not None:
        data = await prepare_data(task_id, storage)
    else:
        data = pd.DataFrame()

    task_uuid: UUID | None = None
    try:
        task_uuid = UUID(task_id)
    except (ValueError, AttributeError):
        pass

    experiment = Experiment(
        experiment_id=uuid4(),
        status="pending",
        task_id=task_uuid,
        arch_config=model_config or None,
        training_config=TrainingConfig(**training_config) if training_config else None,
    )

    result = await train_model(experiment, pruner=pruner, runner=runner)
    await evaluate(result)
    if orcamind_client is not None:
        await log_results(result, orcamind_client)
    return result
