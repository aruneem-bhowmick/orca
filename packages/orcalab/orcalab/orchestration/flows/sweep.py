"""Prefect flow: hyperparameter sweep across n trials."""

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
from orcalab.pruning.asha import ASHAPruner
from orcalab.pruning.base import Pruner
from orcalab.pruning.median import MedianStoppingPruner
from orcalab.pruning.meta_pruner import MetaPruner
from orcalab.search.base import SearchStrategy
from orcalab.search.bayesian import BayesianSearch
from orcalab.search.evolutionary import EvolutionarySearch
from orcalab.search.grid_search import GridSearch
from orcalab.search.random_search import RandomSearch
from orcalab.search_spaces.space import SearchSpace


def _build_strategy(name: str) -> SearchStrategy:
    if name == "random":
        return RandomSearch()
    if name == "grid":
        return GridSearch()
    if name == "evolutionary":
        return EvolutionarySearch()
    return BayesianSearch()


def _build_pruner(name: str, orcamind_client: OrcaMindClient | None) -> Pruner | None:
    if name == "median":
        return MedianStoppingPruner()
    if name == "meta":
        if orcamind_client is not None:
            return MetaPruner(orcamind_client=orcamind_client, base_pruner=ASHAPruner())
        return ASHAPruner()
    if name == "asha":
        return ASHAPruner()
    return None


@flow(name="hyperparameter_sweep")
async def run_sweep(
    task_id: str,
    search_space: SearchSpace,
    n_trials: int = 50,
    strategy: str = "bayesian",
    pruner_name: str = "asha",
    *,
    storage: StorageBackend | None = None,
    runner: ExperimentRunner,
    orcamind_client: OrcaMindClient | None = None,
) -> list[ExperimentResult]:
    data = await prepare_data(task_id, storage) if storage is not None else pd.DataFrame()

    search = _build_strategy(strategy)
    pruner = _build_pruner(pruner_name, orcamind_client)

    task_uuid: UUID | None = None
    try:
        task_uuid = UUID(task_id)
    except (ValueError, AttributeError):
        pass

    results: list[ExperimentResult] = []
    for _ in range(n_trials):
        params = search.suggest(search_space)
        experiment = Experiment(
            experiment_id=uuid4(),
            status="pending",
            task_id=task_uuid,
            arch_config=params or None,
            training_config=TrainingConfig(),
        )
        result = await train_model(experiment, pruner=pruner, runner=runner)
        await evaluate(result)
        if orcamind_client is not None:
            await log_results(result, orcamind_client)
        metric = result.metrics.get("accuracy", result.metrics.get("metric", 0.0)) if result.metrics else 0.0
        search.update(params, metric)
        results.append(result)

    return results
