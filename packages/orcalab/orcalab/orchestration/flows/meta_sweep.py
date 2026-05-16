"""Prefect flow: OrcaMind-warm-started meta-informed hyperparameter sweep."""

from __future__ import annotations

from prefect import flow

from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.schemas.training import ExperimentResult
from orcalab.experiments.runner import ExperimentRunner
from orcalab.orchestration.flows.single_experiment import run_single_experiment
from orcalab.pruning.asha import ASHAPruner
from orcalab.pruning.base import Pruner
from orcalab.pruning.meta_pruner import MetaPruner
from orcalab.search.base import SearchStrategy
from orcalab.search.bayesian import BayesianSearch
from orcalab.search.meta_informed import MetaInformedSearch
from orcalab.search_spaces.space import SearchSpace


@flow(name="meta_informed_sweep")
async def meta_informed_sweep(
    task_id: str,
    n_trials: int = 50,
    use_orcamind: bool = True,
    *,
    search_space: SearchSpace | None = None,
    runner: ExperimentRunner,
    orcamind_client: OrcaMindClient | None = None,
) -> list[ExperimentResult]:
    if search_space is None:
        search_space = SearchSpace(name=task_id)

    strategy: SearchStrategy
    pruner: Pruner | None
    is_meta: bool

    if use_orcamind and orcamind_client is not None:
        strategy = MetaInformedSearch(orcamind_client=orcamind_client)
        await strategy.initialize_from_orcamind(task_id, search_space)
        pruner = MetaPruner(orcamind_client=orcamind_client, base_pruner=ASHAPruner())
        is_meta = True
    else:
        strategy = BayesianSearch()
        pruner = ASHAPruner()
        is_meta = False

    results: list[ExperimentResult] = []
    for _ in range(n_trials):
        params = strategy.suggest(search_space)
        result = await run_single_experiment(
            task_id,
            params,
            {},
            pruner=pruner,
            runner=runner,
            orcamind_client=orcamind_client if use_orcamind else None,
        )
        metric = result.metrics.get("accuracy", result.metrics.get("metric", 0.0)) if result.metrics else 0.0
        strategy.update(params, metric)
        results.append(result)

    if is_meta:
        await strategy.flush_results_to_orcamind(task_id)

    top_n = min(5, len(results))
    if top_n == 0:
        return []
    return sorted(
        results,
        key=lambda r: (r.metrics or {}).get("accuracy", (r.metrics or {}).get("metric", 0.0)),
        reverse=True,
    )[:top_n]
