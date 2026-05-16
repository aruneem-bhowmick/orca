"""Unit tests for Prefect orchestration flow functions."""

from __future__ import annotations

import io
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from orca_shared.schemas.training import ExperimentResult, TrainingConfig
from orcalab.experiments.experiment import Experiment
from orcalab.orchestration.flows.continuous_learning import continuous_learning_loop
from orcalab.orchestration.flows.meta_sweep import meta_informed_sweep
from orcalab.orchestration.flows.single_experiment import run_single_experiment
from orcalab.orchestration.flows.sweep import _build_pruner, _build_strategy, run_sweep
from orcalab.pruning.asha import ASHAPruner
from orcalab.pruning.median import MedianStoppingPruner
from orcalab.pruning.meta_pruner import MetaPruner
from orcalab.search.bayesian import BayesianSearch
from orcalab.search.evolutionary import EvolutionarySearch
from orcalab.search.grid_search import GridSearch
from orcalab.search.random_search import RandomSearch
from orcalab.search_spaces.parameters import FloatParameter
from orcalab.search_spaces.space import SearchSpace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(metrics: dict | None = None) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=uuid.uuid4(),
        status="completed",
        metrics=metrics or {"accuracy": 0.9, "loss": 0.1, "metric": 0.9},
    )


def _make_runner(result: ExperimentResult | None = None) -> AsyncMock:
    runner = AsyncMock()
    runner.run = AsyncMock(return_value=result or _make_result())
    return runner


def _make_search_space() -> SearchSpace:
    space = SearchSpace(name="test")
    space.add(FloatParameter(name="lr", low=1e-4, high=1e-1))
    return space


def _make_parquet_bytes() -> bytes:
    buf = io.BytesIO()
    pd.DataFrame({"x": [1, 2]}).to_parquet(buf, index=False)
    return buf.getvalue()


def _make_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.download = AsyncMock(return_value=_make_parquet_bytes())
    return storage


# ---------------------------------------------------------------------------
# sweep flow helpers (pure unit tests, no Prefect overhead)
# ---------------------------------------------------------------------------


class TestBuildStrategy:
    def test_bayesian_is_default(self) -> None:
        assert isinstance(_build_strategy("bayesian"), BayesianSearch)

    def test_random_strategy(self) -> None:
        assert isinstance(_build_strategy("random"), RandomSearch)

    def test_grid_strategy(self) -> None:
        assert isinstance(_build_strategy("grid"), GridSearch)

    def test_evolutionary_strategy(self) -> None:
        assert isinstance(_build_strategy("evolutionary"), EvolutionarySearch)

    def test_unknown_name_falls_back_to_bayesian(self) -> None:
        assert isinstance(_build_strategy("unknown"), BayesianSearch)


class TestBuildPruner:
    def test_asha_pruner(self) -> None:
        assert isinstance(_build_pruner("asha", None), ASHAPruner)

    def test_median_pruner(self) -> None:
        assert isinstance(_build_pruner("median", None), MedianStoppingPruner)

    def test_meta_pruner_with_client(self) -> None:
        client = MagicMock()
        pruner = _build_pruner("meta", client)
        assert isinstance(pruner, MetaPruner)

    def test_meta_pruner_without_client_falls_back_to_asha(self) -> None:
        pruner = _build_pruner("meta", None)
        assert isinstance(pruner, ASHAPruner)

    def test_unknown_name_returns_none(self) -> None:
        assert _build_pruner("none", None) is None


# ---------------------------------------------------------------------------
# single_experiment flow
# ---------------------------------------------------------------------------


class TestSingleExperimentFlow:
    async def test_returns_experiment_result(self) -> None:
        runner = _make_runner()
        result = await run_single_experiment.fn("task-abc", {}, {}, runner=runner)
        assert isinstance(result, ExperimentResult)

    async def test_completes_without_storage(self) -> None:
        runner = _make_runner()
        result = await run_single_experiment.fn(
            "task-abc", {}, {}, storage=None, runner=runner
        )
        assert result.status == "completed"

    async def test_calls_runner_run(self) -> None:
        runner = _make_runner()
        await run_single_experiment.fn("task-abc", {}, {}, runner=runner)
        runner.run.assert_awaited_once()

    async def test_no_log_when_orcamind_client_is_none(self) -> None:
        runner = _make_runner()
        # Should complete without any OrcaMind client calls
        result = await run_single_experiment.fn(
            "task-abc", {}, {}, runner=runner, orcamind_client=None
        )
        assert result is not None

    async def test_logs_when_orcamind_provided(self) -> None:
        runner = _make_runner()
        client = AsyncMock()
        client.submit_feedback = AsyncMock(side_effect=NotImplementedError)
        await run_single_experiment.fn(
            "task-abc", {}, {}, runner=runner, orcamind_client=client
        )
        client.submit_feedback.assert_awaited_once()

    async def test_with_storage_downloads_correct_key(self) -> None:
        runner = _make_runner()
        storage = _make_storage()
        await run_single_experiment.fn(
            "task-abc", {}, {}, storage=storage, runner=runner
        )
        storage.download.assert_awaited_once_with("datasets/task-abc/data.parquet")

    async def test_invalid_task_id_does_not_raise(self) -> None:
        runner = _make_runner()
        result = await run_single_experiment.fn(
            "not-a-valid-uuid", {}, {}, runner=runner
        )
        assert result is not None

    async def test_training_config_passed_to_experiment(self) -> None:
        runner = _make_runner()
        await run_single_experiment.fn(
            "task-abc",
            {},
            {"epochs": 7, "lr": 0.01, "batch_size": 16},
            runner=runner,
        )
        experiment_passed = runner.run.call_args[0][0]
        assert isinstance(experiment_passed, Experiment)
        assert experiment_passed.training_config is not None
        assert experiment_passed.training_config.epochs == 7


# ---------------------------------------------------------------------------
# hyperparameter sweep flow
# ---------------------------------------------------------------------------


class TestSweepFlow:
    async def test_returns_n_results(self) -> None:
        runner = _make_runner()
        space = _make_search_space()
        results = await run_sweep.fn(
            "task-abc", space, n_trials=3, strategy="random", runner=runner
        )
        assert len(results) == 3

    async def test_all_results_are_experiment_results(self) -> None:
        runner = _make_runner()
        space = _make_search_space()
        results = await run_sweep.fn(
            "task-abc", space, n_trials=2, strategy="bayesian", runner=runner
        )
        assert all(isinstance(r, ExperimentResult) for r in results)

    async def test_runner_called_once_per_trial(self) -> None:
        runner = _make_runner()
        space = _make_search_space()
        await run_sweep.fn("task-abc", space, n_trials=4, runner=runner)
        assert runner.run.await_count == 4

    async def test_strategy_name_dispatches_correctly(self) -> None:
        runner = _make_runner()
        space = _make_search_space()
        captured: list[object] = []
        real_build = _build_strategy

        def recording_build(name: str):
            strat = real_build(name)
            captured.append(strat)
            return strat

        with patch("orcalab.orchestration.flows.sweep._build_strategy", side_effect=recording_build):
            await run_sweep.fn("task-abc", space, n_trials=2, strategy="random", runner=runner)
        assert isinstance(captured[0], RandomSearch)

    async def test_pruner_name_dispatches_correctly(self) -> None:
        runner = _make_runner()
        space = _make_search_space()
        captured: list[object] = []
        real_build = _build_pruner

        def recording_build(name: str, client):
            pruner = real_build(name, client)
            captured.append(pruner)
            return pruner

        with patch("orcalab.orchestration.flows.sweep._build_pruner", side_effect=recording_build):
            await run_sweep.fn(
                "task-abc", space, n_trials=2, pruner_name="median", runner=runner
            )
        assert isinstance(captured[0], MedianStoppingPruner)


# ---------------------------------------------------------------------------
# meta_informed_sweep flow
# ---------------------------------------------------------------------------


class TestMetaInformedSweepFlow:
    async def test_completes_5_trials_without_orcamind(self) -> None:
        runner = _make_runner()
        space = _make_search_space()
        with patch(
            "orcalab.orchestration.flows.meta_sweep.run_single_experiment",
            new_callable=AsyncMock,
            return_value=_make_result(),
        ) as mock_exp:
            results = await meta_informed_sweep.fn(
                "task-abc",
                n_trials=5,
                use_orcamind=False,
                search_space=space,
                runner=runner,
            )
        assert mock_exp.call_count == 5
        assert len(results) <= 5

    async def test_no_orcamind_calls_when_disabled(self) -> None:
        runner = _make_runner()
        space = _make_search_space()
        client = AsyncMock()
        with patch(
            "orcalab.orchestration.flows.meta_sweep.run_single_experiment",
            new_callable=AsyncMock,
            return_value=_make_result(),
        ):
            await meta_informed_sweep.fn(
                "task-abc",
                n_trials=3,
                use_orcamind=False,
                search_space=space,
                runner=runner,
                orcamind_client=client,
            )
        client.submit_feedback.assert_not_awaited()
        client.embed_task.assert_not_awaited()

    async def test_returns_at_most_5_results(self) -> None:
        runner = _make_runner()
        space = _make_search_space()
        with patch(
            "orcalab.orchestration.flows.meta_sweep.run_single_experiment",
            new_callable=AsyncMock,
            return_value=_make_result(),
        ):
            results = await meta_informed_sweep.fn(
                "task-abc",
                n_trials=10,
                use_orcamind=False,
                search_space=space,
                runner=runner,
            )
        assert len(results) <= 5

    async def test_creates_default_search_space_when_none(self) -> None:
        runner = _make_runner()
        with patch(
            "orcalab.orchestration.flows.meta_sweep.run_single_experiment",
            new_callable=AsyncMock,
            return_value=_make_result(),
        ):
            results = await meta_informed_sweep.fn(
                "task-xyz",
                n_trials=2,
                use_orcamind=False,
                runner=runner,
            )
        assert isinstance(results, list)

    async def test_orcamind_strategy_used_when_enabled(self) -> None:
        runner = _make_runner()
        space = _make_search_space()
        client = AsyncMock()
        task_id = str(uuid.uuid4())
        with patch(
            "orcalab.orchestration.flows.meta_sweep.run_single_experiment",
            new_callable=AsyncMock,
            return_value=_make_result(),
        ) as mock_exp:
            results = await meta_informed_sweep.fn(
                task_id,
                n_trials=3,
                use_orcamind=True,
                search_space=space,
                runner=runner,
                orcamind_client=client,
            )
        # Verify OrcaMind initialization path was taken
        client.embed_task.assert_awaited_once()
        client.recommend_model.assert_awaited_once()
        client.find_similar_tasks.assert_awaited_once()
        # All trials ran and results flushed back to OrcaMind
        assert mock_exp.call_count == 3
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# continuous_learning_loop flow
# ---------------------------------------------------------------------------


class TestContinuousLearningFlow:
    async def test_iterates_correct_number_of_times(self) -> None:
        runner = _make_runner()
        task_ids = ["task-a", "task-b"]
        iterations = 3
        with patch(
            "orcalab.orchestration.flows.continuous_learning.meta_informed_sweep",
            new_callable=AsyncMock,
        ) as mock_sweep, patch(
            "orcalab.orchestration.flows.continuous_learning.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            mock_sweep.return_value = []
            await continuous_learning_loop.fn(
                task_ids,
                iterations=iterations,
                trials_per_iteration=2,
                iteration_sleep_seconds=0.0,
                runner=runner,
            )
        assert mock_sweep.call_count == iterations * len(task_ids)

    async def test_sleeps_between_iterations_not_after_last(self) -> None:
        runner = _make_runner()
        iterations = 4
        with patch(
            "orcalab.orchestration.flows.continuous_learning.meta_informed_sweep",
            new_callable=AsyncMock,
        ) as mock_sweep, patch(
            "orcalab.orchestration.flows.continuous_learning.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep:
            mock_sweep.return_value = []
            await continuous_learning_loop.fn(
                ["task-a"],
                iterations=iterations,
                trials_per_iteration=2,
                iteration_sleep_seconds=1.0,
                runner=runner,
            )
        assert mock_sleep.call_count == iterations - 1

    async def test_single_iteration_does_not_sleep(self) -> None:
        runner = _make_runner()
        with patch(
            "orcalab.orchestration.flows.continuous_learning.meta_informed_sweep",
            new_callable=AsyncMock,
        ) as mock_sweep, patch(
            "orcalab.orchestration.flows.continuous_learning.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep:
            mock_sweep.return_value = []
            await continuous_learning_loop.fn(
                ["task-a"],
                iterations=1,
                trials_per_iteration=2,
                iteration_sleep_seconds=60.0,
                runner=runner,
            )
        mock_sleep.assert_not_called()

    async def test_calls_each_task_id_in_every_iteration(self) -> None:
        runner = _make_runner()
        task_ids = ["t1", "t2", "t3"]
        iterations = 2
        seen_calls: list[str] = []

        async def record_call(tid, *, runner, **kwargs):
            seen_calls.append(tid)
            return []

        with patch(
            "orcalab.orchestration.flows.continuous_learning.meta_informed_sweep",
            side_effect=record_call,
        ), patch(
            "orcalab.orchestration.flows.continuous_learning.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await continuous_learning_loop.fn(
                task_ids,
                iterations=iterations,
                trials_per_iteration=2,
                iteration_sleep_seconds=0.0,
                runner=runner,
            )

        assert seen_calls.count("t1") == iterations
        assert seen_calls.count("t2") == iterations
        assert seen_calls.count("t3") == iterations
