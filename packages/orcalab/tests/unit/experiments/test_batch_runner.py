"""Unit tests for BatchExperimentRunner: concurrency cap and ordering."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orca_shared.schemas.training import ExperimentResult, TrainingConfig
from orcalab.experiments.batch_runner import BatchExperimentRunner
from orcalab.experiments.experiment import Experiment
from orcalab.experiments.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment(idx: int = 0) -> Experiment:
    return Experiment(
        experiment_id=uuid.UUID(int=idx + 1),
        status="queued",
        training_config=TrainingConfig(epochs=1),
        tags={"idx": str(idx)},
    )


def _make_result(exp: Experiment, status: str = "completed") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=exp.experiment_id,
        status=status,
    )


def _make_batch_runner(
    run_side_effect: Any = None,
    max_parallel: int = 4,
) -> BatchExperimentRunner:
    runner = AsyncMock(spec=ExperimentRunner)
    if run_side_effect is not None:
        runner.run.side_effect = run_side_effect
    batch = BatchExperimentRunner(runner=runner, max_parallel=max_parallel)
    return batch


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


class TestConcurrencyCap:
    async def test_peak_concurrency_does_not_exceed_max_parallel(self) -> None:
        """8 experiments with max_parallel=4: peak active must be ≤ 4."""
        n = 8
        max_parallel = 4
        experiments = [_make_experiment(i) for i in range(n)]

        active = [0]
        peak = [0]

        async def _slow_run(exp: Experiment, pruner: Any = None) -> ExperimentResult:
            active[0] += 1
            if active[0] > peak[0]:
                peak[0] = active[0]
            await asyncio.sleep(0.05)
            active[0] -= 1
            return _make_result(exp)

        runner_mock = AsyncMock(spec=ExperimentRunner)
        runner_mock.run.side_effect = _slow_run
        batch = BatchExperimentRunner(runner=runner_mock, max_parallel=max_parallel)

        await batch.run_batch(experiments)
        assert peak[0] <= max_parallel

    async def test_max_parallel_1_forces_sequential_execution(self) -> None:
        """With max_parallel=1 the peak concurrency must stay at 1."""
        experiments = [_make_experiment(i) for i in range(5)]
        active = [0]
        peak = [0]

        async def _run(exp: Experiment, pruner: Any = None) -> ExperimentResult:
            active[0] += 1
            if active[0] > peak[0]:
                peak[0] = active[0]
            await asyncio.sleep(0.01)
            active[0] -= 1
            return _make_result(exp)

        runner_mock = AsyncMock(spec=ExperimentRunner)
        runner_mock.run.side_effect = _run
        batch = BatchExperimentRunner(runner=runner_mock, max_parallel=1)

        await batch.run_batch(experiments)
        assert peak[0] == 1

    async def test_max_parallel_matches_n_experiments_all_run_concurrently(self) -> None:
        """When max_parallel >= n all n start without waiting."""
        n = 4
        experiments = [_make_experiment(i) for i in range(n)]
        started: list[int] = []
        barrier = asyncio.Event()

        async def _run(exp: Experiment, pruner: Any = None) -> ExperimentResult:
            started.append(1)
            if len(started) == n:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), timeout=2.0)
            return _make_result(exp)

        runner_mock = AsyncMock(spec=ExperimentRunner)
        runner_mock.run.side_effect = _run
        batch = BatchExperimentRunner(runner=runner_mock, max_parallel=n)

        await batch.run_batch(experiments)
        assert len(started) == n


# ---------------------------------------------------------------------------
# Result ordering and correctness
# ---------------------------------------------------------------------------


class TestResultOrdering:
    async def test_results_length_matches_input(self) -> None:
        n = 8
        experiments = [_make_experiment(i) for i in range(n)]
        runner_mock = AsyncMock(spec=ExperimentRunner)
        runner_mock.run.side_effect = lambda exp, pruner=None: asyncio.coroutine(
            lambda: _make_result(exp)
        )()

        async def _run(exp: Experiment, pruner: Any = None) -> ExperimentResult:
            return _make_result(exp)

        runner_mock.run.side_effect = _run
        batch = BatchExperimentRunner(runner=runner_mock, max_parallel=4)
        results = await batch.run_batch(experiments)
        assert len(results) == n

    async def test_results_preserve_input_order(self) -> None:
        """Results must be in the same order as the input experiments, even when
        faster experiments finish first."""
        n = 6
        experiments = [_make_experiment(i) for i in range(n)]

        async def _run(exp: Experiment, pruner: Any = None) -> ExperimentResult:
            # Reverse delay: later experiments finish faster
            delay = (n - int(exp.tags["idx"])) * 0.01  # type: ignore[index]
            await asyncio.sleep(delay)
            return _make_result(exp)

        runner_mock = AsyncMock(spec=ExperimentRunner)
        runner_mock.run.side_effect = _run
        batch = BatchExperimentRunner(runner=runner_mock, max_parallel=n)

        results = await batch.run_batch(experiments)
        for i, (exp, result) in enumerate(zip(experiments, results)):
            assert result.experiment_id == exp.experiment_id, (
                f"Result at index {i} has wrong experiment_id"
            )

    async def test_all_experiments_are_run_exactly_once(self) -> None:
        n = 5
        experiments = [_make_experiment(i) for i in range(n)]
        called_ids: list[uuid.UUID] = []

        async def _run(exp: Experiment, pruner: Any = None) -> ExperimentResult:
            called_ids.append(exp.experiment_id)
            return _make_result(exp)

        runner_mock = AsyncMock(spec=ExperimentRunner)
        runner_mock.run.side_effect = _run
        batch = BatchExperimentRunner(runner=runner_mock, max_parallel=3)

        await batch.run_batch(experiments)
        assert sorted(called_ids) == sorted(e.experiment_id for e in experiments)


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    async def test_individual_failure_does_not_abort_batch(self) -> None:
        experiments = [_make_experiment(i) for i in range(4)]

        async def _run(exp: Experiment, pruner: Any = None) -> ExperimentResult:
            return _make_result(exp, status="failed")

        runner_mock = AsyncMock(spec=ExperimentRunner)
        runner_mock.run.side_effect = _run
        batch = BatchExperimentRunner(runner=runner_mock, max_parallel=4)

        results = await batch.run_batch(experiments)
        assert len(results) == 4
        assert all(r.status == "failed" for r in results)

    async def test_mixed_success_and_failure_results_returned(self) -> None:
        experiments = [_make_experiment(i) for i in range(4)]

        async def _run(exp: Experiment, pruner: Any = None) -> ExperimentResult:
            idx = int(exp.tags["idx"])  # type: ignore[index]
            status = "completed" if idx % 2 == 0 else "failed"
            return _make_result(exp, status=status)

        runner_mock = AsyncMock(spec=ExperimentRunner)
        runner_mock.run.side_effect = _run
        batch = BatchExperimentRunner(runner=runner_mock, max_parallel=4)

        results = await batch.run_batch(experiments)
        statuses = [r.status for r in results]
        assert statuses == ["completed", "failed", "completed", "failed"]

    async def test_empty_experiment_list_returns_empty_results(self) -> None:
        runner_mock = AsyncMock(spec=ExperimentRunner)
        batch = BatchExperimentRunner(runner=runner_mock, max_parallel=4)
        results = await batch.run_batch([])
        assert results == []
        runner_mock.run.assert_not_awaited()


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_max_parallel_zero_raises(self) -> None:
        runner_mock = AsyncMock(spec=ExperimentRunner)
        with pytest.raises(ValueError, match="max_parallel"):
            BatchExperimentRunner(runner=runner_mock, max_parallel=0)

    def test_invalid_max_parallel_negative_raises(self) -> None:
        runner_mock = AsyncMock(spec=ExperimentRunner)
        with pytest.raises(ValueError, match="max_parallel"):
            BatchExperimentRunner(runner=runner_mock, max_parallel=-1)

    def test_default_max_parallel_is_four(self) -> None:
        runner_mock = AsyncMock(spec=ExperimentRunner)
        batch = BatchExperimentRunner(runner=runner_mock)
        assert batch._max_parallel == 4
