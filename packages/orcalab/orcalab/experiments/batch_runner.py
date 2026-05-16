"""Concurrent batch runner that caps parallelism via asyncio.Semaphore."""

from __future__ import annotations

import asyncio

from orca_shared.schemas.training import ExperimentResult

from orcalab.experiments.experiment import Experiment
from orcalab.experiments.runner import ExperimentRunner
from orcalab.pruning.base import Pruner


class BatchExperimentRunner:
    """Runs a list of experiments concurrently, capping simultaneous trials.

    Results are returned in the same order as the input list.  Individual
    failures are captured as FAILED ExperimentResult entries without aborting
    the rest of the batch.
    """

    def __init__(self, runner: ExperimentRunner, max_parallel: int = 4) -> None:
        if max_parallel < 1:
            raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
        self._runner = runner
        self._max_parallel = max_parallel

    async def run_batch(
        self,
        experiments: list[Experiment],
        pruner: Pruner | None = None,
    ) -> list[ExperimentResult]:
        """Execute *experiments* concurrently, at most *max_parallel* at a time."""
        sem = asyncio.Semaphore(self._max_parallel)
        results: list[ExperimentResult | None] = [None] * len(experiments)

        async def _run_one(idx: int, exp: Experiment) -> None:
            async with sem:
                results[idx] = await self._runner.run(exp, pruner)

        await asyncio.gather(*(_run_one(i, e) for i, e in enumerate(experiments)))
        return results  # type: ignore[return-value]
