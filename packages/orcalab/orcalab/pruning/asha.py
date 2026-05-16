"""ASHAPruner — Asynchronous Successive Halving Algorithm (Li et al. 2018)."""

from __future__ import annotations

import logging

from orcalab.pruning.base import Pruner

logger = logging.getLogger(__name__)


class ASHAPruner(Pruner):
    """ASHA early-stopping pruner.

    Rungs are located at ``min_resource * reduction_factor^k`` for k = 0, 1, 2, ...
    up to ``max_resource``.  At each rung the bottom
    ``(1 - 1/reduction_factor)`` fraction of trials is pruned; only the top
    ``1/reduction_factor`` fraction is promoted to the next rung.
    """

    def __init__(
        self,
        min_resource: int = 1,
        max_resource: int = 81,
        reduction_factor: int = 3,
    ) -> None:
        if min_resource < 1:
            raise ValueError(f"min_resource must be >= 1, got {min_resource}")
        if max_resource < min_resource:
            raise ValueError(
                f"max_resource ({max_resource}) must be >= min_resource ({min_resource})"
            )
        if reduction_factor < 2:
            raise ValueError(f"reduction_factor must be >= 2, got {reduction_factor}")
        self._min_resource = min_resource
        self._max_resource = max_resource
        self._reduction_factor = reduction_factor
        self._promoted: dict[int, list[str]] = {}

    @property
    def name(self) -> str:
        return "asha"

    def _rungs(self) -> list[tuple[int, int]]:
        """Return (rung_index, step) pairs for every rung up to max_resource."""
        rungs: list[tuple[int, int]] = []
        resource = self._min_resource
        rung_idx = 0
        while resource <= self._max_resource:
            rungs.append((rung_idx, resource))
            resource *= self._reduction_factor
            rung_idx += 1
        return rungs

    def _rung_for_step(self, step: int) -> int | None:
        """Return the rung index if *step* is a rung level, else None."""
        for rung_idx, rung_step in self._rungs():
            if rung_step == step:
                return rung_idx
        return None

    def should_prune(
        self,
        trial_id: str,
        step: int,
        current_value: float,
        all_trial_values: dict[str, list[float]],
    ) -> bool:
        rung_idx = self._rung_for_step(step)
        if rung_idx is None:
            return False
        if trial_id in self._promoted.get(rung_idx, []):
            return False

        trials_at_rung: dict[str, float] = {trial_id: current_value}
        for tid, values in all_trial_values.items():
            if tid == trial_id:
                continue
            if len(values) >= step:
                trials_at_rung[tid] = values[step - 1]

        n = len(trials_at_rung)
        keep = max(1, n // self._reduction_factor)

        sorted_trials = sorted(trials_at_rung.items(), key=lambda x: x[1], reverse=True)
        top_ids = {tid for tid, _ in sorted_trials[:keep]}

        if trial_id in top_ids:
            if rung_idx not in self._promoted:
                self._promoted[rung_idx] = []
            if trial_id not in self._promoted[rung_idx]:
                self._promoted[rung_idx].append(trial_id)
            return False

        logger.debug(
            "Pruning trial %r at rung %d (step %d): not in top-%d of %d trials",
            trial_id,
            rung_idx,
            step,
            keep,
            n,
        )
        return True
