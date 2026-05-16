"""MedianStoppingPruner — prune trials below the median of peers."""

from __future__ import annotations

import logging
import statistics

from orcalab.pruning.base import Pruner

logger = logging.getLogger(__name__)


class MedianStoppingPruner(Pruner):
    """Prune a trial if its value at step *s* falls below the median best
    value of all other trials up to step *s*.

    During the first ``warmup_steps`` steps no pruning decision is made,
    giving every trial a fair chance to warm up before being compared.
    """

    def __init__(self, warmup_steps: int = 5) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        self._warmup_steps = warmup_steps

    @property
    def name(self) -> str:
        return "median_stopping"

    def should_prune(
        self,
        trial_id: str,
        step: int,
        current_value: float,
        all_trial_values: dict[str, list[float]],
    ) -> bool:
        if step < self._warmup_steps:
            return False

        peer_bests: list[float] = []
        for tid, values in all_trial_values.items():
            if tid == trial_id or not values:
                continue
            available = values[:step]
            if available:
                peer_bests.append(max(available))

        if not peer_bests:
            return False

        median = statistics.median(peer_bests)
        should = current_value < median
        if should:
            logger.debug(
                "Pruning trial %r at step %d: value %.4f < median %.4f",
                trial_id,
                step,
                current_value,
                median,
            )
        return should
