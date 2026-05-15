"""RandomSearch strategy backed by Optuna's RandomSampler."""

from __future__ import annotations

from collections import deque
from typing import Any

import optuna

from orcalab.search.base import SearchStrategy
from orcalab.search_spaces.space import SearchSpace

class RandomSearch(SearchStrategy):
    """Search strategy that samples uniformly at random using Optuna's RandomSampler.

    Note: constructing this class sets Optuna's global logging level to WARNING,
    suppressing INFO-level trial output for all Optuna code in the process.
    """

    def __init__(self, random_state: int = 42) -> None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.RandomSampler(seed=random_state)
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._pending: deque[tuple[dict[str, Any], optuna.Trial]] = deque()

    def suggest(self, search_space: SearchSpace) -> dict[str, Any]:
        trial = self._study.ask()
        params = search_space.sample(trial)
        self._pending.append((params, trial))
        return params

    def update(self, params: dict[str, Any], result: float) -> None:
        _, trial = self._pending.popleft()
        self._study.tell(trial, result)

    def get_best(self, n: int = 1) -> list[tuple[dict, float]]:
        completed = [
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)  # type: ignore[arg-type]
        return [(t.params, t.value) for t in sorted_trials[:n]]

    @property
    def n_trials(self) -> int:
        return sum(
            1 for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
