"""BayesianSearch strategy backed by Optuna's TPE sampler."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import optuna
import optuna.distributions
import optuna.study
import optuna.trial

from orcalab.search.base import SearchStrategy
from orcalab.search_spaces.parameters import (
    CategoricalParameter,
    DiscreteUniformParameter,
    FloatParameter,
    IntParameter,
)
from orcalab.search_spaces.space import SearchSpace


def _build_distributions(
    space: SearchSpace,
) -> dict[str, optuna.distributions.BaseDistribution]:
    distributions: dict[str, optuna.distributions.BaseDistribution] = {}
    for param in space._params.values():
        if isinstance(param, CategoricalParameter):
            distributions[param.name] = optuna.distributions.CategoricalDistribution(
                param.choices
            )
        elif isinstance(param, DiscreteUniformParameter):
            distributions[param.name] = optuna.distributions.FloatDistribution(
                param.low, param.high, step=param.q
            )
        elif isinstance(param, FloatParameter):
            distributions[param.name] = optuna.distributions.FloatDistribution(
                param.low, param.high, log=param.log
            )
        elif isinstance(param, IntParameter):
            distributions[param.name] = optuna.distributions.IntDistribution(
                param.low, param.high, step=param.step, log=param.log
            )
    return distributions


class BayesianSearch(SearchStrategy):
    """Search strategy using Optuna's Tree-structured Parzen Estimator (TPE) sampler.

    Supports warm-starting via inject_priors(), direction-aware get_best(), and optional
    persistence to any Optuna storage backend (e.g. SQLite, PostgreSQL).
    """

    def __init__(
        self,
        study_name: str = "orcalab_bayesian",
        direction: str = "maximize",
        sampler: optuna.samplers.BaseSampler | None = None,
        storage: str | None = None,
        warm_start_trials: list[tuple[dict, float]] | None = None,
    ) -> None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler or optuna.samplers.TPESampler(),
            storage=storage,
            load_if_exists=True,
        )
        self._pending: deque[tuple[dict[str, Any], optuna.Trial]] = deque()
        self._search_space: SearchSpace | None = None
        self._deferred_priors: list[tuple[dict, float]] = list(warm_start_trials or [])

    def suggest(self, search_space: SearchSpace) -> dict[str, Any]:
        if self._search_space is None:
            self._search_space = search_space
            if self._deferred_priors:
                self.inject_priors(self._deferred_priors, search_space)
                self._deferred_priors = []
        elif set(_build_distributions(search_space)) != set(_build_distributions(self._search_space)):
            raise ValueError(
                "SearchSpace schema changed across suggest() calls for the same "
                "BayesianSearch instance."
            )
        trial = self._study.ask()
        params = search_space.sample(trial)
        self._pending.append((params, trial))
        return params

    def update(self, params: dict[str, Any], result: float) -> None:
        """Record the result for the next pending trial.

        Must be called in the same FIFO order as suggest(). Raises ValueError if params
        do not match the oldest pending trial. NaN or Inf results are recorded as FAIL.
        """
        if not self._pending:
            raise ValueError("No pending trials; call suggest() before update().")
        pending_params, trial = self._pending[0]
        if params != pending_params:
            raise ValueError(
                f"params do not match the next pending trial. "
                f"Expected {pending_params!r}, got {params!r}."
            )
        self._pending.popleft()
        if math.isnan(result) or math.isinf(result):
            self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            self._study.tell(trial, result)

    def get_best(self, n: int = 1) -> list[tuple[dict, float]]:
        completed = [
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        reverse = self._study.direction == optuna.study.StudyDirection.MAXIMIZE
        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=reverse)  # type: ignore[arg-type]
        return [(t.params, t.value) for t in sorted_trials[:n]]

    @property
    def n_trials(self) -> int:
        return sum(
            1 for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )

    @property
    def study(self) -> optuna.Study:
        return self._study

    def inject_priors(
        self,
        warm_trials: list[tuple[dict, float]],
        search_space: SearchSpace | None = None,
    ) -> None:
        """Seed the study with historical (params, value) observations as completed trials.

        search_space must be provided when called before the first suggest(); afterwards the
        internally stored search space is used automatically.
        """
        space = search_space or self._search_space
        if space is None:
            raise ValueError(
                "search_space must be provided when inject_priors is called before suggest()."
            )
        distributions = _build_distributions(space)
        for params, value in warm_trials:
            trial_params = {k: v for k, v in params.items() if k in distributions}
            trial_dists = {k: distributions[k] for k in trial_params}
            frozen = optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=value,
                params=trial_params,
                distributions=trial_dists,
            )
            self._study.add_trial(frozen)
