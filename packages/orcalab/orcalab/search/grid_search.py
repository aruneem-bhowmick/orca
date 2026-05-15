"""GridSearch strategy that exhaustively covers a pre-computed Cartesian product."""

from __future__ import annotations

import itertools
import math
from typing import Any

from orcalab.search.base import SearchStrategy
from orcalab.search_spaces.parameters import (
    CategoricalParameter,
    DiscreteUniformParameter,
    FloatParameter,
    IntParameter,
    Parameter,
)
from orcalab.search_spaces.space import SearchSpace


def _grid_values(param: Parameter, n_steps: int) -> list[Any]:
    if isinstance(param, CategoricalParameter):
        return list(param.choices)

    if isinstance(param, DiscreteUniformParameter):
        n = round((param.high - param.low) / param.q) + 1
        return [round(param.low + i * param.q, 10) for i in range(n)]

    if isinstance(param, IntParameter):
        if param.step > 1:
            return list(range(param.low, param.high + 1, param.step))
        total = param.high - param.low
        if n_steps >= total + 1:
            return list(range(param.low, param.high + 1))
        return [param.low + round(i * total / (n_steps - 1)) for i in range(n_steps)]

    if isinstance(param, FloatParameter):
        if n_steps == 1:
            return [param.low]
        if param.log:
            log_low = math.log(param.low)
            log_high = math.log(param.high)
            return [
                math.exp(log_low + i * (log_high - log_low) / (n_steps - 1))
                for i in range(n_steps)
            ]
        return [
            param.low + i * (param.high - param.low) / (n_steps - 1)
            for i in range(n_steps)
        ]

    raise TypeError(f"Unsupported parameter type for grid search: {type(param).__name__!r}")


class GridSearch(SearchStrategy):
    def __init__(self, n_steps: int = 5) -> None:
        self._n_steps = n_steps
        self._grid: list[dict[str, Any]] = []
        self._index: int = 0
        self._results: list[tuple[dict[str, Any], float]] = []

    def _build_grid(self, search_space: SearchSpace) -> None:
        params = [Parameter.from_dict(p) for p in search_space.to_dict()["parameters"]]
        names = [p.name for p in params]
        value_lists = [_grid_values(p, self._n_steps) for p in params]
        self._grid = [
            dict(zip(names, combo)) for combo in itertools.product(*value_lists)
        ]

    def suggest(self, search_space: SearchSpace) -> dict[str, Any]:
        if not self._grid:
            self._build_grid(search_space)
        if self._index >= len(self._grid):
            raise StopIteration
        params = self._grid[self._index]
        self._index += 1
        return params

    def update(self, params: dict[str, Any], result: float) -> None:
        self._results.append((params, result))

    def get_best(self, n: int = 1) -> list[tuple[dict, float]]:
        return sorted(self._results, key=lambda x: x[1], reverse=True)[:n]

    @property
    def n_trials(self) -> int:
        return len(self._results)
