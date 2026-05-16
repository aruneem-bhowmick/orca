"""EvolutionarySearch strategy backed by CMA-ES via the cma library."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import cma
import numpy as np

from orcalab.search.base import SearchStrategy
from orcalab.search_spaces.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
)
from orcalab.search_spaces.space import SearchSpace

logger = logging.getLogger(__name__)


def _build_dim_map(space: SearchSpace) -> list[tuple[Any, slice]]:
    """Build a mapping from each parameter to its slice in the normalized vector.

    CategoricalParameter with N choices occupies N dimensions (one-hot).
    All other parameter types occupy 1 dimension each.
    """
    dim_map: list[tuple[Any, slice]] = []
    cursor = 0
    for param in space._params.values():
        if isinstance(param, CategoricalParameter):
            n = len(param.choices)
            dim_map.append((param, slice(cursor, cursor + n)))
            cursor += n
        else:
            dim_map.append((param, slice(cursor, cursor + 1)))
            cursor += 1
    return dim_map


def _total_dim(dim_map: list[tuple[Any, slice]]) -> int:
    """Return the total number of dimensions in the normalized vector."""
    if not dim_map:
        return 0
    return dim_map[-1][1].stop


def _encode(params: dict[str, Any], dim_map: list[tuple[Any, slice]]) -> np.ndarray:
    """Encode a parameter dict into a normalized [0, 1]^d vector."""
    d = _total_dim(dim_map)
    vec = np.zeros(d)
    for param, sl in dim_map:
        val = params[param.name]
        if isinstance(param, CategoricalParameter):
            idx = param.choices.index(val)
            vec[sl.start + idx] = 1.0
        elif isinstance(param, IntParameter):
            if param.high == param.low:
                vec[sl.start] = 0.5
            elif param.log:
                vec[sl.start] = (math.log(val) - math.log(param.low)) / (
                    math.log(param.high) - math.log(param.low)
                )
            else:
                vec[sl.start] = (val - param.low) / (param.high - param.low)
        elif isinstance(param, FloatParameter):
            if param.high == param.low:
                vec[sl.start] = 0.5
            elif param.log:
                vec[sl.start] = (math.log(val) - math.log(param.low)) / (
                    math.log(param.high) - math.log(param.low)
                )
            else:
                vec[sl.start] = (val - param.low) / (param.high - param.low)
        else:
            # DiscreteUniformParameter and any future linear parameter types
            if param.high == param.low:  # type: ignore[attr-defined]
                vec[sl.start] = 0.5
            else:
                vec[sl.start] = (val - param.low) / (param.high - param.low)  # type: ignore[attr-defined]
    return vec


def _decode(vec: np.ndarray, dim_map: list[tuple[Any, slice]]) -> dict[str, Any]:
    """Decode a normalized vector back to a parameter dict."""
    params: dict[str, Any] = {}
    for param, sl in dim_map:
        if isinstance(param, CategoricalParameter):
            idx = int(np.argmax(vec[sl]))
            params[param.name] = param.choices[idx]
        elif isinstance(param, IntParameter):
            v = float(np.clip(vec[sl.start], 0.0, 1.0))
            if param.log:
                raw = math.exp(
                    v * (math.log(param.high) - math.log(param.low)) + math.log(param.low)
                )
            else:
                raw = v * (param.high - param.low) + param.low
            params[param.name] = int(np.clip(round(raw), param.low, param.high))
        elif isinstance(param, FloatParameter):
            v = float(np.clip(vec[sl.start], 0.0, 1.0))
            if param.log:
                raw = math.exp(
                    v * (math.log(param.high) - math.log(param.low)) + math.log(param.low)
                )
            else:
                raw = v * (param.high - param.low) + param.low
            params[param.name] = float(np.clip(raw, param.low, param.high))
        else:
            # DiscreteUniformParameter: treat as linear float
            v = float(np.clip(vec[sl.start], 0.0, 1.0))
            raw = v * (param.high - param.low) + param.low  # type: ignore[attr-defined]
            params[param.name] = float(np.clip(raw, param.low, param.high))  # type: ignore[attr-defined]
    return params


class EvolutionarySearch(SearchStrategy):
    """Search strategy using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    Parameters are encoded into a normalized [0, 1]^d vector; CategoricalParameters
    are one-hot encoded. CMA-ES updates its distribution every `population_size`
    evaluations. Because CMA-ES minimizes internally, fitnesses are negated when
    direction is "maximize".
    """

    def __init__(
        self,
        population_size: int = 10,
        sigma0: float = 0.3,
        seed: int = 42,
        direction: str = "maximize",
    ) -> None:
        """Initialize EvolutionarySearch with CMA-ES hyperparameters and direction."""
        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")
        if population_size <= 0:
            raise ValueError(f"population_size must be > 0, got {population_size}")
        if sigma0 <= 0.0:
            raise ValueError(f"sigma0 must be > 0, got {sigma0}")
        self._population_size = population_size
        self._sigma0 = sigma0
        self._seed = seed
        self._direction = direction
        self._es: cma.CMAEvolutionStrategy | None = None
        self._dim_map: list[tuple[Any, slice]] | None = None
        self._solution_queue: deque[tuple[np.ndarray, dict[str, Any]]] = deque()
        self._pending: deque[tuple[dict[str, Any], np.ndarray]] = deque()
        self._gen_accumulator: list[tuple[np.ndarray, float]] = []
        self._history: list[tuple[dict[str, Any], float]] = []
        self._stopped: bool = False
        self._rng = np.random.default_rng(seed)
        self._restart_count: int = 0
        self._search_space: SearchSpace | None = None

    def _make_es(self, x0: list[float]) -> cma.CMAEvolutionStrategy:
        """Instantiate a new CMAEvolutionStrategy from the given starting point."""
        return cma.CMAEvolutionStrategy(
            x0,
            self._sigma0,
            {
                "seed": self._seed + self._restart_count,
                "popsize": self._population_size,
                "verbose": -9,
            },
        )

    def _repopulate(self) -> None:
        """Ask CMA-ES for the next population and fill the solution queue."""
        assert self._es is not None
        assert self._dim_map is not None
        solutions = self._es.ask()
        for sol in solutions:
            arr = np.asarray(sol, dtype=float)
            self._solution_queue.append((arr, _decode(arr, self._dim_map)))

    def suggest(self, search_space: SearchSpace) -> dict[str, Any]:
        """Return the next parameter dict to evaluate, sampled from the CMA-ES distribution."""
        if self._search_space is None:
            self._search_space = search_space
        elif search_space is not self._search_space:
            raise ValueError(
                "EvolutionarySearch must be used with the same SearchSpace instance across suggest() calls."
            )

        if self._dim_map is None:
            self._dim_map = _build_dim_map(search_space)
            d = _total_dim(self._dim_map)
            self._es = self._make_es([0.5] * d)
            self._repopulate()

        if self._stopped and self._pending:
            raise ValueError(
                "Cannot restart while pending trials exist; call update() for all pending suggestions first."
            )

        if self._stopped:
            assert self._dim_map is not None
            d = _total_dim(self._dim_map)
            if self._history:
                reverse = self._direction == "maximize"
                best_params = sorted(self._history, key=lambda x: x[1], reverse=reverse)[0][0]
                best_vec = _encode(best_params, self._dim_map)
                noise = self._rng.standard_normal(d) * self._sigma0
                x0 = list(np.clip(best_vec + noise, 0.0, 1.0))
            else:
                x0 = list(self._rng.uniform(0.0, 1.0, d))
            self._restart_count += 1
            self._es = self._make_es(x0)
            self._stopped = False
            self._gen_accumulator.clear()
            self._solution_queue.clear()
            self._repopulate()

        if not self._solution_queue:
            self._repopulate()

        vec, params = self._solution_queue.popleft()
        self._pending.append((params, vec))
        return params

    def update(self, params: dict[str, Any], result: float) -> None:
        """Record the result for the next pending trial.

        Must be called in FIFO order matching suggest(). NaN and Inf results are
        dropped silently and do not contribute to the CMA-ES update or history.
        """
        if not self._pending:
            raise ValueError("No pending trials; call suggest() before update().")
        pending_params, vec = self._pending[0]
        if params != pending_params:
            raise ValueError(
                f"params do not match the next pending trial. "
                f"Expected {pending_params!r}, got {params!r}."
            )
        self._pending.popleft()

        if math.isnan(result) or math.isinf(result):
            return

        self._history.append((params, result))
        self._gen_accumulator.append((vec, result))

        if len(self._gen_accumulator) == self._population_size:
            assert self._es is not None
            solutions = [v for v, _ in self._gen_accumulator]
            raw_fits = [f for _, f in self._gen_accumulator]
            fits = [-f for f in raw_fits] if self._direction == "maximize" else raw_fits
            self._es.tell(solutions, fits)
            self._gen_accumulator.clear()
            stop_dict = self._es.stop()
            if stop_dict:
                logger.info("CMA-ES converged: %s", stop_dict)
                self._stopped = True

    def get_best(self, n: int = 1) -> list[tuple[dict, float]]:
        """Return the top n (params, result) pairs sorted by result in the optimization direction."""
        reverse = self._direction == "maximize"
        return sorted(self._history, key=lambda x: x[1], reverse=reverse)[:n]

    @property
    def n_trials(self) -> int:
        """Return the number of valid (non-NaN/Inf) trials recorded so far."""
        return len(self._history)
