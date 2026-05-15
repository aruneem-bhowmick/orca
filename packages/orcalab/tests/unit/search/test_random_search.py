"""Unit tests for RandomSearch."""

from __future__ import annotations

import optuna
import pytest

from orcalab.search.random_search import RandomSearch
from orcalab.search_spaces.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
)
from orcalab.search_spaces.space import SearchSpace

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _mixed_space() -> SearchSpace:
    return (
        SearchSpace("test")
        .add(IntParameter("layers", low=2, high=10))
        .add(FloatParameter("lr", low=1e-4, high=1e-1))
        .add(CategoricalParameter("optimizer", choices=["adam", "sgd"]))
    )


class TestRandomSearchBounds:
    def test_100_suggestions_within_bounds(self) -> None:
        space = _mixed_space()
        searcher = RandomSearch(random_state=42)
        for i in range(100):
            params = searcher.suggest(space)
            assert 2 <= params["layers"] <= 10
            assert 1e-4 <= params["lr"] <= 1e-1
            assert params["optimizer"] in ["adam", "sgd"]
            searcher.update(params, result=float(i))


class TestRandomSearchGetBest:
    def test_get_best_3_returns_top_3_by_result_value(self) -> None:
        space = SearchSpace("test").add(CategoricalParameter("x", choices=list(range(20))))
        searcher = RandomSearch(random_state=7)
        all_results: list[float] = []
        for i in range(20):
            params = searcher.suggest(space)
            result = float(i)
            all_results.append(result)
            searcher.update(params, result=result)

        best = searcher.get_best(3)
        assert len(best) == 3
        values = [v for _, v in best]
        assert values == sorted(values, reverse=True), "get_best must return results in descending order"
        assert values[0] == max(all_results)

    def test_get_best_returns_fewer_when_fewer_trials_exist(self) -> None:
        space = SearchSpace("test").add(CategoricalParameter("x", choices=["a", "b"]))
        searcher = RandomSearch(random_state=42)
        params = searcher.suggest(space)
        searcher.update(params, result=1.0)
        assert len(searcher.get_best(5)) == 1


class TestRandomSearchDeterminism:
    def test_seeded_runs_are_deterministic(self) -> None:
        space = (
            SearchSpace("test")
            .add(FloatParameter("lr", low=1e-5, high=1e-1))
            .add(IntParameter("layers", low=2, high=20))
        )
        searcher_a = RandomSearch(random_state=99)
        searcher_b = RandomSearch(random_state=99)
        suggestions_a = [searcher_a.suggest(space) for _ in range(10)]
        suggestions_b = [searcher_b.suggest(space) for _ in range(10)]
        assert suggestions_a == suggestions_b
