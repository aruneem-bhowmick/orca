"""Unit tests for GridSearch."""

from __future__ import annotations

import pytest

from orcalab.search.grid_search import GridSearch
from orcalab.search_spaces.parameters import (
    CategoricalParameter,
    DiscreteUniformParameter,
    FloatParameter,
    IntParameter,
)
from orcalab.search_spaces.space import SearchSpace


class TestGridSearchCoverage:
    def test_all_combinations_generated_exactly_once(self) -> None:
        space = (
            SearchSpace("test")
            .add(CategoricalParameter("optimizer", choices=["adam", "sgd"]))
            .add(CategoricalParameter("activation", choices=["relu", "tanh", "gelu"]))
        )
        searcher = GridSearch()
        combos: list[dict] = []
        try:
            while True:
                params = searcher.suggest(space)
                combos.append(params)
                searcher.update(params, result=0.0)
        except StopIteration:
            pass

        assert len(combos) == 6
        unique = {(c["optimizer"], c["activation"]) for c in combos}
        assert len(unique) == 6

    def test_continuous_params_discretized_to_n_steps(self) -> None:
        space = SearchSpace("test").add(FloatParameter("lr", low=0.0, high=1.0))
        searcher = GridSearch(n_steps=5)
        suggestions = []
        try:
            while True:
                params = searcher.suggest(space)
                suggestions.append(params["lr"])
                searcher.update(params, result=0.0)
        except StopIteration:
            pass

        assert len(suggestions) == 5
        assert suggestions[0] == pytest.approx(0.0)
        assert suggestions[-1] == pytest.approx(1.0)


class TestGridSearchExhaustion:
    def test_stop_iteration_raised_after_grid_exhausted(self) -> None:
        space = SearchSpace("test").add(CategoricalParameter("x", choices=["a", "b"]))
        searcher = GridSearch()
        searcher.suggest(space)
        searcher.suggest(space)
        with pytest.raises(StopIteration):
            searcher.suggest(space)

    def test_discrete_uniform_exhaustion(self) -> None:
        space = SearchSpace("test").add(
            DiscreteUniformParameter("momentum", low=0.8, high=0.82, q=0.01)
        )
        searcher = GridSearch()
        count = 0
        try:
            while True:
                params = searcher.suggest(space)
                searcher.update(params, result=0.0)
                count += 1
        except StopIteration:
            pass
        assert count == 3  # 0.80, 0.81, 0.82


class TestGridSearchGetBest:
    def test_get_best_1_returns_highest_result(self) -> None:
        space = SearchSpace("test").add(
            CategoricalParameter("x", choices=["a", "b", "c"])
        )
        searcher = GridSearch()
        results = [0.3, 0.9, 0.1]
        for result in results:
            params = searcher.suggest(space)
            searcher.update(params, result=result)

        best = searcher.get_best(1)
        assert len(best) == 1
        assert best[0][1] == pytest.approx(0.9)
        assert best[0][0]["x"] == "b"

    def test_get_best_n_returns_descending_order(self) -> None:
        space = SearchSpace("test").add(
            CategoricalParameter("v", choices=[10, 20, 30, 40, 50])
        )
        searcher = GridSearch()
        for i, result in enumerate([0.1, 0.5, 0.3, 0.9, 0.7]):
            params = searcher.suggest(space)
            searcher.update(params, result=result)

        best = searcher.get_best(3)
        values = [v for _, v in best]
        assert values == sorted(values, reverse=True)
        assert values[0] == pytest.approx(0.9)

    def test_get_history_returns_all_results_sorted(self) -> None:
        space = SearchSpace("test").add(
            CategoricalParameter("x", choices=["a", "b", "c"])
        )
        searcher = GridSearch()
        for result in [0.2, 0.8, 0.5]:
            params = searcher.suggest(space)
            searcher.update(params, result=result)

        history = searcher.get_history()
        assert len(history) == 3
        values = [v for _, v in history]
        assert values == sorted(values, reverse=True)


class TestGridSearchIntParameter:
    def test_int_param_with_step_uses_step(self) -> None:
        space = SearchSpace("test").add(
            IntParameter("batch", low=16, high=64, step=16)
        )
        searcher = GridSearch()
        combos = []
        try:
            while True:
                params = searcher.suggest(space)
                combos.append(params["batch"])
                searcher.update(params, result=0.0)
        except StopIteration:
            pass

        assert combos == [16, 32, 48, 64]

    def test_int_param_without_step_produces_n_steps(self) -> None:
        space = SearchSpace("test").add(IntParameter("layers", low=2, high=20))
        searcher = GridSearch(n_steps=5)
        combos = []
        try:
            while True:
                params = searcher.suggest(space)
                combos.append(params["layers"])
                searcher.update(params, result=0.0)
        except StopIteration:
            pass

        assert len(combos) == 5
        assert combos[0] == 2
        assert combos[-1] == 20
