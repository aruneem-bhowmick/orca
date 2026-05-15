"""Unit tests for BayesianSearch."""

from __future__ import annotations

import math

import optuna
import pytest

from orcalab.search.bayesian import BayesianSearch
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


def _float_space() -> SearchSpace:
    return SearchSpace("float_test").add(FloatParameter("lr", low=1e-5, high=1e-1))


class TestBayesianSearchCycle:
    def test_20_trial_cycle_runs_without_error(self) -> None:
        space = _mixed_space()
        searcher = BayesianSearch()
        for i in range(20):
            params = searcher.suggest(space)
            searcher.update(params, result=float(i) * 0.05)
        assert searcher.n_trials == 20

    def test_get_best_returns_descending_order(self) -> None:
        space = SearchSpace("test").add(CategoricalParameter("x", choices=list(range(10))))
        searcher = BayesianSearch()
        results = [float(i) * 0.1 for i in range(10)]
        for r in results:
            params = searcher.suggest(space)
            searcher.update(params, result=r)

        best = searcher.get_best(3)
        assert len(best) == 3
        values = [v for _, v in best]
        assert values == sorted(values, reverse=True)
        assert values[0] == pytest.approx(max(results))

    def test_get_best_returns_fewer_when_fewer_trials_exist(self) -> None:
        space = _float_space()
        searcher = BayesianSearch()
        params = searcher.suggest(space)
        searcher.update(params, result=0.5)
        assert len(searcher.get_best(10)) == 1


class TestBayesianInjectPriors:
    def test_inject_priors_seeds_study(self) -> None:
        space = _float_space()
        searcher = BayesianSearch()
        priors = [
            ({"lr": 0.01}, 0.95),
            ({"lr": 0.001}, 0.80),
            ({"lr": 0.1}, 0.60),
        ]
        searcher.inject_priors(priors, search_space=space)
        assert searcher.n_trials == 3

    def test_get_best_matches_best_injected_prior(self) -> None:
        space = _float_space()
        searcher = BayesianSearch()
        priors = [
            ({"lr": 0.01}, 0.95),
            ({"lr": 0.001}, 0.80),
            ({"lr": 0.1}, 0.60),
        ]
        searcher.inject_priors(priors, search_space=space)
        best = searcher.get_best(1)
        assert len(best) == 1
        assert best[0][1] == pytest.approx(0.95)
        assert best[0][0]["lr"] == pytest.approx(0.01)

    def test_inject_priors_without_search_space_raises(self) -> None:
        searcher = BayesianSearch()
        with pytest.raises(ValueError, match="search_space must be provided"):
            searcher.inject_priors([({"lr": 0.01}, 0.9)])

    def test_inject_priors_uses_stored_space_after_suggest(self) -> None:
        space = _float_space()
        searcher = BayesianSearch()
        params = searcher.suggest(space)
        searcher.update(params, result=0.5)
        # No search_space arg needed — stored from suggest()
        searcher.inject_priors([({"lr": 0.05}, 0.99)])
        assert searcher.n_trials == 2

    def test_warm_start_trials_constructor_deferred(self) -> None:
        space = _float_space()
        priors = [({"lr": 0.01}, 0.95), ({"lr": 0.001}, 0.80)]
        searcher = BayesianSearch(warm_start_trials=priors)
        # Deferred: priors not yet injected before first suggest()
        assert searcher.n_trials == 0
        params = searcher.suggest(space)
        searcher.update(params, result=0.5)
        # 2 injected priors + 1 new trial
        assert searcher.n_trials == 3


class TestBayesianPersistence:
    def test_persist_and_resume_from_sqlite(self, tmp_path: object) -> None:
        db_path = tmp_path / "study.db"  # type: ignore[operator]
        storage = f"sqlite:///{db_path}"
        space = _float_space()

        searcher = BayesianSearch(study_name="test_persist", storage=storage)
        for i in range(5):
            params = searcher.suggest(space)
            searcher.update(params, result=float(i) * 0.1)
        assert searcher.n_trials == 5

        searcher2 = BayesianSearch(study_name="test_persist", storage=storage)
        assert searcher2.n_trials == 5


class TestBayesianNaNHandling:
    def test_nan_result_handled_gracefully(self) -> None:
        space = _float_space()
        searcher = BayesianSearch()
        params = searcher.suggest(space)
        searcher.update(params, result=float("nan"))
        assert searcher.n_trials == 0

    def test_inf_result_handled_gracefully(self) -> None:
        space = _float_space()
        searcher = BayesianSearch()
        params = searcher.suggest(space)
        searcher.update(params, result=float("inf"))
        assert searcher.n_trials == 0

    def test_negative_inf_result_handled_gracefully(self) -> None:
        space = _float_space()
        searcher = BayesianSearch()
        params = searcher.suggest(space)
        searcher.update(params, result=float("-inf"))
        assert searcher.n_trials == 0

    def test_valid_results_after_nan_still_recorded(self) -> None:
        space = _float_space()
        searcher = BayesianSearch()
        bad = searcher.suggest(space)
        searcher.update(bad, result=float("nan"))
        good = searcher.suggest(space)
        searcher.update(good, result=0.8)
        assert searcher.n_trials == 1


class TestBayesianStudyProperty:
    def test_study_property_exposes_optuna_study(self) -> None:
        searcher = BayesianSearch()
        assert isinstance(searcher.study, optuna.Study)

    def test_custom_study_name(self) -> None:
        searcher = BayesianSearch(study_name="my_custom_study")
        assert searcher.study.study_name == "my_custom_study"

    def test_default_direction_is_maximize(self) -> None:
        searcher = BayesianSearch()
        assert searcher.study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_minimize_direction(self) -> None:
        space = _float_space()
        searcher = BayesianSearch(direction="minimize")
        assert searcher.study.direction == optuna.study.StudyDirection.MINIMIZE
        params = searcher.suggest(space)
        searcher.update(params, result=0.01)
        # get_best for minimize should return the lowest value
        best = searcher.get_best(1)
        assert best[0][1] == pytest.approx(0.01)

    def test_update_mismatch_raises_value_error(self) -> None:
        space = _float_space()
        searcher = BayesianSearch()
        searcher.suggest(space)
        with pytest.raises(ValueError, match="params do not match"):
            searcher.update({"lr": 9999.0}, result=0.5)

    def test_update_without_pending_raises_value_error(self) -> None:
        searcher = BayesianSearch()
        with pytest.raises(ValueError, match="No pending trials"):
            searcher.update({"lr": 0.01}, result=0.5)
