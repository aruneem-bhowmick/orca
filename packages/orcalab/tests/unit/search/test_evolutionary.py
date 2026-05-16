"""Unit tests for EvolutionarySearch."""

from __future__ import annotations

import random

import numpy as np
import pytest

from orcalab.search.base import SearchStrategy
from orcalab.search.evolutionary import (
    EvolutionarySearch,
    _build_dim_map,
    _decode,
    _encode,
    _total_dim,
)
from orcalab.search_spaces.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
)
from orcalab.search_spaces.space import SearchSpace


def _float_space() -> SearchSpace:
    return SearchSpace("float").add(FloatParameter("lr", low=1e-4, high=1e-1))


def _log_float_space() -> SearchSpace:
    return SearchSpace("logfloat").add(FloatParameter("lr", low=1e-4, high=1e-1, log=True))


def _int_space() -> SearchSpace:
    return SearchSpace("int").add(IntParameter("layers", low=2, high=20))


def _cat_space() -> SearchSpace:
    return SearchSpace("cat").add(
        CategoricalParameter("optimizer", choices=["adam", "sgd", "rmsprop"])
    )


def _mixed_space() -> SearchSpace:
    return (
        SearchSpace("mixed")
        .add(FloatParameter("lr", low=1e-4, high=1e-1))
        .add(IntParameter("layers", low=2, high=20))
        .add(CategoricalParameter("optimizer", choices=["adam", "sgd"]))
    )


class TestEncodingDecoding:
    def test_int_roundtrip(self) -> None:
        space = _int_space()
        dim_map = _build_dim_map(space)
        for v in [2, 5, 10, 15, 20]:
            vec = _encode({"layers": v}, dim_map)
            assert 0.0 <= vec[0] <= 1.0
            decoded = _decode(vec, dim_map)
            assert decoded["layers"] == v

    def test_float_linear_roundtrip(self) -> None:
        space = _float_space()
        dim_map = _build_dim_map(space)
        for v in [1e-4, 1e-3, 1e-2, 1e-1]:
            vec = _encode({"lr": v}, dim_map)
            assert 0.0 <= vec[0] <= 1.0
            decoded = _decode(vec, dim_map)
            assert decoded["lr"] == pytest.approx(v, rel=1e-6)

    def test_float_log_roundtrip(self) -> None:
        space = _log_float_space()
        dim_map = _build_dim_map(space)
        for v in [1e-4, 1e-3, 1e-2, 1e-1]:
            vec = _encode({"lr": v}, dim_map)
            assert 0.0 <= vec[0] <= 1.0
            decoded = _decode(vec, dim_map)
            assert decoded["lr"] == pytest.approx(v, rel=1e-6)

    def test_categorical_roundtrip(self) -> None:
        space = _cat_space()
        dim_map = _build_dim_map(space)
        for choice in ["adam", "sgd", "rmsprop"]:
            vec = _encode({"optimizer": choice}, dim_map)
            assert vec.sum() == pytest.approx(1.0)
            decoded = _decode(vec, dim_map)
            assert decoded["optimizer"] == choice

    def test_mixed_space_roundtrip(self) -> None:
        space = _mixed_space()
        dim_map = _build_dim_map(space)
        assert _total_dim(dim_map) == 4  # 1 + 1 + 2

        params = {"lr": 0.01, "layers": 8, "optimizer": "sgd"}
        vec = _encode(params, dim_map)
        assert len(vec) == 4
        decoded = _decode(vec, dim_map)
        assert decoded["lr"] == pytest.approx(0.01, rel=1e-6)
        assert decoded["layers"] == 8
        assert decoded["optimizer"] == "sgd"

    def test_int_boundary_values_roundtrip(self) -> None:
        space = _int_space()
        dim_map = _build_dim_map(space)
        for boundary in [2, 20]:
            vec = _encode({"layers": boundary}, dim_map)
            decoded = _decode(vec, dim_map)
            assert decoded["layers"] == boundary

    def test_float_boundary_values_roundtrip(self) -> None:
        space = _float_space()
        dim_map = _build_dim_map(space)
        for boundary in [1e-4, 1e-1]:
            vec = _encode({"lr": boundary}, dim_map)
            decoded = _decode(vec, dim_map)
            assert decoded["lr"] == pytest.approx(boundary, rel=1e-6)

    def test_decoded_value_stays_in_bounds_when_vec_is_clipped(self) -> None:
        space = _float_space()
        dim_map = _build_dim_map(space)
        decoded_low = _decode(np.array([-0.5]), dim_map)
        decoded_high = _decode(np.array([1.5]), dim_map)
        assert decoded_low["lr"] == pytest.approx(1e-4, rel=1e-6)
        assert decoded_high["lr"] == pytest.approx(1e-1, rel=1e-6)


class TestEvolutionarySearchABCCompliance:
    def test_isinstance_search_strategy(self) -> None:
        assert isinstance(EvolutionarySearch(), SearchStrategy)

    def test_suggest_is_callable(self) -> None:
        assert callable(EvolutionarySearch().suggest)

    def test_update_is_callable(self) -> None:
        assert callable(EvolutionarySearch().update)

    def test_get_best_is_callable(self) -> None:
        assert callable(EvolutionarySearch().get_best)

    def test_n_trials_is_int_on_fresh_instance(self) -> None:
        evo = EvolutionarySearch()
        assert isinstance(evo.n_trials, int)
        assert evo.n_trials == 0

    def test_get_history_returns_all_results_sorted_descending(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=10, seed=0)
        for i in range(10):
            p = evo.suggest(space)
            evo.update(p, float(i) * 0.1)
        history = evo.get_history()
        assert len(history) == 10
        values = [v for _, v in history]
        assert values == sorted(values, reverse=True)

    def test_n_trials_increments_per_valid_update(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=10, seed=0)
        for i in range(5):
            p = evo.suggest(space)
            evo.update(p, float(i))
        assert evo.n_trials == 5


class TestEvolutionarySearchCycle:
    def test_20_trial_cycle_no_error(self) -> None:
        space = _mixed_space()
        evo = EvolutionarySearch(population_size=10, seed=42)
        for i in range(20):
            p = evo.suggest(space)
            evo.update(p, float(i) * 0.05)
        assert evo.n_trials == 20

    def test_get_best_n_returns_descending_values(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=10, seed=0)
        for i in range(10):
            p = evo.suggest(space)
            evo.update(p, float(i))
        best = evo.get_best(3)
        assert len(best) == 3
        vals = [v for _, v in best]
        assert vals == sorted(vals, reverse=True)
        assert vals[0] == pytest.approx(9.0)

    def test_get_best_returns_fewer_when_fewer_trials_exist(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=10, seed=0)
        p = evo.suggest(space)
        evo.update(p, 0.5)
        assert len(evo.get_best(100)) == 1

    def test_direction_minimize_get_best_returns_lowest(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=5, direction="minimize", seed=0)
        results = [10.0, 3.0, 7.0, 1.0, 5.0]
        for r in results:
            p = evo.suggest(space)
            evo.update(p, r)
        best = evo.get_best(1)
        assert best[0][1] == pytest.approx(1.0)

    def test_multiple_generations_run_without_error(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=10, seed=0)
        for i in range(25):
            p = evo.suggest(space)
            evo.update(p, float(i))
        assert evo.n_trials == 25

    def test_suggest_returns_params_within_space_bounds(self) -> None:
        space = _mixed_space()
        evo = EvolutionarySearch(population_size=5, seed=7)
        for _ in range(20):
            p = evo.suggest(space)
            assert 1e-4 <= p["lr"] <= 1e-1
            assert 2 <= p["layers"] <= 20
            assert p["optimizer"] in ["adam", "sgd"]
            evo.update(p, 0.5)


class TestEvolutionarySearchOptimization:
    def test_50_trials_beats_random_on_quadratic(self) -> None:
        space = (
            SearchSpace("quad")
            .add(FloatParameter("lr", low=1e-4, high=1e-1))
            .add(IntParameter("layers", low=2, high=20))
        )

        def objective(params: dict) -> float:
            return -((params["lr"] - 0.05) ** 2) - ((params["layers"] - 10) ** 2 / 100.0)

        evo = EvolutionarySearch(population_size=10, sigma0=0.3, seed=42)
        for _ in range(50):
            p = evo.suggest(space)
            evo.update(p, objective(p))

        rng = random.Random(42)
        random_best = max(
            objective({"lr": rng.uniform(1e-4, 1e-1), "layers": rng.randint(2, 20)})
            for _ in range(50)
        )
        assert evo.get_best(1)[0][1] > random_best


class TestEvolutionarySearchSpaceContract:
    def test_suggest_with_different_space_instance_raises_value_error(self) -> None:
        space1 = _float_space()
        space2 = _float_space()
        evo = EvolutionarySearch(population_size=5, seed=0)
        evo.suggest(space1)
        with pytest.raises(ValueError, match="same SearchSpace instance"):
            evo.suggest(space2)

    def test_suggest_with_same_space_instance_does_not_raise(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=5, seed=0)
        p = evo.suggest(space)
        evo.update(p, 0.5)
        p2 = evo.suggest(space)
        evo.update(p2, 0.6)
        assert evo.n_trials == 2

    def test_suggest_with_stopped_and_pending_raises_value_error(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=5, seed=0)
        p = evo.suggest(space)
        evo._stopped = True  # force converged state while trial is still pending
        with pytest.raises(ValueError, match="Cannot restart while pending"):
            evo.suggest(space)


class TestEvolutionarySearchUpdateValidation:
    def test_update_with_partial_population_does_not_fail(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=10, seed=42)
        for _ in range(5):
            p = evo.suggest(space)
            evo.update(p, 0.5)
        assert evo.n_trials == 5

    def test_update_without_pending_raises_value_error(self) -> None:
        evo = EvolutionarySearch()
        with pytest.raises(ValueError, match="No pending trials"):
            evo.update({"lr": 0.01}, result=0.5)

    def test_update_with_mismatched_params_raises_value_error(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=5, seed=0)
        evo.suggest(space)
        with pytest.raises(ValueError, match="params do not match"):
            evo.update({"lr": 9999.0}, result=0.5)

    def test_nan_result_not_recorded_in_history(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=5, seed=0)
        p = evo.suggest(space)
        evo.update(p, float("nan"))
        assert evo.n_trials == 0

    def test_inf_result_not_recorded_in_history(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=5, seed=0)
        p = evo.suggest(space)
        evo.update(p, float("inf"))
        assert evo.n_trials == 0

    def test_negative_inf_result_not_recorded_in_history(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=5, seed=0)
        p = evo.suggest(space)
        evo.update(p, float("-inf"))
        assert evo.n_trials == 0

    def test_valid_result_after_nan_is_recorded(self) -> None:
        space = _float_space()
        evo = EvolutionarySearch(population_size=5, seed=0)
        bad = evo.suggest(space)
        evo.update(bad, float("nan"))
        good = evo.suggest(space)
        evo.update(good, 0.8)
        assert evo.n_trials == 1

    def test_invalid_direction_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="direction must be"):
            EvolutionarySearch(direction="sideways")

    def test_zero_population_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="population_size must be > 0"):
            EvolutionarySearch(population_size=0)

    def test_negative_population_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="population_size must be > 0"):
            EvolutionarySearch(population_size=-5)

    def test_zero_sigma0_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sigma0 must be > 0"):
            EvolutionarySearch(sigma0=0.0)

    def test_negative_sigma0_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sigma0 must be > 0"):
            EvolutionarySearch(sigma0=-0.1)
