"""Unit tests for SearchSpace sampling, conditional parameters, and serialization."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest

from orcalab.search_spaces.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    LogUniformParameter,
)
from orcalab.search_spaces.space import SearchSpace

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _trial() -> optuna.Trial:
    return optuna.create_study().ask()


def _space_with_three_params() -> SearchSpace:
    return (
        SearchSpace(name="test_space")
        .add(IntParameter("num_layers", low=4, high=32))
        .add(LogUniformParameter("learning_rate", low=1e-5, high=1e-1))
        .add(CategoricalParameter("optimizer", choices=["adam", "sgd"]))
    )


class TestSearchSpaceSampling:
    def test_sample_returns_all_param_names(self) -> None:
        space = _space_with_three_params()
        result = space.sample(_trial())
        assert set(result.keys()) == {"num_layers", "learning_rate", "optimizer"}

    def test_sample_values_are_within_parameter_ranges(self) -> None:
        space = _space_with_three_params()
        result = space.sample(_trial())
        assert 4 <= result["num_layers"] <= 32
        assert 1e-5 <= result["learning_rate"] <= 1e-1
        assert result["optimizer"] in ["adam", "sgd"]

    def test_add_returns_self_for_fluent_chaining(self) -> None:
        space = SearchSpace(name="chain_test")
        param = IntParameter("x", low=0, high=10)
        returned = space.add(param)
        assert returned is space


class TestConditionalParameters:
    def test_conditional_param_excluded_when_condition_false(self) -> None:
        space = SearchSpace(name="conditional_space")
        space.add(CategoricalParameter("arch", choices=["cnn", "mlp"]))
        # Condition always returns False — dropout should never appear
        space.add_condition(lambda sampled: False, FloatParameter("dropout", low=0.0, high=0.5))

        study = optuna.create_study()
        for _ in range(10):
            trial = study.ask()
            result = space.sample(trial)
            assert "dropout" not in result
            study.tell(trial, 0.0)

    def test_conditional_param_included_when_condition_true(self) -> None:
        space = SearchSpace(name="conditional_space")
        space.add(CategoricalParameter("arch", choices=["mlp"]))
        # Condition always returns True — dropout should always appear
        space.add_condition(lambda sampled: True, FloatParameter("dropout", low=0.0, high=0.5))

        result = space.sample(_trial())
        assert "dropout" in result
        assert 0.0 <= result["dropout"] <= 0.5

    def test_condition_receives_sampled_values(self) -> None:
        space = SearchSpace(name="data_driven_condition")
        space.add(CategoricalParameter("arch", choices=["mlp", "cnn"]))
        space.add_condition(
            lambda s: s.get("arch") == "mlp",
            FloatParameter("dropout", low=0.1, high=0.5),
        )

        study = optuna.create_study()
        for _ in range(30):
            trial = study.ask()
            result = space.sample(trial)
            if result["arch"] == "mlp":
                assert "dropout" in result
            else:
                assert "dropout" not in result
            study.tell(trial, 0.0)

    def test_add_condition_returns_self_for_fluent_chaining(self) -> None:
        space = SearchSpace(name="chain_test")
        returned = space.add_condition(lambda _: True, FloatParameter("x", low=0.0, high=1.0))
        assert returned is space


class TestSearchSpaceSerialization:
    def test_to_dict_excludes_conditional_parameters(self, tmp_path: Path) -> None:
        space = SearchSpace(name="conditional_serialization")
        space.add(IntParameter("layers", low=2, high=10))
        space.add_condition(lambda _: True, FloatParameter("dropout", low=0.0, high=0.5))

        d = space.to_dict()
        assert [p["name"] for p in d["parameters"]] == ["layers"]

        reconstructed = SearchSpace.from_dict(d)
        assert set(reconstructed._params.keys()) == {"layers"}
        assert len(reconstructed._conditions) == 0

        path = str(tmp_path / "space.json")
        space.save(path)
        loaded = SearchSpace.load(path)
        assert [p["name"] for p in loaded.to_dict()["parameters"]] == ["layers"]
        assert len(loaded._conditions) == 0

    def test_to_dict_contains_name_description_and_parameters(self) -> None:
        space = SearchSpace(name="my_space", description="test desc")
        space.add(IntParameter("layers", low=2, high=10))
        d = space.to_dict()
        assert d["name"] == "my_space"
        assert d["description"] == "test desc"
        assert len(d["parameters"]) == 1
        assert d["parameters"][0]["name"] == "layers"

    def test_from_dict_reconstructs_space(self) -> None:
        space = _space_with_three_params()
        reconstructed = SearchSpace.from_dict(space.to_dict())
        assert reconstructed.name == space.name
        assert set(reconstructed._params.keys()) == set(space._params.keys())

    def test_save_load_roundtrip_preserves_params_and_types(self, tmp_path: Path) -> None:
        space = SearchSpace(name="saved_space", description="round-trip test")
        space.add(IntParameter("batch_size", low=16, high=256, step=16))
        space.add(LogUniformParameter("lr", low=1e-4, high=1e-1))
        space.add(CategoricalParameter("opt", choices=["adam", "sgd"]))

        path = str(tmp_path / "space.json")
        space.save(path)
        loaded = SearchSpace.load(path)

        assert loaded.name == space.name
        assert loaded.description == space.description
        assert loaded.to_dict() == space.to_dict()

    def test_save_creates_valid_json_file(self, tmp_path: Path) -> None:
        import json

        space = SearchSpace(name="json_test")
        space.add(FloatParameter("lr", low=1e-4, high=1e-1))
        path = tmp_path / "space.json"
        space.save(str(path))
        assert path.exists()
        with open(path) as fh:
            data = json.load(fh)
        assert data["name"] == "json_test"
