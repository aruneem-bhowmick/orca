"""Unit tests for search space parameter type classes."""

from __future__ import annotations

import math

import optuna
import pytest

from orcalab.search_spaces.parameters import (
    CategoricalParameter,
    DiscreteUniformParameter,
    FloatParameter,
    IntParameter,
    LogUniformParameter,
    Parameter,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _ask(study: optuna.Study | None = None) -> optuna.Trial:
    """Return a fresh trial from a new (or given) in-memory study."""
    if study is None:
        study = optuna.create_study()
    return study.ask()


class TestIntParameter:
    def test_samples_within_bounds(self) -> None:
        param = IntParameter("layers", low=4, high=20)
        trial = _ask()
        value = param.to_optuna(trial)
        assert isinstance(value, int)
        assert 4 <= value <= 20

    def test_step_constrains_values(self) -> None:
        param = IntParameter("batch", low=16, high=256, step=16)
        study = optuna.create_study()
        for _ in range(20):
            trial = _ask(study)
            value = param.to_optuna(trial)
            assert value % 16 == 0, f"{value} is not a multiple of 16"
            study.tell(trial, 0.0)

    def test_to_dict_round_trip(self) -> None:
        param = IntParameter("layers", low=2, high=10, step=2, log=False)
        reconstructed = Parameter.from_dict(param.to_dict())
        assert isinstance(reconstructed, IntParameter)
        assert reconstructed.name == param.name
        assert reconstructed.low == param.low
        assert reconstructed.high == param.high
        assert reconstructed.step == param.step
        assert reconstructed.log == param.log


class TestFloatParameter:
    def test_samples_within_bounds(self) -> None:
        param = FloatParameter("dropout", low=0.0, high=0.5)
        trial = _ask()
        value = param.to_optuna(trial)
        assert isinstance(value, float)
        assert 0.0 <= value <= 0.5

    def test_to_dict_round_trip(self) -> None:
        param = FloatParameter("weight_decay", low=1e-6, high=1e-2)
        reconstructed = Parameter.from_dict(param.to_dict())
        assert isinstance(reconstructed, FloatParameter)
        assert reconstructed.name == param.name
        assert math.isclose(reconstructed.low, param.low)
        assert math.isclose(reconstructed.high, param.high)
        assert reconstructed.log == param.log


class TestLogUniformParameter:
    def test_produces_values_spanning_multiple_orders_of_magnitude(self) -> None:
        param = LogUniformParameter("lr", low=1e-5, high=1e-1)
        study = optuna.create_study()
        samples: list[float] = []
        for _ in range(50):
            trial = _ask(study)
            samples.append(param.to_optuna(trial))
            study.tell(trial, 0.0)
        assert max(samples) / min(samples) > 10, (
            "Expected log-uniform samples to span more than one order of magnitude"
        )

    def test_is_float_parameter_with_log_true(self) -> None:
        param = LogUniformParameter("lr", low=1e-4, high=1e-1)
        assert isinstance(param, FloatParameter)
        assert param.log is True

    def test_to_dict_round_trip(self) -> None:
        param = LogUniformParameter("lr", low=1e-5, high=1e-1)
        reconstructed = Parameter.from_dict(param.to_dict())
        assert isinstance(reconstructed, LogUniformParameter)
        assert reconstructed.name == param.name
        assert math.isclose(reconstructed.low, param.low)
        assert math.isclose(reconstructed.high, param.high)


class TestDiscreteUniformParameter:
    def test_samples_on_grid(self) -> None:
        param = DiscreteUniformParameter("momentum", low=0.8, high=0.99, q=0.01)
        study = optuna.create_study()
        for _ in range(20):
            trial = _ask(study)
            value = param.to_optuna(trial)
            # value should be a multiple of q within floating-point tolerance
            remainder = (value - param.low) % param.q
            assert remainder < 1e-9 or abs(remainder - param.q) < 1e-9, (
                f"{value} is not on the grid [low={param.low}, q={param.q}]"
            )
            study.tell(trial, 0.0)

    def test_to_dict_round_trip(self) -> None:
        param = DiscreteUniformParameter("momentum", low=0.8, high=0.99, q=0.01)
        reconstructed = Parameter.from_dict(param.to_dict())
        assert isinstance(reconstructed, DiscreteUniformParameter)
        assert reconstructed.name == param.name
        assert math.isclose(reconstructed.low, param.low)
        assert math.isclose(reconstructed.high, param.high)
        assert math.isclose(reconstructed.q, param.q)


class TestCategoricalParameter:
    def test_samples_from_choices(self) -> None:
        choices = ["adam", "sgd", "rmsprop"]
        param = CategoricalParameter("optimizer", choices=choices)
        study = optuna.create_study()
        for _ in range(20):
            trial = _ask(study)
            value = param.to_optuna(trial)
            assert value in choices
            study.tell(trial, 0.0)

    def test_to_dict_round_trip(self) -> None:
        param = CategoricalParameter("optimizer", choices=["adam", "sgd"])
        reconstructed = Parameter.from_dict(param.to_dict())
        assert isinstance(reconstructed, CategoricalParameter)
        assert reconstructed.name == param.name
        assert reconstructed.choices == param.choices


class TestParameterFromDict:
    def test_raises_on_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter type"):
            Parameter.from_dict({"type": "nonexistent", "name": "x"})

    def test_raises_on_missing_type_key(self) -> None:
        with pytest.raises(ValueError, match="missing required 'type' key"):
            Parameter.from_dict({"name": "x", "low": 0, "high": 10})

    @pytest.mark.parametrize(
        "param",
        [
            IntParameter("a", low=1, high=10),
            FloatParameter("b", low=0.0, high=1.0),
            LogUniformParameter("c", low=1e-4, high=1e-1),
            DiscreteUniformParameter("d", low=0.1, high=0.9, q=0.1),
            CategoricalParameter("e", choices=[1, 2, 3]),
        ],
    )
    def test_all_types_round_trip(self, param: Parameter) -> None:
        reconstructed = Parameter.from_dict(param.to_dict())
        assert type(reconstructed) is type(param)
        assert reconstructed.name == param.name
