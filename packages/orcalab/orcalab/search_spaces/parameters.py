"""Parameter type classes for composable search space definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import optuna


class Parameter(ABC):
    """Abstract base class for all hyperparameter types."""

    name: str

    @abstractmethod
    def to_optuna(self, trial: optuna.Trial) -> Any:
        """Sample a value from this parameter using an Optuna trial."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize this parameter to a JSON-compatible dictionary."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Parameter:
        """Reconstruct a Parameter from its serialized dictionary representation."""
        type_map: dict[str, type[Parameter]] = {
            "int": IntParameter,
            "float": FloatParameter,
            "log_uniform": LogUniformParameter,
            "discrete_uniform": DiscreteUniformParameter,
            "categorical": CategoricalParameter,
        }
        param_type = d["type"]
        if param_type not in type_map:
            raise ValueError(f"Unknown parameter type: {param_type!r}")
        klass = type_map[param_type]
        d = {k: v for k, v in d.items() if k != "type"}
        return klass(**d)


class IntParameter(Parameter):
    """Integer parameter sampled uniformly (or log-uniformly) over [low, high]."""

    def __init__(
        self,
        name: str,
        low: int,
        high: int,
        step: int = 1,
        log: bool = False,
    ) -> None:
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def to_optuna(self, trial: optuna.Trial) -> int:
        return trial.suggest_int(self.name, self.low, self.high, step=self.step, log=self.log)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "int",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
        }


class FloatParameter(Parameter):
    """Continuous float parameter sampled uniformly (or log-uniformly) over [low, high]."""

    def __init__(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> None:
        self.name = name
        self.low = low
        self.high = high
        self.log = log

    def to_optuna(self, trial: optuna.Trial) -> float:
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "float",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "log": self.log,
        }


class LogUniformParameter(FloatParameter):
    """Convenience wrapper for FloatParameter sampled on a log scale."""

    def __init__(self, name: str, low: float, high: float) -> None:
        super().__init__(name, low, high, log=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "log_uniform",
            "name": self.name,
            "low": self.low,
            "high": self.high,
        }


class DiscreteUniformParameter(Parameter):
    """Float parameter sampled on a uniform grid with step size q over [low, high]."""

    def __init__(self, name: str, low: float, high: float, q: float) -> None:
        self.name = name
        self.low = low
        self.high = high
        self.q = q

    def to_optuna(self, trial: optuna.Trial) -> float:
        # suggest_discrete_uniform was removed in Optuna 3.x; use suggest_float with step
        return trial.suggest_float(self.name, self.low, self.high, step=self.q)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "discrete_uniform",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "q": self.q,
        }


class CategoricalParameter(Parameter):
    """Parameter sampled uniformly from a fixed set of choices."""

    def __init__(self, name: str, choices: list[Any]) -> None:
        self.name = name
        self.choices = choices

    def to_optuna(self, trial: optuna.Trial) -> Any:
        return trial.suggest_categorical(self.name, self.choices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "categorical",
            "name": self.name,
            "choices": self.choices,
        }
