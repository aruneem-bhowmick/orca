"""SearchSpace class for defining and sampling hyperparameter search spaces."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import optuna

from orcalab.search_spaces.parameters import Parameter


class SearchSpace:
    """A named collection of parameters that can be jointly sampled via an Optuna trial.

    Supports conditional parameters (included only when a predicate over already-sampled
    values is True) and fluent construction:

        space = SearchSpace(name="resnet_search")
        space.add(IntParameter("num_layers", low=8, high=50))
        space.add(LogUniformParameter("learning_rate", low=1e-5, high=1e-1))
        space.add(CategoricalParameter("optimizer", choices=["adam", "sgd"]))
        space.add(IntParameter("batch_size", low=16, high=256, step=16))

    Conditions are callable closures and are NOT persisted by save()/load(); only the
    unconditional parameters survive a round-trip.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._params: dict[str, Parameter] = {}
        self._conditions: list[tuple[Callable[[dict[str, Any]], bool], Parameter]] = []

    def add(self, param: Parameter) -> SearchSpace:
        """Register an unconditional parameter. Returns self for fluent chaining."""
        self._params[param.name] = param
        return self

    def add_condition(
        self,
        condition: Callable[[dict[str, Any]], bool],
        param: Parameter,
    ) -> SearchSpace:
        """Register a conditional parameter sampled only when condition(sampled) is True.

        The condition receives the dict of values sampled so far from unconditional
        parameters and any previously evaluated conditional ones.
        Returns self for fluent chaining.
        """
        self._conditions.append((condition, param))
        return self

    def sample(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample all parameters using the given Optuna trial.

        Unconditional parameters are sampled first; conditional parameters are sampled
        in registration order, each receiving the accumulated sampled dict for predicate
        evaluation.
        """
        sampled: dict[str, Any] = {}
        for param in self._params.values():
            sampled[param.name] = param.to_optuna(trial)
        for condition, param in self._conditions:
            if condition(sampled):
                sampled[param.name] = param.to_optuna(trial)
        return sampled

    def to_dict(self) -> dict[str, Any]:
        """Serialize this search space to a JSON-compatible dictionary.

        Conditions (callable closures) are not included.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self._params.values()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SearchSpace:
        """Reconstruct a SearchSpace from its serialized dictionary representation."""
        space = cls(name=d["name"], description=d.get("description", ""))
        for param_dict in d.get("parameters", []):
            space.add(Parameter.from_dict(param_dict))
        return space

    def save(self, path: str) -> None:
        """Persist this search space's parameters to a JSON file at the given path."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> SearchSpace:
        """Load a SearchSpace from a JSON file previously written by save()."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
