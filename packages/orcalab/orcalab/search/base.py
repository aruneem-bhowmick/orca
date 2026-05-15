"""Abstract base class for hyperparameter search strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from orcalab.search_spaces.space import SearchSpace


class SearchStrategy(ABC):
    @abstractmethod
    def suggest(self, search_space: SearchSpace) -> dict[str, Any]: ...

    @abstractmethod
    def update(self, params: dict[str, Any], result: float) -> None: ...

    @abstractmethod
    def get_best(self, n: int = 1) -> list[tuple[dict, float]]: ...

    @property
    @abstractmethod
    def n_trials(self) -> int: ...

    def get_history(self) -> list[tuple[dict, float]]:
        """Return all (params, result) pairs seen so far, sorted by result descending."""
        if self.n_trials == 0:
            return []
        return self.get_best(self.n_trials)
