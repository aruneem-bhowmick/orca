"""Abstract base class for trial pruning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Pruner(ABC):
    @abstractmethod
    def should_prune(
        self,
        trial_id: str,
        step: int,
        current_value: float,
        all_trial_values: dict[str, list[float]],
    ) -> bool: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
