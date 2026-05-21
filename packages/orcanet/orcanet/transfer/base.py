"""Abstract base class for all OrcaNet transfer strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

from orca_shared.schemas.task import Task

from .types import TransferScore


class TransferStrategy(ABC):
    """Contract that every knowledge-transfer strategy must satisfy."""

    @abstractmethod
    def score_transfer(self, source: Task, target: Task) -> TransferScore:
        """Compute a transferability score between *source* and *target* tasks."""
        ...

    @abstractmethod
    def execute_transfer(
        self,
        source: Task,
        target: Task,
        source_model: nn.Module,
    ) -> nn.Module:
        """Transfer knowledge from *source_model* and return an adapted model for the target task."""
        ...

    @abstractmethod
    def get_transfer_metadata(self) -> dict:
        """Return a dict describing this strategy's configuration."""
        ...
