"""Abstract base classes for meta-learning algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


@dataclass
class Task:
    """A single few-shot task with support and query splits."""

    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor


class MetaLearner(ABC):
    """Abstract base for all meta-learning algorithms."""

    @abstractmethod
    def inner_loop(
        self,
        support_x: Tensor,
        support_y: Tensor,
        num_steps: int | None = None,
    ) -> tuple[nn.Module, list[float]]:
        """Adapt the model to a support set; return (adapted_model, per-step losses)."""
        ...

    @abstractmethod
    def meta_update(self, task_batch: list[Task]) -> dict[str, float]:
        """Perform one outer-loop optimiser step over a batch of tasks; return training metrics."""
        ...

    @abstractmethod
    def adapt(self, support_x: Tensor, support_y: Tensor) -> nn.Module:
        """Return a task-adapted copy of the model without modifying the original."""
        ...

    @abstractmethod
    def evaluate_task(
        self,
        adapted_model: nn.Module,
        query_x: Tensor,
        query_y: Tensor,
    ) -> dict[str, float]:
        """Compute loss and accuracy on a query set with an already-adapted model."""
        ...
