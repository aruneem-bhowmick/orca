"""On First-Order Meta-Learning Algorithms (Reptile) — Nichol et al. 2018."""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import MetaLearner, Task


class Reptile(MetaLearner):
    """Reptile meta-learner (Nichol et al. 2018).

    Simpler than MAML: no second-order gradients, no differentiable inner loop.
    The outer update interpolates meta-parameters toward task-adapted parameters:
    θ ← θ + ε * (φ − θ), where φ are the adapted parameters and ε is outer_lr.

    Args:
        model: Base model to meta-learn.
        inner_lr: Learning rate for inner-loop SGD adaptation.
        outer_lr: Interpolation factor ε for the Reptile outer update.
        inner_steps: Number of gradient steps in the inner loop.
        loss_fn: Loss function applied to model predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.02,
        outer_lr: float = 0.1,
        inner_steps: int = 10,
        loss_fn: Callable = F.cross_entropy,
    ) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.loss_fn = loss_fn

    # ------------------------------------------------------------------
    # MetaLearner interface
    # ------------------------------------------------------------------

    def inner_loop(
        self,
        support_x: Tensor,
        support_y: Tensor,
        num_steps: int | None = None,
    ) -> tuple[nn.Module, list[float]]:
        """Adapt a copy of the model to a support set using standard SGD.

        Reptile does not require a differentiable inner loop, so no higher
        context is needed — the adapted copy is returned as a plain nn.Module.
        """
        steps = num_steps if num_steps is not None else self.inner_steps
        adapted = copy.deepcopy(self.model)
        adapted.train()
        inner_opt = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
        losses: list[float] = []

        for _ in range(steps):
            inner_opt.zero_grad()
            pred = adapted(support_x)
            loss = self.loss_fn(pred, support_y)
            losses.append(loss.item())
            loss.backward()
            inner_opt.step()

        return adapted, losses

    def meta_update(self, task_batch: list[Task]) -> dict[str, float]:
        """Perform sequential Reptile updates over a batch of tasks.

        For each task the model is adapted via inner_loop, then meta-parameters
        are interpolated toward the adapted parameters: θ ← θ + ε * (φ − θ).
        """
        if not task_batch:
            return {"meta_train_loss": 0.0, "meta_train_accuracy": float("nan")}

        self.model.train()
        n_tasks = len(task_batch)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        is_classification = False

        for task in task_batch:
            adapted, _ = self.inner_loop(task.support_x, task.support_y)

            # Reptile interpolation: θ ← θ + ε * (φ − θ)
            with torch.no_grad():
                for p, phi in zip(self.model.parameters(), adapted.parameters(), strict=True):
                    p.data.add_(self.outer_lr * (phi.data - p.data))

            # Query metrics from the adapted model (no-grad, eval only)
            adapted.eval()
            with torch.no_grad():
                query_pred = adapted(task.query_x)
                task_loss = self.loss_fn(query_pred, task.query_y)
                total_loss += task_loss.item()
                n = task.query_y.size(0)
                total_samples += n
                if query_pred.dim() > 1 and query_pred.size(1) > 1:
                    is_classification = True
                    total_correct += (query_pred.argmax(dim=1) == task.query_y).sum().item()

        if is_classification and total_samples > 0:
            accuracy: float = total_correct / total_samples
        else:
            accuracy = float("nan")
        return {
            "meta_train_loss": total_loss / n_tasks,
            "meta_train_accuracy": accuracy,
        }

    def adapt(self, support_x: Tensor, support_y: Tensor) -> nn.Module:
        """Return a task-adapted copy of the model without modifying the original."""
        adapted, _ = self.inner_loop(support_x, support_y)
        adapted.eval()
        return adapted

    def evaluate_task(
        self,
        adapted_model: nn.Module,
        query_x: Tensor,
        query_y: Tensor,
    ) -> dict[str, float]:
        """Compute loss (and accuracy for classification) on a query set."""
        adapted_model.eval()
        with torch.no_grad():
            pred = adapted_model(query_x)
            loss = self.loss_fn(pred, query_y)

        result: dict[str, float] = {"loss": loss.item()}
        if pred.dim() > 1 and pred.size(1) > 1:
            result["accuracy"] = (pred.argmax(dim=1) == query_y).float().mean().item()
        else:
            result["accuracy"] = float("nan")
        return result
