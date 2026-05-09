"""Model-Agnostic Meta-Learning (MAML) — Finn et al. 2017."""

from __future__ import annotations

import copy
from typing import Callable

import higher
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import MetaLearner, Task


class MAML(MetaLearner):
    """MAML meta-learner supporting both second-order and first-order (FOMAML) variants.

    Args:
        model: Base model to meta-learn.
        inner_lr: Learning rate for inner-loop (task-level) adaptation.
        outer_lr: Learning rate for the outer meta-optimizer (Adam).
        inner_steps: Number of gradient steps in the inner loop.
        first_order: If True, use FOMAML (skip second-order gradients).
        loss_fn: Loss function applied to model predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = False,
        loss_fn: Callable = F.cross_entropy,
    ) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.loss_fn = loss_fn
        self._outer_opt = torch.optim.Adam(model.parameters(), lr=outer_lr)

    # ------------------------------------------------------------------
    # MetaLearner interface
    # ------------------------------------------------------------------

    def inner_loop(
        self,
        support_x: Tensor,
        support_y: Tensor,
        num_steps: int | None = None,
    ) -> tuple[nn.Module, list[float]]:
        """Adapt the model to a support set; return (adapted_model, inner_losses).

        Uses higher's innerloop_ctx so the returned functional model retains the
        adaptation computation graph when first_order=False.
        """
        steps = num_steps if num_steps is not None else self.inner_steps
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        losses: list[float] = []

        with higher.innerloop_ctx(
            self.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
        ) as (fmodel, diffopt):
            for _ in range(steps):
                pred = fmodel(support_x)
                loss = self.loss_fn(pred, support_y)
                losses.append(loss.item())
                diffopt.step(loss)

        return fmodel, losses

    def meta_update(self, task_batch: list[Task]) -> dict[str, float]:
        """Run one outer-loop update over a batch of tasks.

        The inner loop context is kept open during query-loss computation so
        that second-order gradients flow correctly back to self.model when
        first_order=False.  For first_order=True (FOMAML), first-order
        meta-gradients are computed manually and accumulated on self.model.
        """
        if not task_batch:
            return {"meta_train_loss": 0.0, "meta_train_accuracy": 0.0}

        self.model.train()
        self._outer_opt.zero_grad()

        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        n_tasks = len(task_batch)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for task in task_batch:
            with higher.innerloop_ctx(
                self.model,
                inner_opt,
                copy_initial_weights=False,
                track_higher_grads=not self.first_order,
            ) as (fmodel, diffopt):
                for _ in range(self.inner_steps):
                    inner_pred = fmodel(task.support_x)
                    inner_loss = self.loss_fn(inner_pred, task.support_y)
                    diffopt.step(inner_loss)

                query_pred = fmodel(task.query_x)
                task_loss = self.loss_fn(query_pred, task.query_y)
                total_loss += task_loss.item()

                if self.first_order:
                    fo_grads = torch.autograd.grad(task_loss, list(fmodel.parameters()))
                    for orig_p, grad in zip(self.model.parameters(), fo_grads):
                        scaled = grad.detach() / n_tasks
                        orig_p.grad = scaled if orig_p.grad is None else orig_p.grad + scaled
                else:
                    (task_loss / n_tasks).backward()

                with torch.no_grad():
                    n = task.query_y.size(0)
                    total_samples += n
                    if query_pred.dim() > 1 and query_pred.size(1) > 1:
                        total_correct += (query_pred.argmax(dim=1) == task.query_y).sum().item()

        self._outer_opt.step()

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {
            "meta_train_loss": total_loss / n_tasks,
            "meta_train_accuracy": accuracy,
        }

    def adapt(self, support_x: Tensor, support_y: Tensor) -> nn.Module:
        """Return a task-adapted copy of the model without tracking gradients.

        Does not modify self.model.
        """
        adapted = copy.deepcopy(self.model)
        adapted.train()
        inner_opt = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            inner_opt.zero_grad()
            loss = self.loss_fn(adapted(support_x), support_y)
            loss.backward()
            inner_opt.step()
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
            result["accuracy"] = 0.0
        return result
