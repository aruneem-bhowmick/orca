"""Meta-SGD — Li et al. 2017.  Per-parameter learnable inner learning rates."""

from __future__ import annotations

import copy
from typing import Callable

import higher
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import MetaLearner, Task


class MetaSGD(nn.Module, MetaLearner):
    """Meta-SGD meta-learner (Li et al. 2017).

    Extends MAML by making the inner-loop learning rates learnable: each parameter
    element has its own α co-optimised with the model weights during meta-training.
    The inner update is p ← p − α_p * grad_p (element-wise multiplication).

    Args:
        model: Base model to meta-learn.
        outer_lr: Learning rate for the outer Adam meta-optimizer.
        inner_steps: Number of gradient steps in the inner loop.
        loss_fn: Loss function applied to model predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        loss_fn: Callable = F.cross_entropy,
    ) -> None:
        super().__init__()
        self.model = model
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.loss_fn = loss_fn

        # One learnable per-element learning rate per model parameter
        self.lrs: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(torch.ones_like(p) * 0.01) for p in model.parameters()]
        )

        # Outer optimizer updates both model weights and the learnable rates
        self._outer_opt = torch.optim.Adam(
            list(model.parameters()) + list(self.lrs.parameters()),
            lr=outer_lr,
        )

    # ------------------------------------------------------------------
    # MetaLearner interface
    # ------------------------------------------------------------------

    def inner_loop(
        self,
        support_x: Tensor,
        support_y: Tensor,
        num_steps: int | None = None,
    ) -> tuple[nn.Module, list[float]]:
        """Adapt the model using element-wise learnable rates; return (fmodel, inner_losses).

        Uses higher to keep the computation graph intact for meta-gradient flow.
        The per-element update p ← p − α_p * grad_p is applied manually because
        higher's diffopt only supports scalar per-group learning rates.
        """
        steps = num_steps if num_steps is not None else self.inner_steps
        # The base lr of this optimizer is unused — updates are applied manually.
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        losses: list[float] = []

        with higher.innerloop_ctx(
            self.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
        ) as (fmodel, _diffopt):
            for _ in range(steps):
                pred = fmodel(support_x)
                loss = self.loss_fn(pred, support_y)
                losses.append(loss.item())

                grads = torch.autograd.grad(
                    loss,
                    list(fmodel.fast_params),
                    create_graph=True,
                    allow_unused=True,
                )
                fmodel.fast_params = [
                    fp - lr * g if g is not None else fp
                    for fp, lr, g in zip(fmodel.fast_params, self.lrs, grads)
                ]

        return fmodel, losses

    def meta_update(self, task_batch: list[Task]) -> dict[str, float]:
        """Perform one outer-loop update over a batch of tasks.

        Both model parameters and per-parameter learning rates are updated by
        the outer Adam optimizer, so gradients must flow through the inner loop.
        """
        if not task_batch:
            return {"meta_train_loss": 0.0, "meta_train_accuracy": 0.0}

        self.model.train()
        self._outer_opt.zero_grad()

        inner_opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        n_tasks = len(task_batch)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for task in task_batch:
            with higher.innerloop_ctx(
                self.model,
                inner_opt,
                copy_initial_weights=False,
                track_higher_grads=True,
            ) as (fmodel, _diffopt):
                for _ in range(self.inner_steps):
                    inner_pred = fmodel(task.support_x)
                    inner_loss = self.loss_fn(inner_pred, task.support_y)
                    grads = torch.autograd.grad(
                        inner_loss,
                        list(fmodel.fast_params),
                        create_graph=True,
                        allow_unused=True,
                    )
                    fmodel.fast_params = [
                        fp - lr * g if g is not None else fp
                        for fp, lr, g in zip(fmodel.fast_params, self.lrs, grads)
                    ]

                query_pred = fmodel(task.query_x)
                task_loss = self.loss_fn(query_pred, task.query_y)
                (task_loss / n_tasks).backward()

                total_loss += task_loss.item()
                with torch.no_grad():
                    n = task.query_y.size(0)
                    total_samples += n
                    if query_pred.dim() > 1 and query_pred.size(1) > 1:
                        total_correct += (
                            query_pred.argmax(dim=1) == task.query_y
                        ).sum().item()

        self._outer_opt.step()

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {
            "meta_train_loss": total_loss / n_tasks,
            "meta_train_accuracy": accuracy,
        }

    def adapt(self, support_x: Tensor, support_y: Tensor) -> nn.Module:
        """Return a task-adapted copy of the model without modifying the original.

        Applies the element-wise per-parameter update using the current (trained)
        learning rates, detached so no computation graph is built.
        """
        adapted = copy.deepcopy(self.model)
        adapted.train()
        adapted_lrs = [lr.detach() for lr in self.lrs]

        for _ in range(self.inner_steps):
            adapted.zero_grad()
            loss = self.loss_fn(adapted(support_x), support_y)
            loss.backward()
            with torch.no_grad():
                for p, lr in zip(adapted.parameters(), adapted_lrs):
                    if p.grad is not None:
                        p.data.sub_(lr * p.grad)

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
