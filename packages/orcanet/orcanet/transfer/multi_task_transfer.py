"""Multi-task transfer strategy.

Joint training across multiple related tasks using a shared backbone and
task-specific heads.  Task compatibility is scored via cosine similarity of
CrossDomainEmbedder embeddings, and three task-weighting schemes are supported:
equal, uncertainty (Kendall et al. 2018), and gradnorm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from orca_shared.schemas.task import Task

from .base import TransferStrategy
from .types import TransferScore

if TYPE_CHECKING:
    from orcanet.embeddings.cross_domain import CrossDomainEmbedder

_VALID_WEIGHTINGS = frozenset({"equal", "uncertainty", "gradnorm"})


def _get_backbone_out_dim(backbone: nn.Module) -> int:
    """Return the output dimension of *backbone* by inspecting its last ``nn.Linear``.

    Raises
    ------
    ValueError
        If *backbone* contains no ``nn.Linear`` layers.
    """
    last_linear: nn.Linear | None = None
    for module in backbone.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError(
            "Cannot infer backbone output dim: no nn.Linear found in backbone."
        )
    return last_linear.out_features


class MultiTaskModel(nn.Module):
    """Shared backbone with per-task output heads.

    Parameters
    ----------
    backbone:
        Shared feature extractor used for all tasks.
    task_heads:
        Mapping from task id (str) to the task-specific output head.
    task_weighting:
        The weighting scheme in use.  Stored for reference; does not affect
        ``forward`` — only ``compute_loss`` / ``compute_uncertainty_loss``.
    log_sigmas:
        Per-task learnable log-variance parameters (uncertainty weighting).
        Pass an empty dict or ``None`` when not using uncertainty weighting.
    """

    def __init__(
        self,
        backbone: nn.Module,
        task_heads: dict[str, nn.Module],
        task_weighting: str = "equal",
        log_sigmas: dict[str, nn.Parameter] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # nn.ModuleDict ensures heads are registered as submodules (parameters tracked).
        self.task_heads = nn.ModuleDict(task_heads)
        self.task_weighting = task_weighting
        # nn.ParameterDict ensures log_sigma params are included in model.parameters().
        self.log_sigmas = nn.ParameterDict(log_sigmas or {})

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        """Extract shared features then route to the head for *task_id*.

        Raises
        ------
        KeyError
            If *task_id* was not registered as a task head.
        """
        features = self.backbone(x)
        return self.task_heads[task_id](features)

    def compute_loss(
        self,
        batch: dict[str, tuple[Tensor, Tensor]],
        weights: dict[str, float],
    ) -> Tensor:
        """Weighted sum of per-task cross-entropy losses.

        Parameters
        ----------
        batch:
            Maps each task id to an ``(x, y)`` tuple of inputs and integer
            class labels.
        weights:
            Per-task scalar multipliers.  Typically produced by
            ``MultiTaskTransfer.task_weights``.

        Returns
        -------
        Tensor
            Scalar loss tensor suitable for ``loss.backward()``.
        """
        losses = [
            weights[tid] * F.cross_entropy(self(x, tid), y)
            for tid, (x, y) in batch.items()
        ]
        return torch.stack(losses).sum()

    def compute_uncertainty_loss(
        self,
        batch: dict[str, tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Uncertainty-weighted multi-task loss (Kendall et al. 2018).

        For each task *i* with learnable ``log_sigma_i``:

            L_total = sum_i( exp(-2 * log_s_i) * CE_i + log_s_i )

        The ``exp(-2 * log_s_i)`` term down-weights tasks with high learned
        noise; ``log_s_i`` regularises against trivially large sigmas.

        Parameters
        ----------
        batch:
            Maps each task id to ``(x, y)``.

        Returns
        -------
        Tensor
            Scalar loss tensor.  Gradients flow into ``log_sigmas`` so they
            are updated by a standard optimiser step.
        """
        losses: list[Tensor] = []
        for tid, (x, y) in batch.items():
            log_s = self.log_sigmas[tid]
            task_loss = F.cross_entropy(self(x, tid), y)
            losses.append(torch.exp(-2.0 * log_s) * task_loss + log_s)
        return torch.stack(losses).sum().squeeze()
