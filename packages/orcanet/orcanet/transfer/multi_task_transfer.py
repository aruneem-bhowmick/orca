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


class MultiTaskTransfer(TransferStrategy):
    """Transfer strategy that jointly trains a shared backbone across multiple tasks.

    Workflow
    --------
    1. Construct with a backbone and optional ``task_weighting`` scheme.
    2. Call :meth:`add_task` for each task (source and target at minimum).
    3. Optionally call :meth:`register_task_features` to enable similarity
       scoring in :meth:`score_transfer`.
    4. Call :meth:`execute_transfer` to obtain a :class:`MultiTaskModel`.

    Parameters
    ----------
    backbone:
        Shared feature extractor.  Its output dimensionality is inferred
        automatically from the last ``nn.Linear`` layer.
    task_weighting:
        ``"equal"`` — uniform ``1/n`` weights; ``"uncertainty"`` — learnable
        Kendall et al. 2018 log-variance per task; ``"gradnorm"`` — caller
        supplies per-task gradient norms via :meth:`update_gradnorm_weights`.
    task_head_hidden_dim:
        Hidden size of each task-specific two-layer head.
    embedder:
        Optional :class:`~orcanet.embeddings.cross_domain.CrossDomainEmbedder`
        used in :meth:`score_transfer`.  A default instance is created when
        not supplied.
    """

    def __init__(
        self,
        backbone: nn.Module,
        task_weighting: str = "equal",
        task_head_hidden_dim: int = 64,
        embedder: CrossDomainEmbedder | None = None,
    ) -> None:
        if task_weighting not in _VALID_WEIGHTINGS:
            raise ValueError(
                f"task_weighting must be one of {sorted(_VALID_WEIGHTINGS)}; "
                f"got {task_weighting!r}"
            )
        self.backbone = backbone
        self.task_weighting = task_weighting
        self.task_head_hidden_dim = task_head_hidden_dim
        self._backbone_out_dim: int = _get_backbone_out_dim(backbone)
        self._task_heads: dict[str, nn.Module] = {}
        self._task_weights: dict[str, float] = {}
        self._log_sigmas: dict[str, nn.Parameter] = {}
        self._task_features: dict[str, Tensor] = {}

        # Defer import to avoid circular dependency at module load time.
        if embedder is not None:
            self._embedder = embedder
        else:
            from orcanet.embeddings.cross_domain import CrossDomainEmbedder as _CDE
            self._embedder = _CDE()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def register_task_features(self, task_id: str, features: Tensor) -> None:
        """Store a meta-feature tensor for *task_id* used by :meth:`score_transfer`.

        Parameters
        ----------
        task_id:
            String task identifier.
        features:
            Tensor of shape ``(1, embedder_input_dim)`` or ``(embedder_input_dim,)``.
            Must match the embedder's expected input dimensionality (default 25).
        """
        self._task_features[str(task_id)] = features

    def add_task(self, task: Task, head_output_dim: int) -> None:
        """Register a new task and create its two-layer output head.

        Parameters
        ----------
        task:
            Task to register.  Its ``task_id`` is used as the head key.
        head_output_dim:
            Number of output units in the task head (typically ``n_classes``).
        """
        task_id = str(task.task_id)
        if task_id in self._task_heads:
            raise ValueError(f"Task {task_id!r} is already registered.")
        head = nn.Sequential(
            nn.Linear(self._backbone_out_dim, self.task_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.task_head_hidden_dim, head_output_dim),
        )
        self._task_heads[task_id] = head
        if self.task_weighting == "uncertainty":
            self._log_sigmas[task_id] = nn.Parameter(torch.zeros(1))
        self._update_weights()

    @property
    def task_weights(self) -> dict[str, float]:
        """Current per-task weight dictionary (a copy)."""
        return dict(self._task_weights)

    def update_gradnorm_weights(self, grad_norms: dict[str, float]) -> None:
        """Renormalise task weights from per-task gradient norms (GradNorm).

        Weights are set so that tasks with larger gradient norms receive
        proportionally higher weight, normalised to sum to 1.  The caller is
        responsible for computing ``grad_norms`` (e.g. via
        ``torch.autograd.grad``) on the backbone's final layer.

        Parameters
        ----------
        grad_norms:
            Maps each task id to its gradient norm scalar.  Missing task ids
            retain their existing weight.
        """
        if not grad_norms:
            return
        total = sum(grad_norms.values()) + 1e-8
        n = len(grad_norms)
        for tid, norm in grad_norms.items():
            self._task_weights[tid] = (norm / total) * n
        w_sum = sum(self._task_weights.values()) + 1e-8
        for tid in self._task_weights:
            self._task_weights[tid] /= w_sum

    # ------------------------------------------------------------------
    # TransferStrategy interface
    # ------------------------------------------------------------------

    def score_transfer(self, source: Task, target: Task) -> TransferScore:
        """Score joint-training compatibility via embedding cosine similarity.

        If task features have been registered for both tasks (via
        :meth:`register_task_features`), passes them through the embedder and
        computes the cosine similarity of their L2-normalised representations.
        Falls back to a neutral score of ``0.5`` when features are absent.

        Returns
        -------
        TransferScore
            ``overall`` is the clamped cosine similarity in ``[0, 1]``.
            ``reasoning`` uses the exact wording specified in the prompt when
            similarity exceeds the ``0.5`` threshold.
        """
        source_id = str(source.task_id)
        target_id = str(target.task_id)

        src_feat = self._task_features.get(source_id)
        tgt_feat = self._task_features.get(target_id)

        if src_feat is None or tgt_feat is None:
            return TransferScore(
                overall=0.5,
                layer_scores={},
                recommended_layers=[],
                reasoning="No task features registered for similarity computation.",
            )

        # Ensure (1, D) shape for the embedder.
        if src_feat.ndim == 1:
            src_feat = src_feat.unsqueeze(0)
        if tgt_feat.ndim == 1:
            tgt_feat = tgt_feat.unsqueeze(0)

        src_emb = self._embedder.embed(src_feat)   # L2-normalised (1, emb_dim)
        tgt_emb = self._embedder.embed(tgt_feat)   # L2-normalised (1, emb_dim)

        # Cosine similarity of L2-normalised vectors = dot product.
        similarity = float(torch.clamp(torch.sum(src_emb * tgt_emb), 0.0, 1.0).item())

        if similarity > 0.5:
            reasoning = (
                f"Multi-task training beneficial: similarity {similarity:.2f} > threshold 0.5"
            )
        else:
            reasoning = (
                f"Multi-task training marginal: similarity {similarity:.2f} <= threshold 0.5"
            )

        return TransferScore(
            overall=similarity,
            layer_scores={"cosine_similarity": similarity},
            recommended_layers=["backbone"] if similarity > 0.5 else [],
            reasoning=reasoning,
        )

    def execute_transfer(
        self,
        source: Task,
        target: Task,
        source_model: nn.Module,
    ) -> nn.Module:
        """Build a :class:`MultiTaskModel` covering both *source* and *target* tasks.

        Tasks not yet registered via :meth:`add_task` are auto-registered
        using ``task.n_classes`` (falling back to ``1``) as the head output
        dimension.

        Returns
        -------
        MultiTaskModel
            A model sharing ``self.backbone`` with a head per registered task.
        """
        for task in (source, target):
            task_id = str(task.task_id)
            if task_id not in self._task_heads:
                head_dim = int(task.n_classes) if task.n_classes is not None else 1
                self.add_task(task, head_dim)

        return MultiTaskModel(
            backbone=self.backbone,
            task_heads=dict(self._task_heads),
            task_weighting=self.task_weighting,
            log_sigmas=dict(self._log_sigmas) if self._log_sigmas else None,
        )

    def get_transfer_metadata(self) -> dict:
        """Return a dict describing this strategy's configuration."""
        return {
            "strategy": "multi_task_transfer",
            "task_weighting": self.task_weighting,
            "task_head_hidden_dim": self.task_head_hidden_dim,
            "n_registered_tasks": len(self._task_heads),
            "backbone_out_dim": self._backbone_out_dim,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_weights(self) -> None:
        """Recompute ``_task_weights`` from the current task set and weighting scheme."""
        n = len(self._task_heads)
        if n == 0:
            self._task_weights = {}
            return
        if self.task_weighting == "equal":
            self._task_weights = {tid: 1.0 / n for tid in self._task_heads}
        elif self.task_weighting == "uncertainty":
            # Actual weighting is embedded in compute_uncertainty_loss via log_sigmas.
            # _task_weights is a unit-weight placeholder for API consistency.
            self._task_weights = {tid: 1.0 for tid in self._task_heads}
        elif self.task_weighting == "gradnorm":
            # Initialise uniformly; caller updates via update_gradnorm_weights().
            self._task_weights = {tid: 1.0 / n for tid in self._task_heads}
