"""Weight-based transfer strategy.

Matches parameter tensors between source and target models by name, shape,
or both; copies matched weights; reinitialises unmatched ones; and provides
an Adam optimizer with per-layer LR decay for transferred parameters.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from orca_shared.schemas.task import Task

from .base import TransferStrategy
from .types import TransferScore


class WeightTransfer(TransferStrategy):
    """Transfer strategy that copies matched parameter tensors from source to target.

    Workflow
    --------
    1. Register source and target ``nn.Module`` instances via
       :meth:`register_model` (keyed by ``str(task_id)``).
    2. Call :meth:`score_transfer` to obtain a :class:`TransferScore` with
       per-parameter match flags and an overall match ratio.
    3. Call :meth:`execute_transfer` to produce an adapted model; the second
       return value is the list of transferred parameter names, which can be
       passed directly to :func:`get_optimizer_with_layer_lr`.
    """

    def __init__(
        self,
        match_by: str = "name",
        frozen_epochs: int = 5,
        layer_lr_decay: float = 0.1,
    ) -> None:
        """Initialise a WeightTransfer strategy.

        Parameters
        ----------
        match_by:
            Matching criterion — ``"name"`` (parameter name only),
            ``"shape"`` (tensor shape only), or ``"both"`` (name and shape).
        frozen_epochs:
            Number of epochs to freeze transferred layers after transfer
            (stored for downstream use; not enforced internally).
        layer_lr_decay:
            Default decay factor used by :func:`get_optimizer_with_layer_lr`
            when called without an explicit ``decay`` argument.
        """
        if match_by not in ("name", "shape", "both"):
            raise ValueError(f"match_by must be 'name', 'shape', or 'both'; got {match_by!r}")
        self.match_by = match_by
        self.frozen_epochs = frozen_epochs
        self.layer_lr_decay = layer_lr_decay
        self._model_registry: dict[str, nn.Module] = {}

    def register_model(self, task_id: str, model: nn.Module) -> None:
        """Register an ``nn.Module`` for weight scoring under *task_id*."""
        self._model_registry[str(task_id)] = model

    def _matches(self, name: str, param: torch.Tensor, source_state: dict) -> bool:
        """Return True if *param* (named *name*) has a candidate in *source_state*."""
        if self.match_by == "name":
            return name in source_state
        if self.match_by == "shape":
            return any(v.shape == param.shape for v in source_state.values())
        # "both"
        return name in source_state and source_state[name].shape == param.shape

    def _find_source_tensor(
        self,
        name: str,
        param: torch.Tensor,
        source_state: dict,
    ) -> torch.Tensor | None:
        """Resolve the source tensor that should be copied into *param*.

        Returns ``None`` when no safe copy is possible (shape mismatch or no
        matching tensor found), so the caller can reinitialise instead.
        """
        if self.match_by == "shape":
            return next(
                (v for v in source_state.values() if v.shape == param.shape),
                None,
            )
        # "name" or "both": require same name AND matching shape
        src = source_state.get(name)
        if src is not None and src.shape == param.shape:
            return src
        return None

    def score_transfer(self, source: Task, target: Task) -> TransferScore:
        """Compute a weight-level transferability score between two registered models.

        Raises
        ------
        ValueError
            If models have not been registered for both task IDs.
        """
        source_id = str(source.task_id)
        target_id = str(target.task_id)

        if source_id not in self._model_registry or target_id not in self._model_registry:
            raise ValueError(
                f"Models not registered. Call register_model() for task IDs "
                f"'{source_id}' and '{target_id}' before scoring."
            )

        source_state = self._model_registry[source_id].state_dict()
        target_state = self._model_registry[target_id].state_dict()

        layer_scores: dict[str, float] = {}
        for name, param in target_state.items():
            layer_scores[name] = 1.0 if self._matches(name, param, source_state) else 0.0

        n_total = len(layer_scores)
        n_matched = sum(1 for v in layer_scores.values() if v == 1.0)
        overall = n_matched / n_total if n_total > 0 else 0.0
        recommended_layers = [n for n, s in layer_scores.items() if s == 1.0]
        reasoning = f"Matched {n_matched}/{n_total} layers by {self.match_by}"

        return TransferScore(
            overall=overall,
            layer_scores=layer_scores,
            recommended_layers=recommended_layers,
            reasoning=reasoning,
        )

    def execute_transfer(
        self,
        source: Task,
        target: Task,
        source_model: nn.Module,
    ) -> nn.Module:
        """Placeholder — implemented in the weight-execution commit."""
        raise NotImplementedError

    def get_transfer_metadata(self) -> dict:
        """Return a dict of strategy configuration for logging and inspection."""
        return {
            "strategy": "weight_transfer",
            "match_by": self.match_by,
            "frozen_epochs": self.frozen_epochs,
            "layer_lr_decay": self.layer_lr_decay,
        }
