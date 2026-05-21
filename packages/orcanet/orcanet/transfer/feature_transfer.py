"""CKA-based feature transfer strategy.

Uses Centered Kernel Alignment (Kornblith et al. 2019) to identify which
feature layers are most transferable between a source and target task.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from orca_shared.schemas.task import Task

from .base import TransferStrategy
from .types import TransferScore

if TYPE_CHECKING:
    from orca_shared.clients.orcamind_client import OrcaMindClient


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between feature matrices X (n×p) and Y (n×q).

    CKA = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)

    where the subscript *c* denotes column-mean centering.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro")
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro")

    # Clamp to [0, 1] to absorb floating-point noise around the theoretical bounds
    return float(min(hsic_xy / (hsic_xx * hsic_yy + 1e-8), 1.0))


class FeatureTransfer(TransferStrategy):
    """Transfer strategy that scores transferability with per-layer CKA.

    Workflow
    --------
    1. Register source and target ``nn.Module`` instances via
       :meth:`register_model` (keyed by ``str(task_id)``).
    2. Provide a ``probe_data`` array (``n_samples × input_dim``) shared
       by both models.
    3. Call :meth:`score_transfer` to obtain a :class:`TransferScore` with
       per-layer CKA values and weighted-mean overall score.

    An optional ``OrcaMindClient`` is accepted and stored for future
    registry-backed model resolution; it is not used in the current scoring
    path.
    """

    def __init__(
        self,
        orcamind_client: OrcaMindClient | None = None,
        probe_data: np.ndarray | None = None,
        cka_threshold: float = 0.5,
    ) -> None:
        self.orcamind_client = orcamind_client
        self.probe_data = probe_data
        self.cka_threshold = cka_threshold
        self._model_registry: dict[str, nn.Module] = {}

    def register_model(self, task_id: str, model: nn.Module) -> None:
        """Register an ``nn.Module`` for transfer scoring under *task_id*."""
        self._model_registry[str(task_id)] = model

    def _collect_activations(
        self, model: nn.Module, data: torch.Tensor
    ) -> dict[str, np.ndarray]:
        """Run *data* through *model* and return per-named-module output activations."""
        activations: dict[str, np.ndarray] = {}
        hooks: list = []

        def _make_hook(name: str):
            def _hook(module, input, output):  # noqa: ARG001
                if isinstance(output, torch.Tensor) and output.ndim >= 2:
                    activations[name] = output.detach().cpu().numpy()

            return _hook

        for name, module in model.named_modules():
            if name:  # skip the root wrapper itself
                hooks.append(module.register_forward_hook(_make_hook(name)))

        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(data)
        model.train(was_training)

        for h in hooks:
            h.remove()

        return activations

    def score_transfer(self, source: Task, target: Task) -> TransferScore:
        """Compute CKA-based layer-wise transferability between two registered models.

        Raises
        ------
        ValueError
            If models have not been registered for both task IDs, or if no
            ``probe_data`` has been supplied.
        """
        source_id = str(source.task_id)
        target_id = str(target.task_id)

        if source_id not in self._model_registry or target_id not in self._model_registry:
            raise ValueError(
                f"Models not registered. Call register_model() for task IDs "
                f"'{source_id}' and '{target_id}' before scoring."
            )
        if self.probe_data is None:
            raise ValueError(
                "probe_data must be provided to collect activations for CKA computation."
            )

        source_model = self._model_registry[source_id]
        target_model = self._model_registry[target_id]
        probe_tensor = torch.tensor(self.probe_data, dtype=torch.float32)

        source_acts = self._collect_activations(source_model, probe_tensor)
        target_acts = self._collect_activations(target_model, probe_tensor)

        common_layers = sorted(set(source_acts.keys()) & set(target_acts.keys()))

        layer_scores: dict[str, float] = {}
        for name in common_layers:
            src = source_acts[name]
            tgt = target_acts[name]
            # Flatten spatial / sequence dims → (batch, features)
            if src.ndim > 2:
                src = src.reshape(src.shape[0], -1)
            if tgt.ndim > 2:
                tgt = tgt.reshape(tgt.shape[0], -1)
            if src.shape[0] > 1 and tgt.shape[0] == src.shape[0]:
                layer_scores[name] = linear_cka(src, tgt)

        if not layer_scores:
            return TransferScore(
                overall=0.0,
                layer_scores={},
                recommended_layers=[],
                reasoning="No comparable layers found between source and target models.",
            )

        # Shallower layers get higher weight (deeper layers weighted less)
        names = list(layer_scores.keys())
        weights = np.array([1.0 / (i + 1) for i in range(len(names))])
        weights /= weights.sum()
        scores = np.array([layer_scores[n] for n in names])
        overall = float(np.dot(weights, scores))

        recommended_layers = [n for n, s in layer_scores.items() if s > self.cka_threshold]

        n_rec = len(recommended_layers)
        n_tot = len(layer_scores)
        reasoning = (
            f"CKA analysis: {n_rec}/{n_tot} layers exceed threshold "
            f"{self.cka_threshold:.1f}. Weighted overall CKA: {overall:.3f}."
        )

        return TransferScore(
            overall=overall,
            layer_scores=layer_scores,
            recommended_layers=recommended_layers,
            reasoning=reasoning,
        )

    def execute_transfer(
        self, source: Task, target: Task, source_model: nn.Module
    ) -> nn.Module:
        """Copy weights from CKA-recommended layers into a clone of *source_model*."""
        score = self.score_transfer(source, target)
        target_model = deepcopy(source_model)
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()

        for param_name, param in target_state.items():
            # Derive the layer name by stripping the trailing ".weight" / ".bias"
            layer_name = ".".join(param_name.split(".")[:-1])
            if (
                layer_name in score.recommended_layers
                and param_name in source_state
                and source_state[param_name].shape == param.shape
            ):
                target_state[param_name].copy_(source_state[param_name])

        target_model.load_state_dict(target_state)
        return target_model

    def get_transfer_metadata(self) -> dict:
        return {
            "strategy": "feature_cka",
            "cka_threshold": self.cka_threshold,
            "n_registered_models": len(self._model_registry),
            "has_probe_data": self.probe_data is not None,
        }
