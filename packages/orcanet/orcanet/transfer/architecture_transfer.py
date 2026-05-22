"""Architecture-based transfer strategy.

Recommends and adapts model architectures for a target domain by comparing
architecture graph embeddings produced by ArchitectureEmbedder.  Uses the
OrcaMind service to retrieve the source task's best-known architecture, then
scores all locally registered candidate configs by cosine similarity and
returns the best match.
"""

from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from orca_shared.schemas.task import Task

from .base import TransferStrategy
from .types import TransferScore

if TYPE_CHECKING:
    from orcanet.embeddings.architecture_embedder import ArchitectureEmbedder
    from orca_shared.clients.orcamind_client import OrcaMindClient

# Architecture config dict: {"input_dim": int, "layers": [{"type": str, "size": int, "activation": str}, ...]}
ArchConfig = dict[str, Any]

_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def adapt_architecture(config: ArchConfig, target_task: Task) -> ArchConfig:
    """Return a copy of *config* with input/output dimensions updated for *target_task*.

    Only the first layer's input size and the last layer's output size change;
    all hidden (middle) layers are left untouched.

    Parameters
    ----------
    config:
        Architecture config dict with ``"input_dim"`` and ``"layers"`` keys.
    target_task:
        Task whose ``n_features`` (input) and ``n_classes`` (output) define
        the new dimensions.

    Returns
    -------
    ArchConfig
        A deep-copied config with updated ``input_dim`` and last-layer ``size``.
    """
    adapted = deepcopy(config)
    layers: list[dict[str, Any]] = adapted.get("layers", [])

    if target_task.n_features is not None:
        adapted["input_dim"] = int(target_task.n_features)

    if layers and target_task.n_classes is not None:
        last = dict(layers[-1])
        last["size"] = int(target_task.n_classes)
        adapted["layers"] = list(layers[:-1]) + [last]

    return adapted


def _build_sequential_from_config(config: ArchConfig) -> nn.Sequential:
    """Build an ``nn.Sequential`` model from an architecture config dict.

    Parameters
    ----------
    config:
        Dict with ``"input_dim": int`` and ``"layers": list``.  Each layer
        entry must have ``"size"`` (output width) and optionally ``"type"``
        (default ``"linear"``) and ``"activation"`` (default ``"none"``).

    Returns
    -------
    nn.Sequential
        A sequential model whose first ``nn.Linear`` maps from ``input_dim``
        and whose last ``nn.Linear`` maps to the final layer's ``size``.
    """
    layers_spec: list[dict[str, Any]] = config.get("layers", [])
    current_in = int(config.get("input_dim", 64))

    modules: list[nn.Module] = []
    for layer_spec in layers_spec:
        layer_type = str(layer_spec.get("type", "linear")).lower()
        out_size = int(layer_spec.get("size", 64))
        activation = str(layer_spec.get("activation", "none")).lower()

        if layer_type == "linear":
            modules.append(nn.Linear(current_in, out_size))

        if activation in _ACTIVATION_MAP:
            modules.append(_ACTIVATION_MAP[activation]())

        current_in = out_size

    return nn.Sequential(*modules)


def _run_coro(coro) -> Any:
    """Run *coro* synchronously, even when called from within an event loop.

    Uses a dedicated background thread when a running loop is detected so that
    the caller's loop is not re-entered.
    """
    try:
        asyncio.get_running_loop()
        # Already inside a loop — run in a fresh thread with its own loop.
        result: list[Any] = []
        exc: list[BaseException] = []

        def _target():
            try:
                result.append(asyncio.run(coro))
            except BaseException as e:  # noqa: BLE001
                exc.append(e)

        t = threading.Thread(target=_target)
        t.start()
        t.join()
        if exc:
            raise exc[0]
        return result[0]
    except RuntimeError:
        # No running loop — safe to use asyncio.run directly.
        return asyncio.run(coro)


class ArchitectureTransfer(TransferStrategy):
    """Transfer strategy that recommends architectures via graph-embedding similarity.

    Workflow
    --------
    1. Register candidate architecture configs with :meth:`register_config`.
    2. Call :meth:`score_transfer` to find the candidate most similar to the
       source task's best-known architecture (retrieved from OrcaMind).
    3. Call :meth:`execute_transfer` to build and return an adapted model whose
       input/output dimensions match the target task.

    Parameters
    ----------
    architecture_embedder:
        Computes cosine similarity between two architecture config dicts.
    orcamind_client:
        Async client used to retrieve the source task's best model (name only;
        the full config is looked up from the local registry).
    top_k_candidates:
        Stored for metadata reporting; all registered configs are always scored.
    """

    def __init__(
        self,
        architecture_embedder: ArchitectureEmbedder,
        orcamind_client: OrcaMindClient,
        top_k_candidates: int = 10,
    ) -> None:
        """Initialise the strategy with an embedder, client, and optional top-k config.

        Parameters
        ----------
        architecture_embedder:
            ``ArchitectureEmbedder`` instance that computes cosine similarity
            between two architecture config dicts via graph-embedding dot product.
        orcamind_client:
            Async ``OrcaMindClient`` used to resolve the source task's best model
            name from the OrcaMind registry.
        top_k_candidates:
            Maximum number of candidates to report; stored for metadata only —
            all registered configs are always scored and the best is selected.
        """
        self.architecture_embedder = architecture_embedder
        self.orcamind_client = orcamind_client
        self.top_k_candidates = top_k_candidates
        self._config_registry: dict[str, ArchConfig] = {}
        self._last_best_match: tuple[str, ArchConfig] | None = None

    def register_config(self, name: str, config: ArchConfig) -> None:
        """Register a named architecture config for similarity scoring."""
        self._config_registry[name] = config

    # ------------------------------------------------------------------
    # TransferStrategy interface
    # ------------------------------------------------------------------

    def score_transfer(self, source: Task, target: Task) -> TransferScore:
        """Score architecture transferability by comparing graph embeddings.

        Calls OrcaMind to identify the source task's best architecture name,
        then scores every registered candidate config against it using the
        architecture embedder.  The highest similarity becomes ``overall``.
        """
        return _run_coro(self._score_transfer_async(source, target))

    async def _score_transfer_async(
        self, source: Task, target: Task
    ) -> TransferScore:
        """Async implementation delegated to by :meth:`score_transfer`.

        Fetches the source task's best architecture name from OrcaMind, then
        scores every registered candidate config against it using cosine
        similarity from the architecture embedder.

        Parameters
        ----------
        source:
            Task whose best-known model architecture is looked up via OrcaMind.
        target:
            Target task; reserved for future task-conditioned similarity — not
            used in the current scoring path.

        Returns
        -------
        TransferScore
            ``overall`` is the highest per-candidate cosine similarity (clamped
            to 0); ``layer_scores`` maps each registered config name to its
            similarity value; ``recommended_layers`` is always ``[]`` because
            architecture transfer is a model-level rather than layer-level
            decision.
        """
        source_summary = await self.orcamind_client.get_best_model(source.task_id)
        source_config = self._config_registry.get(
            source_summary.name or "", {"layers": []}
        )

        if not self._config_registry:
            self._last_best_match = (source_summary.name or "", source_config)
            return TransferScore(
                overall=0.0,
                layer_scores={},
                recommended_layers=[],
                reasoning="No candidate architecture configs registered.",
            )

        best_sim = -1.0
        best_name = source_summary.name or ""
        best_config: ArchConfig = source_config
        layer_scores: dict[str, float] = {}

        for name, cand_config in self._config_registry.items():
            sim = float(
                self.architecture_embedder.similarity(source_config, cand_config)
            )
            layer_scores[name] = sim
            if sim > best_sim:
                best_sim = sim
                best_name = name
                best_config = cand_config

        self._last_best_match = (best_name, best_config)

        return TransferScore(
            overall=max(0.0, best_sim),
            layer_scores=layer_scores,
            recommended_layers=[],
            reasoning=f"Architecture {best_name} is most similar to source architecture",
        )

    def execute_transfer(
        self,
        source: Task,
        target: Task,
        source_model: nn.Module,
    ) -> nn.Module:
        """Build and return an architecture-adapted model for the target task.

        1. If ``score_transfer`` has not been called yet, calls it now to
           determine the best matching candidate config.
        2. Adapts the best config to the target task's input/output dims.
        3. Initialises all ``nn.Linear`` layers with ``kaiming_uniform_``
           (weights) and ``zeros_`` (biases).
        4. Copies middle-layer (not first, not last ``nn.Linear``) parameters
           from *source_model* where shapes match.
        """
        if self._last_best_match is None:
            self.score_transfer(source, target)
        assert self._last_best_match is not None  # guaranteed by score_transfer

        _, best_config = self._last_best_match
        adapted_config = adapt_architecture(best_config, target)
        adapted_model = _build_sequential_from_config(adapted_config)

        # Initialise all linear params.
        for module in adapted_model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Identify sequential positions of nn.Linear submodules.
        linear_positions = [
            i for i, m in enumerate(adapted_model) if isinstance(m, nn.Linear)
        ]
        middle_positions = set(linear_positions[1:-1]) if len(linear_positions) >= 3 else set()

        if middle_positions:
            source_state = source_model.state_dict()
            adapted_state = adapted_model.state_dict()

            for param_name, param in adapted_state.items():
                parts = param_name.split(".")
                try:
                    layer_idx = int(parts[0])
                except (ValueError, IndexError):
                    continue
                if layer_idx in middle_positions:
                    src = source_state.get(param_name)
                    if src is not None and src.shape == param.shape:
                        adapted_state[param_name].copy_(src)

            adapted_model.load_state_dict(adapted_state)

        return adapted_model

    def get_transfer_metadata(self) -> dict:
        """Return a dict describing this strategy's configuration."""
        return {
            "strategy": "architecture_transfer",
            "top_k_candidates": self.top_k_candidates,
            "n_registered_configs": len(self._config_registry),
        }
