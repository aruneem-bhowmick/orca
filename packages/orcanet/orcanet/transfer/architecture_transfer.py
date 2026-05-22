"""Architecture-based transfer strategy.

Recommends and adapts model architectures for a target domain by comparing
architecture graph embeddings produced by ArchitectureEmbedder.  Uses the
OrcaMind service to retrieve the source task's best-known architecture, then
scores all locally registered candidate configs by cosine similarity and
returns the best match.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TYPE_CHECKING

import torch.nn as nn

from orca_shared.schemas.task import Task

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
