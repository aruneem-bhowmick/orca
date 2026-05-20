"""GNN-based neural network architecture embedder.

Encodes a model architecture config dict into a fixed-size, L2-normalised embedding
by representing the architecture as a graph (nodes = layers, edges = connectivity)
and applying manual adjacency-matrix message passing.

If the optional ``torch-geometric`` package is installed (``pip install orcanet[gnn]``),
``GCNConv`` is used for message passing instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ModelConfig = dict[str, Any]

_LAYER_TYPES: tuple[str, ...] = (
    "linear",
    "conv2d",
    "lstm",
    "attention",
    "batchnorm",
    "dropout",
    "pooling",
    "embedding",
)
_ACTIVATION_TYPES: tuple[str, ...] = (
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "selu",
    "softmax",
    "none",
)
# node_dim = 8 (layer-type one-hot) + 1 (log-size) + 7 (activation one-hot) = 16
_NODE_DIM: int = 16
_GLOBAL_DIM: int = 3  # log(total_params), depth, log(max_width)


@dataclass
class ArchitectureGraph:
    """Graph representation of a neural network architecture config.

    Attributes:
        node_features:  ``(n_nodes, node_dim)`` float32 array — one-hot layer type,
                        log-scaled size, one-hot activation type.
        edge_index:     ``(2, n_edges)`` int64 array — ``[sources, targets]``.
        graph_features: ``(3,)`` float32 array — global statistics:
                        ``[log1p(Σsize), depth, log1p(max_size)]``.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    graph_features: np.ndarray

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "ArchitectureGraph":
        """Convert an architecture config dict into a graph.

        The ``config`` dict is expected to contain:
        - ``"layers"``: list of dicts with optional keys ``"type"``, ``"size"``,
          ``"activation"``.
        - ``"skip_connections"``: optional list of ``[src, dst]`` pairs for
          non-sequential edges.

        Args:
            config: Architecture configuration dictionary.

        Returns:
            An :class:`ArchitectureGraph` suitable for GNN processing.
        """
        layers: list[dict[str, Any]] = config.get("layers", [])

        if not layers:
            return cls(
                node_features=np.zeros((1, _NODE_DIM), dtype=np.float32),
                edge_index=np.zeros((2, 0), dtype=np.int64),
                graph_features=np.zeros(_GLOBAL_DIM, dtype=np.float32),
            )

        n_nodes = len(layers)
        node_features = np.zeros((n_nodes, _NODE_DIM), dtype=np.float32)

        sizes: list[int] = []
        for i, layer in enumerate(layers):
            layer_type = str(layer.get("type", "linear")).lower()
            layer_size = int(layer.get("size", 1))
            activation = str(layer.get("activation", "none")).lower()

            if layer_type in _LAYER_TYPES:
                node_features[i, _LAYER_TYPES.index(layer_type)] = 1.0

            node_features[i, len(_LAYER_TYPES)] = float(np.log1p(layer_size))

            if activation in _ACTIVATION_TYPES:
                node_features[i, len(_LAYER_TYPES) + 1 + _ACTIVATION_TYPES.index(activation)] = 1.0

            sizes.append(layer_size)

        # Sequential edges: i → i+1
        src: list[int] = list(range(n_nodes - 1))
        dst: list[int] = list(range(1, n_nodes))

        # Optional skip connections
        for pair in config.get("skip_connections", []):
            s, d = int(pair[0]), int(pair[1])
            if 0 <= s < n_nodes and 0 <= d < n_nodes:
                src.append(s)
                dst.append(d)

        edge_index = (
            np.array([src, dst], dtype=np.int64)
            if src
            else np.zeros((2, 0), dtype=np.int64)
        )

        total_size = sum(sizes)
        max_size = max(sizes) if sizes else 0
        graph_features = np.array(
            [float(np.log1p(total_size)), float(n_nodes), float(np.log1p(max_size))],
            dtype=np.float32,
        )

        return cls(
            node_features=node_features,
            edge_index=edge_index,
            graph_features=graph_features,
        )
