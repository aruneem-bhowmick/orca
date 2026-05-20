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
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Normalise log1p(size) to ~[0, 1] so it doesn't dominate the one-hot features
# in cosine-similarity calculations.  log1p(4096) ≈ 8.32 serves as the upper bound.
_LOG_SIZE_SCALE: float = float(np.log1p(4096.0))


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
            try:
                layer_size = int(layer.get("size", 1))
            except (TypeError, ValueError):
                layer_size = 1
            if layer_size < 0:
                logger.debug("Layer %d has negative size %d; clamping to 0.", i, layer_size)
                layer_size = 0
            activation = str(layer.get("activation", "none")).lower()

            if layer_type in _LAYER_TYPES:
                node_features[i, _LAYER_TYPES.index(layer_type)] = 1.0

            node_features[i, len(_LAYER_TYPES)] = float(np.log1p(layer_size) / _LOG_SIZE_SCALE)

            if activation in _ACTIVATION_TYPES:
                node_features[i, len(_LAYER_TYPES) + 1 + _ACTIVATION_TYPES.index(activation)] = 1.0

            sizes.append(layer_size)

        # Sequential edges: i → i+1
        src: list[int] = list(range(n_nodes - 1))
        dst: list[int] = list(range(1, n_nodes))

        # Optional skip connections
        for pair in config.get("skip_connections", []):
            try:
                if len(pair) < 2:
                    raise ValueError("skip_connection entry must have at least two elements")
                s, d = int(pair[0]), int(pair[1])
            except (TypeError, ValueError, IndexError):
                logger.debug("Skipping malformed skip_connection entry: %r", pair)
                continue
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


def _try_import_gcnconv() -> type | None:
    """Return GCNConv class if torch_geometric is installed, else None."""
    try:
        from torch_geometric.nn import GCNConv  # type: ignore[import-untyped]

        return GCNConv
    except ImportError:
        return None


class ArchitectureEmbedder(nn.Module):
    """GNN embedder that encodes architecture configs into fixed-size vector embeddings.

    Uses manual adjacency-matrix message passing by default.  If the optional
    ``torch-geometric`` package is available, ``GCNConv`` layers are used instead.

    Args:
        node_dim:   Dimensionality of input node features (must match ``_NODE_DIM`` = 16).
        hidden_dim: Hidden dimensionality inside the GNN.
        output_dim: Output embedding dimensionality (default 32).
    """

    def __init__(
        self,
        node_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ) -> None:
        """Build the GNN: node encoder, 3 message-passing layers, and readout projection."""
        super().__init__()
        self._node_dim = node_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        GCNConv = _try_import_gcnconv()

        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        if GCNConv is not None:
            self.message_passing: nn.Module = nn.ModuleList(
                [GCNConv(hidden_dim, hidden_dim) for _ in range(3)]
            )
            self._use_gcnconv = True
        else:
            self.message_passing = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(3)]
            )
            self._use_gcnconv = False

        self.readout = nn.Linear(hidden_dim, output_dim)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed(self, config: ModelConfig) -> np.ndarray:
        """Encode an architecture config dict into an L2-normalised ``(output_dim,)`` vector.

        Args:
            config: Architecture configuration dict (see :class:`ArchitectureGraph`).

        Returns:
            A float32 numpy array of shape ``(output_dim,)`` with unit L2 norm.
        """
        graph = ArchitectureGraph.from_model_config(config)

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                out = self._forward_graph(graph)
        finally:
            if was_training:
                self.train()

        out_np: np.ndarray = out.cpu().numpy()
        norm = float(np.linalg.norm(out_np))
        return out_np / norm if norm > 0.0 else out_np

    def similarity(self, config_a: ModelConfig, config_b: ModelConfig) -> float:
        """Return the cosine similarity between the embeddings of two architecture configs.

        Args:
            config_a: First architecture config.
            config_b: Second architecture config.

        Returns:
            Cosine similarity in ``[-1, 1]`` (1.0 for identical configs).
        """
        emb_a = self.embed(config_a)
        emb_b = self.embed(config_b)
        return float(np.dot(emb_a, emb_b))

    def find_similar_architectures(
        self,
        query: ModelConfig,
        candidates: list[ModelConfig],
        top_k: int = 5,
    ) -> list[tuple[ModelConfig, float]]:
        """Return the top-k most similar candidate architectures to *query*.

        Args:
            query:      Architecture config to search for.
            candidates: Pool of architecture configs to rank.
            top_k:      Number of results to return.

        Returns:
            List of ``(config, similarity_score)`` tuples sorted by similarity descending.
            Length is ``min(top_k, len(candidates))``.

        Raises:
            ValueError: If ``top_k`` is negative.
        """
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        query_emb = self.embed(query)
        scored: list[tuple[ModelConfig, float]] = [
            (c, float(np.dot(query_emb, self.embed(c)))) for c in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward_graph(self, graph: ArchitectureGraph) -> torch.Tensor:
        """Run the full GNN pipeline and return the un-normalised readout vector."""
        device = self.node_encoder.weight.device
        x = torch.from_numpy(graph.node_features).to(device=device)  # (n_nodes, node_dim)
        edge_index = torch.from_numpy(graph.edge_index).to(device=device)  # (2, n_edges)
        n_nodes = x.shape[0]

        h = F.relu(self.node_encoder(x))  # (n_nodes, hidden_dim)

        if self._use_gcnconv:
            for conv in self.message_passing:  # type: ignore[union-attr]
                h = h + F.relu(conv(h, edge_index))
        else:
            # Residual message passing: each round adds a correction to the existing
            # representation so nodes retain their original identity after smoothing.
            # Without residuals, repeated adjacency averaging collapses all sequential-graph
            # embeddings towards a common direction regardless of layer types.
            adj_norm = self._build_adj_norm(n_nodes, edge_index)
            for linear in self.message_passing:  # type: ignore[union-attr]
                h = h + linear(adj_norm @ h)

        graph_emb = h.mean(dim=0)  # (hidden_dim,)
        return self.readout(graph_emb)  # (output_dim,)

    @staticmethod
    def _build_adj_norm(n_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
        """Build a row-normalised adjacency matrix with self-loops.

        Returns a ``(n_nodes, n_nodes)`` float32 tensor where each row sums to 1.
        """
        adj = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=edge_index.device)

        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            adj[dst, src] = 1.0
            adj[src, dst] = 1.0  # treat as undirected

        adj = adj + torch.eye(n_nodes, device=edge_index.device)  # self-loops
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        return adj / deg
