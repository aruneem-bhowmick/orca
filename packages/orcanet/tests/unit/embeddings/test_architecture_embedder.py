"""Unit tests for ArchitectureGraph and ArchitectureEmbedder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from orcanet.embeddings.architecture_embedder import (
    ArchitectureGraph,
    ArchitectureEmbedder,
    ModelConfig,
    _LAYER_TYPES,
    _ACTIVATION_TYPES,
    _LOG_SIZE_SCALE,
    _NODE_DIM,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _mlp_config() -> ModelConfig:
    return {
        "layers": [
            {"type": "linear", "size": 512, "activation": "relu"},
            {"type": "linear", "size": 256, "activation": "relu"},
            {"type": "linear", "size": 128, "activation": "relu"},
            {"type": "linear", "size": 10, "activation": "softmax"},
        ]
    }


def _cnn_config() -> ModelConfig:
    """Predominantly convolutional architecture — deliberately distinct from MLP.

    Uses conv2d and pooling layers (types 1 and 6) with small sizes so that
    both the layer-type one-hot bits and the normalised log-size feature differ
    substantially from the all-linear, large-size MLP, giving raw-feature
    cosine similarity well below 0.9.
    """
    return {
        "layers": [
            {"type": "conv2d", "size": 32, "activation": "relu"},
            {"type": "conv2d", "size": 64, "activation": "relu"},
            {"type": "pooling", "size": 64},
            {"type": "conv2d", "size": 64, "activation": "relu"},
            {"type": "pooling", "size": 64},
            {"type": "linear", "size": 10, "activation": "softmax"},
        ]
    }


# ---------------------------------------------------------------------------
# TestArchitectureGraph
# ---------------------------------------------------------------------------


class TestArchitectureGraphNodeCount:
    def test_mlp_node_count(self) -> None:
        config = _mlp_config()
        graph = ArchitectureGraph.from_model_config(config)
        assert graph.node_features.shape[0] == 4

    def test_single_layer_node_count(self) -> None:
        config: ModelConfig = {"layers": [{"type": "linear", "size": 10}]}
        graph = ArchitectureGraph.from_model_config(config)
        assert graph.node_features.shape[0] == 1

    def test_cnn_node_count(self) -> None:
        config = _cnn_config()
        graph = ArchitectureGraph.from_model_config(config)
        assert graph.node_features.shape[0] == 6

    def test_empty_layers_gives_one_degenerate_node(self) -> None:
        graph = ArchitectureGraph.from_model_config({})
        assert graph.node_features.shape[0] == 1


class TestArchitectureGraphNodeFeatures:
    def test_node_features_shape(self) -> None:
        graph = ArchitectureGraph.from_model_config(_mlp_config())
        assert graph.node_features.shape == (4, _NODE_DIM)

    def test_node_features_dtype(self) -> None:
        graph = ArchitectureGraph.from_model_config(_mlp_config())
        assert graph.node_features.dtype == np.float32

    def test_layer_type_one_hot_linear(self) -> None:
        config: ModelConfig = {"layers": [{"type": "linear", "size": 64, "activation": "relu"}]}
        graph = ArchitectureGraph.from_model_config(config)
        linear_idx = _LAYER_TYPES.index("linear")
        assert graph.node_features[0, linear_idx] == 1.0
        other_type_indices = [i for i in range(len(_LAYER_TYPES)) if i != linear_idx]
        assert all(graph.node_features[0, i] == 0.0 for i in other_type_indices)

    def test_layer_type_one_hot_conv2d(self) -> None:
        config: ModelConfig = {"layers": [{"type": "conv2d", "size": 64}]}
        graph = ArchitectureGraph.from_model_config(config)
        conv_idx = _LAYER_TYPES.index("conv2d")
        assert graph.node_features[0, conv_idx] == 1.0

    def test_log_size_encoding(self) -> None:
        config: ModelConfig = {"layers": [{"type": "linear", "size": 512}]}
        graph = ArchitectureGraph.from_model_config(config)
        expected = float(np.log1p(512) / _LOG_SIZE_SCALE)
        assert np.isclose(graph.node_features[0, len(_LAYER_TYPES)], expected, atol=1e-5)

    def test_activation_one_hot_relu(self) -> None:
        config: ModelConfig = {"layers": [{"type": "linear", "size": 64, "activation": "relu"}]}
        graph = ArchitectureGraph.from_model_config(config)
        relu_idx = len(_LAYER_TYPES) + 1 + _ACTIVATION_TYPES.index("relu")
        assert graph.node_features[0, relu_idx] == 1.0

    def test_unknown_layer_type_all_zeros_in_type_dims(self) -> None:
        config: ModelConfig = {"layers": [{"type": "unknown_layer", "size": 32}]}
        graph = ArchitectureGraph.from_model_config(config)
        assert all(graph.node_features[0, i] == 0.0 for i in range(len(_LAYER_TYPES)))

    def test_default_activation_is_none(self) -> None:
        config: ModelConfig = {"layers": [{"type": "batchnorm", "size": 128}]}
        graph = ArchitectureGraph.from_model_config(config)
        none_idx = len(_LAYER_TYPES) + 1 + _ACTIVATION_TYPES.index("none")
        assert graph.node_features[0, none_idx] == 1.0


class TestArchitectureGraphEdgeIndex:
    def test_sequential_edge_count_for_mlp(self) -> None:
        graph = ArchitectureGraph.from_model_config(_mlp_config())
        # 4 layers → 3 sequential edges
        assert graph.edge_index.shape == (2, 3)

    def test_sequential_edge_count_single_layer(self) -> None:
        config: ModelConfig = {"layers": [{"type": "linear", "size": 10}]}
        graph = ArchitectureGraph.from_model_config(config)
        assert graph.edge_index.shape == (2, 0)

    def test_edge_index_dtype(self) -> None:
        graph = ArchitectureGraph.from_model_config(_mlp_config())
        assert graph.edge_index.dtype == np.int64

    def test_skip_connections_add_extra_edges(self) -> None:
        config: ModelConfig = {
            "layers": [
                {"type": "linear", "size": 64},
                {"type": "linear", "size": 64},
                {"type": "linear", "size": 64},
                {"type": "linear", "size": 10},
            ],
            "skip_connections": [[0, 2]],
        }
        graph = ArchitectureGraph.from_model_config(config)
        # 3 sequential + 1 skip = 4 edges
        assert graph.edge_index.shape[1] == 4

    def test_empty_config_has_zero_edges(self) -> None:
        graph = ArchitectureGraph.from_model_config({})
        assert graph.edge_index.shape == (2, 0)


class TestArchitectureGraphGlobalFeatures:
    def test_graph_features_shape(self) -> None:
        graph = ArchitectureGraph.from_model_config(_mlp_config())
        assert graph.graph_features.shape == (3,)

    def test_graph_features_dtype(self) -> None:
        graph = ArchitectureGraph.from_model_config(_mlp_config())
        assert graph.graph_features.dtype == np.float32

    def test_depth_equals_layer_count(self) -> None:
        graph = ArchitectureGraph.from_model_config(_mlp_config())
        assert graph.graph_features[1] == pytest.approx(4.0)

    def test_log_total_size_positive(self) -> None:
        graph = ArchitectureGraph.from_model_config(_mlp_config())
        assert graph.graph_features[0] > 0.0

    def test_log_max_width_matches_largest_layer(self) -> None:
        config: ModelConfig = {
            "layers": [
                {"type": "linear", "size": 128},
                {"type": "linear", "size": 512},
                {"type": "linear", "size": 64},
            ]
        }
        graph = ArchitectureGraph.from_model_config(config)
        expected_log_width = float(np.log1p(512))
        assert np.isclose(graph.graph_features[2], expected_log_width, atol=1e-5)

    def test_empty_config_graph_features_are_zero(self) -> None:
        graph = ArchitectureGraph.from_model_config({})
        assert np.all(graph.graph_features == 0.0)


# ---------------------------------------------------------------------------
# TestArchitectureEmbedderEmbed
# ---------------------------------------------------------------------------


class TestArchitectureEmbedderEmbed:
    def test_embed_returns_correct_shape(self) -> None:
        embedder = ArchitectureEmbedder()
        out = embedder.embed(_mlp_config())
        assert out.shape == (32,)

    def test_embed_is_l2_normalized(self) -> None:
        embedder = ArchitectureEmbedder()
        out = embedder.embed(_mlp_config())
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)

    def test_embed_returns_numpy_array(self) -> None:
        embedder = ArchitectureEmbedder()
        out = embedder.embed(_mlp_config())
        assert isinstance(out, np.ndarray)

    def test_embed_dtype_float32(self) -> None:
        embedder = ArchitectureEmbedder()
        out = embedder.embed(_mlp_config())
        assert out.dtype == np.float32

    def test_embed_custom_output_dim(self) -> None:
        embedder = ArchitectureEmbedder(output_dim=16)
        out = embedder.embed(_mlp_config())
        assert out.shape == (16,)

    def test_embed_preserves_training_mode(self) -> None:
        embedder = ArchitectureEmbedder()
        embedder.train()
        embedder.embed(_mlp_config())
        assert embedder.training

    def test_embed_preserves_eval_mode(self) -> None:
        embedder = ArchitectureEmbedder()
        embedder.eval()
        embedder.embed(_mlp_config())
        assert not embedder.training

    def test_embed_single_layer_config(self) -> None:
        embedder = ArchitectureEmbedder()
        config: ModelConfig = {"layers": [{"type": "linear", "size": 10}]}
        out = embedder.embed(config)
        assert out.shape == (32,)
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)

    def test_embed_empty_config(self) -> None:
        embedder = ArchitectureEmbedder()
        out = embedder.embed({})
        assert out.shape == (32,)
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)

    def test_embed_is_deterministic(self) -> None:
        embedder = ArchitectureEmbedder()
        out1 = embedder.embed(_mlp_config())
        out2 = embedder.embed(_mlp_config())
        assert np.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# TestArchitectureEmbedderSimilarity
# ---------------------------------------------------------------------------


class TestArchitectureEmbedderSimilarity:
    def test_similarity_identical_configs_is_one(self) -> None:
        torch.manual_seed(42)
        embedder = ArchitectureEmbedder()
        config = _mlp_config()
        sim = embedder.similarity(config, config)
        assert np.isclose(sim, 1.0, atol=1e-5)

    def test_similarity_mlp_vs_cnn_less_than_self_similarity(self) -> None:
        torch.manual_seed(42)
        embedder = ArchitectureEmbedder()
        sim_same = embedder.similarity(_mlp_config(), _mlp_config())
        sim_diff = embedder.similarity(_mlp_config(), _cnn_config())
        assert sim_same > sim_diff, (
            f"Expected MLP self-similarity ({sim_same:.4f}) > MLP-CNN cross-similarity ({sim_diff:.4f})"
        )

    def test_similarity_returns_float(self) -> None:
        torch.manual_seed(0)
        embedder = ArchitectureEmbedder()
        sim = embedder.similarity(_mlp_config(), _cnn_config())
        assert isinstance(sim, float)

    def test_similarity_is_symmetric(self) -> None:
        torch.manual_seed(0)
        embedder = ArchitectureEmbedder()
        sim_ab = embedder.similarity(_mlp_config(), _cnn_config())
        sim_ba = embedder.similarity(_cnn_config(), _mlp_config())
        assert np.isclose(sim_ab, sim_ba, atol=1e-6)


# ---------------------------------------------------------------------------
# TestArchitectureEmbedderRetrieval
# ---------------------------------------------------------------------------


class TestArchitectureEmbedderRetrieval:
    def _make_candidates(self) -> list[ModelConfig]:
        return [
            {
                "layers": [
                    {"type": "linear", "size": 512, "activation": "relu"},
                    {"type": "linear", "size": 10},
                ]
            },
            {
                "layers": [
                    {"type": "conv2d", "size": 64, "activation": "relu"},
                    {"type": "pooling", "size": 64},
                    {"type": "linear", "size": 10},
                ]
            },
            {
                "layers": [
                    {"type": "lstm", "size": 128, "activation": "tanh"},
                    {"type": "linear", "size": 10},
                ]
            },
            {
                "layers": [
                    {"type": "embedding", "size": 64},
                    {"type": "attention", "size": 64},
                    {"type": "linear", "size": 10},
                ]
            },
            {
                "layers": [
                    {"type": "linear", "size": 256, "activation": "relu"},
                    {"type": "batchnorm", "size": 256},
                    {"type": "dropout", "size": 256},
                    {"type": "linear", "size": 10},
                ]
            },
            {
                "layers": [
                    {"type": "conv2d", "size": 32, "activation": "relu"},
                    {"type": "conv2d", "size": 64, "activation": "relu"},
                    {"type": "linear", "size": 10},
                ]
            },
        ]

    def test_find_similar_returns_top_k_results(self) -> None:
        embedder = ArchitectureEmbedder()
        results = embedder.find_similar_architectures(_mlp_config(), self._make_candidates(), top_k=3)
        assert len(results) == 3

    def test_find_similar_returns_all_when_fewer_candidates(self) -> None:
        embedder = ArchitectureEmbedder()
        candidates = self._make_candidates()[:2]
        results = embedder.find_similar_architectures(_mlp_config(), candidates, top_k=5)
        assert len(results) == 2

    def test_find_similar_sorted_descending(self) -> None:
        embedder = ArchitectureEmbedder()
        results = embedder.find_similar_architectures(_mlp_config(), self._make_candidates(), top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by similarity descending"

    def test_find_similar_returns_tuples_of_config_and_float(self) -> None:
        embedder = ArchitectureEmbedder()
        results = embedder.find_similar_architectures(_mlp_config(), self._make_candidates(), top_k=2)
        for config, score in results:
            assert isinstance(config, dict)
            assert isinstance(score, float)

    def test_find_similar_identical_query_is_top_result(self) -> None:
        embedder = ArchitectureEmbedder()
        query = _mlp_config()
        results = embedder.find_similar_architectures(query, [_cnn_config(), query], top_k=2)
        _, top_score = results[0]
        assert np.isclose(top_score, 1.0, atol=1e-5), (
            f"Identical config should be top result with similarity 1.0, got {top_score}"
        )

    def test_find_similar_default_top_k_five(self) -> None:
        embedder = ArchitectureEmbedder()
        results = embedder.find_similar_architectures(_mlp_config(), self._make_candidates())
        assert len(results) == 5

    def test_find_similar_top_k_zero_returns_empty(self) -> None:
        embedder = ArchitectureEmbedder()
        results = embedder.find_similar_architectures(_mlp_config(), self._make_candidates(), top_k=0)
        assert results == []

    def test_find_similar_negative_top_k_raises(self) -> None:
        embedder = ArchitectureEmbedder()
        with pytest.raises(ValueError):
            embedder.find_similar_architectures(_mlp_config(), self._make_candidates(), top_k=-1)
