"""Unit tests for TextTaskEmbedder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orcanet.embeddings.text_features import TextTaskEmbedder


class TestEmbedFromDescription:
    def test_returns_384_dim_vector(self) -> None:
        embedder = TextTaskEmbedder()
        out = embedder.embed_from_description("binary image classification task")
        assert out.shape == (384,)

    def test_is_l2_normalized(self) -> None:
        embedder = TextTaskEmbedder()
        out = embedder.embed_from_description("some task description")
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)

    def test_returns_numpy_array(self) -> None:
        embedder = TextTaskEmbedder()
        out = embedder.embed_from_description("classification task")
        assert isinstance(out, np.ndarray)

    def test_invalid_fusion_raises(self) -> None:
        with pytest.raises(ValueError, match="fusion must be one of"):
            TextTaskEmbedder(fusion="invalid")


class TestEmbedWithStats:
    def test_returns_output_dim_concat(
        self, sample_classification_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_classification_dataset
        embedder = TextTaskEmbedder(fusion="concat", output_dim=128)
        out = embedder.embed_with_stats("binary classification", X, y)
        assert out.shape == (128,)

    def test_returns_output_dim_add(
        self, sample_classification_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_classification_dataset
        embedder = TextTaskEmbedder(fusion="add", output_dim=64)
        out = embedder.embed_with_stats("binary classification", X, y)
        assert out.shape == (64,)

    def test_returns_output_dim_attention(
        self, sample_regression_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_regression_dataset
        embedder = TextTaskEmbedder(fusion="attention", output_dim=32)
        out = embedder.embed_with_stats("financial time series regression", X, y)
        assert out.shape == (32,)

    def test_no_labels_regression_path(
        self, sample_regression_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, _ = sample_regression_dataset
        embedder = TextTaskEmbedder(fusion="concat", output_dim=128)
        out = embedder.embed_with_stats("regression task", X, labels=None)
        assert out.shape == (128,)

    def test_output_is_l2_normalized(
        self, sample_classification_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_classification_dataset
        embedder = TextTaskEmbedder(fusion="concat", output_dim=64)
        out = embedder.embed_with_stats("classification", X, y)
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)

    def test_custom_output_dim(
        self, sample_classification_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_classification_dataset
        embedder = TextTaskEmbedder(fusion="add", output_dim=16)
        out = embedder.embed_with_stats("task", X, y)
        assert out.shape == (16,)


class TestSemanticSimilarity:
    def test_image_tasks_more_similar_than_financial(self) -> None:
        embedder = TextTaskEmbedder()
        v_cls = embedder.embed_from_description("binary image classification task")
        v_recog = embedder.embed_from_description("multi-class image recognition")
        v_finance = embedder.embed_from_description("financial time series regression")
        # L2-normalised vectors → dot product = cosine similarity
        sim_image = float(np.dot(v_cls, v_recog))
        sim_cross_cls = float(np.dot(v_cls, v_finance))
        sim_cross_recog = float(np.dot(v_recog, v_finance))
        assert sim_image > sim_cross_cls
        assert sim_image > sim_cross_recog

    def test_identical_descriptions_have_similarity_one(self) -> None:
        embedder = TextTaskEmbedder()
        desc = "multi-label medical image segmentation"
        v1 = embedder.embed_from_description(desc)
        v2 = embedder.embed_from_description(desc)
        assert np.isclose(float(np.dot(v1, v2)), 1.0, atol=1e-5)


class TestEmbedBatchDescriptions:
    def test_batch_returns_correct_shape(self) -> None:
        embedder = TextTaskEmbedder()
        descriptions = ["task one", "task two", "task three"]
        out = embedder.embed_batch_descriptions(descriptions)
        assert out.shape == (3, 384)

    def test_batch_rows_are_l2_normalized(self) -> None:
        embedder = TextTaskEmbedder()
        descriptions = ["classification", "regression", "clustering"]
        out = embedder.embed_batch_descriptions(descriptions)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_batch_matches_individual_calls(self) -> None:
        embedder = TextTaskEmbedder()
        descriptions = ["binary classification", "regression", "time series clustering"]
        batch_out = embedder.embed_batch_descriptions(descriptions)
        for i, desc in enumerate(descriptions):
            individual = embedder.embed_from_description(desc)
            assert np.allclose(batch_out[i], individual, atol=1e-5), (
                f"Batch vs individual mismatch at index {i} for {desc!r}"
            )

    def test_single_item_list(self) -> None:
        embedder = TextTaskEmbedder()
        out = embedder.embed_batch_descriptions(["single description"])
        assert out.shape == (1, 384)
