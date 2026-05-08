"""Tests for StatisticalEmbedder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orcamind.embedders.statistical import StatisticalEmbedder

_EMBED_DIM = 25


@pytest.fixture()
def embedder() -> StatisticalEmbedder:
    return StatisticalEmbedder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classification_dataset(
    n_rows: int, n_cols: int, n_classes: int = 2, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])
    y = pd.Series(rng.integers(0, n_classes, size=n_rows), name="target")
    return X, y


def _make_regression_dataset(
    n_rows: int, n_cols: int, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])
    y = pd.Series(rng.standard_normal(n_rows), name="target")
    return X, y


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_standard_classification(self, embedder):
        X, y = _make_classification_dataset(200, 10, n_classes=3)
        vec = embedder.embed(X, y)
        assert vec.shape == (_EMBED_DIM,)

    def test_standard_regression(self, embedder):
        X, y = _make_regression_dataset(200, 10)
        vec = embedder.embed(X, y)
        assert vec.shape == (_EMBED_DIM,)

    def test_no_labels(self, embedder):
        X, _ = _make_classification_dataset(100, 5)
        vec = embedder.embed(X, None)
        assert vec.shape == (_EMBED_DIM,)

    def test_single_column(self, embedder):
        X = pd.DataFrame({"a": np.arange(50, dtype=float)})
        vec = embedder.embed(X, None)
        assert vec.shape == (_EMBED_DIM,)

    def test_embedding_dim_property(self, embedder):
        assert embedder.embedding_dim == _EMBED_DIM


# ---------------------------------------------------------------------------
# All values finite (edge cases)
# ---------------------------------------------------------------------------

class TestFiniteValues:
    def test_all_nan_column(self, embedder):
        X = pd.DataFrame({"a": np.full(20, np.nan), "b": np.arange(20, dtype=float)})
        vec = embedder.embed(X, None)
        assert np.all(np.isfinite(vec))

    def test_constant_column(self, embedder):
        X = pd.DataFrame({"const": np.ones(30), "vary": np.arange(30, dtype=float)})
        vec = embedder.embed(X, None)
        assert np.all(np.isfinite(vec))

    def test_single_class_labels(self, embedder):
        X, _ = _make_classification_dataset(40, 4)
        y = pd.Series(np.zeros(40, dtype=int))
        vec = embedder.embed(X, y)
        assert np.all(np.isfinite(vec))

    def test_tiny_dataset_less_than_5_rows(self, embedder):
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        y = pd.Series([0, 1])
        vec = embedder.embed(X, y)
        assert np.all(np.isfinite(vec))

    def test_single_row(self, embedder):
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        vec = embedder.embed(X, None)
        assert np.all(np.isfinite(vec))

    def test_mixed_dtypes_with_nan(self, embedder):
        X = pd.DataFrame(
            {
                "num": [1.0, np.nan, 3.0, 4.0, 5.0],
                "cat": pd.Categorical(["a", "b", "a", None, "b"]),
            }
        )
        vec = embedder.embed(X, None)
        assert np.all(np.isfinite(vec))


# ---------------------------------------------------------------------------
# log_n_samples increases monotonically with dataset size
# ---------------------------------------------------------------------------

class TestLogNSamples:
    def test_monotonic_with_size(self, embedder):
        sizes = [10, 100, 1000, 10_000]
        log_n = [embedder.embed(pd.DataFrame({"x": np.ones(n)}), None)[0] for n in sizes]
        for a, b in zip(log_n, log_n[1:]):
            assert b > a, f"log_n_samples not increasing: {log_n}"


# ---------------------------------------------------------------------------
# class_balance_entropy (index 3)
# ---------------------------------------------------------------------------

class TestClassBalanceEntropy:
    def test_zero_for_single_class(self, embedder):
        X, _ = _make_classification_dataset(50, 3)
        y = pd.Series(np.zeros(50, dtype=int))
        vec = embedder.embed(X, y)
        assert vec[3] == pytest.approx(0.0, abs=1e-9)

    def test_maximized_for_balanced_binary(self, embedder):
        X, _ = _make_classification_dataset(100, 3)
        y_balanced = pd.Series(np.tile([0, 1], 50))
        y_unbalanced = pd.Series(np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)]))
        e_bal = embedder.embed(X, y_balanced)[3]
        e_unbal = embedder.embed(X, y_unbalanced)[3]
        assert e_bal > e_unbal

    def test_entropy_increases_with_more_classes(self, embedder):
        X, _ = _make_classification_dataset(120, 3)
        entropies = []
        for k in [2, 3, 4]:
            y = pd.Series(np.tile(np.arange(k), 120 // k + 1)[:120])
            entropies.append(embedder.embed(X, y)[3])
        for a, b in zip(entropies, entropies[1:]):
            assert b >= a


# ---------------------------------------------------------------------------
# embed_batch
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    def test_shape_n_by_25(self, embedder):
        datasets = [_make_classification_dataset(50, 5, seed=i) for i in range(7)]
        result = embedder.embed_batch(datasets)
        assert result.shape == (7, _EMBED_DIM)

    def test_batch_matches_individual(self, embedder):
        ds = [_make_classification_dataset(50, 4, seed=i) for i in range(3)]
        batch = embedder.embed_batch(ds)
        for i, (X, y) in enumerate(ds):
            np.testing.assert_array_equal(batch[i], embedder.embed(X, y))

    def test_all_finite_in_batch(self, embedder):
        datasets = [_make_regression_dataset(20, 3, seed=i) for i in range(5)]
        result = embedder.embed_batch(datasets)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Specific feature checks
# ---------------------------------------------------------------------------

class TestSpecificFeatures:
    def test_missing_ratio_all_present(self, embedder):
        X = pd.DataFrame(np.ones((30, 4)))
        vec = embedder.embed(X, None)
        assert vec[4] == pytest.approx(0.0, abs=1e-9)

    def test_missing_ratio_all_missing(self, embedder):
        X = pd.DataFrame(np.full((30, 4), np.nan))
        vec = embedder.embed(X, None)
        assert vec[4] == pytest.approx(1.0, abs=1e-9)

    def test_categorical_ratio_all_numeric(self, embedder):
        X = pd.DataFrame(np.ones((20, 3)))
        vec = embedder.embed(X, None)
        assert vec[5] == pytest.approx(0.0, abs=1e-9)

    def test_categorical_ratio_all_categorical(self, embedder):
        X = pd.DataFrame(
            {
                "a": pd.Categorical(["x", "y"] * 10),
                "b": pd.Categorical(["p", "q"] * 10),
            }
        )
        vec = embedder.embed(X, None)
        assert vec[5] == pytest.approx(1.0, abs=1e-9)

    def test_n_classes_zero_for_regression(self, embedder):
        X, y = _make_regression_dataset(50, 3)
        vec = embedder.embed(X, y)
        assert vec[2] == 0.0

    def test_n_classes_correct(self, embedder):
        X, _ = _make_classification_dataset(60, 4, n_classes=5)
        y = pd.Series(np.tile(np.arange(5), 12))
        vec = embedder.embed(X, y)
        assert vec[2] == pytest.approx(5.0, abs=1e-9)
