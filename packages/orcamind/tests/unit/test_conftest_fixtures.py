"""Tests for the shared fixtures defined in tests/conftest.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ── sample_classification_dataset ─────────────────────────────────────────────

def test_classification_X_is_dataframe(sample_classification_dataset) -> None:
    X, _ = sample_classification_dataset
    assert isinstance(X, pd.DataFrame)


def test_classification_y_is_series(sample_classification_dataset) -> None:
    _, y = sample_classification_dataset
    assert isinstance(y, pd.Series)


def test_classification_X_shape(sample_classification_dataset) -> None:
    X, _ = sample_classification_dataset
    assert X.shape == (120, 4)


def test_classification_y_length(sample_classification_dataset) -> None:
    _, y = sample_classification_dataset
    assert len(y) == 120


def test_classification_column_names(sample_classification_dataset) -> None:
    X, _ = sample_classification_dataset
    assert list(X.columns) == ["feature_0", "feature_1", "feature_2", "feature_3"]


def test_classification_y_name(sample_classification_dataset) -> None:
    _, y = sample_classification_dataset
    assert y.name == "target"


def test_classification_has_three_classes(sample_classification_dataset) -> None:
    _, y = sample_classification_dataset
    assert y.nunique() == 3


def test_classification_y_values_in_range(sample_classification_dataset) -> None:
    _, y = sample_classification_dataset
    assert set(y.unique()).issubset({0, 1, 2})


def test_classification_no_nulls_in_X(sample_classification_dataset) -> None:
    X, _ = sample_classification_dataset
    assert not X.isnull().any().any()


def test_classification_no_nulls_in_y(sample_classification_dataset) -> None:
    _, y = sample_classification_dataset
    assert not y.isnull().any()


def test_classification_is_deterministic(sample_classification_dataset) -> None:
    """Fixture uses a fixed seed — calling it twice must return identical data."""
    X1, y1 = sample_classification_dataset
    # Re-produce manually with same seed to confirm reproducibility
    rng = np.random.default_rng(42)
    n = 120
    X2 = pd.DataFrame(
        {
            "feature_0": rng.normal(0, 1, n),
            "feature_1": rng.normal(1, 2, n),
            "feature_2": rng.uniform(0, 10, n),
            "feature_3": rng.exponential(1, n),
        }
    )
    y2 = pd.Series(rng.integers(0, 3, n), name="target")
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


# ── sample_regression_dataset ─────────────────────────────────────────────────

def test_regression_X_shape(sample_regression_dataset) -> None:
    X, _ = sample_regression_dataset
    assert X.shape == (80, 2)


def test_regression_y_length(sample_regression_dataset) -> None:
    _, y = sample_regression_dataset
    assert len(y) == 80


def test_regression_column_names(sample_regression_dataset) -> None:
    X, _ = sample_regression_dataset
    assert list(X.columns) == ["x0", "x1"]


def test_regression_y_name(sample_regression_dataset) -> None:
    _, y = sample_regression_dataset
    assert y.name == "target"


def test_regression_y_is_continuous(sample_regression_dataset) -> None:
    _, y = sample_regression_dataset
    assert y.dtype == float


def test_regression_y_has_many_unique_values(sample_regression_dataset) -> None:
    _, y = sample_regression_dataset
    # Continuous target should have nearly all unique values
    assert y.nunique() > 70


def test_regression_no_nulls(sample_regression_dataset) -> None:
    X, y = sample_regression_dataset
    assert not X.isnull().any().any()
    assert not y.isnull().any()


# ── empty_dataset ─────────────────────────────────────────────────────────────

def test_empty_dataset_X_shape(empty_dataset) -> None:
    X, _ = empty_dataset
    assert X.shape == (2, 2)


def test_empty_dataset_y_length(empty_dataset) -> None:
    _, y = empty_dataset
    assert len(y) == 2


def test_empty_dataset_column_a_has_no_nans(empty_dataset) -> None:
    X, _ = empty_dataset
    assert not X["a"].isnull().any()


def test_empty_dataset_column_b_all_nan(empty_dataset) -> None:
    X, _ = empty_dataset
    assert X["b"].isnull().all()


def test_empty_dataset_y_all_zeros(empty_dataset) -> None:
    _, y = empty_dataset
    assert (y == 0).all()


def test_empty_dataset_y_name(empty_dataset) -> None:
    _, y = empty_dataset
    assert y.name == "target"
