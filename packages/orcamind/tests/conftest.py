"""Pytest configuration and shared fixtures for orcamind tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """A small synthetic 3-class classification dataset."""
    rng = np.random.default_rng(42)
    n = 120
    X = pd.DataFrame(
        {
            "feature_0": rng.normal(0, 1, n),
            "feature_1": rng.normal(1, 2, n),
            "feature_2": rng.uniform(0, 10, n),
            "feature_3": rng.exponential(1, n),
        }
    )
    y = pd.Series(rng.integers(0, 3, n), name="target")
    return X, y


@pytest.fixture
def sample_regression_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """A small synthetic regression dataset."""
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "x0": rng.normal(0, 1, n),
            "x1": rng.normal(5, 0.5, n),
        }
    )
    y = pd.Series(2.0 * X["x0"] + 0.5 * X["x1"] + rng.normal(0, 0.1, n), name="target")
    return X, y


@pytest.fixture
def empty_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """A near-empty dataset to exercise edge cases."""
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [float("nan"), float("nan")]})
    y = pd.Series([0, 0], name="target")
    return X, y
