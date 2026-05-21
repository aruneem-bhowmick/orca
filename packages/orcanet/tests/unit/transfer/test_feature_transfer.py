"""Unit tests for linear_cka and FeatureTransfer."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
import torch
import torch.nn as nn

from orcanet.transfer.feature_transfer import FeatureTransfer, linear_cka
from orcanet.transfer.types import TransferScore
from orca_shared.schemas.task import Task


# ---------------------------------------------------------------------------
# Helpers shared by FeatureTransfer tests
# ---------------------------------------------------------------------------


def _make_task(task_id=None) -> Task:
    now = datetime.now(timezone.utc)
    return Task(
        task_id=task_id or uuid4(),
        name="test_task",
        task_type="classification",
        created_at=now,
        updated_at=now,
    )


def _make_mlp(in_dim: int = 10, hidden: int = 20, out_dim: int = 5) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# TestLinearCKA
# ---------------------------------------------------------------------------


class TestLinearCKAIdentical:
    """CKA of a matrix with itself must equal 1.0."""

    def test_square_identical(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 10))
        assert abs(linear_cka(X, X) - 1.0) < 1e-6

    def test_tall_identical(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 5))
        assert abs(linear_cka(X, X) - 1.0) < 1e-6

    def test_wide_identical(self) -> None:
        rng = np.random.default_rng(13)
        X = rng.standard_normal((30, 20))
        assert abs(linear_cka(X, X) - 1.0) < 1e-6


class TestLinearCKAOrthogonal:
    """CKA of two matrices spanning orthogonal subspaces must be near zero."""

    def _orthogonal_pair(self, n: int = 80, p: int = 6) -> tuple[np.ndarray, np.ndarray]:
        """Return two (n, p) submatrices drawn from non-overlapping columns of a
        random orthonormal basis so that col(X) ⊥ col(Y) exactly."""
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        return Q[:, :p], Q[:, p : 2 * p]

    def test_orthogonal_subspaces_near_zero(self) -> None:
        X, Y = self._orthogonal_pair()
        assert linear_cka(X, Y) < 0.1

    def test_orthogonal_is_symmetric(self) -> None:
        X, Y = self._orthogonal_pair()
        assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-10

    def test_orthogonal_larger_n(self) -> None:
        X, Y = self._orthogonal_pair(n=200, p=8)
        assert linear_cka(X, Y) < 0.01


class TestLinearCKAReturnType:
    def test_returns_float(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 8))
        assert isinstance(linear_cka(X, X), float)

    def test_value_in_unit_interval_identical(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 8))
        val = linear_cka(X, X)
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_value_in_unit_interval_random(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 10))
        Y = rng.standard_normal((60, 15))
        val = linear_cka(X, Y)
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_different_feature_dims_allowed(self) -> None:
        """CKA is defined for X (n×p) and Y (n×q) with p ≠ q."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((50, 8))
        Y = rng.standard_normal((50, 20))
        val = linear_cka(X, Y)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# TestFeatureTransferScoring — identical vs. random models
# ---------------------------------------------------------------------------


class TestFeatureTransferIdenticalModels:
    """score_transfer with the same model registered for both tasks must return overall ≈ 1."""

    def _setup(self) -> tuple[FeatureTransfer, Task, Task]:
        probe = np.random.default_rng(0).standard_normal((100, 10)).astype(np.float32)
        transfer = FeatureTransfer(probe_data=probe, cka_threshold=0.5)
        source = _make_task()
        target = _make_task()
        model = _make_mlp()
        transfer.register_model(str(source.task_id), model)
        transfer.register_model(str(target.task_id), model)
        return transfer, source, target

    def test_overall_near_one(self) -> None:
        transfer, source, target = self._setup()
        score = transfer.score_transfer(source, target)
        assert score.overall > 0.9

    def test_returns_transfer_score_instance(self) -> None:
        transfer, source, target = self._setup()
        score = transfer.score_transfer(source, target)
        assert isinstance(score, TransferScore)

    def test_all_layers_recommended(self) -> None:
        transfer, source, target = self._setup()
        score = transfer.score_transfer(source, target)
        assert len(score.recommended_layers) == len(score.layer_scores)

    def test_all_layer_scores_near_one(self) -> None:
        transfer, source, target = self._setup()
        score = transfer.score_transfer(source, target)
        for name, s in score.layer_scores.items():
            assert s > 0.9, f"Layer '{name}' CKA {s:.4f} unexpectedly low for identical models"


class TestFeatureTransferRandomModels:
    """score_transfer with two independently initialised models must score well below identical.

    Identical models produce overall ≈ 1.0.  Two different random initialisations
    share no learned structure, so their overall CKA should be substantially lower.
    """

    def _setup(self) -> tuple[FeatureTransfer, Task, Task]:
        probe = np.random.default_rng(1).standard_normal((100, 10)).astype(np.float32)
        transfer = FeatureTransfer(probe_data=probe, cka_threshold=0.5)
        source = _make_task()
        target = _make_task()
        torch.manual_seed(0)
        model_a = _make_mlp()
        torch.manual_seed(9999)
        model_b = _make_mlp()
        transfer.register_model(str(source.task_id), model_a)
        transfer.register_model(str(target.task_id), model_b)
        return transfer, source, target

    def test_overall_below_identical(self) -> None:
        transfer, source, target = self._setup()
        score = transfer.score_transfer(source, target)
        # Random initialisations share no learned structure; shallow networks
        # still exhibit moderate CKA due to the shared input distribution, but
        # the score should remain well below the ≈1.0 of identical models.
        assert score.overall < 0.8

    def test_reasoning_non_empty(self) -> None:
        transfer, source, target = self._setup()
        score = transfer.score_transfer(source, target)
        assert len(score.reasoning) > 0

    def test_layer_scores_populated(self) -> None:
        transfer, source, target = self._setup()
        score = transfer.score_transfer(source, target)
        assert len(score.layer_scores) > 0
