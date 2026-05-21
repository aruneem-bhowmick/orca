"""Unit tests for linear_cka and FeatureTransfer."""

from __future__ import annotations

import numpy as np
import pytest

from orcanet.transfer.feature_transfer import linear_cka


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
