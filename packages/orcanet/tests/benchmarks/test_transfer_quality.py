"""Transfer recommendation quality benchmark.

Validates that ``FeatureTransfer.score_transfer`` produces CKA-based
transferability scores that correlate with actual transfer performance,
as measured by Spearman rank correlation across 20 synthetic task pairs.

Design
------
* 10 **high-transfer** pairs: source and target share identical weights
  (CKA ≈ 1.0) → actual performance label = 1.0.
* 10 **low-transfer** pairs: source and target are independently
  randomly initialised (CKA typically in [0.05, 0.40]) → label = 0.0.

Spearman(CKA scores, labels) > 0.60 confirms that CKA is a reliable
proxy for transfer performance across the pair population.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from orca_shared.schemas.task import Task
from orcanet.transfer.feature_transfer import FeatureTransfer

# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------

_N_HIGH: int = 10
_N_LOW: int = 10
_SPEARMAN_THRESHOLD: float = 0.60

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)

# Probe data: 64 samples × 8 features (shared by all model pairs)
_PROBE_DATA: np.ndarray = np.random.default_rng(0).standard_normal((64, 8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Minimal MLP for activation capture
# ---------------------------------------------------------------------------


def _make_mlp(seed: int = 0) -> nn.Module:
    """Return a small deterministic MLP with 3 linear layers.

    Args:
        seed: Random seed used to initialise weights.  Different seeds
              produce uncorrelated activation patterns (low CKA).

    Returns:
        An ``nn.Sequential`` with shape 8 → 32 → 16 → 8.
    """
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )


def _make_task() -> Task:
    """Return a minimal Task object with a fresh UUID."""
    return Task(
        task_id=uuid4(),
        name="bench-task",
        task_type="classification",
        created_at=_NOW,
        updated_at=_NOW,
    )


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestTransferRecommendationQuality:
    """Spearman-correlation benchmark for FeatureTransfer.score_transfer."""

    def _build_strategy(self) -> FeatureTransfer:
        """Return a fresh FeatureTransfer with shared probe data."""
        return FeatureTransfer(probe_data=_PROBE_DATA, cka_threshold=0.5)

    def test_spearman_correlation_exceeds_threshold(self) -> None:
        """Spearman(CKA_score, actual_label) across 20 pairs must be > 0.60.

        High-transfer pairs (identical weights) must rank above low-transfer
        pairs (independently random weights) consistently enough to exceed
        the rank-correlation threshold.
        """
        strategy = self._build_strategy()
        cka_scores: list[float] = []
        actual_labels: list[float] = []

        # ---- High-transfer pairs (identical weights → CKA ≈ 1.0) ----
        for seed in range(_N_HIGH):
            source_task = _make_task()
            target_task = _make_task()
            model = _make_mlp(seed=seed)
            clone = deepcopy(model)  # identical weights

            strategy.register_model(str(source_task.task_id), model)
            strategy.register_model(str(target_task.task_id), clone)

            score = strategy.score_transfer(source_task, target_task)
            cka_scores.append(score.overall)
            actual_labels.append(1.0)

        # ---- Low-transfer pairs (different random seeds → low CKA) ----
        for seed in range(_N_LOW):
            source_task = _make_task()
            target_task = _make_task()
            # Different seeds → uncorrelated activation patterns
            source_model = _make_mlp(seed=seed * 100 + 7)
            target_model = _make_mlp(seed=seed * 100 + 77)

            strategy.register_model(str(source_task.task_id), source_model)
            strategy.register_model(str(target_task.task_id), target_model)

            score = strategy.score_transfer(source_task, target_task)
            cka_scores.append(score.overall)
            actual_labels.append(0.0)

        rho, _ = spearmanr(cka_scores, actual_labels)
        assert rho > _SPEARMAN_THRESHOLD, (
            f"Spearman ρ = {rho:.3f} is below the required threshold "
            f"{_SPEARMAN_THRESHOLD}. CKA is not reliably ordering transfer "
            "quality across the 20 synthetic pairs."
        )

    def test_high_transfer_pairs_have_higher_cka_than_low(self) -> None:
        """Mean CKA of identical-weight pairs must exceed mean CKA of random pairs."""
        strategy = self._build_strategy()
        high_scores: list[float] = []
        low_scores: list[float] = []

        for seed in range(_N_HIGH):
            src, tgt = _make_task(), _make_task()
            m = _make_mlp(seed=seed)
            strategy.register_model(str(src.task_id), m)
            strategy.register_model(str(tgt.task_id), deepcopy(m))
            high_scores.append(strategy.score_transfer(src, tgt).overall)

        for seed in range(_N_LOW):
            src, tgt = _make_task(), _make_task()
            strategy.register_model(str(src.task_id), _make_mlp(seed=seed * 100 + 7))
            strategy.register_model(str(tgt.task_id), _make_mlp(seed=seed * 100 + 77))
            low_scores.append(strategy.score_transfer(src, tgt).overall)

        mean_high = float(np.mean(high_scores))
        mean_low = float(np.mean(low_scores))
        assert mean_high > mean_low, (
            f"Mean CKA for identical-weight pairs ({mean_high:.3f}) must exceed "
            f"mean CKA for random pairs ({mean_low:.3f})."
        )

    def test_identical_models_approach_cka_one(self) -> None:
        """CKA for identical-weight pairs must be close to 1.0 (> 0.85)."""
        strategy = self._build_strategy()
        scores: list[float] = []

        for seed in range(5):
            src, tgt = _make_task(), _make_task()
            m = _make_mlp(seed=seed)
            strategy.register_model(str(src.task_id), m)
            strategy.register_model(str(tgt.task_id), deepcopy(m))
            scores.append(strategy.score_transfer(src, tgt).overall)

        mean_score = float(np.mean(scores))
        assert mean_score > 0.85, (
            f"Mean CKA for identical models = {mean_score:.3f}; expected > 0.85"
        )

    def test_random_models_produce_low_cka(self) -> None:
        """CKA for independently random-initialised pairs must be < 0.50."""
        strategy = self._build_strategy()
        scores: list[float] = []

        for seed in range(5):
            src, tgt = _make_task(), _make_task()
            strategy.register_model(str(src.task_id), _make_mlp(seed=seed * 200 + 3))
            strategy.register_model(str(tgt.task_id), _make_mlp(seed=seed * 200 + 99))
            scores.append(strategy.score_transfer(src, tgt).overall)

        mean_score = float(np.mean(scores))
        assert mean_score < 0.50, (
            f"Mean CKA for random models = {mean_score:.3f}; expected < 0.50"
        )

    def test_score_overall_is_in_valid_range(self) -> None:
        """TransferScore.overall must be in [0, 1] for all 20 pairs."""
        strategy = self._build_strategy()

        for seed in range(_N_HIGH):
            src, tgt = _make_task(), _make_task()
            m = _make_mlp(seed=seed)
            strategy.register_model(str(src.task_id), m)
            strategy.register_model(str(tgt.task_id), deepcopy(m))
            score = strategy.score_transfer(src, tgt)
            assert 0.0 <= score.overall <= 1.0, (
                f"score.overall = {score.overall} is outside [0, 1]"
            )

        for seed in range(_N_LOW):
            src, tgt = _make_task(), _make_task()
            strategy.register_model(str(src.task_id), _make_mlp(seed=seed * 100 + 7))
            strategy.register_model(str(tgt.task_id), _make_mlp(seed=seed * 100 + 77))
            score = strategy.score_transfer(src, tgt)
            assert 0.0 <= score.overall <= 1.0, (
                f"score.overall = {score.overall} is outside [0, 1]"
            )
