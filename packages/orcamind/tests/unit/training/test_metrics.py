"""Unit tests for meta-learning evaluation metrics."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, call

import pytest
import torch

from orcamind.core.base import Task
from orcamind.training.metrics import (
    adaptation_efficiency,
    catastrophic_forgetting,
    k_shot_accuracy,
)


def _make_task(k: int = 5, q: int = 15, features: int = 4) -> Task:
    torch.manual_seed(1)
    return Task(
        support_x=torch.randn(k, features),
        support_y=torch.randint(0, 2, (k,)),
        query_x=torch.randn(q, features),
        query_y=torch.randint(0, 2, (q,)),
    )


def _mock_learner(accuracy: float = 0.75) -> MagicMock:
    m = MagicMock()
    m.adapt.return_value = MagicMock()
    m.evaluate_task.return_value = {"loss": 0.3, "accuracy": accuracy}
    return m


class TestKShotAccuracy:
    def test_returns_float(self):
        """Returns a plain float, not a tensor."""
        learner = _mock_learner()
        result = k_shot_accuracy(learner, [_make_task()], k_shot=3, n_query=5)
        assert isinstance(result, float)

    def test_result_in_zero_one_range(self):
        """Result is in [0, 1]."""
        learner = _mock_learner(accuracy=0.8)
        result = k_shot_accuracy(learner, [_make_task()] * 5, k_shot=3, n_query=5)
        assert 0.0 <= result <= 1.0

    def test_calls_adapt_once_per_task(self):
        """adapt is called exactly once for each task."""
        learner = _mock_learner()
        tasks = [_make_task()] * 4
        k_shot_accuracy(learner, tasks, k_shot=3, n_query=5)
        assert learner.adapt.call_count == 4

    def test_calls_evaluate_task_once_per_task(self):
        """evaluate_task is called exactly once for each task."""
        learner = _mock_learner()
        tasks = [_make_task()] * 3
        k_shot_accuracy(learner, tasks, k_shot=3, n_query=5)
        assert learner.evaluate_task.call_count == 3

    def test_uses_k_shot_support_examples(self):
        """adapt receives support_x[:k_shot] and support_y[:k_shot]."""
        learner = _mock_learner()
        task = _make_task(k=10, q=10)
        k_shot_accuracy(learner, [task], k_shot=3, n_query=5)
        args = learner.adapt.call_args[0]
        assert args[0].shape[0] == 3
        assert args[1].shape[0] == 3

    def test_uses_n_query_examples(self):
        """evaluate_task receives query_x[:n_query] and query_y[:n_query]."""
        learner = _mock_learner()
        task = _make_task(k=10, q=15)
        k_shot_accuracy(learner, [task], k_shot=3, n_query=7)
        _, args, _ = learner.evaluate_task.mock_calls[0]
        assert args[1].shape[0] == 7

    def test_handles_nan_accuracy_gracefully(self):
        """NaN accuracy values are skipped; result is 0.0 if all NaN."""
        learner = _mock_learner()
        learner.evaluate_task.return_value = {
            "loss": 0.5,
            "accuracy": float("nan"),
        }
        result = k_shot_accuracy(learner, [_make_task()] * 3, k_shot=3, n_query=5)
        assert result == 0.0
        assert not math.isnan(result)


class TestAdaptationEfficiency:
    def test_returns_one_for_constantly_zero_losses(self):
        """A model that already has zero loss everywhere has efficiency 1.0."""
        losses = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert adaptation_efficiency(losses) == 1.0

    def test_value_in_zero_one_range(self):
        """Result is always in [0, 1]."""
        losses = [[1.0, 0.8, 0.6], [1.0, 0.9, 0.7]]
        result = adaptation_efficiency(losses)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize(
        "losses",
        [
            [[1.0, 0.5, 0.1]],
            [[2.0, 1.5, 1.0, 0.5], [3.0, 2.0, 1.0, 0.5]],
            [[0.9, 0.1]],
        ],
    )
    def test_valid_result_for_various_curves(self, losses):
        """Result is a finite float in [0, 1] for various curves."""
        result = adaptation_efficiency(losses)
        assert 0.0 <= result <= 1.0
        assert not math.isnan(result)

    def test_single_step_returns_valid_value(self):
        """Single-step curves don't cause division by zero."""
        result = adaptation_efficiency([[0.5], [0.5]])
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_raises_on_unequal_step_lengths(self):
        """ValueError when inner lists differ in length."""
        with pytest.raises(ValueError, match="equal length"):
            adaptation_efficiency([[1.0, 0.5], [1.0, 0.5, 0.2]])

    def test_raises_on_empty_input(self):
        """ValueError when input is empty."""
        with pytest.raises(ValueError):
            adaptation_efficiency([])

    def test_raises_on_empty_trajectory(self):
        """ValueError when each task's step list is empty."""
        with pytest.raises(ValueError, match="at least one step"):
            adaptation_efficiency([[], []])


class TestCatastrophicForgetting:
    def test_returns_non_negative_float(self):
        """Result is always >= 0.0."""
        learner = _mock_learner(accuracy=0.8)
        old_tasks = [_make_task()] * 3
        new_tasks = [_make_task()] * 3
        result = catastrophic_forgetting(learner, old_tasks, new_tasks)
        assert result >= 0.0
        assert isinstance(result, float)

    def test_calls_meta_update_with_new_tasks(self):
        """meta_update is called once with the new_tasks list."""
        learner = _mock_learner()
        old_tasks = [_make_task()] * 2
        new_tasks = [_make_task()] * 2
        catastrophic_forgetting(learner, old_tasks, new_tasks)
        learner.meta_update.assert_called_once_with(new_tasks)

    def test_returns_zero_when_no_forgetting(self):
        """Returns 0.0 when acc_after >= acc_before."""
        learner = MagicMock()
        # First two adapt/evaluate_task calls (before): accuracy 0.5
        # Next two calls (after): accuracy 0.9 (improved)
        learner.adapt.return_value = MagicMock()
        learner.evaluate_task.side_effect = [
            {"loss": 0.5, "accuracy": 0.5},
            {"loss": 0.5, "accuracy": 0.5},
            {"loss": 0.3, "accuracy": 0.9},
            {"loss": 0.3, "accuracy": 0.9},
        ]
        old_tasks = [_make_task()] * 2
        new_tasks = [_make_task()] * 2
        result = catastrophic_forgetting(
            learner, old_tasks, new_tasks, k_shot=3, n_query=5
        )
        assert result == 0.0

    def test_measures_forgetting_when_performance_drops(self):
        """Returns positive value when accuracy decreases after training."""
        learner = MagicMock()
        learner.adapt.return_value = MagicMock()
        # Before: accuracy 0.9; After: accuracy 0.6
        learner.evaluate_task.side_effect = [
            {"loss": 0.2, "accuracy": 0.9},
            {"loss": 0.5, "accuracy": 0.6},
        ]
        old_tasks = [_make_task()]
        new_tasks = [_make_task()]
        result = catastrophic_forgetting(
            learner, old_tasks, new_tasks, k_shot=3, n_query=5
        )
        assert result == pytest.approx(0.3, abs=1e-5)
