"""Unit tests for Meta-SGD meta-learner."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from orcamind.core.base import Task
from orcamind.core.meta_sgd import MetaSGD


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_maml.py to keep tests self-contained)
# ---------------------------------------------------------------------------


class _SinusoidModel(nn.Module):
    """Small five-layer MLP for sinusoidal regression; outputs shape (N,)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, squeezing last dimension to match regression targets."""
        return self.net(x).squeeze(-1)


def _make_sin_task(
    seed: int = 0,
    support_size: int = 10,
    query_size: int = 15,
) -> Task:
    """Create a sinusoidal regression Task with a fixed random amplitude and phase."""
    rng = np.random.default_rng(seed)
    amplitude = float(rng.uniform(0.1, 5.0))
    phase = float(rng.uniform(0.0, np.pi))
    support_x = torch.FloatTensor(support_size, 1).uniform_(-5, 5)
    support_y = (amplitude * torch.sin(support_x + phase)).squeeze(-1)
    query_x = torch.FloatTensor(query_size, 1).uniform_(-5, 5)
    query_y = (amplitude * torch.sin(query_x + phase)).squeeze(-1)
    return Task(support_x=support_x, support_y=support_y, query_x=query_x, query_y=query_y)


@pytest.fixture()
def sin_model() -> _SinusoidModel:
    """A freshly initialised _SinusoidModel seeded at 42."""
    torch.manual_seed(42)
    return _SinusoidModel()


@pytest.fixture()
def sin_task() -> Task:
    """A sinusoidal regression Task generated from seed 0."""
    return _make_sin_task(seed=0)


@pytest.fixture()
def meta_sgd(sin_model: _SinusoidModel) -> MetaSGD:
    """MetaSGD wrapping sin_model with MSE loss and 3 inner steps."""
    return MetaSGD(sin_model, outer_lr=0.001, inner_steps=3, loss_fn=F.mse_loss)


# ---------------------------------------------------------------------------
# Learning-rate initialisation tests
# ---------------------------------------------------------------------------


class TestLearningRateInit:
    """Tests for Meta-SGD's learnable per-parameter learning rate initialisation."""

    def test_lrs_count_matches_model_param_count(self, meta_sgd: MetaSGD) -> None:
        """self.lrs contains exactly one entry per model parameter tensor."""
        n_params = len(list(meta_sgd.model.parameters()))
        assert len(meta_sgd.lrs) == n_params

    def test_lr_shapes_match_parameter_shapes(self, meta_sgd: MetaSGD) -> None:
        """Each lr tensor has the same shape as its corresponding parameter."""
        for p, lr in zip(meta_sgd.model.parameters(), meta_sgd.lrs):
            assert lr.shape == p.shape, f"lr shape {lr.shape} != param shape {p.shape}"

    def test_initial_lr_values_are_0_01(self, meta_sgd: MetaSGD) -> None:
        """All lr elements initialise to 0.01."""
        for lr in meta_sgd.lrs:
            assert torch.allclose(lr, torch.ones_like(lr) * 0.01)

    def test_lrs_are_nn_parameters(self, meta_sgd: MetaSGD) -> None:
        """All lr tensors are nn.Parameter instances (i.e. registered for grad)."""
        for lr in meta_sgd.lrs:
            assert isinstance(lr, nn.Parameter)


# ---------------------------------------------------------------------------
# inner_loop tests
# ---------------------------------------------------------------------------


class TestInnerLoop:
    """Tests for MetaSGD.inner_loop — adaptation step and return contract."""

    def test_loss_list_length_equals_inner_steps(self, meta_sgd: MetaSGD, sin_task: Task) -> None:
        """Loss list length matches inner_steps and every element is a Python float."""
        _, losses = meta_sgd.inner_loop(sin_task.support_x, sin_task.support_y)
        assert len(losses) == meta_sgd.inner_steps
        assert all(isinstance(v, float) for v in losses)

    def test_num_steps_override(self, meta_sgd: MetaSGD, sin_task: Task) -> None:
        """Passing num_steps=2 returns exactly two loss values."""
        _, losses = meta_sgd.inner_loop(sin_task.support_x, sin_task.support_y, num_steps=2)
        assert len(losses) == 2

    def test_original_model_unchanged_after_inner_loop(self, meta_sgd: MetaSGD, sin_task: Task) -> None:
        """All original model parameters remain unchanged after inner_loop."""
        orig_state = {k: v.clone() for k, v in meta_sgd.model.state_dict().items()}
        meta_sgd.inner_loop(sin_task.support_x, sin_task.support_y)
        for key, orig_val in orig_state.items():
            assert torch.equal(meta_sgd.model.state_dict()[key], orig_val), f"{key} was modified"

    def test_adapted_output_differs_from_original(self, meta_sgd: MetaSGD, sin_task: Task) -> None:
        """Functional model output differs from original after inner-loop steps."""
        with torch.no_grad():
            orig_out = meta_sgd.model(sin_task.support_x)
        fmodel, _ = meta_sgd.inner_loop(sin_task.support_x, sin_task.support_y)
        with torch.no_grad():
            adapted_out = fmodel(sin_task.support_x)
        assert not torch.allclose(orig_out, adapted_out, atol=1e-6)


# ---------------------------------------------------------------------------
# meta_update tests
# ---------------------------------------------------------------------------


class TestMetaUpdate:
    """Tests for MetaSGD.meta_update — outer-loop gradient step and return contract."""

    def test_returns_expected_keys(self, meta_sgd: MetaSGD) -> None:
        """meta_update() return dict contains 'meta_train_loss' and 'meta_train_accuracy'."""
        tasks = [_make_sin_task(seed=i) for i in range(2)]
        result = meta_sgd.meta_update(tasks)
        assert "meta_train_loss" in result
        assert "meta_train_accuracy" in result

    def test_loss_is_non_negative(self, meta_sgd: MetaSGD) -> None:
        """meta_train_loss is non-negative after one meta-update."""
        tasks = [_make_sin_task(seed=i) for i in range(2)]
        result = meta_sgd.meta_update(tasks)
        assert result["meta_train_loss"] >= 0.0

    def test_empty_task_batch_returns_zero_metrics(self, meta_sgd: MetaSGD) -> None:
        """Empty task_batch returns zero loss and NaN accuracy without raising."""
        result = meta_sgd.meta_update([])
        assert result["meta_train_loss"] == 0.0
        assert math.isnan(result["meta_train_accuracy"])

    def test_empty_task_batch_does_not_modify_model(self, meta_sgd: MetaSGD) -> None:
        """Empty task_batch leaves all model parameters unchanged."""
        orig_state = {k: v.clone() for k, v in meta_sgd.model.state_dict().items()}
        meta_sgd.meta_update([])
        for key, orig_val in orig_state.items():
            assert torch.equal(meta_sgd.model.state_dict()[key], orig_val), f"{key} was modified"

    def test_model_params_change_after_update(self, meta_sgd: MetaSGD) -> None:
        """At least one model parameter changes after meta_update()."""
        orig_state = {k: v.clone() for k, v in meta_sgd.model.state_dict().items()}
        tasks = [_make_sin_task(seed=i) for i in range(4)]
        meta_sgd.meta_update(tasks)
        changed = any(
            not torch.equal(meta_sgd.model.state_dict()[k], orig_state[k]) for k in orig_state
        )
        assert changed, "meta_update did not modify model parameters"

    def test_lrs_updated_during_meta_training(self, meta_sgd: MetaSGD) -> None:
        """Per-parameter learning rates are modified by the outer optimizer."""
        orig_lrs = [lr.data.clone() for lr in meta_sgd.lrs]
        tasks = [_make_sin_task(seed=i) for i in range(4)]
        meta_sgd.meta_update(tasks)
        changed = any(
            not torch.equal(lr.data, orig) for lr, orig in zip(meta_sgd.lrs, orig_lrs)
        )
        assert changed, "Per-parameter learning rates were not updated during meta-training"


# ---------------------------------------------------------------------------
# adapt tests
# ---------------------------------------------------------------------------


class TestAdapt:
    """Tests for MetaSGD.adapt — inference-only task adaptation."""

    def test_does_not_modify_original_model(self, meta_sgd: MetaSGD, sin_task: Task) -> None:
        """adapt() leaves every parameter of the original model unchanged."""
        orig_state = {k: v.clone() for k, v in meta_sgd.model.state_dict().items()}
        meta_sgd.adapt(sin_task.support_x, sin_task.support_y)
        for key, orig_val in orig_state.items():
            assert torch.equal(meta_sgd.model.state_dict()[key], orig_val), f"{key} was modified"

    def test_returns_nn_module(self, meta_sgd: MetaSGD, sin_task: Task) -> None:
        """adapt() returns an nn.Module instance."""
        adapted = meta_sgd.adapt(sin_task.support_x, sin_task.support_y)
        assert isinstance(adapted, nn.Module)

    def test_adapted_model_produces_different_outputs(self, meta_sgd: MetaSGD, sin_task: Task) -> None:
        """Adapted model outputs differ from the unmodified model on the same input."""
        adapted = meta_sgd.adapt(sin_task.support_x, sin_task.support_y)
        with torch.no_grad():
            orig_out = meta_sgd.model(sin_task.support_x)
            adapted_out = adapted(sin_task.support_x)
        assert not torch.allclose(orig_out, adapted_out, atol=1e-6)


# ---------------------------------------------------------------------------
# evaluate_task tests
# ---------------------------------------------------------------------------


class TestEvaluateTask:
    """Tests for MetaSGD.evaluate_task — query-set metric computation."""

    def test_returns_expected_keys(self, meta_sgd: MetaSGD, sin_task: Task) -> None:
        """evaluate_task() return dict contains 'loss' and 'accuracy'."""
        adapted = meta_sgd.adapt(sin_task.support_x, sin_task.support_y)
        result = meta_sgd.evaluate_task(adapted, sin_task.query_x, sin_task.query_y)
        assert "loss" in result
        assert "accuracy" in result

    def test_loss_is_non_negative(self, meta_sgd: MetaSGD, sin_task: Task) -> None:
        """Returned loss value is non-negative."""
        adapted = meta_sgd.adapt(sin_task.support_x, sin_task.support_y)
        result = meta_sgd.evaluate_task(adapted, sin_task.query_x, sin_task.query_y)
        assert result["loss"] >= 0.0


# ---------------------------------------------------------------------------
# Positive learning-rate test
# ---------------------------------------------------------------------------


class TestPositiveLearningRates:
    """Per-parameter LRs should remain positive during meta-training."""

    def test_lrs_remain_positive_after_meta_training(self) -> None:
        """All lr elements stay > 0 after several meta-update steps.

        meta_update clamps each lr to ≥ 1e-8 after the outer optimizer step,
        so positivity is guaranteed regardless of Adam's update trajectory.
        """
        torch.manual_seed(0)
        model = _SinusoidModel()
        meta_sgd = MetaSGD(model, outer_lr=0.001, inner_steps=3, loss_fn=F.mse_loss)

        for step in range(5):
            tasks = [_make_sin_task(seed=step * 4 + i) for i in range(4)]
            meta_sgd.meta_update(tasks)

        all_positive = all((lr > 0).all().item() for lr in meta_sgd.lrs)
        assert all_positive, "Some per-parameter learning rates became non-positive"


# ---------------------------------------------------------------------------
# Classification accuracy branch tests
# ---------------------------------------------------------------------------


class _ClassificationModel(nn.Module):
    """Minimal MLP classifier; outputs logits of shape (N, num_classes)."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning per-class logits."""
        return self.net(x)


def _make_cls_task(
    seed: int = 0,
    support_size: int = 20,
    query_size: int = 20,
    num_classes: int = 5,
) -> Task:
    """Create a multi-class classification Task with random integer labels."""
    rng = np.random.default_rng(seed)
    support_x = torch.FloatTensor(rng.standard_normal((support_size, 4)))
    support_y = torch.LongTensor(rng.integers(0, num_classes, support_size))
    query_x = torch.FloatTensor(rng.standard_normal((query_size, 4)))
    query_y = torch.LongTensor(rng.integers(0, num_classes, query_size))
    return Task(support_x=support_x, support_y=support_y, query_x=query_x, query_y=query_y)


class TestClassificationAccuracy:
    """Verify the argmax-based accuracy branch in evaluate_task and meta_update."""

    def test_evaluate_task_classification_accuracy_in_range(self) -> None:
        """evaluate_task returns accuracy in [0, 1] for multi-class classification."""
        torch.manual_seed(0)
        model = _ClassificationModel()
        meta_sgd = MetaSGD(model, outer_lr=0.001, inner_steps=3, loss_fn=F.cross_entropy)
        task = _make_cls_task(seed=0)
        adapted = meta_sgd.adapt(task.support_x, task.support_y)
        result = meta_sgd.evaluate_task(adapted, task.query_x, task.query_y)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_meta_update_classification_accuracy_in_range(self) -> None:
        """meta_update returns meta_train_accuracy in [0, 1] for multi-class classification."""
        torch.manual_seed(0)
        model = _ClassificationModel()
        meta_sgd = MetaSGD(model, outer_lr=0.001, inner_steps=3, loss_fn=F.cross_entropy)
        tasks = [_make_cls_task(seed=i) for i in range(2)]
        result = meta_sgd.meta_update(tasks)
        assert "meta_train_accuracy" in result
        assert 0.0 <= result["meta_train_accuracy"] <= 1.0

    def test_adapted_accuracy_above_chance_on_separable_task(self) -> None:
        """Adapted model achieves accuracy > chance on a trivially separable task.

        Label = (x[:,0] > 0).long() (binary, two-feature input).  After a few
        meta-update steps the adapted model should predict correctly more often
        than random (50% chance for balanced binary labels).
        """
        torch.manual_seed(42)

        class _BinaryModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        def _make_separable_task(seed: int, n: int = 30) -> Task:
            rng = torch.Generator()
            rng.manual_seed(seed)
            x = torch.randn(n, 2, generator=rng)
            y = (x[:, 0] > 0).long()
            mid = n // 2
            return Task(support_x=x[:mid], support_y=y[:mid], query_x=x[mid:], query_y=y[mid:])

        model = _BinaryModel()
        meta_sgd = MetaSGD(model, outer_lr=0.005, inner_steps=5, loss_fn=F.cross_entropy)

        for step in range(10):
            tasks = [_make_separable_task(seed=step * 4 + i) for i in range(4)]
            meta_sgd.meta_update(tasks)

        eval_task = _make_separable_task(seed=999)
        adapted = meta_sgd.adapt(eval_task.support_x, eval_task.support_y)
        result = meta_sgd.evaluate_task(adapted, eval_task.query_x, eval_task.query_y)
        assert result["accuracy"] > 0.5, (
            f"Expected accuracy above chance (0.5) on a separable task, got {result['accuracy']:.3f}"
        )
