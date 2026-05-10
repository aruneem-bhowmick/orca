"""Unit tests for Reptile meta-learner."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from orcamind.core.base import Task
from orcamind.core.reptile import Reptile


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
def reptile(sin_model: _SinusoidModel) -> Reptile:
    """Reptile wrapping sin_model with MSE loss and 5 inner steps."""
    return Reptile(sin_model, inner_lr=0.02, outer_lr=0.1, inner_steps=5, loss_fn=F.mse_loss)


# ---------------------------------------------------------------------------
# inner_loop tests
# ---------------------------------------------------------------------------


class TestInnerLoop:
    """Tests for Reptile.inner_loop — adaptation step and return contract."""

    def test_returns_nn_module(self, reptile: Reptile, sin_task: Task) -> None:
        """inner_loop returns an nn.Module instance."""
        adapted, _ = reptile.inner_loop(sin_task.support_x, sin_task.support_y)
        assert isinstance(adapted, nn.Module)

    def test_loss_list_length_equals_inner_steps(self, reptile: Reptile, sin_task: Task) -> None:
        """Loss list length matches inner_steps and every element is a Python float."""
        _, losses = reptile.inner_loop(sin_task.support_x, sin_task.support_y)
        assert len(losses) == reptile.inner_steps
        assert all(isinstance(v, float) for v in losses)

    def test_num_steps_override(self, reptile: Reptile, sin_task: Task) -> None:
        """Passing num_steps=3 returns exactly three loss values."""
        _, losses = reptile.inner_loop(sin_task.support_x, sin_task.support_y, num_steps=3)
        assert len(losses) == 3

    def test_original_model_unchanged_after_inner_loop(self, reptile: Reptile, sin_task: Task) -> None:
        """All original model parameters remain unchanged after inner_loop."""
        orig_state = {k: v.clone() for k, v in reptile.model.state_dict().items()}
        reptile.inner_loop(sin_task.support_x, sin_task.support_y)
        for key, orig_val in orig_state.items():
            assert torch.equal(reptile.model.state_dict()[key], orig_val), f"{key} was modified"

    def test_adapted_output_differs_from_original(self, reptile: Reptile, sin_task: Task) -> None:
        """Adapted model output differs from original after inner-loop steps."""
        with torch.no_grad():
            orig_out = reptile.model(sin_task.support_x)
        adapted, _ = reptile.inner_loop(sin_task.support_x, sin_task.support_y)
        with torch.no_grad():
            adapted_out = adapted(sin_task.support_x)
        assert not torch.allclose(orig_out, adapted_out, atol=1e-6)


# ---------------------------------------------------------------------------
# meta_update tests
# ---------------------------------------------------------------------------


class TestMetaUpdate:
    """Tests for Reptile.meta_update — interpolation update rule and return contract."""

    def test_returns_expected_keys(self, reptile: Reptile) -> None:
        """meta_update() return dict contains 'meta_train_loss' and 'meta_train_accuracy'."""
        tasks = [_make_sin_task(seed=i) for i in range(2)]
        result = reptile.meta_update(tasks)
        assert "meta_train_loss" in result
        assert "meta_train_accuracy" in result

    def test_loss_is_non_negative(self, reptile: Reptile) -> None:
        """meta_train_loss is non-negative after one meta-update."""
        tasks = [_make_sin_task(seed=i) for i in range(2)]
        result = reptile.meta_update(tasks)
        assert result["meta_train_loss"] >= 0.0

    def test_empty_task_batch_returns_zero_metrics(self, reptile: Reptile) -> None:
        """Empty task_batch returns zero loss and NaN accuracy without raising."""
        result = reptile.meta_update([])
        assert result["meta_train_loss"] == 0.0
        assert math.isnan(result["meta_train_accuracy"])

    def test_empty_task_batch_does_not_modify_model(self, reptile: Reptile) -> None:
        """Empty task_batch leaves all model parameters unchanged."""
        orig_state = {k: v.clone() for k, v in reptile.model.state_dict().items()}
        reptile.meta_update([])
        for key, orig_val in orig_state.items():
            assert torch.equal(reptile.model.state_dict()[key], orig_val), f"{key} was modified"

    def test_model_params_change_after_update(self, reptile: Reptile) -> None:
        """At least one model parameter changes after meta_update()."""
        orig_state = {k: v.clone() for k, v in reptile.model.state_dict().items()}
        tasks = [_make_sin_task(seed=i) for i in range(4)]
        reptile.meta_update(tasks)
        changed = any(
            not torch.equal(reptile.model.state_dict()[k], orig_state[k]) for k in orig_state
        )
        assert changed, "meta_update did not modify model parameters"

    def test_interpolation_moves_params_toward_adapted(self, sin_task: Task) -> None:
        """Reptile outer update reduces the distance to the adapted parameters.

        With outer_lr=0.5, the interpolation is halfway, so distance must halve.

        The test calls inner_loop *before* meta_update while θ is still at its
        initial value.  meta_update internally re-runs inner_loop from the same
        θ, so both invocations produce the same φ — a guarantee that holds as
        long as the inner optimiser (SGD on this deterministic MLP) is
        reproducible.  torch.use_deterministic_algorithms(True) makes that
        requirement explicit and turns accidental non-determinism into an error.
        """
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        try:
            model = _SinusoidModel()
            reptile = Reptile(model, inner_lr=0.02, outer_lr=0.5, inner_steps=5, loss_fn=F.mse_loss)

            # φ computed from initial θ; meta_update will produce the same φ
            # internally because θ has not been modified yet.
            adapted, _ = reptile.inner_loop(sin_task.support_x, sin_task.support_y)
            adapted_params = [p.data.clone() for p in adapted.parameters()]

            dist_before = sum(
                (p.data - phi).norm().item()
                for p, phi in zip(reptile.model.parameters(), adapted_params)
            )

            reptile.meta_update([sin_task])

            dist_after = sum(
                (p.data - phi).norm().item()
                for p, phi in zip(reptile.model.parameters(), adapted_params)
            )

            assert dist_after < dist_before, (
                f"Reptile did not move params toward adapted: before={dist_before:.4f}, after={dist_after:.4f}"
            )
        finally:
            torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# adapt tests
# ---------------------------------------------------------------------------


class TestAdapt:
    """Tests for Reptile.adapt — inference-only task adaptation."""

    def test_does_not_modify_original_model(self, reptile: Reptile, sin_task: Task) -> None:
        """adapt() leaves every parameter of the original model unchanged."""
        orig_state = {k: v.clone() for k, v in reptile.model.state_dict().items()}
        reptile.adapt(sin_task.support_x, sin_task.support_y)
        for key, orig_val in orig_state.items():
            assert torch.equal(reptile.model.state_dict()[key], orig_val), f"{key} was modified"

    def test_returns_nn_module(self, reptile: Reptile, sin_task: Task) -> None:
        """adapt() returns an nn.Module instance."""
        adapted = reptile.adapt(sin_task.support_x, sin_task.support_y)
        assert isinstance(adapted, nn.Module)

    def test_adapted_model_produces_different_outputs(self, reptile: Reptile, sin_task: Task) -> None:
        """Adapted model outputs differ from the unmodified model on the same input."""
        adapted = reptile.adapt(sin_task.support_x, sin_task.support_y)
        with torch.no_grad():
            orig_out = reptile.model(sin_task.support_x)
            adapted_out = adapted(sin_task.support_x)
        assert not torch.allclose(orig_out, adapted_out, atol=1e-6)


# ---------------------------------------------------------------------------
# evaluate_task tests
# ---------------------------------------------------------------------------


class TestEvaluateTask:
    """Tests for Reptile.evaluate_task — query-set metric computation."""

    def test_returns_expected_keys(self, reptile: Reptile, sin_task: Task) -> None:
        """evaluate_task() return dict contains 'loss' and 'accuracy'."""
        adapted = reptile.adapt(sin_task.support_x, sin_task.support_y)
        result = reptile.evaluate_task(adapted, sin_task.query_x, sin_task.query_y)
        assert "loss" in result
        assert "accuracy" in result

    def test_loss_is_non_negative(self, reptile: Reptile, sin_task: Task) -> None:
        """Returned loss value is non-negative."""
        adapted = reptile.adapt(sin_task.support_x, sin_task.support_y)
        result = reptile.evaluate_task(adapted, sin_task.query_x, sin_task.query_y)
        assert result["loss"] >= 0.0


# ---------------------------------------------------------------------------
# Meta-training convergence test
# ---------------------------------------------------------------------------


class TestConvergence:
    """Convergence test: meta-loss should decrease over 20 outer-loop steps."""

    def test_meta_loss_decreases_over_20_steps(self) -> None:
        """Mean query loss over steps 15–19 is below mean over steps 0–4.

        torch.use_deterministic_algorithms(True) is set to ensure the seed-based
        reproducibility that the 5%-improvement threshold depends on.
        """
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        try:
            model = _SinusoidModel()
            reptile = Reptile(model, inner_lr=0.02, outer_lr=0.1, inner_steps=5, loss_fn=F.mse_loss)

            losses: list[float] = []
            for step in range(20):
                tasks = [_make_sin_task(seed=step * 4 + i) for i in range(4)]
                metrics = reptile.meta_update(tasks)
                losses.append(metrics["meta_train_loss"])

            first_five = sum(losses[:5]) / 5
            last_five = sum(losses[15:]) / 5
            assert last_five < 0.95 * first_five, (
                f"Reptile meta-loss did not decrease by ≥5%: first-5 avg={first_five:.4f}, last-5 avg={last_five:.4f}"
            )
        finally:
            torch.use_deterministic_algorithms(False)
