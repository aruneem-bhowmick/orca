"""Unit tests for MAML meta-learner."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from orcamind.core.base import Task
from orcamind.core.maml import MAML


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SinusoidModel(nn.Module):
    """Small MLP for sinusoidal regression; outputs shape (N,)."""

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
        return self.net(x).squeeze(-1)


def _make_sin_task(
    seed: int = 0,
    support_size: int = 10,
    query_size: int = 15,
) -> Task:
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
    torch.manual_seed(42)
    return _SinusoidModel()


@pytest.fixture()
def sin_task() -> Task:
    return _make_sin_task(seed=0)


@pytest.fixture()
def maml(sin_model: _SinusoidModel) -> MAML:
    return MAML(sin_model, inner_lr=0.01, outer_lr=0.001, inner_steps=3, loss_fn=F.mse_loss)


# ---------------------------------------------------------------------------
# inner_loop tests
# ---------------------------------------------------------------------------


class TestInnerLoop:
    def test_returns_adapted_outputs(self, maml: MAML, sin_task: Task) -> None:
        with torch.no_grad():
            orig_out = maml.model(sin_task.support_x)

        fmodel, _ = maml.inner_loop(sin_task.support_x, sin_task.support_y)

        with torch.no_grad():
            adapted_out = fmodel(sin_task.support_x)

        assert not torch.allclose(orig_out, adapted_out, atol=1e-6)

    def test_loss_list_length_equals_inner_steps(self, maml: MAML, sin_task: Task) -> None:
        _, losses = maml.inner_loop(sin_task.support_x, sin_task.support_y)
        assert len(losses) == maml.inner_steps
        assert all(isinstance(v, float) for v in losses)

    def test_num_steps_override(self, maml: MAML, sin_task: Task) -> None:
        _, losses = maml.inner_loop(sin_task.support_x, sin_task.support_y, num_steps=2)
        assert len(losses) == 2

    def test_original_model_unchanged_after_inner_loop(self, maml: MAML, sin_task: Task) -> None:
        orig_state = {k: v.clone() for k, v in maml.model.state_dict().items()}
        maml.inner_loop(sin_task.support_x, sin_task.support_y)
        for key, orig_val in orig_state.items():
            assert torch.equal(maml.model.state_dict()[key], orig_val), f"{key} was modified"


# ---------------------------------------------------------------------------
# adapt tests
# ---------------------------------------------------------------------------


class TestAdapt:
    def test_does_not_modify_original_model(self, maml: MAML, sin_task: Task) -> None:
        orig_state = {k: v.clone() for k, v in maml.model.state_dict().items()}
        maml.adapt(sin_task.support_x, sin_task.support_y)
        for key, orig_val in orig_state.items():
            assert torch.equal(maml.model.state_dict()[key], orig_val), f"{key} was modified"

    def test_adapted_model_produces_different_outputs(self, maml: MAML, sin_task: Task) -> None:
        adapted = maml.adapt(sin_task.support_x, sin_task.support_y)
        with torch.no_grad():
            orig_out = maml.model(sin_task.support_x)
            adapted_out = adapted(sin_task.support_x)
        assert not torch.allclose(orig_out, adapted_out, atol=1e-6)

    def test_returns_nn_module(self, maml: MAML, sin_task: Task) -> None:
        adapted = maml.adapt(sin_task.support_x, sin_task.support_y)
        assert isinstance(adapted, nn.Module)


# ---------------------------------------------------------------------------
# evaluate_task tests
# ---------------------------------------------------------------------------


class TestEvaluateTask:
    def test_returns_expected_keys(self, maml: MAML, sin_task: Task) -> None:
        adapted = maml.adapt(sin_task.support_x, sin_task.support_y)
        result = maml.evaluate_task(adapted, sin_task.query_x, sin_task.query_y)
        assert "loss" in result
        assert "accuracy" in result

    def test_loss_is_non_negative(self, maml: MAML, sin_task: Task) -> None:
        adapted = maml.adapt(sin_task.support_x, sin_task.support_y)
        result = maml.evaluate_task(adapted, sin_task.query_x, sin_task.query_y)
        assert result["loss"] >= 0.0


# ---------------------------------------------------------------------------
# meta_update tests
# ---------------------------------------------------------------------------


class TestMetaUpdate:
    def test_returns_expected_keys(self, maml: MAML) -> None:
        tasks = [_make_sin_task(seed=i) for i in range(2)]
        result = maml.meta_update(tasks)
        assert "meta_train_loss" in result
        assert "meta_train_accuracy" in result

    def test_loss_is_non_negative(self, maml: MAML) -> None:
        tasks = [_make_sin_task(seed=i) for i in range(2)]
        result = maml.meta_update(tasks)
        assert result["meta_train_loss"] >= 0.0

    def test_model_params_change_after_update(self, maml: MAML) -> None:
        orig_state = {k: v.clone() for k, v in maml.model.state_dict().items()}
        tasks = [_make_sin_task(seed=i) for i in range(4)]
        maml.meta_update(tasks)
        changed = any(
            not torch.equal(maml.model.state_dict()[k], orig_state[k])
            for k in orig_state
        )
        assert changed, "meta_update did not modify model parameters"

    def test_fomaml_updates_model_params(self, sin_model: _SinusoidModel) -> None:
        maml_fo = MAML(sin_model, inner_steps=3, first_order=True, loss_fn=F.mse_loss)
        orig_state = {k: v.clone() for k, v in sin_model.state_dict().items()}
        tasks = [_make_sin_task(seed=i) for i in range(4)]
        maml_fo.meta_update(tasks)
        changed = any(
            not torch.equal(sin_model.state_dict()[k], orig_state[k])
            for k in orig_state
        )
        assert changed

    def test_empty_task_batch_returns_zero_metrics(self, maml: MAML) -> None:
        result = maml.meta_update([])
        assert result["meta_train_loss"] == 0.0
        assert result["meta_train_accuracy"] == 0.0

    def test_empty_task_batch_does_not_modify_model(self, maml: MAML) -> None:
        orig_state = {k: v.clone() for k, v in maml.model.state_dict().items()}
        maml.meta_update([])
        for key, orig_val in orig_state.items():
            assert torch.equal(maml.model.state_dict()[key], orig_val), f"{key} was modified"


# ---------------------------------------------------------------------------
# Meta-training convergence test
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_meta_loss_decreases_over_20_steps(self) -> None:
        """Meta-loss averaged over the last 5 steps should be lower than the first 5."""
        torch.manual_seed(0)
        model = _SinusoidModel()
        maml = MAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=5, loss_fn=F.mse_loss)

        losses: list[float] = []
        for step in range(20):
            tasks = [_make_sin_task(seed=step * 4 + i) for i in range(4)]
            metrics = maml.meta_update(tasks)
            losses.append(metrics["meta_train_loss"])

        first_five = sum(losses[:5]) / 5
        last_five = sum(losses[15:]) / 5
        assert last_five < first_five, (
            f"Meta-loss did not decrease: first-5 avg={first_five:.4f}, last-5 avg={last_five:.4f}"
        )


# ---------------------------------------------------------------------------
# FOMAML gradient direction test
# ---------------------------------------------------------------------------


class TestFOMAMLGradientDirection:
    def test_fomaml_update_direction_correlates_with_full_maml(self) -> None:
        """FOMAML and second-order MAML meta-gradients should have Pearson correlation > 0.9.

        With inner_steps=1, the two-order terms vanish and both variants produce
        nearly identical gradients, making correlation > 0.9 reliably achievable.
        """
        torch.manual_seed(42)
        model_so = _SinusoidModel()
        model_fo = copy.deepcopy(model_so)

        tasks = [_make_sin_task(seed=i) for i in range(4)]

        maml_so = MAML(model_so, inner_lr=0.01, outer_lr=0.001, inner_steps=1, first_order=False, loss_fn=F.mse_loss)
        maml_fo = MAML(model_fo, inner_lr=0.01, outer_lr=0.001, inner_steps=1, first_order=True, loss_fn=F.mse_loss)

        maml_so.meta_update(tasks)
        maml_fo.meta_update(tasks)

        # optimizer.step() does not clear .grad, so gradients are still available
        grads_so = torch.cat([p.grad.flatten() for p in model_so.parameters() if p.grad is not None])
        grads_fo = torch.cat([p.grad.flatten() for p in model_fo.parameters() if p.grad is not None])

        assert grads_so.numel() > 0, "No gradients found for second-order MAML"
        assert grads_fo.numel() > 0, "No gradients found for FOMAML"

        corr = torch.corrcoef(torch.stack([grads_so, grads_fo]))[0, 1].item()
        assert corr > 0.9, f"FOMAML/MAML gradient correlation {corr:.4f} not > 0.9"
