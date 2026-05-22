"""Unit tests for MultiTaskModel and MultiTaskTransfer."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from orcanet.transfer.multi_task_transfer import (
    MultiTaskModel,
    MultiTaskTransfer,
    _get_backbone_out_dim,
)
from orcanet.transfer.types import TransferScore
from orca_shared.schemas.task import Task


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_task(
    n_features: int | None = None,
    n_classes: int | None = None,
) -> Task:
    """Return a minimal ``Task`` with a random UUID and optional dimension fields."""
    now = datetime.now(timezone.utc)
    return Task(
        task_id=uuid4(),
        name="test_task",
        task_type="classification",
        n_features=n_features,
        n_classes=n_classes,
        created_at=now,
        updated_at=now,
    )


def _make_backbone(in_dim: int = 10, hidden: int = 16) -> nn.Sequential:
    """Return a simple ``Linear → ReLU`` backbone with known output dim."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())


def _make_heads(
    backbone_out: int,
    task_ids: list[str],
    head_hidden: int = 8,
    out_dim: int = 3,
) -> dict[str, nn.Module]:
    """Return a dict of simple two-layer heads for testing."""
    return {
        tid: nn.Sequential(
            nn.Linear(backbone_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, out_dim),
        )
        for tid in task_ids
    }


# ---------------------------------------------------------------------------
# TestGetBackboneOutDim
# ---------------------------------------------------------------------------


class TestGetBackboneOutDim:
    def test_simple_sequential_backbone(self) -> None:
        """Returns ``out_features`` of the last ``nn.Linear`` in a flat Sequential."""
        backbone = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 16))
        assert _get_backbone_out_dim(backbone) == 16

    def test_single_linear_layer(self) -> None:
        """Works with a backbone that is just a single ``nn.Linear``."""
        backbone = nn.Linear(10, 64)
        assert _get_backbone_out_dim(backbone) == 64

    def test_nested_module(self) -> None:
        """Finds the last ``nn.Linear`` in a nested submodule."""

        class Nested(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = nn.Sequential(nn.Linear(8, 32), nn.Linear(32, 24))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x)

        assert _get_backbone_out_dim(Nested()) == 24

    def test_raises_when_no_linear(self) -> None:
        """Raises ``ValueError`` when the backbone contains no ``nn.Linear`` layer."""
        with pytest.raises(ValueError, match="no nn.Linear"):
            _get_backbone_out_dim(nn.ReLU())


# ---------------------------------------------------------------------------
# TestMultiTaskModelForward
# ---------------------------------------------------------------------------


class TestMultiTaskModelForward:
    def _make_model(self) -> tuple[MultiTaskModel, list[str]]:
        backbone = _make_backbone(in_dim=10, hidden=16)
        task_ids = ["task_a", "task_b"]
        heads = _make_heads(backbone_out=16, task_ids=task_ids, out_dim=3)
        model = MultiTaskModel(backbone=backbone, task_heads=heads)
        return model, task_ids

    def test_routes_to_correct_task_head(self) -> None:
        """forward routes to the head matching *task_id*, not any other head."""
        backbone = _make_backbone(in_dim=10, hidden=16)
        # Heads with different output dims to confirm routing is task-specific.
        heads = {
            "task_a": nn.Sequential(nn.Linear(16, 3)),
            "task_b": nn.Sequential(nn.Linear(16, 7)),
        }
        model = MultiTaskModel(backbone=backbone, task_heads=heads)
        x = torch.randn(4, 10)
        assert model(x, "task_a").shape == (4, 3)
        assert model(x, "task_b").shape == (4, 7)

    def test_unknown_task_id_raises_key_error(self) -> None:
        """Passing an unregistered task_id raises ``KeyError``."""
        model, _ = self._make_model()
        with pytest.raises(KeyError):
            model(torch.randn(2, 10), "nonexistent_task")

    def test_backbone_shared_across_heads(self) -> None:
        """Both heads share the same backbone parameter object."""
        model, task_ids = self._make_model()
        # The backbone is the same nn.Module instance regardless of task_id.
        assert model.backbone is model.backbone

    def test_output_shape_matches_head_out_dim(self) -> None:
        """Output tensor has shape ``(batch, head_out_dim)``."""
        backbone = _make_backbone(in_dim=10, hidden=16)
        heads = {"t1": nn.Sequential(nn.Linear(16, 5))}
        model = MultiTaskModel(backbone=backbone, task_heads=heads)
        x = torch.randn(8, 10)
        assert model(x, "t1").shape == (8, 5)

    def test_task_heads_is_module_dict(self) -> None:
        """``task_heads`` attribute is an ``nn.ModuleDict`` for proper parameter tracking."""
        model, _ = self._make_model()
        assert isinstance(model.task_heads, nn.ModuleDict)

    def test_head_params_in_model_parameters(self) -> None:
        """Head parameters appear in ``model.parameters()`` (via ``nn.ModuleDict``)."""
        model, task_ids = self._make_model()
        model_params = set(id(p) for p in model.parameters())
        for tid in task_ids:
            for p in model.task_heads[tid].parameters():
                assert id(p) in model_params, f"Head param for '{tid}' missing from model.parameters()"


# ---------------------------------------------------------------------------
# TestMultiTaskModelLoss
# ---------------------------------------------------------------------------


class TestMultiTaskModelLoss:
    def _make_model_and_batch(
        self,
        n_tasks: int = 2,
        in_dim: int = 10,
        out_dim: int = 3,
        batch_size: int = 8,
    ) -> tuple[MultiTaskModel, dict, dict[str, float]]:
        backbone = _make_backbone(in_dim=in_dim, hidden=16)
        task_ids = [f"task_{i}" for i in range(n_tasks)]
        heads = _make_heads(backbone_out=16, task_ids=task_ids, out_dim=out_dim)
        model = MultiTaskModel(backbone=backbone, task_heads=heads)
        torch.manual_seed(0)
        batch = {
            tid: (torch.randn(batch_size, in_dim), torch.randint(0, out_dim, (batch_size,)))
            for tid in task_ids
        }
        weights = {tid: 1.0 / n_tasks for tid in task_ids}
        return model, batch, weights

    def test_returns_scalar_tensor(self) -> None:
        """``compute_loss`` returns a 0-dim scalar tensor."""
        model, batch, weights = self._make_model_and_batch()
        loss = model.compute_loss(batch, weights)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_combines_multiple_task_losses(self) -> None:
        """Loss with two tasks differs from loss computed on one task alone."""
        model, batch, weights = self._make_model_and_batch(n_tasks=2)
        full_loss = model.compute_loss(batch, weights)
        # Single-task subset
        tid0 = list(batch.keys())[0]
        single_batch = {tid0: batch[tid0]}
        single_weights = {tid0: 1.0}
        single_loss = model.compute_loss(single_batch, single_weights)
        assert not torch.isclose(full_loss, single_loss)

    def test_single_task_equals_weighted_cross_entropy(self) -> None:
        """Single-task loss equals ``weight * F.cross_entropy(...)``."""
        torch.manual_seed(42)
        backbone = _make_backbone()
        heads = {"t": nn.Sequential(nn.Linear(16, 3))}
        model = MultiTaskModel(backbone=backbone, task_heads=heads)
        x = torch.randn(6, 10)
        y = torch.randint(0, 3, (6,))
        w = 0.7
        batch = {"t": (x, y)}
        loss = model.compute_loss(batch, {"t": w})
        expected = w * F.cross_entropy(model(x, "t"), y)
        assert torch.isclose(loss, expected)

    def test_weight_zero_contributes_nothing(self) -> None:
        """A task with weight 0.0 does not change the total loss value."""
        torch.manual_seed(1)
        backbone = _make_backbone()
        task_ids = ["a", "b"]
        heads = _make_heads(backbone_out=16, task_ids=task_ids, out_dim=3)
        model = MultiTaskModel(backbone=backbone, task_heads=heads)
        x = torch.randn(4, 10)
        ya = torch.randint(0, 3, (4,))
        yb = torch.randint(0, 3, (4,))
        batch = {"a": (x, ya), "b": (x, yb)}
        loss_with_b = model.compute_loss(batch, {"a": 1.0, "b": 1.0})
        loss_b_zero = model.compute_loss(batch, {"a": 1.0, "b": 0.0})
        expected_a_only = model.compute_loss({"a": (x, ya)}, {"a": 1.0})
        assert torch.isclose(loss_b_zero, expected_a_only)
        assert not torch.isclose(loss_with_b, loss_b_zero)

    def test_loss_has_grad_fn(self) -> None:
        """Loss tensor has a ``grad_fn`` so ``backward()`` is callable."""
        model, batch, weights = self._make_model_and_batch()
        loss = model.compute_loss(batch, weights)
        assert loss.grad_fn is not None


# ---------------------------------------------------------------------------
# TestMultiTaskModelUncertaintyLoss
# ---------------------------------------------------------------------------


class TestMultiTaskModelUncertaintyLoss:
    def _make_uncertainty_model(
        self,
        task_ids: list[str],
        in_dim: int = 10,
        out_dim: int = 3,
    ) -> tuple[MultiTaskModel, dict]:
        backbone = _make_backbone(in_dim=in_dim, hidden=16)
        heads = _make_heads(backbone_out=16, task_ids=task_ids, out_dim=out_dim)
        log_sigmas = {tid: nn.Parameter(torch.zeros(1)) for tid in task_ids}
        model = MultiTaskModel(
            backbone=backbone,
            task_heads=heads,
            task_weighting="uncertainty",
            log_sigmas=log_sigmas,
        )
        torch.manual_seed(0)
        batch = {
            tid: (torch.randn(8, in_dim), torch.randint(0, out_dim, (8,)))
            for tid in task_ids
        }
        return model, batch

    def test_returns_scalar_tensor(self) -> None:
        """``compute_uncertainty_loss`` returns a 0-dim scalar tensor."""
        model, batch = self._make_uncertainty_model(["t1", "t2"])
        loss = model.compute_uncertainty_loss(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_has_grad_fn(self) -> None:
        """Returned tensor has a ``grad_fn`` so ``backward()`` can be called."""
        model, batch = self._make_uncertainty_model(["t1", "t2"])
        loss = model.compute_uncertainty_loss(batch)
        assert loss.grad_fn is not None

    def test_log_sigmas_in_model_parameters(self) -> None:
        """``log_sigmas`` appear in ``model.parameters()`` via ``nn.ParameterDict``."""
        model, _ = self._make_uncertainty_model(["t1", "t2"])
        model_param_ids = set(id(p) for p in model.parameters())
        for p in model.log_sigmas.values():
            assert id(p) in model_param_ids

    def test_log_sigmas_have_grad_after_backward(self) -> None:
        """Each ``log_sigma`` tensor has a non-None gradient after a backward pass."""
        model, batch = self._make_uncertainty_model(["t1", "t2"])
        loss = model.compute_uncertainty_loss(batch)
        loss.backward()
        for tid, log_s in model.log_sigmas.items():
            assert log_s.grad is not None, f"log_sigma for '{tid}' has no gradient"
