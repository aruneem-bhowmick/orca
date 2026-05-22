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


# ---------------------------------------------------------------------------
# TestAddTask
# ---------------------------------------------------------------------------


class TestAddTask:
    def _strategy(self, weighting: str = "equal") -> MultiTaskTransfer:
        return MultiTaskTransfer(
            backbone=_make_backbone(in_dim=10, hidden=16),
            task_weighting=weighting,
            task_head_hidden_dim=8,
        )

    def test_head_output_dim_matches_argument(self) -> None:
        """The last ``nn.Linear`` in the created head has ``out_features == head_output_dim``."""
        mt = self._strategy()
        task = _make_task(n_classes=5)
        mt.add_task(task, head_output_dim=5)
        head = mt._task_heads[str(task.task_id)]
        linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
        assert linears[-1].out_features == 5

    def test_head_input_dim_matches_backbone_out(self) -> None:
        """The first ``nn.Linear`` in the head reads from the backbone's output dim."""
        mt = self._strategy()
        task = _make_task()
        mt.add_task(task, head_output_dim=3)
        head = mt._task_heads[str(task.task_id)]
        linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
        assert linears[0].in_features == 16  # backbone hidden=16

    def test_head_architecture_is_linear_relu_linear(self) -> None:
        """The created head is ``nn.Sequential`` with Linear → ReLU → Linear."""
        mt = self._strategy()
        task = _make_task()
        mt.add_task(task, head_output_dim=3)
        head = mt._task_heads[str(task.task_id)]
        assert isinstance(head, nn.Sequential)
        modules = list(head.children())
        assert isinstance(modules[0], nn.Linear)
        assert isinstance(modules[1], nn.ReLU)
        assert isinstance(modules[2], nn.Linear)

    def test_equal_weights_after_single_task(self) -> None:
        """After one ``add_task`` call, weight is 1.0 (= 1/1)."""
        mt = self._strategy("equal")
        task = _make_task()
        mt.add_task(task, 3)
        assert abs(mt.task_weights[str(task.task_id)] - 1.0) < 1e-9

    def test_equal_weights_after_two_tasks(self) -> None:
        """After two ``add_task`` calls, each weight is 0.5."""
        mt = self._strategy("equal")
        t1, t2 = _make_task(), _make_task()
        mt.add_task(t1, 3)
        mt.add_task(t2, 5)
        weights = mt.task_weights
        assert abs(weights[str(t1.task_id)] - 0.5) < 1e-9
        assert abs(weights[str(t2.task_id)] - 0.5) < 1e-9

    def test_uncertainty_creates_log_sigma_per_task(self) -> None:
        """Uncertainty weighting creates one ``nn.Parameter`` in ``_log_sigmas`` per task."""
        mt = self._strategy("uncertainty")
        t1, t2 = _make_task(), _make_task()
        mt.add_task(t1, 3)
        mt.add_task(t2, 5)
        assert str(t1.task_id) in mt._log_sigmas
        assert str(t2.task_id) in mt._log_sigmas
        assert isinstance(mt._log_sigmas[str(t1.task_id)], nn.Parameter)

    def test_uncertainty_log_sigma_initialised_to_zero(self) -> None:
        """Log-sigma parameters start at zero (sigma=1, equal initial weighting)."""
        mt = self._strategy("uncertainty")
        task = _make_task()
        mt.add_task(task, 3)
        log_s = mt._log_sigmas[str(task.task_id)]
        assert torch.allclose(log_s, torch.zeros(1))

    def test_task_id_stored_as_string(self) -> None:
        """Task heads are keyed by ``str(task.task_id)``, not a UUID object."""
        mt = self._strategy()
        task = _make_task()
        mt.add_task(task, 3)
        assert str(task.task_id) in mt._task_heads

    def test_raises_on_invalid_task_weighting(self) -> None:
        """``ValueError`` is raised when an unsupported ``task_weighting`` is supplied."""
        with pytest.raises(ValueError, match="task_weighting"):
            MultiTaskTransfer(backbone=_make_backbone(), task_weighting="unknown")


# ---------------------------------------------------------------------------
# TestScoreTransfer
# ---------------------------------------------------------------------------


class TestScoreTransfer:
    def _strategy(self) -> MultiTaskTransfer:
        return MultiTaskTransfer(backbone=_make_backbone())

    def test_returns_transfer_score_instance(self) -> None:
        """``score_transfer`` always returns a ``TransferScore`` instance."""
        mt = self._strategy()
        score = mt.score_transfer(_make_task(), _make_task())
        assert isinstance(score, TransferScore)

    def test_no_features_returns_neutral_score(self) -> None:
        """Without registered features, ``overall`` is the neutral value 0.5."""
        mt = self._strategy()
        score = mt.score_transfer(_make_task(), _make_task())
        assert score.overall == 0.5

    def test_no_features_reasoning_mentions_registration(self) -> None:
        """The fallback reasoning string informs the caller to register features."""
        mt = self._strategy()
        score = mt.score_transfer(_make_task(), _make_task())
        assert "No task features" in score.reasoning

    def test_registered_features_produce_nonzero_score(self) -> None:
        """After registering feature vectors, ``score_transfer`` computes a real similarity."""
        mt = self._strategy()
        t1, t2 = _make_task(), _make_task()
        # Use 25-dim input to match CrossDomainEmbedder default input_dim.
        torch.manual_seed(0)
        mt.register_task_features(str(t1.task_id), torch.randn(1, 25))
        mt.register_task_features(str(t2.task_id), torch.randn(1, 25))
        score = mt.score_transfer(t1, t2)
        assert isinstance(score.overall, float)
        assert 0.0 <= score.overall <= 1.0

    def test_identical_features_produce_score_one(self) -> None:
        """Two identical feature vectors yield cosine similarity 1.0 (clamped)."""
        mt = self._strategy()
        t1, t2 = _make_task(), _make_task()
        feat = torch.ones(1, 25)
        mt.register_task_features(str(t1.task_id), feat)
        mt.register_task_features(str(t2.task_id), feat)
        score = mt.score_transfer(t1, t2)
        assert abs(score.overall - 1.0) < 1e-5

    def test_high_similarity_beneficial_reasoning(self) -> None:
        """Similarity > 0.5 produces the exact spec reasoning string."""
        mt = self._strategy()
        t1, t2 = _make_task(), _make_task()
        feat = torch.ones(1, 25)  # identical → similarity == 1.0
        mt.register_task_features(str(t1.task_id), feat)
        mt.register_task_features(str(t2.task_id), feat)
        score = mt.score_transfer(t1, t2)
        assert "Multi-task training beneficial" in score.reasoning
        assert "> threshold 0.5" in score.reasoning

    def test_overall_in_unit_interval(self) -> None:
        """``overall`` is always in ``[0.0, 1.0]``."""
        mt = self._strategy()
        t1, t2 = _make_task(), _make_task()
        mt.register_task_features(str(t1.task_id), torch.randn(1, 25))
        mt.register_task_features(str(t2.task_id), torch.randn(1, 25))
        score = mt.score_transfer(t1, t2)
        assert 0.0 <= score.overall <= 1.0

    def test_layer_scores_contains_cosine_similarity(self) -> None:
        """When features are registered, ``layer_scores`` has a ``cosine_similarity`` key."""
        mt = self._strategy()
        t1, t2 = _make_task(), _make_task()
        mt.register_task_features(str(t1.task_id), torch.ones(1, 25))
        mt.register_task_features(str(t2.task_id), torch.ones(1, 25))
        score = mt.score_transfer(t1, t2)
        assert "cosine_similarity" in score.layer_scores


# ---------------------------------------------------------------------------
# TestExecuteTransfer
# ---------------------------------------------------------------------------


class TestExecuteTransfer:
    def _setup(
        self,
        weighting: str = "equal",
        n_classes_src: int = 3,
        n_classes_tgt: int = 5,
    ) -> tuple[MultiTaskTransfer, Task, Task]:
        backbone = _make_backbone(in_dim=10, hidden=16)
        mt = MultiTaskTransfer(backbone=backbone, task_weighting=weighting, task_head_hidden_dim=8)
        source = _make_task(n_features=10, n_classes=n_classes_src)
        target = _make_task(n_features=10, n_classes=n_classes_tgt)
        return mt, source, target

    def test_returns_multi_task_model_instance(self) -> None:
        """``execute_transfer`` returns a ``MultiTaskModel``."""
        mt, source, target = self._setup()
        mt.add_task(source, 3)
        mt.add_task(target, 5)
        model = mt.execute_transfer(source, target, mt.backbone)
        assert isinstance(model, MultiTaskModel)

    def test_model_has_both_task_heads(self) -> None:
        """The returned model has a head registered for both source and target."""
        mt, source, target = self._setup()
        mt.add_task(source, 3)
        mt.add_task(target, 5)
        model = mt.execute_transfer(source, target, mt.backbone)
        assert str(source.task_id) in model.task_heads
        assert str(target.task_id) in model.task_heads

    def test_model_shares_backbone(self) -> None:
        """The ``MultiTaskModel`` uses the same backbone passed to ``MultiTaskTransfer``."""
        mt, source, target = self._setup()
        mt.add_task(source, 3)
        mt.add_task(target, 5)
        model = mt.execute_transfer(source, target, mt.backbone)
        assert model.backbone is mt.backbone

    def test_auto_registers_tasks_from_n_classes(self) -> None:
        """Unregistered tasks are auto-added using ``task.n_classes`` as head output dim."""
        mt, source, target = self._setup(n_classes_src=3, n_classes_tgt=7)
        # Do NOT pre-register; execute_transfer should auto-register.
        model = mt.execute_transfer(source, target, mt.backbone)
        assert str(source.task_id) in model.task_heads
        assert str(target.task_id) in model.task_heads

    def test_auto_registered_head_output_matches_n_classes(self) -> None:
        """Auto-registered head has ``out_features == task.n_classes``."""
        mt, source, target = self._setup(n_classes_tgt=7)
        model = mt.execute_transfer(source, target, mt.backbone)
        tgt_head = model.task_heads[str(target.task_id)]
        linears = [m for m in tgt_head.modules() if isinstance(m, nn.Linear)]
        assert linears[-1].out_features == 7

    def test_forward_runs_on_returned_model(self) -> None:
        """The returned ``MultiTaskModel`` can run a forward pass without error."""
        mt, source, target = self._setup()
        mt.add_task(source, 3)
        mt.add_task(target, 5)
        model = mt.execute_transfer(source, target, mt.backbone)
        x = torch.randn(4, 10)
        out = model(x, str(source.task_id))
        assert out.shape == (4, 3)

    def test_uncertainty_execute_passes_log_sigmas(self) -> None:
        """With uncertainty weighting, ``MultiTaskModel.log_sigmas`` is populated."""
        mt, source, target = self._setup(weighting="uncertainty")
        mt.add_task(source, 3)
        mt.add_task(target, 5)
        model = mt.execute_transfer(source, target, mt.backbone)
        assert len(model.log_sigmas) == 2
        assert str(source.task_id) in model.log_sigmas
        assert str(target.task_id) in model.log_sigmas


# ---------------------------------------------------------------------------
# TestTransferMetadata
# ---------------------------------------------------------------------------


class TestTransferMetadata:
    def _strategy(self, **kwargs) -> MultiTaskTransfer:
        return MultiTaskTransfer(backbone=_make_backbone(), **kwargs)

    def test_strategy_name(self) -> None:
        """``strategy`` key is ``"multi_task_transfer"``."""
        assert self._strategy().get_transfer_metadata()["strategy"] == "multi_task_transfer"

    def test_default_task_weighting(self) -> None:
        """Default weighting is ``"equal"``."""
        assert self._strategy().get_transfer_metadata()["task_weighting"] == "equal"

    def test_custom_task_weighting_reflected(self) -> None:
        """Custom ``task_weighting`` is reflected in metadata."""
        meta = self._strategy(task_weighting="uncertainty").get_transfer_metadata()
        assert meta["task_weighting"] == "uncertainty"

    def test_task_head_hidden_dim_reflected(self) -> None:
        """``task_head_hidden_dim`` value is reflected in metadata."""
        meta = self._strategy(task_head_hidden_dim=128).get_transfer_metadata()
        assert meta["task_head_hidden_dim"] == 128

    def test_n_tasks_increments_with_add_task(self) -> None:
        """``n_registered_tasks`` equals the number of ``add_task`` calls."""
        mt = self._strategy()
        assert mt.get_transfer_metadata()["n_registered_tasks"] == 0
        mt.add_task(_make_task(), 3)
        assert mt.get_transfer_metadata()["n_registered_tasks"] == 1
        mt.add_task(_make_task(), 5)
        assert mt.get_transfer_metadata()["n_registered_tasks"] == 2

    def test_backbone_out_dim_in_metadata(self) -> None:
        """``backbone_out_dim`` reflects the inferred backbone output dimension."""
        meta = self._strategy().get_transfer_metadata()
        assert meta["backbone_out_dim"] == 16  # _make_backbone hidden=16

    def test_default_hidden_dim_is_64(self) -> None:
        """Default ``task_head_hidden_dim`` is 64."""
        assert self._strategy().get_transfer_metadata()["task_head_hidden_dim"] == 64


# ---------------------------------------------------------------------------
# TestGradnormWeighting
# ---------------------------------------------------------------------------


class TestGradnormWeighting:
    def _strategy_with_two_tasks(self) -> tuple[MultiTaskTransfer, Task, Task]:
        mt = MultiTaskTransfer(
            backbone=_make_backbone(), task_weighting="gradnorm", task_head_hidden_dim=8
        )
        t1, t2 = _make_task(), _make_task()
        mt.add_task(t1, 3)
        mt.add_task(t2, 5)
        return mt, t1, t2

    def test_initial_weights_are_equal(self) -> None:
        """GradNorm strategy initialises with uniform equal weights."""
        mt, t1, t2 = self._strategy_with_two_tasks()
        w = mt.task_weights
        assert abs(w[str(t1.task_id)] - 0.5) < 1e-9
        assert abs(w[str(t2.task_id)] - 0.5) < 1e-9

    def test_update_gradnorm_weights_renormalises(self) -> None:
        """After calling ``update_gradnorm_weights``, weights reflect gradient norms."""
        mt, t1, t2 = self._strategy_with_two_tasks()
        tid1, tid2 = str(t1.task_id), str(t2.task_id)
        mt.update_gradnorm_weights({tid1: 1.0, tid2: 3.0})
        # Higher gradient norm → higher weight target after normalisation.
        w = mt.task_weights
        assert w[tid2] > w[tid1]

    def test_weights_sum_to_one_after_update(self) -> None:
        """Updated gradnorm weights sum to 1.0."""
        mt, t1, t2 = self._strategy_with_two_tasks()
        tid1, tid2 = str(t1.task_id), str(t2.task_id)
        mt.update_gradnorm_weights({tid1: 2.0, tid2: 5.0})
        assert abs(sum(mt.task_weights.values()) - 1.0) < 1e-6

    def test_empty_grad_norms_leaves_weights_unchanged(self) -> None:
        """Calling with an empty dict does not alter the current weights."""
        mt, t1, t2 = self._strategy_with_two_tasks()
        before = dict(mt.task_weights)
        mt.update_gradnorm_weights({})
        assert mt.task_weights == before


# ---------------------------------------------------------------------------
# TestUncertaintyWeighting
# ---------------------------------------------------------------------------


class TestUncertaintyWeighting:
    """Verify that uncertainty weighting learns higher log_sigma for noisier tasks."""

    def _build(
        self,
        in_dim: int = 4,
        hidden: int = 8,
        out_dim: int = 2,
    ) -> tuple[MultiTaskTransfer, MultiTaskModel, Task, Task]:
        """Return a fully wired (strategy, model, task_easy, task_hard) tuple."""
        backbone = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        mt = MultiTaskTransfer(
            backbone=backbone,
            task_weighting="uncertainty",
            task_head_hidden_dim=8,
        )
        task_easy = _make_task(n_features=in_dim, n_classes=out_dim)
        task_hard = _make_task(n_features=in_dim, n_classes=out_dim)
        mt.add_task(task_easy, out_dim)
        mt.add_task(task_hard, out_dim)
        model = mt.execute_transfer(task_easy, task_hard, backbone)
        return mt, model, task_easy, task_hard

    def test_log_sigma_gradients_nonzero_after_single_backward(self) -> None:
        """Each log_sigma has a non-None, non-zero gradient after one backward pass."""
        torch.manual_seed(7)
        _, model, task_easy, task_hard = self._build()
        tid_e = str(task_easy.task_id)
        tid_h = str(task_hard.task_id)

        x = torch.randn(8, 4)
        batch = {
            tid_e: (x, torch.zeros(8, dtype=torch.long)),   # easy: constant label
            tid_h: (x, torch.randint(0, 2, (8,))),           # hard: random labels
        }
        loss = model.compute_uncertainty_loss(batch)
        loss.backward()

        for tid in (tid_e, tid_h):
            log_s = model.log_sigmas[tid]
            assert log_s.grad is not None, f"log_sigma['{tid}'] has no gradient"
            assert log_s.grad.abs().item() > 0.0, f"log_sigma['{tid}'] gradient is zero"

    def test_noisy_task_learns_higher_log_sigma(self) -> None:
        """After training, the high-noise task accumulates a larger log_sigma value.

        Setup: task_hard gets random (irreducible) labels → persistently high
        cross-entropy; task_easy gets a fixed all-zero label → the model can
        fit it and drive CE down.  After 10 Adam steps the Kendall objective
        should push log_sigma_hard > log_sigma_easy.
        """
        torch.manual_seed(42)
        _, model, task_easy, task_hard = self._build(in_dim=4, hidden=8, out_dim=2)
        tid_e = str(task_easy.task_id)
        tid_h = str(task_hard.task_id)

        opt = torch.optim.Adam(model.parameters(), lr=0.05)

        for _ in range(10):
            opt.zero_grad()
            x = torch.randn(16, 4)
            y_easy = torch.zeros(16, dtype=torch.long)        # constant, learnable
            y_hard = torch.randint(0, 2, (16,))               # random, irreducible noise
            batch = {tid_e: (x, y_easy), tid_h: (x, y_hard)}
            model.compute_uncertainty_loss(batch).backward()
            opt.step()

        log_s_easy = model.log_sigmas[tid_e].item()
        log_s_hard = model.log_sigmas[tid_h].item()
        assert log_s_hard > log_s_easy, (
            f"Expected log_sigma_hard ({log_s_hard:.4f}) > log_sigma_easy ({log_s_easy:.4f}): "
            "the noisy task should receive a larger learned variance"
        )
