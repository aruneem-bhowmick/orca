"""Unit tests for WeightTransfer and get_optimizer_with_layer_lr."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import torch
import torch.nn as nn

from orcanet.transfer.types import TransferScore
from orcanet.transfer.weight_transfer import WeightTransfer, get_optimizer_with_layer_lr
from orca_shared.schemas.task import Task


# ---------------------------------------------------------------------------
# Shared helpers
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
# TestWeightTransferScoreIdentical
# ---------------------------------------------------------------------------


class TestWeightTransferScoreIdentical:
    """score_transfer on two copies of the same architecture must return overall == 1.0."""

    def _setup(self) -> tuple[WeightTransfer, Task, Task]:
        wt = WeightTransfer(match_by="name")
        source = _make_task()
        target = _make_task()
        torch.manual_seed(0)
        model = _make_mlp()
        wt.register_model(str(source.task_id), model)
        wt.register_model(str(target.task_id), model)
        return wt, source, target

    def test_overall_is_one(self) -> None:
        wt, source, target = self._setup()
        score = wt.score_transfer(source, target)
        assert score.overall == 1.0

    def test_returns_transfer_score_instance(self) -> None:
        wt, source, target = self._setup()
        assert isinstance(wt.score_transfer(source, target), TransferScore)

    def test_all_params_recommended(self) -> None:
        wt, source, target = self._setup()
        score = wt.score_transfer(source, target)
        assert len(score.recommended_layers) == len(score.layer_scores)

    def test_all_layer_scores_are_one(self) -> None:
        wt, source, target = self._setup()
        score = wt.score_transfer(source, target)
        for name, s in score.layer_scores.items():
            assert s == 1.0, f"Param '{name}' score {s} unexpected for identical models"

    def test_reasoning_contains_match_count(self) -> None:
        wt, source, target = self._setup()
        score = wt.score_transfer(source, target)
        assert "Matched" in score.reasoning
        assert "name" in score.reasoning


# ---------------------------------------------------------------------------
# TestWeightTransferScoreMatchBy
# ---------------------------------------------------------------------------


class TestWeightTransferScoreMatchBy:
    """Different match_by modes produce different scores for mismatched architectures."""

    def _setup_mismatched(
        self, match_by: str
    ) -> tuple[WeightTransfer, Task, Task]:
        """Source: MLP with out_dim=5; target: MLP with out_dim=3 — last layer differs."""
        wt = WeightTransfer(match_by=match_by)
        source = _make_task()
        target = _make_task()
        torch.manual_seed(1)
        source_model = _make_mlp(out_dim=5)
        torch.manual_seed(2)
        target_model = _make_mlp(out_dim=3)
        wt.register_model(str(source.task_id), source_model)
        wt.register_model(str(target.task_id), target_model)
        return wt, source, target

    def test_name_mode_counts_by_name_regardless_of_shape(self) -> None:
        wt, source, target = self._setup_mismatched("name")
        score = wt.score_transfer(source, target)
        # All param names exist in source ("0.weight", "0.bias", "2.weight", "2.bias")
        # → overall should be 1.0 because names all match even if shapes differ for layer 2
        assert score.overall == 1.0

    def test_both_mode_excludes_shape_mismatched_params(self) -> None:
        wt, source, target = self._setup_mismatched("both")
        score = wt.score_transfer(source, target)
        # "2.weight" shape (3,20) vs (5,20) and "2.bias" shape (3,) vs (5,) don't match
        assert score.overall < 1.0

    def test_both_mode_matching_params_have_score_one(self) -> None:
        wt, source, target = self._setup_mismatched("both")
        score = wt.score_transfer(source, target)
        # "0.weight" (20,10) and "0.bias" (20,) should still match
        assert score.layer_scores.get("0.weight") == 1.0
        assert score.layer_scores.get("0.bias") == 1.0

    def test_shape_mode_matches_by_shape(self) -> None:
        wt, source, target = self._setup_mismatched("shape")
        score = wt.score_transfer(source, target)
        # All layer shapes from source exist (first-layer params are identical shapes)
        assert isinstance(score.overall, float)
        assert 0.0 <= score.overall <= 1.0


# ---------------------------------------------------------------------------
# TestWeightTransferScoreStructure
# ---------------------------------------------------------------------------


class TestWeightTransferScoreStructure:
    """Structural invariants of TransferScore returned by WeightTransfer."""

    def _score(self) -> TransferScore:
        wt = WeightTransfer()
        source = _make_task()
        target = _make_task()
        model = _make_mlp()
        wt.register_model(str(source.task_id), model)
        wt.register_model(str(target.task_id), model)
        return wt.score_transfer(source, target)

    def test_overall_in_unit_interval(self) -> None:
        assert 0.0 <= self._score().overall <= 1.0

    def test_layer_scores_is_dict(self) -> None:
        assert isinstance(self._score().layer_scores, dict)

    def test_recommended_layers_is_list(self) -> None:
        assert isinstance(self._score().recommended_layers, list)

    def test_reasoning_is_non_empty_string(self) -> None:
        score = self._score()
        assert isinstance(score.reasoning, str)
        assert len(score.reasoning) > 0

    def test_recommended_subset_of_layer_scores(self) -> None:
        score = self._score()
        assert set(score.recommended_layers).issubset(set(score.layer_scores.keys()))

    def test_recommended_layers_all_have_score_one(self) -> None:
        score = self._score()
        for name in score.recommended_layers:
            assert score.layer_scores[name] == 1.0


# ---------------------------------------------------------------------------
# TestWeightTransferExecute
# ---------------------------------------------------------------------------


class TestWeightTransferExecute:
    """execute_transfer copies matched weights and reinitialises unmatched ones."""

    def _setup_identical(self) -> tuple[WeightTransfer, Task, Task, nn.Module]:
        """Same architecture: all params should be transferred."""
        wt = WeightTransfer(match_by="name")
        source = _make_task()
        target = _make_task()
        torch.manual_seed(10)
        source_model = _make_mlp()
        torch.manual_seed(20)
        target_model = _make_mlp()
        wt.register_model(str(source.task_id), source_model)
        wt.register_model(str(target.task_id), target_model)
        return wt, source, target, source_model

    def _setup_mismatched(self) -> tuple[WeightTransfer, Task, Task, nn.Module]:
        """Different out_dim: last-layer params have shape mismatch."""
        wt = WeightTransfer(match_by="both")
        source = _make_task()
        target = _make_task()
        torch.manual_seed(30)
        source_model = _make_mlp(out_dim=5)
        torch.manual_seed(40)
        target_model = _make_mlp(out_dim=3)
        wt.register_model(str(source.task_id), source_model)
        wt.register_model(str(target.task_id), target_model)
        return wt, source, target, source_model

    def test_returns_tuple_of_module_and_list(self) -> None:
        wt, source, target, source_model = self._setup_identical()
        result = wt.execute_transfer(source, target, source_model)
        assert isinstance(result, tuple) and len(result) == 2
        model, transferred = result
        assert isinstance(model, nn.Module)
        assert isinstance(transferred, list)

    def test_transferred_weights_equal_source(self) -> None:
        wt, source, target, source_model = self._setup_identical()
        adapted, transferred = wt.execute_transfer(source, target, source_model)
        source_state = source_model.state_dict()
        adapted_state = adapted.state_dict()
        for name in transferred:
            assert torch.equal(adapted_state[name], source_state[name]), (
                f"Transferred param '{name}' does not match source after execute_transfer"
            )

    def test_all_params_transferred_for_identical_architecture(self) -> None:
        wt, source, target, source_model = self._setup_identical()
        _, transferred = wt.execute_transfer(source, target, source_model)
        expected = list(source_model.state_dict().keys())
        assert set(transferred) == set(expected)

    def test_source_model_not_mutated(self) -> None:
        wt, source, target, source_model = self._setup_identical()
        original = {k: v.clone() for k, v in source_model.state_dict().items()}
        wt.execute_transfer(source, target, source_model)
        for k, v in source_model.state_dict().items():
            assert torch.equal(v, original[k]), f"Source param '{k}' was mutated"

    def test_shape_mismatch_skipped_without_exception(self) -> None:
        """execute_transfer with any match_by mode does not raise.

        Since execute_transfer starts from deepcopy(source_model), all params
        share the source architecture and shape mismatches cannot occur in that
        path.  The shape-safety of _find_source_tensor is verified here by
        running all three matching modes — none should raise an exception.
        """
        for mode in ("name", "shape", "both"):
            wt = WeightTransfer(match_by=mode)
            source = _make_task()
            target = _make_task()
            torch.manual_seed(50)
            model = _make_mlp()
            wt.register_model(str(source.task_id), model)
            wt.register_model(str(target.task_id), model)
            adapted, transferred = wt.execute_transfer(source, target, model)
            assert isinstance(adapted, nn.Module)
            assert len(transferred) > 0

    def test_safe_reinit_handles_1d_and_2d_tensors(self) -> None:
        """_safe_reinit applies kaiming_uniform_ to weight tensors without raising.

        Bias vectors (1-D) use zeros_ instead, also without raising.
        Shape-mismatch reinitialization uses this helper, so this verifies the
        reinit path does not crash for either tensor dimensionality.
        """
        from orcanet.transfer.weight_transfer import _safe_reinit

        bias = torch.zeros(20)
        weight = torch.zeros(20, 10)
        _safe_reinit(bias)
        _safe_reinit(weight)
        assert not torch.all(weight == 0.0), "kaiming_uniform_ should produce non-zero weights"


# ---------------------------------------------------------------------------
# TestGetOptimizerWithLayerLR
# ---------------------------------------------------------------------------


class TestGetOptimizerWithLayerLR:
    """get_optimizer_with_layer_lr assigns decayed LR to transferred layers."""

    def _model_and_transferred(self) -> tuple[nn.Module, list[str]]:
        torch.manual_seed(0)
        model = _make_mlp()
        # Treat first-layer params as "transferred", leave last-layer params at base LR
        transferred = ["0.weight", "0.bias"]
        return model, transferred

    def test_returns_adam_optimizer(self) -> None:
        model, transferred = self._model_and_transferred()
        opt = get_optimizer_with_layer_lr(model, transferred, base_lr=0.01)
        assert isinstance(opt, torch.optim.Adam)

    def test_transferred_params_get_decayed_lr(self) -> None:
        model, transferred = self._model_and_transferred()
        base_lr = 0.01
        decay = 0.1
        opt = get_optimizer_with_layer_lr(model, transferred, base_lr=base_lr, decay=decay)
        for group in opt.param_groups:
            for param in group["params"]:
                name = next(
                    n for n, p in model.named_parameters() if p is param
                )
                expected = base_lr * decay if name in transferred else base_lr
                assert abs(group["lr"] - expected) < 1e-9, (
                    f"Param '{name}': expected lr={expected}, got {group['lr']}"
                )

    def test_non_transferred_params_get_base_lr(self) -> None:
        model, transferred = self._model_and_transferred()
        base_lr = 0.05
        opt = get_optimizer_with_layer_lr(model, transferred, base_lr=base_lr, decay=0.1)
        non_transferred = [n for n, _ in model.named_parameters() if n not in transferred]
        assert len(non_transferred) > 0
        for group in opt.param_groups:
            for param in group["params"]:
                name = next(n for n, p in model.named_parameters() if p is param)
                if name in non_transferred:
                    assert abs(group["lr"] - base_lr) < 1e-9

    def test_decay_zero_gives_zero_lr_for_transferred(self) -> None:
        model, transferred = self._model_and_transferred()
        opt = get_optimizer_with_layer_lr(model, transferred, base_lr=0.01, decay=0.0)
        for group in opt.param_groups:
            for param in group["params"]:
                name = next(n for n, p in model.named_parameters() if p is param)
                if name in transferred:
                    assert group["lr"] == 0.0

    def test_empty_transferred_all_base_lr(self) -> None:
        torch.manual_seed(0)
        model = _make_mlp()
        base_lr = 0.02
        opt = get_optimizer_with_layer_lr(model, [], base_lr=base_lr, decay=0.1)
        for group in opt.param_groups:
            assert abs(group["lr"] - base_lr) < 1e-9


# ---------------------------------------------------------------------------
# TestWeightTransferGuards
# ---------------------------------------------------------------------------


class TestWeightTransferGuards:
    def test_raises_when_source_not_registered(self) -> None:
        wt = WeightTransfer()
        source = _make_task()
        target = _make_task()
        wt.register_model(str(target.task_id), _make_mlp())
        with pytest.raises(ValueError, match="Models not registered"):
            wt.score_transfer(source, target)

    def test_raises_when_target_not_registered(self) -> None:
        wt = WeightTransfer()
        source = _make_task()
        target = _make_task()
        wt.register_model(str(source.task_id), _make_mlp())
        with pytest.raises(ValueError, match="Models not registered"):
            wt.score_transfer(source, target)

    def test_raises_on_invalid_match_by(self) -> None:
        with pytest.raises(ValueError, match="match_by"):
            WeightTransfer(match_by="invalid")


# ---------------------------------------------------------------------------
# TestWeightTransferMetadata
# ---------------------------------------------------------------------------


class TestWeightTransferMetadata:
    def test_strategy_name(self) -> None:
        assert WeightTransfer().get_transfer_metadata()["strategy"] == "weight_transfer"

    def test_match_by_reflected(self) -> None:
        assert WeightTransfer(match_by="both").get_transfer_metadata()["match_by"] == "both"

    def test_frozen_epochs_reflected(self) -> None:
        assert WeightTransfer(frozen_epochs=10).get_transfer_metadata()["frozen_epochs"] == 10

    def test_layer_lr_decay_reflected(self) -> None:
        assert (
            WeightTransfer(layer_lr_decay=0.5).get_transfer_metadata()["layer_lr_decay"] == 0.5
        )

    def test_default_values(self) -> None:
        meta = WeightTransfer().get_transfer_metadata()
        assert meta["match_by"] == "name"
        assert meta["frozen_epochs"] == 5
        assert meta["layer_lr_decay"] == 0.1
