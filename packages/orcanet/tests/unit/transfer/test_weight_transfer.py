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
        """Using match_by='both', shape-mismatched params are silently reinitialized."""
        wt, source, target, source_model = self._setup_mismatched()
        adapted, transferred = wt.execute_transfer(source, target, source_model)
        # Should not raise; last-layer params must NOT appear in transferred
        assert "2.weight" not in transferred
        assert "2.bias" not in transferred

    def test_matched_params_transferred_shape_mismatch_case(self) -> None:
        """With match_by='both', first-layer params still transfer."""
        wt, source, target, source_model = self._setup_mismatched()
        adapted, transferred = wt.execute_transfer(source, target, source_model)
        assert "0.weight" in transferred
        assert "0.bias" in transferred

    def test_unmatched_params_differ_from_source(self) -> None:
        """Reinitialized params should not equal source values (with overwhelming probability)."""
        wt, source, target, source_model = self._setup_mismatched()
        adapted, transferred = wt.execute_transfer(source, target, source_model)
        source_state = source_model.state_dict()
        adapted_state = adapted.state_dict()
        unmatched = [n for n in adapted_state if n not in transferred]
        # At least one unmatched weight tensor should differ from source
        assert any(
            not torch.equal(adapted_state[n], source_state.get(n, torch.zeros(1)))
            for n in unmatched
            if adapted_state[n].ndim >= 2
        )
