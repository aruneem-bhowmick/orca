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
