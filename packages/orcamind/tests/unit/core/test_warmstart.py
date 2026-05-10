"""Unit tests for WarmStartTransfer — weight transfer strategies, scheduling, and orchestration."""

from __future__ import annotations

import logging
import uuid

import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import AsyncMock, MagicMock

from orcamind.core.warmstart import WarmStartTransfer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SplitModel(nn.Module):
    """Two-layer model whose parameter names contain 'encoder' and 'head'."""

    def __init__(self, in_features: int = 4, hidden: int = 8, out_features: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_features, hidden)
        self.head = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.encoder(x)))


class _WideHeadModel(nn.Module):
    """Same encoder as _SplitModel but a wider head — forces a shape mismatch."""

    def __init__(self, in_features: int = 4, hidden: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_features, hidden)
        self.head = nn.Linear(hidden, 10)  # mismatch: _SplitModel head is (hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.encoder(x)))


def _fill(model: nn.Module, value: float) -> None:
    with torch.no_grad():
        for p in model.parameters():
            nn.init.constant_(p, value)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_index() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_artifacts() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def wst(mock_index: MagicMock, mock_artifacts: MagicMock, mock_repo: MagicMock) -> WarmStartTransfer:
    return WarmStartTransfer(
        similarity_index=mock_index,
        artifact_manager=mock_artifacts,
        task_repository=mock_repo,
    )


# ---------------------------------------------------------------------------
# transfer_weights — strategy "all"
# ---------------------------------------------------------------------------


class TestTransferWeightsAll:
    """All matching parameters are copied when strategy='all'."""

    def test_all_params_copied(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        target = _SplitModel()
        _fill(source, 1.0)
        _fill(target, 0.0)
        wst.transfer_weights(source, target, strategy="all")
        for name, param in target.named_parameters():
            assert torch.all(param == 1.0), f"{name} was not copied"

    def test_returns_target_identity(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        target = _SplitModel()
        result = wst.transfer_weights(source, target, strategy="all")
        assert result is target


# ---------------------------------------------------------------------------
# transfer_weights — strategy "encoder_only"
# ---------------------------------------------------------------------------


class TestTransferWeightsEncoderOnly:
    """Only encoder-keyword params are copied when strategy='encoder_only'."""

    def test_encoder_params_copied(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        target = _SplitModel()
        _fill(source, 1.0)
        _fill(target, 0.0)
        wst.transfer_weights(source, target, strategy="encoder_only")
        assert torch.all(target.encoder.weight == 1.0)
        assert torch.all(target.encoder.bias == 1.0)

    def test_head_params_untouched(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        target = _SplitModel()
        _fill(source, 1.0)
        _fill(target, 0.0)
        wst.transfer_weights(source, target, strategy="encoder_only")
        assert torch.all(target.head.weight == 0.0)
        assert torch.all(target.head.bias == 0.0)


# ---------------------------------------------------------------------------
# transfer_weights — strategy "head_only"
# ---------------------------------------------------------------------------


class TestTransferWeightsHeadOnly:
    """Only head-keyword params are copied when strategy='head_only'."""

    def test_head_params_copied(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        target = _SplitModel()
        _fill(source, 1.0)
        _fill(target, 0.0)
        wst.transfer_weights(source, target, strategy="head_only")
        assert torch.all(target.head.weight == 1.0)
        assert torch.all(target.head.bias == 1.0)

    def test_encoder_params_untouched(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        target = _SplitModel()
        _fill(source, 1.0)
        _fill(target, 0.0)
        wst.transfer_weights(source, target, strategy="head_only")
        assert torch.all(target.encoder.weight == 0.0)
        assert torch.all(target.encoder.bias == 0.0)


# ---------------------------------------------------------------------------
# transfer_weights — shape mismatch
# ---------------------------------------------------------------------------


class TestTransferWeightsShapeMismatch:
    """Mismatched-shape parameters are skipped with a warning, not an exception."""

    def test_no_exception_on_mismatch(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        target = _WideHeadModel()
        result = wst.transfer_weights(source, target, strategy="all")
        assert result is target

    def test_warning_logged_on_mismatch(self, wst: WarmStartTransfer, caplog: pytest.LogCaptureFixture) -> None:
        source = _SplitModel()
        target = _WideHeadModel()
        with caplog.at_level(logging.WARNING, logger="orcamind.core.warmstart"):
            wst.transfer_weights(source, target, strategy="all")
        assert any("mismatch" in r.message.lower() for r in caplog.records)

    def test_matching_params_still_transferred_despite_mismatch(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        target = _WideHeadModel()
        _fill(source, 1.0)
        _fill(target, 0.0)
        wst.transfer_weights(source, target, strategy="all")
        # encoder shapes match → copied
        assert torch.all(target.encoder.weight == 1.0)
        assert torch.all(target.encoder.bias == 1.0)
        # head shapes mismatch → NOT copied
        assert torch.all(target.head.weight == 0.0)
        assert torch.all(target.head.bias == 0.0)


# ---------------------------------------------------------------------------
# Helpers for schedule / orchestration tests
# ---------------------------------------------------------------------------


def _make_mock_task(**kwargs: object) -> MagicMock:
    task = MagicMock()
    task.task_type = kwargs.get("task_type", "classification")
    task.n_samples = kwargs.get("n_samples", 100)
    task.n_features = kwargs.get("n_features", 10)
    task.n_classes = kwargs.get("n_classes", 2)
    task.metadata = kwargs.get("metadata", {})
    return task


# ---------------------------------------------------------------------------
# get_adaptive_schedule — similarity bands
# ---------------------------------------------------------------------------


class TestGetAdaptiveSchedule:
    """Schedule dict reflects the correct band for the given similarity score."""

    def test_high_similarity(self, wst: WarmStartTransfer) -> None:
        schedule = wst.get_adaptive_schedule(_make_mock_task(), _make_mock_task(), similarity_score=0.95)
        assert schedule == {"lr_multiplier": 0.1, "freeze_backbone_epochs": 5}

    def test_medium_similarity(self, wst: WarmStartTransfer) -> None:
        schedule = wst.get_adaptive_schedule(_make_mock_task(), _make_mock_task(), similarity_score=0.75)
        assert schedule == {"lr_multiplier": 0.3, "freeze_backbone_epochs": 2}

    def test_low_similarity(self, wst: WarmStartTransfer) -> None:
        schedule = wst.get_adaptive_schedule(_make_mock_task(), _make_mock_task(), similarity_score=0.3)
        assert schedule == {"lr_multiplier": 1.0, "freeze_backbone_epochs": 0}

    def test_boundary_above_0_9_is_high(self, wst: WarmStartTransfer) -> None:
        schedule = wst.get_adaptive_schedule(_make_mock_task(), _make_mock_task(), similarity_score=0.91)
        assert schedule["lr_multiplier"] == pytest.approx(0.1)

    def test_boundary_exactly_0_9_is_medium(self, wst: WarmStartTransfer) -> None:
        # score > 0.9 is high; score == 0.9 falls into medium band
        schedule = wst.get_adaptive_schedule(_make_mock_task(), _make_mock_task(), similarity_score=0.9)
        assert schedule["lr_multiplier"] == pytest.approx(0.3)

    def test_boundary_exactly_0_6_is_medium(self, wst: WarmStartTransfer) -> None:
        schedule = wst.get_adaptive_schedule(_make_mock_task(), _make_mock_task(), similarity_score=0.6)
        assert schedule["lr_multiplier"] == pytest.approx(0.3)

    def test_boundary_below_0_6_is_low(self, wst: WarmStartTransfer) -> None:
        schedule = wst.get_adaptive_schedule(_make_mock_task(), _make_mock_task(), similarity_score=0.59)
        assert schedule["lr_multiplier"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# find_source_task — delegation
# ---------------------------------------------------------------------------


class TestFindSourceTask:
    """find_source_task delegates correctly to the similarity index."""

    def test_delegates_search_call(self, wst: WarmStartTransfer, mock_index: MagicMock) -> None:
        embedding = np.random.default_rng(0).random(64).astype(np.float32)
        mock_index.search.return_value = [("task-a", 0.9), ("task-b", 0.8)]
        result = wst.find_source_task(embedding, k=5)
        mock_index.search.assert_called_once_with(embedding, k=5)
        assert result == [("task-a", 0.9), ("task-b", 0.8)]

    def test_default_k_is_5(self, wst: WarmStartTransfer, mock_index: MagicMock) -> None:
        embedding = np.random.default_rng(1).random(64).astype(np.float32)
        mock_index.search.return_value = []
        wst.find_source_task(embedding)
        mock_index.search.assert_called_once_with(embedding, k=5)


# ---------------------------------------------------------------------------
# warm_start — orchestration
# ---------------------------------------------------------------------------


class TestWarmStart:
    """warm_start correctly orchestrates the find → download → transfer → schedule flow."""

    async def test_full_flow_returns_initialized_model_and_schedule(
        self,
        wst: WarmStartTransfer,
        mock_index: MagicMock,
        mock_artifacts: MagicMock,
        mock_repo: MagicMock,
    ) -> None:
        src_id = str(uuid.uuid4())
        tgt_id = str(uuid.uuid4())
        embedding = np.random.default_rng(0).random(64).astype(np.float32)

        mock_index.search.return_value = [(src_id, 0.95)]
        source_task = _make_mock_task(metadata={"checkpoint_uri": f"models/{src_id}/ckpt"})
        target_task = _make_mock_task()
        mock_repo.get_by_id = AsyncMock(side_effect=[source_task, target_task])
        source_model = _SplitModel()
        mock_artifacts.download_model = AsyncMock(return_value=source_model)
        target_model = _SplitModel()

        result_model, schedule = await wst.warm_start(tgt_id, target_model, embedding)

        assert result_model is target_model
        assert "lr_multiplier" in schedule
        assert "freeze_backbone_epochs" in schedule
        # score 0.95 > 0.9 → high-similarity schedule
        assert schedule["lr_multiplier"] == pytest.approx(0.1)
        assert schedule["freeze_backbone_epochs"] == 5

    async def test_no_candidates_returns_default_schedule(
        self,
        wst: WarmStartTransfer,
        mock_index: MagicMock,
    ) -> None:
        mock_index.search.return_value = []
        target_model = _SplitModel()
        tgt_id = str(uuid.uuid4())
        embedding = np.random.default_rng(2).random(64).astype(np.float32)

        result_model, schedule = await wst.warm_start(tgt_id, target_model, embedding)

        assert result_model is target_model
        assert schedule == {"lr_multiplier": 1.0, "freeze_backbone_epochs": 0}

    async def test_layer_selection_respected(
        self,
        mock_index: MagicMock,
        mock_artifacts: MagicMock,
        mock_repo: MagicMock,
    ) -> None:
        """With layer_selection='encoder_only', only encoder params are transferred."""
        wst_enc = WarmStartTransfer(
            similarity_index=mock_index,
            artifact_manager=mock_artifacts,
            task_repository=mock_repo,
            layer_selection="encoder_only",
        )
        src_id = str(uuid.uuid4())
        tgt_id = str(uuid.uuid4())
        embedding = np.random.default_rng(3).random(64).astype(np.float32)

        mock_index.search.return_value = [(src_id, 0.7)]
        mock_repo.get_by_id = AsyncMock(side_effect=[_make_mock_task(), _make_mock_task()])
        source_model = _SplitModel()
        _fill(source_model, 1.0)
        mock_artifacts.download_model = AsyncMock(return_value=source_model)
        target_model = _SplitModel()
        _fill(target_model, 0.0)

        await wst_enc.warm_start(tgt_id, target_model, embedding)

        assert torch.all(target_model.encoder.weight == 1.0), "encoder not transferred"
        assert torch.all(target_model.head.weight == 0.0), "head should not be transferred"


# ---------------------------------------------------------------------------
# strategy validation
# ---------------------------------------------------------------------------


class TestStrategyValidation:
    """transfer_weights rejects unrecognised strategy values immediately."""

    def test_invalid_strategy_raises_value_error(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        target = _SplitModel()
        with pytest.raises(ValueError, match="Unknown strategy"):
            wst.transfer_weights(source, target, strategy="backbone_only")

    def test_valid_strategies_do_not_raise(self, wst: WarmStartTransfer) -> None:
        source = _SplitModel()
        for strategy in ("all", "encoder_only", "head_only"):
            target = _SplitModel()
            wst.transfer_weights(source, target, strategy=strategy)  # must not raise


# ---------------------------------------------------------------------------
# UUID validation in warm_start
# ---------------------------------------------------------------------------


class TestUUIDValidation:
    """warm_start raises ValueError with context for malformed UUID strings."""

    async def test_invalid_source_uuid_raises_value_error(
        self,
        wst: WarmStartTransfer,
        mock_index: MagicMock,
    ) -> None:
        mock_index.search.return_value = [("not-a-valid-uuid", 0.9)]
        embedding = np.random.default_rng(0).random(64).astype(np.float32)
        tgt_id = str(uuid.uuid4())
        with pytest.raises(ValueError, match="source task"):
            await wst.warm_start(tgt_id, _SplitModel(), embedding)

    async def test_invalid_target_uuid_raises_value_error(
        self,
        wst: WarmStartTransfer,
        mock_index: MagicMock,
        mock_repo: MagicMock,
    ) -> None:
        src_id = str(uuid.uuid4())
        mock_index.search.return_value = [(src_id, 0.9)]
        mock_repo.get_by_id = AsyncMock(return_value=_make_mock_task())
        embedding = np.random.default_rng(0).random(64).astype(np.float32)
        with pytest.raises(ValueError, match="target task"):
            await wst.warm_start("not-a-valid-uuid", _SplitModel(), embedding)


# ---------------------------------------------------------------------------
# Segment-aware layer name matching
# ---------------------------------------------------------------------------


class _PrefixedModel(nn.Module):
    """Module names that *contain* keywords only as substrings, not as exact segments."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_encoder = nn.Linear(4, 8)   # "encoder" is substring of "pre_encoder" only
        self.full_head = nn.Linear(8, 2)     # "head" is substring of "full_head" only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.full_head(torch.relu(self.pre_encoder(x)))


class TestSegmentAwareMatching:
    """Keyword matching operates on exact dot-path segments, not substrings."""

    def test_substring_encoder_not_matched_by_encoder_only(self, wst: WarmStartTransfer) -> None:
        source = _PrefixedModel()
        target = _PrefixedModel()
        _fill(source, 1.0)
        _fill(target, 0.0)
        wst.transfer_weights(source, target, strategy="encoder_only")
        # "pre_encoder" is not an exact match for the "encoder" keyword
        assert torch.all(target.pre_encoder.weight == 0.0)
        assert torch.all(target.pre_encoder.bias == 0.0)

    def test_substring_head_not_matched_by_head_only(self, wst: WarmStartTransfer) -> None:
        source = _PrefixedModel()
        target = _PrefixedModel()
        _fill(source, 1.0)
        _fill(target, 0.0)
        wst.transfer_weights(source, target, strategy="head_only")
        # "full_head" is not an exact match for the "head" keyword
        assert torch.all(target.full_head.weight == 0.0)
        assert torch.all(target.full_head.bias == 0.0)

    def test_exact_segment_encoder_still_matched(self, wst: WarmStartTransfer) -> None:
        """Exact-segment names like "encoder.weight" are still transferred."""
        source = _SplitModel()
        target = _SplitModel()
        _fill(source, 1.0)
        _fill(target, 0.0)
        wst.transfer_weights(source, target, strategy="encoder_only")
        assert torch.all(target.encoder.weight == 1.0)
