"""Unit tests for WarmStartTransfer — weight transfer strategies."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

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
