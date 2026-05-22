"""Unit tests for ArchitectureTransfer, adapt_architecture, and _build_sequential_from_config."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import torch
import torch.nn as nn

from orcanet.transfer.architecture_transfer import (
    ArchConfig,
    ArchitectureTransfer,
    _build_sequential_from_config,
    adapt_architecture,
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


def _simple_config(
    input_dim: int = 8,
    hidden: int = 64,
    output: int = 10,
) -> ArchConfig:
    return {
        "input_dim": input_dim,
        "layers": [
            {"type": "linear", "size": hidden, "activation": "relu"},
            {"type": "linear", "size": output, "activation": "none"},
        ],
    }


def _deep_config(
    input_dim: int = 8,
    hidden1: int = 128,
    hidden2: int = 64,
    output: int = 10,
) -> ArchConfig:
    """Three linear layers: input → hidden1 → hidden2 → output."""
    return {
        "input_dim": input_dim,
        "layers": [
            {"type": "linear", "size": hidden1, "activation": "relu"},
            {"type": "linear", "size": hidden2, "activation": "relu"},
            {"type": "linear", "size": output, "activation": "none"},
        ],
    }


# ---------------------------------------------------------------------------
# TestAdaptArchitecture
# ---------------------------------------------------------------------------


class TestAdaptArchitecture:
    """adapt_architecture updates only first-layer input and last-layer output."""

    def test_input_dim_updated(self) -> None:
        config = _simple_config(input_dim=8)
        task = _make_task(n_features=25)
        adapted = adapt_architecture(config, task)
        assert adapted["input_dim"] == 25

    def test_output_dim_updated(self) -> None:
        config = _simple_config(output=10)
        task = _make_task(n_classes=3)
        adapted = adapt_architecture(config, task)
        assert adapted["layers"][-1]["size"] == 3

    def test_hidden_dims_unchanged(self) -> None:
        config = _deep_config(hidden1=128, hidden2=64, output=10)
        task = _make_task(n_features=25, n_classes=5)
        adapted = adapt_architecture(config, task)
        # Middle layers are at index 0 and 1; only the last (index 2) changes.
        assert adapted["layers"][0]["size"] == 128
        assert adapted["layers"][1]["size"] == 64

    def test_no_layers_returns_copy(self) -> None:
        config: ArchConfig = {"input_dim": 8, "layers": []}
        task = _make_task(n_features=25, n_classes=5)
        adapted = adapt_architecture(config, task)
        assert adapted["layers"] == []
        assert adapted["input_dim"] == 25

    def test_original_not_mutated(self) -> None:
        config = _simple_config(input_dim=8, output=10)
        original_input_dim = config["input_dim"]
        original_last_size = config["layers"][-1]["size"]
        task = _make_task(n_features=100, n_classes=50)
        adapt_architecture(config, task)
        assert config["input_dim"] == original_input_dim
        assert config["layers"][-1]["size"] == original_last_size

    def test_none_n_features_leaves_input_dim_unchanged(self) -> None:
        config = _simple_config(input_dim=8)
        task = _make_task(n_features=None)
        adapted = adapt_architecture(config, task)
        assert adapted["input_dim"] == 8

    def test_none_n_classes_leaves_output_size_unchanged(self) -> None:
        config = _simple_config(output=10)
        task = _make_task(n_classes=None)
        adapted = adapt_architecture(config, task)
        assert adapted["layers"][-1]["size"] == 10
