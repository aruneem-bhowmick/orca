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


def _simple_config(
    input_dim: int = 8,
    hidden: int = 64,
    output: int = 10,
) -> ArchConfig:
    """Return a 2-layer (hidden → output) architecture config for test fixtures."""
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
        """``input_dim`` is overwritten with the task's ``n_features``."""
        config = _simple_config(input_dim=8)
        task = _make_task(n_features=25)
        adapted = adapt_architecture(config, task)
        assert adapted["input_dim"] == 25

    def test_output_dim_updated(self) -> None:
        """Last layer ``size`` is overwritten with the task's ``n_classes``."""
        config = _simple_config(output=10)
        task = _make_task(n_classes=3)
        adapted = adapt_architecture(config, task)
        assert adapted["layers"][-1]["size"] == 3

    def test_hidden_dims_unchanged(self) -> None:
        """Middle layers are never modified; only the first input and last output change."""
        config = _deep_config(hidden1=128, hidden2=64, output=10)
        task = _make_task(n_features=25, n_classes=5)
        adapted = adapt_architecture(config, task)
        # Middle layers are at index 0 and 1; only the last (index 2) changes.
        assert adapted["layers"][0]["size"] == 128
        assert adapted["layers"][1]["size"] == 64

    def test_no_layers_returns_copy(self) -> None:
        """A config with an empty ``layers`` list is returned as-is (with updated ``input_dim``)."""
        config: ArchConfig = {"input_dim": 8, "layers": []}
        task = _make_task(n_features=25, n_classes=5)
        adapted = adapt_architecture(config, task)
        assert adapted["layers"] == []
        assert adapted["input_dim"] == 25

    def test_original_not_mutated(self) -> None:
        """The source config dict is deep-copied; the original is never modified."""
        config = _simple_config(input_dim=8, output=10)
        original_input_dim = config["input_dim"]
        original_last_size = config["layers"][-1]["size"]
        task = _make_task(n_features=100, n_classes=50)
        adapt_architecture(config, task)
        assert config["input_dim"] == original_input_dim
        assert config["layers"][-1]["size"] == original_last_size

    def test_none_n_features_leaves_input_dim_unchanged(self) -> None:
        """When ``n_features`` is ``None``, ``input_dim`` is left at its original value."""
        config = _simple_config(input_dim=8)
        task = _make_task(n_features=None)
        adapted = adapt_architecture(config, task)
        assert adapted["input_dim"] == 8

    def test_none_n_classes_leaves_output_size_unchanged(self) -> None:
        """When ``n_classes`` is ``None``, the last layer's ``size`` is left unchanged."""
        config = _simple_config(output=10)
        task = _make_task(n_classes=None)
        adapted = adapt_architecture(config, task)
        assert adapted["layers"][-1]["size"] == 10


# ---------------------------------------------------------------------------
# Helpers for ArchitectureTransfer tests
# ---------------------------------------------------------------------------


def _mock_client(source_name: str = "mlp_128_64") -> Any:
    """Return an AsyncMock OrcaMindClient pre-configured for score_transfer tests."""
    from unittest.mock import AsyncMock
    from uuid import uuid4 as _uuid4
    from orca_shared.schemas.model import ModelSummary

    mock = AsyncMock()
    mock.get_best_model.return_value = ModelSummary(
        model_id=_uuid4(),
        name=source_name,
        architecture=source_name,
    )
    return mock


def _mock_embedder(sim_value: float = 0.85) -> Any:
    """Return a MagicMock ArchitectureEmbedder with deterministic similarity."""
    from unittest.mock import MagicMock
    from orcanet.embeddings.architecture_embedder import ArchitectureEmbedder

    mock = MagicMock(spec=ArchitectureEmbedder)
    mock.similarity.return_value = sim_value
    return mock


def _transfer_with_registry(
    source_name: str = "mlp_128_64",
    sim_value: float = 0.85,
    configs: dict[str, ArchConfig] | None = None,
) -> tuple[ArchitectureTransfer, Any, Any]:
    """Return (transfer, mock_client, mock_embedder) with one registered config."""
    client = _mock_client(source_name)
    embedder = _mock_embedder(sim_value)
    transfer = ArchitectureTransfer(architecture_embedder=embedder, orcamind_client=client)
    for name, cfg in (configs or {source_name: _simple_config()}).items():
        transfer.register_config(name, cfg)
    return transfer, client, embedder


# ---------------------------------------------------------------------------
# TestArchitectureTransferScore
# ---------------------------------------------------------------------------


class TestArchitectureTransferScore:
    """score_transfer delegates to OrcaMind + ArchitectureEmbedder and returns TransferScore."""

    def test_returns_transfer_score_instance(self) -> None:
        """``score_transfer`` always returns a ``TransferScore`` instance."""
        transfer, _, _ = _transfer_with_registry()
        source = _make_task()
        target = _make_task()
        score = transfer.score_transfer(source, target)
        assert isinstance(score, TransferScore)

    def test_score_calls_orcamind_get_best_model(self) -> None:
        """``get_best_model`` is called exactly once with the source task's ID."""
        transfer, client, _ = _transfer_with_registry()
        source = _make_task()
        target = _make_task()
        transfer.score_transfer(source, target)
        client.get_best_model.assert_called_once_with(source.task_id)

    def test_score_uses_embedder_similarity(self) -> None:
        """The architecture embedder's ``similarity`` is invoked during scoring."""
        transfer, _, embedder = _transfer_with_registry()
        source = _make_task()
        target = _make_task()
        transfer.score_transfer(source, target)
        assert embedder.similarity.called

    def test_overall_equals_max_similarity(self) -> None:
        """``overall`` equals the maximum similarity across all registered candidates."""
        configs = {
            "arch_a": _simple_config(hidden=64),
            "arch_b": _simple_config(hidden=128),
        }
        transfer, _, _ = _transfer_with_registry(sim_value=0.85, configs=configs)
        source = _make_task()
        target = _make_task()
        score = transfer.score_transfer(source, target)
        # Both candidates return 0.85; max = 0.85
        assert abs(score.overall - 0.85) < 1e-6

    def test_recommended_layers_is_always_empty(self) -> None:
        """Architecture transfer is model-level; ``recommended_layers`` is always ``[]``."""
        transfer, _, _ = _transfer_with_registry()
        source = _make_task()
        target = _make_task()
        score = transfer.score_transfer(source, target)
        assert score.recommended_layers == []

    def test_reasoning_contains_best_arch_name(self) -> None:
        """The best candidate's name appears in the ``reasoning`` string."""
        configs = {"best_arch": _simple_config()}
        client = _mock_client(source_name="best_arch")
        embedder = _mock_embedder(0.9)
        transfer = ArchitectureTransfer(architecture_embedder=embedder, orcamind_client=client)
        transfer.register_config("best_arch", _simple_config())
        source = _make_task()
        target = _make_task()
        score = transfer.score_transfer(source, target)
        assert "best_arch" in score.reasoning

    def test_empty_registry_returns_zero_overall(self) -> None:
        """An empty config registry produces ``overall == 0.0`` and an empty ``layer_scores``."""
        client = _mock_client()
        embedder = _mock_embedder()
        transfer = ArchitectureTransfer(architecture_embedder=embedder, orcamind_client=client)
        # No configs registered
        source = _make_task()
        target = _make_task()
        score = transfer.score_transfer(source, target)
        assert score.overall == 0.0

    def test_layer_scores_keys_match_registered_names(self) -> None:
        """``layer_scores`` has exactly one entry per registered config name."""
        configs = {"arch_a": _simple_config(), "arch_b": _simple_config(hidden=128)}
        transfer, _, _ = _transfer_with_registry(configs=configs)
        source = _make_task()
        target = _make_task()
        score = transfer.score_transfer(source, target)
        assert set(score.layer_scores.keys()) == {"arch_a", "arch_b"}

    def test_overall_in_unit_interval(self) -> None:
        """``overall`` is always a float in ``[0.0, 1.0]``."""
        transfer, _, _ = _transfer_with_registry(sim_value=0.5)
        source = _make_task()
        target = _make_task()
        score = transfer.score_transfer(source, target)
        assert 0.0 <= score.overall <= 1.0


# ---------------------------------------------------------------------------
# TestArchitectureTransferExecute
# ---------------------------------------------------------------------------


def _config_with_activations(input_dim: int = 10) -> ArchConfig:
    """Return a config exercising every supported activation type.

    Layers use ``sigmoid``, ``tanh``, ``gelu``, and ``none`` in sequence so
    that ``_build_sequential_from_config`` appends each corresponding
    ``nn.Module`` activation (or skips it for ``none``).
    """
    return {
        "input_dim": input_dim,
        "layers": [
            {"type": "linear", "size": 8, "activation": "sigmoid"},
            {"type": "linear", "size": 6, "activation": "tanh"},
            {"type": "linear", "size": 4, "activation": "gelu"},
            {"type": "linear", "size": 5, "activation": "none"},
        ],
    }


def _transfer_for_execute(
    source_name: str = "arch_net",
    source_n_features: int = 8,
    source_hidden: int = 64,
    source_output: int = 10,
    target_n_features: int = 25,
    target_n_classes: int = 3,
) -> tuple[ArchitectureTransfer, Task, Task, nn.Module]:
    """Return (transfer, source_task, target_task, source_model) pre-wired for execute_transfer."""
    client = _mock_client(source_name)
    embedder = _mock_embedder(0.9)
    transfer = ArchitectureTransfer(architecture_embedder=embedder, orcamind_client=client)

    config = _deep_config(
        input_dim=source_n_features,
        hidden1=source_hidden,
        hidden2=source_hidden // 2,
        output=source_output,
    )
    transfer.register_config(source_name, config)

    source_task = _make_task(n_features=source_n_features, n_classes=source_output)
    target_task = _make_task(n_features=target_n_features, n_classes=target_n_classes)

    # Build a source model matching the source config so middle-layer copies work.
    source_model = _build_sequential_from_config(config)
    torch.manual_seed(42)
    for module in source_model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias)

    return transfer, source_task, target_task, source_model


class TestArchitectureTransferExecute:
    """execute_transfer returns an nn.Module with correct input/output dimensions."""

    def test_returns_nn_module(self) -> None:
        """``execute_transfer`` returns an ``nn.Module`` instance."""
        transfer, source, target, source_model = _transfer_for_execute()
        result = transfer.execute_transfer(source, target, source_model)
        assert isinstance(result, nn.Module)

    def test_correct_input_size(self) -> None:
        """The first ``nn.Linear`` layer in the adapted model has ``in_features == target.n_features``."""
        transfer, source, target, source_model = _transfer_for_execute(
            target_n_features=25
        )
        result = transfer.execute_transfer(source, target, source_model)
        first_linear = next(m for m in result.modules() if isinstance(m, nn.Linear))
        assert first_linear.in_features == 25

    def test_correct_output_size(self) -> None:
        """The last ``nn.Linear`` layer in the adapted model has ``out_features == target.n_classes``."""
        transfer, source, target, source_model = _transfer_for_execute(
            target_n_classes=3
        )
        result = transfer.execute_transfer(source, target, source_model)
        all_linears = [m for m in result.modules() if isinstance(m, nn.Linear)]
        assert all_linears[-1].out_features == 3

    def test_middle_layer_width_preserved(self) -> None:
        """Hidden (middle) layer widths from the source config are preserved after adaptation."""
        transfer, source, target, source_model = _transfer_for_execute(
            source_hidden=64, target_n_features=25, target_n_classes=5
        )
        result = transfer.execute_transfer(source, target, source_model)
        all_linears = [m for m in result.modules() if isinstance(m, nn.Linear)]
        # Middle linear(s) should have the same width as in the source config.
        assert len(all_linears) >= 3
        assert all_linears[1].out_features == 32  # hidden2 = hidden1 // 2 = 64 // 2

    def test_source_model_not_mutated(self) -> None:
        """The source model's parameters are never modified during transfer execution."""
        transfer, source, target, source_model = _transfer_for_execute()
        original_state = {k: v.clone() for k, v in source_model.state_dict().items()}
        transfer.execute_transfer(source, target, source_model)
        for k, v in source_model.state_dict().items():
            assert torch.equal(v, original_state[k]), f"Source param '{k}' was mutated"

    def test_execute_without_prior_score_does_not_raise(self) -> None:
        """execute_transfer calls score_transfer internally if needed."""
        transfer, source, target, source_model = _transfer_for_execute()
        assert transfer._last_best_match is None
        result = transfer.execute_transfer(source, target, source_model)
        assert isinstance(result, nn.Module)

    def test_all_activation_types_supported(self) -> None:
        """``_build_sequential_from_config`` materialises sigmoid, tanh, gelu, and none layers.

        Exercises every activation in ``_ACTIVATION_MAP`` plus the no-op ``"none"``
        branch, runs a forward pass, and asserts the output shape matches the final
        layer size defined in the config.
        """
        config = _config_with_activations(input_dim=10)
        model = _build_sequential_from_config(config)
        assert isinstance(model, nn.Module)
        x = torch.randn(2, 10)
        out = model(x)
        assert out.shape == (2, 5)


# ---------------------------------------------------------------------------
# TestArchitectureTransferMetadata
# ---------------------------------------------------------------------------


class TestArchitectureTransferMetadata:
    """``get_transfer_metadata`` reflects constructor parameters and registry state."""

    def test_strategy_name(self) -> None:
        """``strategy`` key is always ``"architecture_transfer"``."""
        client = _mock_client()
        embedder = _mock_embedder()
        transfer = ArchitectureTransfer(architecture_embedder=embedder, orcamind_client=client)
        assert transfer.get_transfer_metadata()["strategy"] == "architecture_transfer"

    def test_top_k_reflected(self) -> None:
        """Custom ``top_k_candidates`` value is reflected in the metadata dict."""
        client = _mock_client()
        embedder = _mock_embedder()
        transfer = ArchitectureTransfer(
            architecture_embedder=embedder,
            orcamind_client=client,
            top_k_candidates=5,
        )
        assert transfer.get_transfer_metadata()["top_k_candidates"] == 5

    def test_n_registered_configs_increments(self) -> None:
        """``n_registered_configs`` increases by one after each ``register_config`` call."""
        client = _mock_client()
        embedder = _mock_embedder()
        transfer = ArchitectureTransfer(architecture_embedder=embedder, orcamind_client=client)
        assert transfer.get_transfer_metadata()["n_registered_configs"] == 0
        transfer.register_config("arch_a", _simple_config())
        assert transfer.get_transfer_metadata()["n_registered_configs"] == 1
        transfer.register_config("arch_b", _simple_config(hidden=128))
        assert transfer.get_transfer_metadata()["n_registered_configs"] == 2

    def test_default_top_k_is_ten(self) -> None:
        """Default ``top_k_candidates`` is 10 when not specified at construction."""
        client = _mock_client()
        embedder = _mock_embedder()
        transfer = ArchitectureTransfer(architecture_embedder=embedder, orcamind_client=client)
        assert transfer.get_transfer_metadata()["top_k_candidates"] == 10


# ---------------------------------------------------------------------------
# TestArchitectureTransferGuards
# ---------------------------------------------------------------------------


class TestArchitectureTransferGuards:
    """``register_config`` stores and replaces configs as expected."""

    def test_register_config_stores_config(self) -> None:
        """A registered config is retrievable from ``_config_registry`` by name."""
        client = _mock_client()
        embedder = _mock_embedder()
        transfer = ArchitectureTransfer(architecture_embedder=embedder, orcamind_client=client)
        cfg = _simple_config(hidden=256)
        transfer.register_config("my_arch", cfg)
        assert transfer._config_registry["my_arch"] == cfg

    def test_overwrite_registered_config(self) -> None:
        """Registering a second config under the same name replaces the first."""
        client = _mock_client()
        embedder = _mock_embedder()
        transfer = ArchitectureTransfer(architecture_embedder=embedder, orcamind_client=client)
        transfer.register_config("arch", _simple_config(hidden=64))
        new_cfg = _simple_config(hidden=256)
        transfer.register_config("arch", new_cfg)
        assert transfer._config_registry["arch"]["layers"][0]["size"] == 256
