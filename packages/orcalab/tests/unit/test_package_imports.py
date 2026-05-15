"""Tests that all orcalab submodules are importable and properly structured."""

from __future__ import annotations

import importlib
import sys

import pytest

MODULES = [
    "orcalab",
    "orcalab.experiments",
    "orcalab.search",
    "orcalab.search_spaces",
    "orcalab.pruning",
    "orcalab.orchestration",
    "orcalab.orchestration.flows",
    "orcalab.orchestration.tasks",
    "orcalab.visualization",
    "orcalab.api",
    "orcalab.cli",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_is_importable(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", MODULES)
def test_module_has_docstring(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod.__doc__ is not None and mod.__doc__.strip() != ""


def test_cli_module_exposes_app() -> None:
    from orcalab import cli

    assert hasattr(cli, "app")


@pytest.mark.clean_torch_modules
def test_root_package_does_not_import_torch_on_load() -> None:
    import orcalab  # noqa: F401

    torch_modules = [k for k in sys.modules if k == "torch" or k.startswith("torch.")]
    assert torch_modules == [], f"Unexpected torch import on load: {torch_modules}"
