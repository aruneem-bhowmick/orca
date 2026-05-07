"""Verify every orcamind sub-module is importable without side-effects."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "orcamind",
    "orcamind.core",
    "orcamind.models",
    "orcamind.embedders",
    "orcamind.selectors",
    "orcamind.data",
    "orcamind.training",
    "orcamind.api",
    "orcamind.cli",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_is_importable(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", MODULES)
def test_module_has_docstring(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod.__doc__ is not None and mod.__doc__.strip(), (
        f"{module_name} is missing a module-level docstring"
    )


def test_cli_module_exposes_app() -> None:
    from orcamind import cli

    assert hasattr(cli, "app"), "orcamind.cli must expose a top-level `app` object"


def test_root_package_does_not_import_torch_on_load() -> None:
    """Importing orcamind must not pull in heavy ML frameworks eagerly."""
    import sys

    # Remove cached modules so we test a fresh import path
    for key in list(sys.modules.keys()):
        if key.startswith("orcamind"):
            del sys.modules[key]

    importlib.import_module("orcamind")

    # torch is NOT in orcamind/__init__.py, so it must not appear
    assert "torch" not in sys.modules, (
        "torch was imported as a side-effect of `import orcamind` — "
        "defer heavy imports to individual sub-modules"
    )
