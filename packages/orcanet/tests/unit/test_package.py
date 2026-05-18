"""Tests that verify the orcanet package structure and importability."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_package_importable() -> None:
    """orcanet is importable without errors."""
    import orcanet  # noqa: F401


def test_version_attribute() -> None:
    """orcanet.__version__ is a non-empty string."""
    import orcanet

    assert isinstance(orcanet.__version__, str)
    assert orcanet.__version__


@pytest.mark.parametrize(
    "submodule",
    [
        "orcanet.transfer",
        "orcanet.embeddings",
        "orcanet.reasoning",
        "orcanet.reasoning.prompts",
        "orcanet.retrieval",
        "orcanet.api",
    ],
)
def test_submodule_importable(submodule: str) -> None:
    """Every orcanet sub-package is importable without errors."""
    importlib.import_module(submodule)


def test_no_submodule_cross_import_side_effects() -> None:
    """Importing all sub-packages does not introduce unexpected third-party modules."""
    _stdlib = frozenset(sys.stdlib_module_names)  # available on Python 3.10+

    before = set(sys.modules.keys())
    for mod in [
        "orcanet",
        "orcanet.transfer",
        "orcanet.embeddings",
        "orcanet.reasoning",
        "orcanet.reasoning.prompts",
        "orcanet.retrieval",
        "orcanet.api",
    ]:
        importlib.import_module(mod)

    after = set(sys.modules.keys())
    new_mods = after - before
    # Allow orcanet sub-modules and any stdlib module (including private helpers
    # like _collections_abc that Python loads as side effects of stdlib imports).
    unexpected = {
        m
        for m in new_mods
        if not m.startswith("orcanet") and m.split(".")[0].lstrip("_") not in _stdlib
    }
    assert not unexpected, f"Unexpected third-party modules registered: {unexpected}"
