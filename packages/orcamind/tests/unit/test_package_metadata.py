"""Tests for orcamind package-level metadata (__version__, __all__)."""

from __future__ import annotations

import re

import orcamind


def test_version_attribute_exists() -> None:
    assert hasattr(orcamind, "__version__")


def test_version_is_string() -> None:
    assert isinstance(orcamind.__version__, str)


def test_version_matches_semver() -> None:
    assert re.match(r"^\d+\.\d+\.\d+$", orcamind.__version__), (
        f"__version__ '{orcamind.__version__}' does not follow X.Y.Z semver"
    )


def test_version_is_initial_release() -> None:
    assert orcamind.__version__ == "0.1.0"


def test_all_attribute_exists() -> None:
    assert hasattr(orcamind, "__all__")


def test_all_is_list() -> None:
    assert isinstance(orcamind.__all__, list)


def test_all_contains_version() -> None:
    assert "__version__" in orcamind.__all__


def test_all_members_are_accessible() -> None:
    for name in orcamind.__all__:
        assert hasattr(orcamind, name), (
            f"'{name}' is listed in __all__ but not accessible on the package"
        )
