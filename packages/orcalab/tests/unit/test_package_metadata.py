"""Tests for orcalab package-level metadata."""

from __future__ import annotations

import re

import orcalab


class TestVersion:
    def test_version_attribute_exists(self) -> None:
        assert hasattr(orcalab, "__version__")

    def test_version_is_string(self) -> None:
        assert isinstance(orcalab.__version__, str)

    def test_version_is_initial_release(self) -> None:
        assert orcalab.__version__ == "0.1.0"

    def test_version_matches_semver(self) -> None:
        semver_pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(semver_pattern, orcalab.__version__), (
            f"Version {orcalab.__version__!r} does not match semver pattern"
        )


class TestAll:
    def test_all_attribute_exists(self) -> None:
        assert hasattr(orcalab, "__all__")

    def test_all_is_list(self) -> None:
        assert isinstance(orcalab.__all__, list)

    def test_version_in_all(self) -> None:
        assert "__version__" in orcalab.__all__

    def test_all_members_are_accessible(self) -> None:
        for name in orcalab.__all__:
            assert hasattr(orcalab, name), f"__all__ member {name!r} not accessible on orcalab"
