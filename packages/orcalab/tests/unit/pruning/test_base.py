"""Unit tests for the abstract Pruner base class."""

from __future__ import annotations

import pytest

from orcalab.pruning.base import Pruner


# ---------------------------------------------------------------------------
# Concrete stub implementations used only within this module
# ---------------------------------------------------------------------------


class _AlwaysPrunePruner(Pruner):
    """Minimal concrete Pruner that always votes to prune."""

    @property
    def name(self) -> str:
        return "always_prune"

    def should_prune(
        self,
        trial_id: str,
        step: int,
        current_value: float,
        all_trial_values: dict[str, list[float]],
    ) -> bool:
        return True


class _NeverPrunePruner(Pruner):
    """Minimal concrete Pruner that never votes to prune."""

    @property
    def name(self) -> str:
        return "never_prune"

    def should_prune(
        self,
        trial_id: str,
        step: int,
        current_value: float,
        all_trial_values: dict[str, list[float]],
    ) -> bool:
        return False


class _MissingNamePruner(Pruner):
    """Pruner that implements should_prune but omits the name property."""

    def should_prune(
        self,
        trial_id: str,
        step: int,
        current_value: float,
        all_trial_values: dict[str, list[float]],
    ) -> bool:
        return False


class _MissingShouldPrunePruner(Pruner):
    """Pruner that implements name but omits should_prune."""

    @property
    def name(self) -> str:
        return "missing_method"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrunerABCEnforcement:
    def test_cannot_instantiate_pruner_directly(self) -> None:
        """Pruner is abstract and must not be instantiatable directly."""
        with pytest.raises(TypeError):
            Pruner()  # type: ignore[abstract]

    def test_missing_name_raises_on_instantiation(self) -> None:
        """A subclass without the name property cannot be instantiated."""
        with pytest.raises(TypeError):
            _MissingNamePruner()  # type: ignore[abstract]

    def test_missing_should_prune_raises_on_instantiation(self) -> None:
        """A subclass without should_prune cannot be instantiated."""
        with pytest.raises(TypeError):
            _MissingShouldPrunePruner()  # type: ignore[abstract]

    def test_full_implementation_can_be_instantiated(self) -> None:
        """A complete Pruner subclass can be instantiated without error."""
        pruner = _AlwaysPrunePruner()
        assert isinstance(pruner, Pruner)

    def test_name_property_returns_string(self) -> None:
        assert isinstance(_AlwaysPrunePruner().name, str)
        assert isinstance(_NeverPrunePruner().name, str)


class TestPrunerShouldPruneContract:
    def test_should_prune_returns_bool_true(self) -> None:
        pruner = _AlwaysPrunePruner()
        result = pruner.should_prune("trial_1", 1, 0.5, {})
        assert result is True

    def test_should_prune_returns_bool_false(self) -> None:
        pruner = _NeverPrunePruner()
        result = pruner.should_prune("trial_1", 1, 0.5, {})
        assert result is False

    def test_should_prune_accepts_empty_all_trial_values(self) -> None:
        pruner = _AlwaysPrunePruner()
        result = pruner.should_prune("trial_1", 5, 0.9, {})
        assert isinstance(result, bool)

    def test_should_prune_accepts_populated_all_trial_values(self) -> None:
        pruner = _NeverPrunePruner()
        all_values = {"trial_a": [0.1, 0.2], "trial_b": [0.3]}
        result = pruner.should_prune("trial_c", 2, 0.5, all_values)
        assert isinstance(result, bool)

    def test_should_prune_signature_accepts_float_current_value(self) -> None:
        pruner = _AlwaysPrunePruner()
        assert pruner.should_prune("t", 1, 0.0, {}) is True
        assert pruner.should_prune("t", 1, 1.0, {}) is True
        assert pruner.should_prune("t", 1, -99.9, {}) is True
