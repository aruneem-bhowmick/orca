"""Unit tests for SearchSpaceComposer merge, inherit, and restrict operations."""

from __future__ import annotations

import optuna
import pytest

from orcalab.search_spaces.composer import SearchSpaceComposer
from orcalab.search_spaces.parameters import (
    FloatParameter,
    IntParameter,
    LogUniformParameter,
)
from orcalab.search_spaces.space import SearchSpace

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _trial() -> optuna.Trial:
    return optuna.create_study().ask()


def _make_space(name: str, *params: IntParameter | FloatParameter | LogUniformParameter) -> SearchSpace:
    space = SearchSpace(name=name)
    for p in params:
        space.add(p)
    return space


class TestMerge:
    def test_unions_parameters_from_all_spaces(self) -> None:
        space_a = _make_space("a", IntParameter("layers", low=2, high=10))
        space_b = _make_space("b", FloatParameter("dropout", low=0.0, high=0.5))
        merged = SearchSpaceComposer.merge(space_a, space_b, name="merged")
        assert set(merged._params.keys()) == {"layers", "dropout"}

    def test_later_space_overrides_on_name_conflict(self) -> None:
        space_a = _make_space("a", IntParameter("layers", low=2, high=10))
        space_b = _make_space("b", IntParameter("layers", low=50, high=200))
        merged = SearchSpaceComposer.merge(space_a, space_b, name="merged")
        assert merged._params["layers"].low == 50
        assert merged._params["layers"].high == 200

    def test_merged_space_can_sample(self) -> None:
        space_a = _make_space("a", IntParameter("x", low=1, high=5))
        space_b = _make_space("b", FloatParameter("y", low=0.0, high=1.0))
        merged = SearchSpaceComposer.merge(space_a, space_b, name="merged")
        result = merged.sample(_trial())
        assert "x" in result
        assert "y" in result

    def test_conditions_from_all_spaces_are_preserved_in_order(self) -> None:
        # space_a's condition always fires; space_b's condition requires cond_a
        # to already be in sampled — this proves space_a's condition runs first.
        space_a = SearchSpace(name="a")
        space_a.add_condition(lambda _: True, IntParameter("cond_a", low=1, high=5))
        space_b = SearchSpace(name="b")
        space_b.add_condition(
            lambda sampled: "cond_a" in sampled,
            IntParameter("cond_b", low=10, high=20),
        )
        merged = SearchSpaceComposer.merge(space_a, space_b, name="merged")
        result = merged.sample(_trial())
        assert "cond_a" in result
        assert "cond_b" in result

    def test_merged_name_is_provided_name(self) -> None:
        space_a = _make_space("a", IntParameter("x", low=1, high=5))
        merged = SearchSpaceComposer.merge(space_a, name="custom_name")
        assert merged.name == "custom_name"


class TestInherit:
    def test_child_overrides_parent_parameter(self) -> None:
        parent = _make_space("parent", IntParameter("layers", low=4, high=16))
        child = _make_space("child", IntParameter("layers", low=32, high=64))
        result = SearchSpaceComposer.inherit(parent, child)
        assert result._params["layers"].low == 32
        assert result._params["layers"].high == 64

    def test_child_adds_params_not_in_parent(self) -> None:
        parent = _make_space("parent", IntParameter("layers", low=4, high=16))
        child = _make_space("child", FloatParameter("dropout", low=0.0, high=0.5))
        result = SearchSpaceComposer.inherit(parent, child)
        assert "layers" in result._params
        assert "dropout" in result._params

    def test_parent_params_not_overridden_by_child_are_kept(self) -> None:
        parent = _make_space(
            "parent",
            IntParameter("layers", low=4, high=16),
            LogUniformParameter("lr", low=1e-4, high=1e-1),
        )
        child = _make_space("child", IntParameter("layers", low=32, high=64))
        result = SearchSpaceComposer.inherit(parent, child)
        assert "lr" in result._params
        assert result._params["lr"].low == pytest.approx(1e-4)

    def test_result_takes_name_and_description_from_child(self) -> None:
        parent = SearchSpace(name="parent_name", description="parent desc")
        parent.add(IntParameter("x", low=1, high=5))
        child = SearchSpace(name="child_name", description="child desc")
        child.add(FloatParameter("y", low=0.0, high=1.0))
        result = SearchSpaceComposer.inherit(parent, child)
        assert result.name == "child_name"
        assert result.description == "child desc"
        assert result.description != parent.description

    def test_conditions_from_both_parent_and_child_are_included_in_order(self) -> None:
        # parent's condition always fires; child's condition requires cond_p to
        # already be sampled — this proves parent conditions precede child ones.
        parent = SearchSpace(name="parent")
        parent.add_condition(lambda _: True, IntParameter("cond_p", low=1, high=5))
        child = SearchSpace(name="child")
        child.add_condition(
            lambda sampled: "cond_p" in sampled,
            IntParameter("cond_c", low=10, high=20),
        )
        result = SearchSpaceComposer.inherit(parent, child)
        sampled = result.sample(_trial())
        assert "cond_p" in sampled
        assert "cond_c" in sampled


class TestRestrict:
    def test_keeps_only_allowed_params(self) -> None:
        space = _make_space(
            "full",
            IntParameter("layers", low=2, high=10),
            FloatParameter("dropout", low=0.0, high=0.5),
            LogUniformParameter("lr", low=1e-4, high=1e-1),
        )
        restricted = SearchSpaceComposer.restrict(space, ["layers", "lr"])
        assert set(restricted._params.keys()) == {"layers", "lr"}
        assert "dropout" not in restricted._params

    def test_restricted_space_can_sample(self) -> None:
        space = _make_space(
            "full",
            IntParameter("layers", low=2, high=10),
            FloatParameter("dropout", low=0.0, high=0.5),
        )
        restricted = SearchSpaceComposer.restrict(space, ["layers"])
        result = restricted.sample(_trial())
        assert "layers" in result
        assert "dropout" not in result

    def test_conditions_for_excluded_params_are_dropped(self) -> None:
        space = SearchSpace(name="full")
        space.add(IntParameter("x", low=1, high=5))
        # This condition's parameter ("y") will be excluded
        space.add_condition(lambda _: True, FloatParameter("y", low=0.0, high=1.0))
        restricted = SearchSpaceComposer.restrict(space, ["x"])
        assert len(restricted._conditions) == 0

    def test_conditions_for_allowed_params_are_kept(self) -> None:
        space = SearchSpace(name="full")
        space.add(IntParameter("x", low=1, high=5))
        space.add_condition(lambda _: True, FloatParameter("y", low=0.0, high=1.0))
        restricted = SearchSpaceComposer.restrict(space, ["x", "y"])
        assert len(restricted._conditions) == 1

    def test_preserves_name_and_description(self) -> None:
        space = SearchSpace(name="my_space", description="original desc")
        space.add(IntParameter("x", low=1, high=5))
        restricted = SearchSpaceComposer.restrict(space, ["x"])
        assert restricted.name == "my_space"
        assert restricted.description == "original desc"
