"""Unit tests for EvolutionarySearch."""

from __future__ import annotations

import numpy as np
import pytest

from orcalab.search.evolutionary import (
    _build_dim_map,
    _decode,
    _encode,
    _total_dim,
)
from orcalab.search_spaces.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
)
from orcalab.search_spaces.space import SearchSpace


def _float_space() -> SearchSpace:
    return SearchSpace("float").add(FloatParameter("lr", low=1e-4, high=1e-1))


def _log_float_space() -> SearchSpace:
    return SearchSpace("logfloat").add(FloatParameter("lr", low=1e-4, high=1e-1, log=True))


def _int_space() -> SearchSpace:
    return SearchSpace("int").add(IntParameter("layers", low=2, high=20))


def _cat_space() -> SearchSpace:
    return SearchSpace("cat").add(
        CategoricalParameter("optimizer", choices=["adam", "sgd", "rmsprop"])
    )


def _mixed_space() -> SearchSpace:
    return (
        SearchSpace("mixed")
        .add(FloatParameter("lr", low=1e-4, high=1e-1))
        .add(IntParameter("layers", low=2, high=20))
        .add(CategoricalParameter("optimizer", choices=["adam", "sgd"]))
    )


class TestEncodingDecoding:
    def test_int_roundtrip(self) -> None:
        space = _int_space()
        dim_map = _build_dim_map(space)
        for v in [2, 5, 10, 15, 20]:
            vec = _encode({"layers": v}, dim_map)
            assert 0.0 <= vec[0] <= 1.0
            decoded = _decode(vec, dim_map)
            assert decoded["layers"] == v

    def test_float_linear_roundtrip(self) -> None:
        space = _float_space()
        dim_map = _build_dim_map(space)
        for v in [1e-4, 1e-3, 1e-2, 1e-1]:
            vec = _encode({"lr": v}, dim_map)
            assert 0.0 <= vec[0] <= 1.0
            decoded = _decode(vec, dim_map)
            assert decoded["lr"] == pytest.approx(v, rel=1e-6)

    def test_float_log_roundtrip(self) -> None:
        space = _log_float_space()
        dim_map = _build_dim_map(space)
        for v in [1e-4, 1e-3, 1e-2, 1e-1]:
            vec = _encode({"lr": v}, dim_map)
            assert 0.0 <= vec[0] <= 1.0
            decoded = _decode(vec, dim_map)
            assert decoded["lr"] == pytest.approx(v, rel=1e-6)

    def test_categorical_roundtrip(self) -> None:
        space = _cat_space()
        dim_map = _build_dim_map(space)
        for choice in ["adam", "sgd", "rmsprop"]:
            vec = _encode({"optimizer": choice}, dim_map)
            assert vec.sum() == pytest.approx(1.0)
            decoded = _decode(vec, dim_map)
            assert decoded["optimizer"] == choice

    def test_mixed_space_roundtrip(self) -> None:
        space = _mixed_space()
        dim_map = _build_dim_map(space)
        assert _total_dim(dim_map) == 4  # 1 + 1 + 2

        params = {"lr": 0.01, "layers": 8, "optimizer": "sgd"}
        vec = _encode(params, dim_map)
        assert len(vec) == 4
        decoded = _decode(vec, dim_map)
        assert decoded["lr"] == pytest.approx(0.01, rel=1e-6)
        assert decoded["layers"] == 8
        assert decoded["optimizer"] == "sgd"

    def test_int_boundary_values_roundtrip(self) -> None:
        space = _int_space()
        dim_map = _build_dim_map(space)
        for boundary in [2, 20]:
            vec = _encode({"layers": boundary}, dim_map)
            decoded = _decode(vec, dim_map)
            assert decoded["layers"] == boundary

    def test_float_boundary_values_roundtrip(self) -> None:
        space = _float_space()
        dim_map = _build_dim_map(space)
        for boundary in [1e-4, 1e-1]:
            vec = _encode({"lr": boundary}, dim_map)
            decoded = _decode(vec, dim_map)
            assert decoded["lr"] == pytest.approx(boundary, rel=1e-6)

    def test_decoded_value_stays_in_bounds_when_vec_is_clipped(self) -> None:
        space = _float_space()
        dim_map = _build_dim_map(space)
        decoded_low = _decode(np.array([-0.5]), dim_map)
        decoded_high = _decode(np.array([1.5]), dim_map)
        assert decoded_low["lr"] == pytest.approx(1e-4, rel=1e-6)
        assert decoded_high["lr"] == pytest.approx(1e-1, rel=1e-6)
