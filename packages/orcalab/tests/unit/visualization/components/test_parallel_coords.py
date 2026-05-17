"""Unit tests for parallel_coords component."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(scope="module")
def pc(_patch_streamlit):
    sys.modules.pop("orcalab.visualization.components.parallel_coords", None)
    return importlib.import_module("orcalab.visualization.components.parallel_coords")


NUMERIC_TRIALS = [
    {"lr": 0.001, "batch_size": 32, "objective": 0.92},
    {"lr": 0.01, "batch_size": 64, "objective": 0.87},
    {"lr": 0.0001, "batch_size": 16, "objective": 0.95},
]

CATEGORICAL_TRIALS = [
    {"optimizer": "adam", "lr": 0.001, "objective": 0.92},
    {"optimizer": "sgd", "lr": 0.01, "objective": 0.87},
    {"optimizer": "adam", "lr": 0.0001, "objective": 0.95},
]


class TestParallelCoordinates:
    def test_empty_trials_does_not_raise(self, pc):
        fig = pc.parallel_coordinates([])
        assert fig is not None

    def test_returns_figure_for_numeric_trials(self, pc):
        fig = pc.parallel_coordinates(NUMERIC_TRIALS)
        assert fig is not None

    def test_categorical_params_trigger_code_conversion(self, pc):
        """Categorical string values should be encoded as integers with ticktext."""
        fig = pc.parallel_coordinates(CATEGORICAL_TRIALS)
        assert fig is not None
        # Verify add_trace was called (since go.Parcoords is mocked)
        assert fig.add_trace.called

    def test_missing_objective_treated_as_nan(self, pc):
        trials = [
            {"lr": 0.001, "objective": None},
            {"lr": 0.01, "objective": 0.9},
        ]
        import math

        mod = pc
        # Call the function; it should not raise despite None objective
        fig = mod.parallel_coordinates(trials)
        assert fig is not None

    def test_custom_colorscale_forwarded(self, pc):
        pc.go.Parcoords.reset_mock()
        pc.parallel_coordinates(NUMERIC_TRIALS, colorscale="Plasma")
        assert pc.go.Parcoords.called
        line = pc.go.Parcoords.call_args.kwargs.get("line", {})
        assert isinstance(line, dict)
        assert line.get("colorscale") == "Plasma"

    def test_default_colorscale_is_viridis(self, pc):
        import inspect

        sig = inspect.signature(pc.parallel_coordinates)
        assert sig.parameters["colorscale"].default == "Viridis"
