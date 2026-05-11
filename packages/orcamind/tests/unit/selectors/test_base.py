"""Tests for the ModelSelector ABC contract."""

from __future__ import annotations

import pytest

from orcamind.selectors.base import ModelSelector
from orcamind.selectors.nearest_neighbor import NearestNeighborSelector
from orcamind.selectors.predictor import PerformancePredictor
from orcamind.selectors.ranker import LearningToRankSelector


class TestAbstractInterface:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ModelSelector()  # type: ignore[abstract]

    def test_nearest_neighbor_is_subclass(self) -> None:
        assert issubclass(NearestNeighborSelector, ModelSelector)

    def test_ranker_is_subclass(self) -> None:
        assert issubclass(LearningToRankSelector, ModelSelector)

    def test_predictor_is_subclass(self) -> None:
        assert issubclass(PerformancePredictor, ModelSelector)
