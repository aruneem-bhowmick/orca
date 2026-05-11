"""Model selectors: ranking and recommending model configurations."""

from __future__ import annotations

from .base import ModelSelector

__all__ = [
    "LearningToRankSelector",
    "ModelSelector",
    "NearestNeighborSelector",
    "PerformancePredictor",
]


def __getattr__(name: str) -> object:
    if name == "NearestNeighborSelector":
        from .nearest_neighbor import NearestNeighborSelector

        return NearestNeighborSelector
    if name == "LearningToRankSelector":
        from .ranker import LearningToRankSelector

        return LearningToRankSelector
    if name == "PerformancePredictor":
        from .predictor import PerformancePredictor

        return PerformancePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
