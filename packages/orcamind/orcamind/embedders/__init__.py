"""Task embedders: statistical feature extraction and neural embedding."""

from __future__ import annotations

from .statistical import StatisticalEmbedder

__all__ = ["FaissIndex", "NeuralEmbedder", "StatisticalEmbedder"]


def __getattr__(name: str) -> object:
    if name == "NeuralEmbedder":
        from .neural import NeuralEmbedder
        return NeuralEmbedder
    if name == "FaissIndex":
        from .similarity import FaissIndex
        return FaissIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
