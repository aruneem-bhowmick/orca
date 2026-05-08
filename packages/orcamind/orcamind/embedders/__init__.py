"""Task embedders: statistical feature extraction and neural embedding."""

from .neural import NeuralEmbedder
from .similarity import FaissIndex
from .statistical import StatisticalEmbedder

__all__ = ["FaissIndex", "NeuralEmbedder", "StatisticalEmbedder"]
