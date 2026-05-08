from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class TaskEmbedder(ABC):
    """Abstract base for all task embedding strategies."""

    @abstractmethod
    def embed(self, dataset: pd.DataFrame, labels: pd.Series | None = None) -> np.ndarray: ...

    @abstractmethod
    def embed_batch(self, datasets: list[tuple[pd.DataFrame, pd.Series | None]]) -> np.ndarray: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...
