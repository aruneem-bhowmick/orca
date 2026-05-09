"""FAISS-backed similarity index for task embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    """FAISS-backed nearest-neighbour index over task embeddings."""

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        if metric not in {"cosine", "l2"}:
            raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")
        self._dim = dim
        self._metric = metric
        self._index = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
        self._task_ids: list[str] = []

    def add(self, task_id: str, embedding: np.ndarray) -> None:
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, self._dim).copy()
        if self._metric == "cosine":
            faiss.normalize_L2(vec)
        self._index.add(vec)
        self._task_ids.append(task_id)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if len(self) == 0:
            return []
        k_eff = min(k, len(self))
        q = np.asarray(query, dtype=np.float32).reshape(1, self._dim).copy()
        if self._metric == "cosine":
            faiss.normalize_L2(q)
        distances, indices = self._index.search(q, k_eff)
        results = [
            (self._task_ids[i], float(d))
            for i, d in zip(indices[0], distances[0], strict=False)
            if i != -1
        ]
        return results

    def save(self, path: str) -> None:
        p = Path(path)
        faiss.write_index(self._index, str(p.with_suffix(".index")))
        meta = {"task_ids": self._task_ids, "dim": self._dim, "metric": self._metric}
        with p.with_suffix(".meta").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        p = Path(path)
        idx = faiss.read_index(str(p.with_suffix(".index")))
        with p.with_suffix(".meta").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["dim"] != idx.d:
            raise ValueError(
                f"index dimension {idx.d} does not match metadata dim {meta['dim']}"
            )
        if len(meta["task_ids"]) != idx.ntotal:
            raise ValueError(
                f"task_ids length {len(meta['task_ids'])} does not match index size {idx.ntotal}"
            )
        expected_type = faiss.IndexFlatIP if meta["metric"] == "cosine" else faiss.IndexFlatL2
        if not isinstance(idx, expected_type):
            raise ValueError(
                f"index type {type(idx).__name__!r} does not match metadata metric {meta['metric']!r}"
            )
        self._index = idx
        self._task_ids = meta["task_ids"]
        self._dim = meta["dim"]
        self._metric = meta["metric"]

    def __len__(self) -> int:
        return self._index.ntotal
