"""Neural task embedder: MLP projection of statistical meta-features with contrastive training."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .base import TaskEmbedder
from .statistical import StatisticalEmbedder

logger = logging.getLogger(__name__)


class _MLP(nn.Module):
    """Feed-forward network: input_dim → hidden_dims → output_dim with L2-normalised output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.BatchNorm1d(out_d), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """NT-Xent (supervised contrastive) loss over a labelled embedding batch."""

    def forward(self, embeddings: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor:
        n = embeddings.size(0)
        if n < 2:
            return embeddings.sum() * 0.0

        # Raw similarity matrix — keep separate copy to avoid -inf * 0 = nan in numerator
        sim_raw = torch.mm(embeddings, embeddings.T) / temperature  # (N, N)

        # Masked copy for log-sum-exp denominator (self-pairs excluded)
        sim_denom = sim_raw.clone()
        sim_denom.fill_diagonal_(float("-inf"))
        log_denom = torch.logsumexp(sim_denom, dim=1)  # (N,)

        eye = torch.eye(n, device=embeddings.device, dtype=torch.bool)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye  # (N, N)

        n_pos = pos_mask.float().sum(dim=1)  # (N,)
        pos_sum = (sim_raw * pos_mask.float()).sum(dim=1)  # (N,)

        loss_per = -pos_sum / n_pos.clamp(min=1) + log_denom  # (N,)

        valid = n_pos > 0
        if not valid.any():
            return embeddings.sum() * 0.0
        return loss_per[valid].mean()


class NeuralEmbedder(TaskEmbedder):
    """MLP that projects 25-dim statistical meta-features to a learned embedding space via NT-Xent training."""

    def __init__(
        self,
        input_dim: int = 25,
        hidden_dims: list[int] | None = None,
        output_dim: int = 64,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self._stat_embedder = StatisticalEmbedder()
        self._mlp = _MLP(input_dim, hidden_dims, output_dim)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mlp.to(self._device)

    @property
    def embedding_dim(self) -> int:
        return self._output_dim

    def embed(self, dataset: pd.DataFrame, labels: pd.Series | None = None) -> np.ndarray:
        stat = self._stat_embedder.embed(dataset, labels)
        x = torch.tensor(stat, dtype=torch.float32).unsqueeze(0).to(self._device)
        self._mlp.eval()
        with torch.no_grad():
            out = self._mlp(x)
        return out.squeeze(0).cpu().numpy()

    def embed_batch(self, datasets: list[tuple[pd.DataFrame, pd.Series | None]]) -> np.ndarray:
        stat = self._stat_embedder.embed_batch(datasets)
        if stat.shape[0] == 0:
            return np.empty((0, self._output_dim), dtype=np.float32)
        x = torch.tensor(stat, dtype=torch.float32).to(self._device)
        self._mlp.eval()
        with torch.no_grad():
            out = self._mlp(x)
        return out.cpu().numpy()

    def fit(
        self,
        dataset_list: list[tuple[pd.DataFrame, pd.Series | None]],
        domain_labels: list[int],
        epochs: int = 20,
        lr: float = 1e-3,
    ) -> None:
        if len(dataset_list) != len(domain_labels):
            raise ValueError(
                f"dataset_list length ({len(dataset_list)}) must match domain_labels ({len(domain_labels)})"
            )
        n = len(dataset_list)
        if n < 2:
            raise ValueError("fit() requires at least 2 datasets")

        stat = self._stat_embedder.embed_batch(dataset_list)
        x_tensor = torch.tensor(stat, dtype=torch.float32)
        y_tensor = torch.tensor(domain_labels, dtype=torch.long)

        batch_size = min(32, n)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        loss_fn = ContrastiveLoss()
        optimizer = torch.optim.Adam(self._mlp.parameters(), lr=lr)
        self._mlp.train()
        self._mlp.to(self._device)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                embeddings = self._mlp(xb)
                loss = loss_fn(embeddings, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            mean_loss = epoch_loss / max(n_batches, 1)
            logger.info("epoch %d/%d  loss=%.4f", epoch, epochs, mean_loss)

    def save(self, path: str) -> None:
        checkpoint = {
            "config": {
                "input_dim": self._input_dim,
                "hidden_dims": self._hidden_dims,
                "output_dim": self._output_dim,
            },
            "state_dict": self._mlp.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> NeuralEmbedder:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        cfg = checkpoint["config"]
        obj = cls(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            output_dim=cfg["output_dim"],
        )
        obj._mlp.load_state_dict(checkpoint["state_dict"])
        obj._mlp.eval()
        return obj
