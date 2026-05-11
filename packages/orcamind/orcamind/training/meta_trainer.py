"""MetaTrainer: PyTorch Lightning module for meta-learning training loops."""

from __future__ import annotations

import inspect
import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from orcamind.core.base import MetaLearner, Task


class _TaskBatchDataset(IterableDataset):
    """Iterable dataset that yields pre-sampled task batches."""

    def __init__(
        self,
        sampler: Any,
        task_pool: list[Task],
        batch_size: int,
        num_batches: int,
        epoch: int = 0,
    ) -> None:
        self._sampler = sampler
        self._task_pool = task_pool
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._epoch = epoch

    def __iter__(self):  # type: ignore[override]
        sig = inspect.signature(self._sampler.sample)
        has_epoch = "epoch" in sig.parameters
        for _ in range(self._num_batches):
            if has_epoch:
                yield self._sampler.sample(
                    self._task_pool, self._batch_size, epoch=self._epoch
                )
            else:
                yield self._sampler.sample(self._task_pool, self._batch_size)


class MetaTrainer(pl.LightningModule):
    """Lightning module that wraps a MetaLearner for meta-training.

    Uses manual optimization (``automatic_optimization = False``) because
    all meta-learner implementations manage their own optimizer steps internally.
    """

    automatic_optimization = False

    def __init__(
        self,
        meta_learner: MetaLearner,
        sampler: Any,
        task_pool: list[Task],
        batch_size: int = 4,
        num_batches_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        self.meta_learner = meta_learner
        self._sampler = sampler
        self._task_pool = task_pool
        self._batch_size = batch_size
        self._num_batches = num_batches_per_epoch or max(
            1, len(task_pool) // batch_size
        )

    def training_step(self, batch: list[Task], batch_idx: int) -> Tensor:
        metrics = self.meta_learner.meta_update(batch)
        loss_val = metrics.get("meta_train_loss", 0.0)
        acc_val = metrics.get("meta_train_accuracy", 0.0)
        if math.isnan(acc_val):
            acc_val = 0.0

        self.log("meta_train_loss", loss_val, on_step=True, prog_bar=True)
        self.log("meta_train_accuracy", acc_val, on_step=True, prog_bar=False)

        loss_tensor = torch.tensor(loss_val, dtype=torch.float32)
        if not torch.isfinite(loss_tensor):
            loss_tensor = torch.tensor(0.0, dtype=torch.float32)
        return loss_tensor

    def validation_step(self, batch: list[Task], batch_idx: int) -> None:
        with torch.no_grad():
            for task in batch:
                adapted = self.meta_learner.adapt(task.support_x, task.support_y)
                result = self.meta_learner.evaluate_task(
                    adapted, task.query_x, task.query_y
                )
                val_loss = result.get("loss", 0.0)
                val_acc = result.get("accuracy", 0.0)
                if math.isnan(val_acc):
                    val_acc = 0.0
                self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
                self.log("val_accuracy", val_acc, on_epoch=True, prog_bar=False)

    def configure_optimizers(self) -> Any:
        opt = getattr(self.meta_learner, "_outer_opt", None)
        if opt is not None:
            return opt
        if isinstance(self.meta_learner, nn.Module):
            return torch.optim.Adam(self.meta_learner.parameters(), lr=1e-3)
        dummy = nn.Parameter(torch.zeros(1))
        return torch.optim.Adam([dummy], lr=1e-3)

    def train_dataloader(self) -> DataLoader:
        ds = _TaskBatchDataset(
            self._sampler,
            self._task_pool,
            self._batch_size,
            self._num_batches,
            epoch=self.current_epoch,
        )
        return DataLoader(
            ds, batch_size=None, collate_fn=lambda x: x, num_workers=0
        )

    def val_dataloader(self) -> DataLoader:
        n_val_batches = max(1, len(self._task_pool) // 4)
        ds = _TaskBatchDataset(
            self._sampler,
            self._task_pool,
            1,
            n_val_batches,
            epoch=self.current_epoch,
        )
        return DataLoader(
            ds, batch_size=None, collate_fn=lambda x: x, num_workers=0
        )

    @classmethod
    def make_trainer(cls, **kwargs: Any) -> pl.Trainer:
        """Return a ``pl.Trainer`` with DDP strategy when multiple GPUs are detected."""
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            kwargs.setdefault("strategy", "ddp")
        return pl.Trainer(**kwargs)
