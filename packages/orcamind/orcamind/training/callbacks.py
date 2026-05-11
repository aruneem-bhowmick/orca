"""PyTorch Lightning callbacks for meta-training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from orcamind.core.base import Task
from orcamind.training.metrics import k_shot_accuracy


class MetaValidationCallback(pl.Callback):
    """Run k-shot accuracy evaluation on a held-out task set during validation.

    Logs ``val_k_shot_accuracy`` via ``pl_module.log`` at every
    ``val_frequency`` epochs.
    """

    def __init__(
        self,
        val_tasks: list[Task],
        val_frequency: int = 1,
        k_shot: int = 5,
        n_query: int = 15,
    ) -> None:
        self._val_tasks = val_tasks
        self._val_frequency = val_frequency
        self._k_shot = k_shot
        self._n_query = n_query

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.current_epoch % self._val_frequency != 0:
            return
        acc = k_shot_accuracy(
            pl_module.meta_learner,  # type: ignore[attr-defined]
            self._val_tasks,
            self._k_shot,
            self._n_query,
        )
        pl_module.log("val_k_shot_accuracy", acc, prog_bar=True)


class EarlyStoppingCallback(pl.Callback):
    """Stop training when a monitored metric stops improving.

    Tracks the metric in lower-is-better mode. Use a negated metric name
    for accuracy-based stopping.
    """

    def __init__(self, patience: int, metric: str = "val_loss") -> None:
        self._patience = patience
        self._metric = metric
        self._best: float = float("inf")
        self._wait: int = 0

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        current: Any = trainer.callback_metrics.get(self._metric)
        if current is None:
            return
        value = float(current)
        if value < self._best:
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                trainer.should_stop = True


class CheckpointCallback(pl.Callback):
    """Save a model checkpoint whenever the monitored metric improves.

    Optionally uploads the checkpoint artifact via an ``OrcaTracker`` instance.
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_loss",
        mlflow_tracker: Any | None = None,
    ) -> None:
        self._save_dir = Path(save_dir)
        self._monitor = monitor
        self._mlflow_tracker = mlflow_tracker
        self._best: float = float("inf")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        current: Any = trainer.callback_metrics.get(self._monitor)
        if current is None:
            return
        value = float(current)
        if value < self._best:
            self._best = value
            self._save_checkpoint(trainer, pl_module)

    def _save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        epoch = trainer.current_epoch
        path = self._save_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save(pl_module.state_dict(), str(path))
        if self._mlflow_tracker is not None:
            self._mlflow_tracker.log_artifact(str(path))
