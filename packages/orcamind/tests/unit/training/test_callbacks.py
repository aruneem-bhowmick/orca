"""Unit tests for PyTorch Lightning training callbacks."""

from __future__ import annotations

import pytest

pytest.importorskip("pytorch_lightning")

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orcamind.core.base import Task
from orcamind.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    MetaValidationCallback,
)


def _make_task() -> Task:
    torch.manual_seed(99)
    return Task(
        support_x=torch.randn(5, 4),
        support_y=torch.randint(0, 2, (5,)),
        query_x=torch.randn(5, 4),
        query_y=torch.randint(0, 2, (5,)),
    )


def _mock_trainer(epoch: int = 0, metrics: dict | None = None) -> MagicMock:
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.should_stop = False
    trainer.callback_metrics = metrics or {}
    return trainer


def _mock_pl_module(meta_learner: MagicMock | None = None) -> MagicMock:
    pl_module = MagicMock()
    pl_module.meta_learner = meta_learner or MagicMock()
    pl_module.log = MagicMock()
    pl_module.state_dict.return_value = {"weight": torch.zeros(2, 2)}
    return pl_module


class TestMetaValidationCallback:
    def test_raises_for_non_positive_val_frequency(self):
        """ValueError when val_frequency is zero or negative."""
        with pytest.raises(ValueError, match="val_frequency must be >= 1"):
            MetaValidationCallback(val_tasks=[_make_task()], val_frequency=0)
        with pytest.raises(ValueError, match="val_frequency must be >= 1"):
            MetaValidationCallback(val_tasks=[_make_task()], val_frequency=-3)

    def test_logs_metrics_without_error(self):
        """on_validation_epoch_end runs without raising and calls pl_module.log."""
        learner = MagicMock()
        learner.adapt.return_value = MagicMock()
        learner.evaluate_task.return_value = {"loss": 0.3, "accuracy": 0.8}
        pl_module = _mock_pl_module(meta_learner=learner)
        trainer = _mock_trainer(epoch=0)

        cb = MetaValidationCallback(val_tasks=[_make_task()], val_frequency=1)
        cb.on_validation_epoch_end(trainer, pl_module)

        pl_module.log.assert_called_once()
        logged_key = pl_module.log.call_args[0][0]
        assert logged_key == "val_k_shot_accuracy"

    def test_skips_when_epoch_not_multiple_of_frequency(self):
        """Does not run k_shot_accuracy when current_epoch % val_frequency != 0."""
        learner = MagicMock()
        pl_module = _mock_pl_module(meta_learner=learner)
        trainer = _mock_trainer(epoch=1)

        cb = MetaValidationCallback(val_tasks=[_make_task()], val_frequency=2)
        cb.on_validation_epoch_end(trainer, pl_module)

        learner.adapt.assert_not_called()
        pl_module.log.assert_not_called()

    def test_runs_when_epoch_is_multiple_of_frequency(self):
        """Runs k_shot_accuracy when current_epoch % val_frequency == 0."""
        learner = MagicMock()
        learner.adapt.return_value = MagicMock()
        learner.evaluate_task.return_value = {"loss": 0.2, "accuracy": 0.9}
        pl_module = _mock_pl_module(meta_learner=learner)
        trainer = _mock_trainer(epoch=2)

        cb = MetaValidationCallback(val_tasks=[_make_task()], val_frequency=2)
        cb.on_validation_epoch_end(trainer, pl_module)

        assert learner.adapt.call_count == 1
        pl_module.log.assert_called_once()

    def test_logged_accuracy_matches_evaluate_task_result(self):
        """Logged value reflects the accuracy returned by evaluate_task."""
        learner = MagicMock()
        learner.adapt.return_value = MagicMock()
        learner.evaluate_task.return_value = {"loss": 0.1, "accuracy": 0.95}
        pl_module = _mock_pl_module(meta_learner=learner)

        cb = MetaValidationCallback(val_tasks=[_make_task()], k_shot=3, n_query=5)
        cb.on_validation_epoch_end(_mock_trainer(epoch=0), pl_module)

        logged_val = pl_module.log.call_args[0][1]
        assert logged_val == pytest.approx(0.95, abs=1e-5)


class TestEarlyStoppingCallback:
    def test_sets_should_stop_after_patience_exceeded(self):
        """trainer.should_stop becomes True after patience epochs without improvement."""
        cb = EarlyStoppingCallback(patience=2, metric="val_loss")
        pl_module = _mock_pl_module()

        # First call: establishes best
        cb.on_validation_epoch_end(
            _mock_trainer(metrics={"val_loss": 0.5}), pl_module
        )
        # Second call: no improvement (wait=1)
        trainer = _mock_trainer(metrics={"val_loss": 0.6})
        cb.on_validation_epoch_end(trainer, pl_module)
        assert not trainer.should_stop

        # Third call: no improvement (wait=2 >= patience=2)
        trainer2 = _mock_trainer(metrics={"val_loss": 0.7})
        cb.on_validation_epoch_end(trainer2, pl_module)
        assert trainer2.should_stop

    def test_does_not_stop_when_metric_keeps_improving(self):
        """trainer.should_stop stays False while metric improves each epoch."""
        cb = EarlyStoppingCallback(patience=2, metric="val_loss")
        pl_module = _mock_pl_module()

        for loss in [1.0, 0.8, 0.6, 0.4]:
            trainer = _mock_trainer(metrics={"val_loss": loss})
            cb.on_validation_epoch_end(trainer, pl_module)
            assert not trainer.should_stop

    def test_resets_wait_counter_on_improvement(self):
        """A single improvement resets the patience counter."""
        cb = EarlyStoppingCallback(patience=2, metric="val_loss")
        pl_module = _mock_pl_module()

        cb.on_validation_epoch_end(
            _mock_trainer(metrics={"val_loss": 0.5}), pl_module
        )
        # One epoch without improvement
        cb.on_validation_epoch_end(
            _mock_trainer(metrics={"val_loss": 0.6}), pl_module
        )
        assert cb._wait == 1
        # Improvement resets counter
        cb.on_validation_epoch_end(
            _mock_trainer(metrics={"val_loss": 0.3}), pl_module
        )
        assert cb._wait == 0

    def test_ignores_missing_metric_key(self):
        """Does not raise or modify trainer when the monitored metric is absent."""
        cb = EarlyStoppingCallback(patience=1, metric="val_loss")
        trainer = _mock_trainer(metrics={})
        cb.on_validation_epoch_end(trainer, _mock_pl_module())
        assert not trainer.should_stop


class TestCheckpointCallback:
    def test_writes_pt_file_to_save_dir(self, tmp_path):
        """A .pt checkpoint file is written to save_dir on first improvement."""
        cb = CheckpointCallback(save_dir=tmp_path, monitor="val_loss")
        trainer = _mock_trainer(epoch=0, metrics={"val_loss": 0.4})
        pl_module = _mock_pl_module()

        with patch("orcamind.training.callbacks.torch.save") as mock_save:
            cb.on_validation_epoch_end(trainer, pl_module)
            mock_save.assert_called_once()
            saved_path = mock_save.call_args[0][1]
            assert saved_path.endswith("checkpoint_epoch0000.pt")

    def test_does_not_save_when_metric_does_not_improve(self, tmp_path):
        """No checkpoint is written when val_loss is worse than best."""
        cb = CheckpointCallback(save_dir=tmp_path, monitor="val_loss")
        pl_module = _mock_pl_module()

        with patch("orcamind.training.callbacks.torch.save") as mock_save:
            # First call establishes best=0.4
            cb.on_validation_epoch_end(
                _mock_trainer(epoch=0, metrics={"val_loss": 0.4}), pl_module
            )
            # Second call with worse loss should not save
            cb.on_validation_epoch_end(
                _mock_trainer(epoch=1, metrics={"val_loss": 0.8}), pl_module
            )
            assert mock_save.call_count == 1

    def test_calls_log_artifact_when_tracker_provided(self, tmp_path):
        """mlflow_tracker.log_artifact is called with the checkpoint path."""
        tracker = MagicMock()
        cb = CheckpointCallback(
            save_dir=tmp_path, monitor="val_loss", mlflow_tracker=tracker
        )
        trainer = _mock_trainer(epoch=0, metrics={"val_loss": 0.3})

        with patch("orcamind.training.callbacks.torch.save"):
            cb.on_validation_epoch_end(trainer, _mock_pl_module())

        tracker.log_artifact.assert_called_once()
        logged_path = tracker.log_artifact.call_args[0][0]
        assert "checkpoint_epoch0000.pt" in logged_path

    def test_skips_log_artifact_when_tracker_is_none(self, tmp_path):
        """No artifact logging when mlflow_tracker is not provided."""
        cb = CheckpointCallback(save_dir=tmp_path, monitor="val_loss")
        trainer = _mock_trainer(epoch=0, metrics={"val_loss": 0.3})

        with patch("orcamind.training.callbacks.torch.save"):
            cb.on_validation_epoch_end(trainer, _mock_pl_module())
        # No exception means no tracker was called (tracker is None)

    def test_ignores_missing_monitor_metric(self, tmp_path):
        """Does not raise when the monitored metric is absent from callback_metrics."""
        cb = CheckpointCallback(save_dir=tmp_path, monitor="val_loss")
        trainer = _mock_trainer(epoch=0, metrics={})
        cb.on_validation_epoch_end(trainer, _mock_pl_module())
