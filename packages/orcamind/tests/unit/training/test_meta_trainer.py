"""Unit tests for MetaTrainer Lightning module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orcamind.core.base import Task
from orcamind.training.meta_trainer import MetaTrainer
from orcamind.training.task_sampler import UniformTaskSampler


def _make_tasks(n: int = 6) -> list[Task]:
    torch.manual_seed(5)
    return [
        Task(
            support_x=torch.randn(3, 4),
            support_y=torch.randint(0, 2, (3,)),
            query_x=torch.randn(3, 4),
            query_y=torch.randint(0, 2, (3,)),
        )
        for _ in range(n)
    ]


def _make_trainer(meta_learner=None, task_pool=None) -> MetaTrainer:
    if meta_learner is None:
        meta_learner = MagicMock()
        meta_learner.meta_update.return_value = {
            "meta_train_loss": 0.5,
            "meta_train_accuracy": 0.7,
        }
        meta_learner.adapt.return_value = MagicMock()
        meta_learner.evaluate_task.return_value = {"loss": 0.4, "accuracy": 0.8}
    if task_pool is None:
        task_pool = _make_tasks()
    trainer = MetaTrainer(
        meta_learner=meta_learner,
        sampler=UniformTaskSampler(),
        task_pool=task_pool,
        batch_size=2,
    )
    trainer.log = MagicMock()
    return trainer


class TestTrainingStep:
    def test_returns_finite_loss_tensor(self):
        """training_step returns a scalar tensor with a finite value."""
        mt = _make_trainer()
        task_batch = _make_tasks(2)
        result = mt.training_step(task_batch, 0)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert torch.isfinite(result)

    def test_logs_meta_train_loss(self):
        """training_step logs meta_train_loss."""
        mt = _make_trainer()
        mt.training_step(_make_tasks(2), 0)
        logged_keys = [call[0][0] for call in mt.log.call_args_list]
        assert "meta_train_loss" in logged_keys

    def test_logs_meta_train_accuracy(self):
        """training_step logs meta_train_accuracy."""
        mt = _make_trainer()
        mt.training_step(_make_tasks(2), 0)
        logged_keys = [call[0][0] for call in mt.log.call_args_list]
        assert "meta_train_accuracy" in logged_keys

    def test_calls_meta_update_with_task_batch(self):
        """training_step passes the batch directly to meta_update."""
        learner = MagicMock()
        learner.meta_update.return_value = {
            "meta_train_loss": 0.3,
            "meta_train_accuracy": 0.6,
        }
        mt = _make_trainer(meta_learner=learner)
        task_batch = _make_tasks(2)
        mt.training_step(task_batch, 0)
        learner.meta_update.assert_called_once_with(task_batch)

    def test_replaces_nan_accuracy_with_zero(self):
        """NaN meta_train_accuracy from regression learners is logged as 0.0."""
        learner = MagicMock()
        learner.meta_update.return_value = {
            "meta_train_loss": 0.4,
            "meta_train_accuracy": float("nan"),
        }
        mt = _make_trainer(meta_learner=learner)
        mt.training_step(_make_tasks(2), 0)
        acc_call = next(
            c for c in mt.log.call_args_list if c[0][0] == "meta_train_accuracy"
        )
        assert acc_call[0][1] == 0.0

    def test_returns_zero_tensor_for_inf_loss(self):
        """Infinite loss is replaced with a zero tensor to keep PL happy."""
        learner = MagicMock()
        learner.meta_update.return_value = {
            "meta_train_loss": float("inf"),
            "meta_train_accuracy": 0.0,
        }
        mt = _make_trainer(meta_learner=learner)
        result = mt.training_step(_make_tasks(2), 0)
        assert torch.isfinite(result)


class TestValidationStep:
    def test_calls_evaluate_task_for_each_task_in_batch(self):
        """validate_step calls evaluate_task once per task."""
        learner = MagicMock()
        learner.adapt.return_value = MagicMock()
        learner.evaluate_task.return_value = {"loss": 0.3, "accuracy": 0.7}
        mt = _make_trainer(meta_learner=learner)
        batch = _make_tasks(3)
        mt.validation_step(batch, 0)
        assert learner.evaluate_task.call_count == 3

    def test_logs_val_loss(self):
        """validation_step logs val_loss."""
        mt = _make_trainer()
        mt.validation_step(_make_tasks(1), 0)
        logged_keys = [c[0][0] for c in mt.log.call_args_list]
        assert "val_loss" in logged_keys

    def test_logs_val_accuracy(self):
        """validation_step logs val_accuracy."""
        mt = _make_trainer()
        mt.validation_step(_make_tasks(1), 0)
        logged_keys = [c[0][0] for c in mt.log.call_args_list]
        assert "val_accuracy" in logged_keys


class TestConfigureOptimizers:
    def test_returns_outer_opt_from_meta_learner(self):
        """configure_optimizers returns the _outer_opt stored in the meta-learner."""
        learner = MagicMock()
        expected_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        learner._outer_opt = expected_opt
        mt = _make_trainer(meta_learner=learner)
        result = mt.configure_optimizers()
        assert result is expected_opt

    def test_falls_back_to_adam_for_nn_module_learner(self):
        """Falls back to Adam when meta-learner is an nn.Module without _outer_opt."""

        class _FakeLearner(nn.Module):
            def forward(self, x):
                return x

        learner = _FakeLearner()
        mt = MetaTrainer(
            meta_learner=learner,
            sampler=UniformTaskSampler(),
            task_pool=_make_tasks(),
            batch_size=2,
        )
        mt.log = MagicMock()
        opt = mt.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)

    def test_returns_dummy_adam_when_no_parameters(self):
        """Returns a usable Adam optimizer even for non-Module meta-learners."""
        learner = MagicMock(spec=[])
        mt = MetaTrainer(
            meta_learner=learner,
            sampler=UniformTaskSampler(),
            task_pool=_make_tasks(),
            batch_size=2,
        )
        mt.log = MagicMock()
        opt = mt.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)


class TestMakeTrainer:
    def test_returns_pl_trainer_instance(self):
        """make_trainer returns a pytorch_lightning.Trainer."""
        import pytorch_lightning as pl

        trainer = MetaTrainer.make_trainer(max_epochs=1, enable_progress_bar=False)
        assert isinstance(trainer, pl.Trainer)

    def test_adds_ddp_strategy_when_multiple_gpus(self):
        """strategy='ddp' is injected when multiple CUDA devices are available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            import pytorch_lightning as pl

            trainer = MetaTrainer.make_trainer(max_epochs=1, enable_progress_bar=False)
            assert isinstance(trainer, pl.Trainer)

    def test_no_ddp_strategy_with_single_device(self):
        """strategy is not overridden to 'ddp' on a single-device machine."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.device_count", return_value=0),
        ):
            import pytorch_lightning as pl

            trainer = MetaTrainer.make_trainer(max_epochs=1, enable_progress_bar=False)
            assert isinstance(trainer, pl.Trainer)
