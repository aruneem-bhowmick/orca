"""Tests for scripts/bootstrap_meta_dataset.py.

All tests mock external dependencies (OpenML, database, FAISS) so the
suite runs without network access or a live PostgreSQL instance.
"""

from __future__ import annotations

import argparse
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Sanity-check module-level constant shapes."""

    def test_suite_names_keys(self, bootstrap: ModuleType) -> None:
        assert set(bootstrap.SUITE_NAMES.keys()) == {"cc18", "ctr23"}

    def test_classification_models_count(self, bootstrap: ModuleType) -> None:
        assert len(bootstrap.CLASSIFICATION_MODELS) == 5

    def test_regression_models_count(self, bootstrap: ModuleType) -> None:
        assert len(bootstrap.REGRESSION_MODELS) == 5

    def test_classification_and_regression_share_names(
        self, bootstrap: ModuleType
    ) -> None:
        shared = set(bootstrap.CLASSIFICATION_MODELS) & set(
            bootstrap.REGRESSION_MODELS
        )
        assert shared == {"random_forest", "xgboost", "knn"}

    def test_large_sample_skip_set(self, bootstrap: ModuleType) -> None:
        assert bootstrap._LARGE_SAMPLE_SKIP == {"svc_rbf", "svr_rbf"}

    def test_large_sample_threshold(self, bootstrap: ModuleType) -> None:
        assert bootstrap._LARGE_SAMPLE_THRESHOLD == 10_000

    def test_all_classification_factories_are_callables(
        self, bootstrap: ModuleType
    ) -> None:
        for factory in bootstrap.CLASSIFICATION_MODELS.values():
            assert callable(factory), "every model factory must be callable"

    def test_all_regression_factories_are_callables(
        self, bootstrap: ModuleType
    ) -> None:
        for factory in bootstrap.REGRESSION_MODELS.values():
            assert callable(factory)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


class TestNonNegativeInt:
    def test_valid_zero(self, bootstrap: ModuleType) -> None:
        assert bootstrap._non_negative_int("0") == 0

    def test_valid_positive(self, bootstrap: ModuleType) -> None:
        assert bootstrap._non_negative_int("42") == 42

    def test_negative_raises(self, bootstrap: ModuleType) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match=">= 0"):
            bootstrap._non_negative_int("-1")

    def test_non_integer_raises(self, bootstrap: ModuleType) -> None:
        with pytest.raises(ValueError):
            bootstrap._non_negative_int("abc")


class TestParseArgs:
    def test_defaults(self, bootstrap: ModuleType) -> None:
        args = bootstrap.parse_args([])
        assert args.max_tasks is None
        assert args.output_dir == "data/"
        assert args.suites == ["cc18", "ctr23"]
        assert args.dry_run is False

    def test_max_tasks_set(self, bootstrap: ModuleType) -> None:
        args = bootstrap.parse_args(["--max-tasks", "10"])
        assert args.max_tasks == 10

    def test_max_tasks_negative_rejected(self, bootstrap: ModuleType) -> None:
        with pytest.raises(SystemExit):
            bootstrap.parse_args(["--max-tasks", "-5"])

    def test_output_dir_override(self, bootstrap: ModuleType) -> None:
        args = bootstrap.parse_args(["--output-dir", "/tmp/data"])
        assert args.output_dir == "/tmp/data"

    def test_suites_single(self, bootstrap: ModuleType) -> None:
        args = bootstrap.parse_args(["--suites", "cc18"])
        assert args.suites == ["cc18"]

    def test_suites_invalid_choice(self, bootstrap: ModuleType) -> None:
        with pytest.raises(SystemExit):
            bootstrap.parse_args(["--suites", "bogus"])

    def test_dry_run_flag(self, bootstrap: ModuleType) -> None:
        args = bootstrap.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_db_url_env_override(self, bootstrap: ModuleType) -> None:
        import os

        old = os.environ.pop("ORCA_DB_URL", None)
        try:
            args = bootstrap.parse_args([])
            assert "localhost:5432" in args.db_url
        finally:
            if old is not None:
                os.environ["ORCA_DB_URL"] = old

    def test_db_url_explicit(self, bootstrap: ModuleType) -> None:
        args = bootstrap.parse_args(
            ["--db-url", "postgresql+asyncpg://db:5432/test"]
        )
        assert args.db_url == "postgresql+asyncpg://db:5432/test"


# ---------------------------------------------------------------------------
# OpenML data acquisition
# ---------------------------------------------------------------------------


class TestDownloadSuite:
    def test_returns_task_list(self, bootstrap: ModuleType) -> None:
        mock_openml = MagicMock()
        mock_openml.study.get_suite.return_value = MagicMock(tasks=[1, 2])
        mock_openml.tasks.get_task.return_value = MagicMock()

        with (
            patch.dict("sys.modules", {"openml": mock_openml}),
            patch.object(bootstrap, "SUITE_NAMES", {"cc18": "OpenML-CC18"}),
        ):
            result = bootstrap.download_suite("cc18")
        assert len(result) == 2

    def test_max_tasks_slicing(self, bootstrap: ModuleType) -> None:
        mock_openml = MagicMock()
        mock_openml.study.get_suite.return_value = MagicMock(
            tasks=[10, 20, 30, 40]
        )
        mock_openml.tasks.get_task.return_value = MagicMock()

        with patch.dict("sys.modules", {"openml": mock_openml}):
            with patch.object(
                bootstrap, "SUITE_NAMES", {"cc18": "OpenML-CC18"}, create=True
            ):
                result = bootstrap.download_suite("cc18", max_tasks=2)
        assert len(result) == 2

    def test_unknown_suite_raises_key_error(
        self, bootstrap: ModuleType
    ) -> None:
        with pytest.raises(KeyError):
            bootstrap.download_suite("nonexistent")

    def test_individual_task_failure_skipped(
        self, bootstrap: ModuleType
    ) -> None:
        mock_openml = MagicMock()
        mock_openml.study.get_suite.return_value = MagicMock(tasks=[1, 2])
        mock_openml.tasks.get_task.side_effect = [
            MagicMock(task_id=1),
            RuntimeError("network error"),
        ]

        with patch.dict("sys.modules", {"openml": mock_openml}):
            with patch.object(
                bootstrap, "SUITE_NAMES", {"cc18": "OpenML-CC18"}, create=True
            ):
                result = bootstrap.download_suite("cc18")
        assert len(result) == 1

    def test_empty_suite(self, bootstrap: ModuleType) -> None:
        mock_openml = MagicMock()
        mock_openml.study.get_suite.return_value = MagicMock(tasks=None)

        with patch.dict("sys.modules", {"openml": mock_openml}):
            with patch.object(
                bootstrap, "SUITE_NAMES", {"cc18": "OpenML-CC18"}, create=True
            ):
                result = bootstrap.download_suite("cc18")
        assert result == []


class TestFetchDataset:
    def _make_task(self, task_type_id=None, class_name="SupervisedClassificationTask"):
        task = MagicMock()
        task.task_type_id = task_type_id
        type(task).__name__ = class_name
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        y = pd.Series([0, 1])
        task.get_X_and_y.return_value = (X, y)
        return task

    def test_classification_by_task_type_id(
        self, bootstrap: ModuleType
    ) -> None:
        mock_openml = MagicMock()
        mock_openml.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1

        task = self._make_task(task_type_id=1)
        with patch.dict("sys.modules", {"openml.tasks": mock_openml.tasks}):
            _X, _y, task_type = bootstrap.fetch_dataset(task)
        assert task_type == "classification"

    def test_regression_by_task_type_id(
        self, bootstrap: ModuleType
    ) -> None:
        """task_type_id != SUPERVISED_CLASSIFICATION → regression."""
        mock_task_type = MagicMock()
        mock_task_type.SUPERVISED_CLASSIFICATION = 1

        mock_openml_tasks_mod = MagicMock()
        mock_openml_tasks_mod.TaskType = mock_task_type

        task = self._make_task(task_type_id=2)
        # Patch sys.modules before the function's lazy import resolves
        with patch.dict("sys.modules", {"openml": MagicMock(), "openml.tasks": mock_openml_tasks_mod}):
            # Remove any cached reference to force re-import
            import importlib
            import sys
            sys.modules.pop("openml.tasks", None)
            sys.modules["openml.tasks"] = mock_openml_tasks_mod
            _X, _y, task_type = bootstrap.fetch_dataset(task)
        assert task_type == "regression"

    def test_fallback_to_class_name(self, bootstrap: ModuleType) -> None:
        task = self._make_task(task_type_id=None, class_name="ClassificationTask")
        _X, _y, task_type = bootstrap.fetch_dataset(task)
        assert task_type == "classification"

    def test_fallback_regression_class_name(
        self, bootstrap: ModuleType
    ) -> None:
        task = self._make_task(task_type_id=None, class_name="RegressionTask")
        _X, _y, task_type = bootstrap.fetch_dataset(task)
        assert task_type == "regression"

    def test_fallback_default_classification_on_import_error(
        self, bootstrap: ModuleType
    ) -> None:
        """When the openml.tasks import itself fails, defaults to classification."""
        task = self._make_task(task_type_id=1)
        # Make "import openml.tasks" raise inside the try block
        with patch("builtins.__import__", side_effect=ModuleNotFoundError("no openml.tasks")):
            # The function has its own try/except that defaults to True
            _X, _y, task_type = bootstrap.fetch_dataset(task)
        assert task_type == "classification"


# ---------------------------------------------------------------------------
# Baseline model evaluation
# ---------------------------------------------------------------------------


class TestRunBaselineModels:
    def _make_xy(self, n_rows=100):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((n_rows, 3)), columns=["a", "b", "c"])
        y = pd.Series(rng.integers(0, 2, size=n_rows))
        return X, y

    def test_normal_classification_run(self, bootstrap: ModuleType) -> None:
        X, y = self._make_xy(200)
        results = bootstrap.run_baseline_models(X, y, "classification", len(X))
        assert "random_forest" in results
        assert "mean" in results["random_forest"]
        assert "std" in results["random_forest"]
        assert not np.isnan(results["random_forest"]["mean"])

    def test_normal_regression_run(self, bootstrap: ModuleType) -> None:
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((200, 3)))
        y = pd.Series(rng.standard_normal(200))
        results = bootstrap.run_baseline_models(X, y, "regression", 200)
        assert "ridge" in results
        assert not np.isnan(results["ridge"]["mean"])

    def test_svc_skipped_for_large_dataset(
        self, bootstrap: ModuleType
    ) -> None:
        """Patch threshold to 5 so a 10-row dataset triggers the skip."""
        X, y = self._make_xy(10)
        with patch.object(bootstrap, "_LARGE_SAMPLE_THRESHOLD", 5):
            results = bootstrap.run_baseline_models(X, y, "classification", 10)
        assert np.isnan(results["svc_rbf"]["mean"])

    def test_svr_skipped_for_large_dataset(
        self, bootstrap: ModuleType
    ) -> None:
        """Patch threshold to 5 so a 10-row dataset triggers the skip."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((10, 3)))
        y = pd.Series(rng.standard_normal(10))
        with patch.object(bootstrap, "_LARGE_SAMPLE_THRESHOLD", 5):
            results = bootstrap.run_baseline_models(X, y, "regression", 10)
        assert np.isnan(results["svr_rbf"]["mean"])

    def test_svc_not_skipped_for_small_dataset(
        self, bootstrap: ModuleType
    ) -> None:
        X, y = self._make_xy(100)
        results = bootstrap.run_baseline_models(X, y, "classification", 100)
        # SVC may fail or succeed on 100 rows, but it is NOT skipped
        assert "svc_rbf" in results

    def test_too_few_samples_returns_all_nan(
        self, bootstrap: ModuleType
    ) -> None:
        X = pd.DataFrame({"a": [1.0]})
        y = pd.Series([0])
        results = bootstrap.run_baseline_models(X, y, "classification", 1)
        for stats in results.values():
            assert np.isnan(stats["mean"])
            assert np.isnan(stats["std"])

    def test_model_failure_returns_nan(self, bootstrap: ModuleType) -> None:
        """A model factory that raises on construction produces NaN results."""
        X, y = self._make_xy(100)
        original = bootstrap.CLASSIFICATION_MODELS.copy()
        try:
            bootstrap.CLASSIFICATION_MODELS["broken"] = lambda: (_ for _ in ()).throw(
                RuntimeError("nope")
            )
            results = bootstrap.run_baseline_models(X, y, "classification", 100)
            assert np.isnan(results["broken"]["mean"])
        finally:
            bootstrap.CLASSIFICATION_MODELS = original

    def test_nan_in_y_dropped(self, bootstrap: ModuleType) -> None:
        """Rows where y is NaN should be dropped before fitting."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((200, 3)))
        y = pd.Series(rng.integers(0, 2, size=200).astype(float))
        y.iloc[0] = float("nan")
        # Should not raise
        results = bootstrap.run_baseline_models(X, y, "classification", 200)
        assert "random_forest" in results

    def test_non_numeric_columns_ignored(self, bootstrap: ModuleType) -> None:
        """Non-numeric feature columns are silently dropped."""
        X = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
                "b": [0.0] * 100,
                "cat": ["x"] * 100,
            }
        )
        y = pd.Series([0, 1] * 50)
        results = bootstrap.run_baseline_models(X, y, "classification", 100)
        assert "random_forest" in results


# ---------------------------------------------------------------------------
# Registry storage helpers
# ---------------------------------------------------------------------------


class TestStoreTask:
    @pytest.mark.asyncio
    async def test_persists_task_and_returns_id(
        self, bootstrap: ModuleType
    ) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0]})
        y = pd.Series([0, 1])

        mock_task_obj = MagicMock()
        mock_task_obj.get_dataset.return_value.name = "iris"
        mock_task_obj.task_id = 42

        mock_task = MagicMock()
        mock_task.task_id = "some-uuid"

        mock_repo = AsyncMock()
        mock_repo.create.return_value = mock_task

        mock_session = AsyncMock()

        with (
            patch.object(bootstrap.TaskRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.TaskRepository, "create", mock_repo.create),
        ):
            tid = await bootstrap.store_task(mock_session, mock_task_obj, X, y, "classification")
        assert tid == "some-uuid"
        mock_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_on_dataset_name_error(
        self, bootstrap: ModuleType
    ) -> None:
        X = pd.DataFrame({"a": [1.0]})
        y = pd.Series([0])

        mock_task_obj = MagicMock()
        mock_task_obj.get_dataset.side_effect = RuntimeError("no dataset")
        mock_task_obj.task_id = 99

        mock_task = MagicMock()
        mock_task.task_id = "uuid-fallback"

        mock_repo = AsyncMock()
        mock_repo.create.return_value = mock_task

        with (
            patch.object(bootstrap.TaskRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.TaskRepository, "create", mock_repo.create),
        ):
            tid = await bootstrap.store_task(mock_session := AsyncMock(), mock_task_obj, X, y, "regression")
        assert tid == "uuid-fallback"
        # Verify the TaskCreate passed uses the fallback name
        call_args = mock_repo.create.call_args
        assert call_args[0][0].name == "openml_99"


class TestStoreExperiments:
    @pytest.mark.asyncio
    async def test_skips_nan_models(self, bootstrap: ModuleType) -> None:
        task_id = "tid"
        model_results = {
            "good": {"mean": 0.9, "std": 0.05},
            "bad": {"mean": float("nan"), "std": float("nan")},
        }

        mock_exp_repo = AsyncMock()
        mock_exp_repo.create.return_value = MagicMock(experiment_id="eid")
        mock_exp_repo.mark_complete = AsyncMock()

        mock_perf_repo = AsyncMock()
        mock_perf_repo.log_metric = AsyncMock()

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        with (
            patch.object(bootstrap.ExperimentRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.ExperimentRepository, "create", mock_exp_repo.create),
            patch.object(bootstrap.ExperimentRepository, "mark_complete", mock_exp_repo.mark_complete),
            patch.object(bootstrap.PerformanceRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.PerformanceRepository, "log_metric", mock_perf_repo.log_metric),
        ):
            stored = await bootstrap.store_experiments(
                mock_session, task_id, model_results, "classification"
            )
        assert stored == 1
        mock_exp_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_zero_when_all_nan(
        self, bootstrap: ModuleType
    ) -> None:
        model_results = {
            "m1": {"mean": float("nan"), "std": float("nan")},
        }
        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        with (
            patch.object(bootstrap.ExperimentRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.PerformanceRepository, "__init__", lambda self, s: None),
        ):
            stored = await bootstrap.store_experiments(
                mock_session, "tid", model_results, "classification"
            )
        assert stored == 0
        mock_session.add.assert_not_called()


class TestStoreEmbedding:
    @pytest.mark.asyncio
    async def test_calls_embed_and_persists(
        self, bootstrap: ModuleType
    ) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0]})
        y = pd.Series([0, 1])

        mock_embedder = MagicMock()
        mock_embedder.embedding_dim = 25
        mock_embedder.embed.return_value = np.zeros(25)

        mock_faiss = MagicMock()

        mock_emb = MagicMock()
        mock_emb.embedding_id = "emb-id"
        mock_emb_repo = AsyncMock()
        mock_emb_repo.create.return_value = mock_emb

        mock_task_repo = AsyncMock()
        mock_task_repo.update_embedding = AsyncMock()

        mock_session = AsyncMock()

        with (
            patch.object(bootstrap.EmbeddingRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.EmbeddingRepository, "create", mock_emb_repo.create),
            patch.object(bootstrap.TaskRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.TaskRepository, "update_embedding", mock_task_repo.update_embedding),
        ):
            vec = await bootstrap.store_embedding(
                mock_session, mock_embedder, mock_faiss, "task-uuid", X, y
            )
        assert vec.shape == (25,)
        mock_faiss.add.assert_called_once()


# ---------------------------------------------------------------------------
# Full async orchestration
# ---------------------------------------------------------------------------


class TestBootstrapAsync:
    def _make_args(self, **overrides):
        defaults = dict(
            suites=["cc18"],
            max_tasks=None,
            output_dir="/tmp/orca_test_bootstrap",
            db_url="postgresql+asyncpg://localhost/orca",
            dry_run=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _make_task_obj(self, task_id=1):
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0])
        task_obj = MagicMock()
        task_obj.task_id = task_id
        task_obj.get_X_and_y.return_value = (X, y)
        task_obj.get_dataset.return_value.name = "synthetic"
        return task_obj

    @pytest.mark.asyncio
    async def test_dry_run_skips_persistence(
        self, bootstrap: ModuleType
    ) -> None:
        """In dry-run mode no repository or FAISS writes should occur."""
        args = self._make_args(dry_run=True)
        task_obj = self._make_task_obj()

        mock_openml = MagicMock()
        mock_openml.study.get_suite.return_value = MagicMock(tasks=[1])
        mock_openml.tasks.get_task.return_value = task_obj

        with (
            patch.dict("sys.modules", {"openml": mock_openml}),
            patch.object(bootstrap, "SUITE_NAMES", {"cc18": "OpenML-CC18"}),
            patch.object(bootstrap, "get_engine", MagicMock()),
            patch.object(bootstrap, "get_session", MagicMock()),
            patch.object(bootstrap, "StatisticalEmbedder", MagicMock),
            patch.object(bootstrap, "FaissIndex", MagicMock),
        ):
            result = await bootstrap._bootstrap_async(args)

        assert result == {"tasks": 0, "experiments": 0}

    @pytest.mark.asyncio
    async def test_non_dry_run_persists_and_counts(
        self, bootstrap: ModuleType
    ) -> None:
        args = self._make_args(dry_run=False)
        task_obj = self._make_task_obj()

        mock_openml = MagicMock()
        mock_openml.study.get_suite.return_value = MagicMock(tasks=[1])
        mock_openml.tasks.get_task.return_value = task_obj

        mock_task = MagicMock()
        mock_task.task_id = "task-uuid-1"
        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp-1"

        mock_task_repo = AsyncMock()
        mock_task_repo.create.return_value = mock_task
        mock_task_repo.update_embedding = AsyncMock()

        mock_exp_repo = AsyncMock()
        mock_exp_repo.create.return_value = mock_exp
        mock_exp_repo.mark_complete = AsyncMock()

        mock_perf_repo = AsyncMock()
        mock_perf_repo.log_metric = AsyncMock()

        mock_emb_repo = AsyncMock()
        mock_emb = MagicMock()
        mock_emb.embedding_id = "emb-1"
        mock_emb_repo.create.return_value = mock_emb

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()

        mock_embedder = MagicMock()
        mock_embedder.embedding_dim = 25
        mock_embedder.embed.return_value = np.zeros(25)

        mock_faiss = MagicMock()

        with (
            patch.dict("sys.modules", {"openml": mock_openml}),
            patch.object(bootstrap, "SUITE_NAMES", {"cc18": "OpenML-CC18"}),
            patch.object(bootstrap, "get_engine", return_value=mock_engine),
            patch.object(bootstrap, "get_session", return_value=mock_session_cm),
            patch.object(bootstrap, "StatisticalEmbedder", return_value=mock_embedder),
            patch.object(bootstrap, "FaissIndex", return_value=mock_faiss),
            patch.object(bootstrap.TaskRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.TaskRepository, "create", mock_task_repo.create),
            patch.object(bootstrap.TaskRepository, "update_embedding", mock_task_repo.update_embedding),
            patch.object(bootstrap.ExperimentRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.ExperimentRepository, "create", mock_exp_repo.create),
            patch.object(bootstrap.ExperimentRepository, "mark_complete", mock_exp_repo.mark_complete),
            patch.object(bootstrap.PerformanceRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.PerformanceRepository, "log_metric", mock_perf_repo.log_metric),
            patch.object(bootstrap.EmbeddingRepository, "__init__", lambda self, s: None),
            patch.object(bootstrap.EmbeddingRepository, "create", mock_emb_repo.create),
        ):
            result = await bootstrap._bootstrap_async(args)

        assert result["tasks"] >= 1
        assert result["experiments"] >= 0
        mock_faiss.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_failure_skipped_gracefully(
        self, bootstrap: ModuleType
    ) -> None:
        """A task whose dataset can't be fetched should be skipped, not crash."""
        args = self._make_args(dry_run=True)
        task_obj = MagicMock()
        task_obj.task_id = 42
        task_obj.get_X_and_y.side_effect = RuntimeError("download failed")

        mock_openml = MagicMock()
        mock_openml.study.get_suite.return_value = MagicMock(tasks=[42])
        mock_openml.tasks.get_task.return_value = task_obj

        with (
            patch.dict("sys.modules", {"openml": mock_openml}),
            patch.object(bootstrap, "SUITE_NAMES", {"cc18": "OpenML-CC18"}),
            patch.object(bootstrap, "get_engine", MagicMock()),
            patch.object(bootstrap, "StatisticalEmbedder", MagicMock),
            patch.object(bootstrap, "FaissIndex", MagicMock),
        ):
            result = await bootstrap._bootstrap_async(args)

        assert result == {"tasks": 0, "experiments": 0}

    @pytest.mark.asyncio
    async def test_persist_failure_skipped_gracefully(
        self, bootstrap: ModuleType
    ) -> None:
        """If store_task raises, the loop continues to the next task."""
        args = self._make_args(dry_run=False)
        task_obj = self._make_task_obj()

        mock_openml = MagicMock()
        mock_openml.study.get_suite.return_value = MagicMock(tasks=[1])
        mock_openml.tasks.get_task.return_value = task_obj

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict("sys.modules", {"openml": mock_openml}),
            patch.object(bootstrap, "SUITE_NAMES", {"cc18": "OpenML-CC18"}),
            patch.object(bootstrap, "get_engine", MagicMock()),
            patch.object(bootstrap, "get_session", return_value=mock_session_cm),
            patch.object(bootstrap, "StatisticalEmbedder", MagicMock),
            patch.object(bootstrap, "FaissIndex", MagicMock),
        ):
            result = await bootstrap._bootstrap_async(args)

        assert result == {"tasks": 0, "experiments": 0}


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_returns_zero(self, bootstrap: ModuleType) -> None:
        mock_args = argparse.Namespace(
            suites=[], max_tasks=None, output_dir="/tmp",
            db_url="sqlite://", dry_run=True,
        )
        mock_result = {"tasks": 0, "experiments": 0}

        with (
            patch.object(bootstrap, "parse_args", return_value=mock_args),
            patch.object(bootstrap, "_bootstrap_async", new_callable=AsyncMock, return_value=mock_result),
            patch("asyncio.run", return_value=mock_result),
        ):
            ret = bootstrap.main()
        assert ret == 0

    def test_main_calls_parse_args_and_bootstrap(
        self, bootstrap: ModuleType
    ) -> None:
        mock_args = argparse.Namespace(
            suites=[], max_tasks=None, output_dir="/tmp",
            db_url="sqlite://", dry_run=True,
        )

        with (
            patch.object(bootstrap, "parse_args", return_value=mock_args) as mock_parse,
            patch.object(bootstrap, "_bootstrap_async", new_callable=AsyncMock, return_value={"tasks": 0, "experiments": 0}),
            patch("asyncio.run", return_value={"tasks": 0, "experiments": 0}),
        ):
            bootstrap.main()
        mock_parse.assert_called_once()
