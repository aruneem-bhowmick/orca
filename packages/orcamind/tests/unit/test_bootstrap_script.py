"""Tests for scripts/bootstrap_meta_dataset.py."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest


# ── TestParseArgsDefaults ─────────────────────────────────────────────────────


class TestParseArgsDefaults:
    def test_max_tasks_default_is_none(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert args.max_tasks is None

    def test_output_dir_default(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert args.output_dir == "data/"

    def test_db_url_default_contains_asyncpg(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert "asyncpg" in args.db_url

    def test_db_url_default_contains_orca_registry(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert "orca_registry" in args.db_url

    def test_db_url_default_contains_localhost(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert "localhost" in args.db_url

    def test_dry_run_default_is_false(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert args.dry_run is False

    def test_suites_default_contains_cc18(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert "cc18" in args.suites

    def test_suites_default_contains_ctr23(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert "ctr23" in args.suites


# ── TestParseArgsCustom ───────────────────────────────────────────────────────


class TestParseArgsCustom:
    def test_max_tasks_custom_value(self, bmd) -> None:
        with patch("sys.argv", ["prog", "--max-tasks", "10"]):
            args = bmd.parse_args()
        assert args.max_tasks == 10

    def test_output_dir_custom(self, bmd) -> None:
        with patch("sys.argv", ["prog", "--output-dir", "/tmp/out"]):
            args = bmd.parse_args()
        assert args.output_dir == "/tmp/out"

    def test_db_url_custom(self, bmd) -> None:
        url = "postgresql+asyncpg://user:pass@host:5432/db"
        with patch("sys.argv", ["prog", "--db-url", url]):
            args = bmd.parse_args()
        assert args.db_url == url

    def test_dry_run_flag_set(self, bmd) -> None:
        with patch("sys.argv", ["prog", "--dry-run"]):
            args = bmd.parse_args()
        assert args.dry_run is True

    def test_suites_custom_single(self, bmd) -> None:
        with patch("sys.argv", ["prog", "--suites", "cc18"]):
            args = bmd.parse_args()
        assert args.suites == ["cc18"]

    def test_suites_custom_multiple(self, bmd) -> None:
        with patch("sys.argv", ["prog", "--suites", "cc18", "ctr23"]):
            args = bmd.parse_args()
        assert args.suites == ["cc18", "ctr23"]


# ── TestParseArgsOldArgsRemoved ───────────────────────────────────────────────


class TestParseArgsOldArgsRemoved:
    def test_suite_id_not_an_attribute(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert not hasattr(args, "suite_id")

    def test_data_dir_not_an_attribute(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert not hasattr(args, "data_dir")

    def test_registry_url_not_an_attribute(self, bmd) -> None:
        with patch("sys.argv", ["prog"]):
            args = bmd.parse_args()
        assert not hasattr(args, "registry_url")


# ── TestModuleStructure ───────────────────────────────────────────────────────


class TestModuleStructure:
    def test_suite_names_constant_exists(self, bmd) -> None:
        assert hasattr(bmd, "SUITE_NAMES")

    def test_suite_names_has_cc18(self, bmd) -> None:
        assert "cc18" in bmd.SUITE_NAMES

    def test_suite_names_has_ctr23(self, bmd) -> None:
        assert "ctr23" in bmd.SUITE_NAMES

    def test_classification_models_constant_exists(self, bmd) -> None:
        assert hasattr(bmd, "CLASSIFICATION_MODELS")

    def test_classification_models_has_five_entries(self, bmd) -> None:
        assert len(bmd.CLASSIFICATION_MODELS) == 5

    def test_regression_models_constant_exists(self, bmd) -> None:
        assert hasattr(bmd, "REGRESSION_MODELS")

    def test_regression_models_has_five_entries(self, bmd) -> None:
        assert len(bmd.REGRESSION_MODELS) == 5

    def test_download_suite_callable(self, bmd) -> None:
        assert callable(bmd.download_suite)

    def test_fetch_dataset_callable(self, bmd) -> None:
        assert callable(bmd.fetch_dataset)

    def test_run_baseline_models_callable(self, bmd) -> None:
        assert callable(bmd.run_baseline_models)

    def test_store_task_callable(self, bmd) -> None:
        assert callable(bmd.store_task)

    def test_store_experiments_callable(self, bmd) -> None:
        assert callable(bmd.store_experiments)

    def test_store_embedding_callable(self, bmd) -> None:
        assert callable(bmd.store_embedding)

    def test_bootstrap_async_callable(self, bmd) -> None:
        assert callable(bmd._bootstrap_async)

    def test_main_callable(self, bmd) -> None:
        assert callable(bmd.main)

    def test_module_has_docstring(self, bmd) -> None:
        assert bmd.__doc__ is not None and bmd.__doc__.strip()

    def test_module_docstring_mentions_openml(self, bmd) -> None:
        assert "openml" in bmd.__doc__.lower()

    def test_script_file_exists(self, repo_root: Path) -> None:
        assert (repo_root / "scripts" / "bootstrap_meta_dataset.py").is_file()


# ── TestDownloadSuite ─────────────────────────────────────────────────────────


class TestDownloadSuite:
    def _make_suite(self, n: int = 5) -> MagicMock:
        suite = MagicMock()
        suite.tasks = list(range(n))
        return suite

    def _make_task(self, tid: int) -> MagicMock:
        t = MagicMock()
        t.task_id = tid
        return t

    def test_returns_all_tasks_by_default(self, bmd) -> None:
        suite = self._make_suite(3)
        task_objs = [self._make_task(i) for i in range(3)]
        with patch.dict("sys.modules", {"openml": MagicMock(), "openml.study": MagicMock(), "openml.tasks": MagicMock()}):
            import sys
            sys.modules["openml"].study.get_suite.return_value = suite
            sys.modules["openml"].tasks.get_task.side_effect = task_objs
            result = bmd.download_suite("cc18")
        assert len(result) == 3

    def test_max_tasks_limits_output(self, bmd) -> None:
        suite = self._make_suite(10)
        task_objs = [self._make_task(i) for i in range(10)]
        with patch.dict("sys.modules", {"openml": MagicMock(), "openml.study": MagicMock(), "openml.tasks": MagicMock()}):
            import sys
            sys.modules["openml"].study.get_suite.return_value = suite
            sys.modules["openml"].tasks.get_task.side_effect = task_objs
            result = bmd.download_suite("cc18", max_tasks=3)
        assert len(result) == 3

    def test_failed_task_download_is_skipped(self, bmd) -> None:
        suite = self._make_suite(3)
        good = self._make_task(0)

        def _get_task(tid):
            if tid == 1:
                raise RuntimeError("network error")
            return good

        with patch.dict("sys.modules", {"openml": MagicMock(), "openml.study": MagicMock(), "openml.tasks": MagicMock()}):
            import sys
            sys.modules["openml"].study.get_suite.return_value = suite
            sys.modules["openml"].tasks.get_task.side_effect = _get_task
            result = bmd.download_suite("cc18")
        assert len(result) == 2

    def test_all_tasks_fail_returns_empty_list(self, bmd) -> None:
        suite = self._make_suite(3)
        with patch.dict("sys.modules", {"openml": MagicMock(), "openml.study": MagicMock(), "openml.tasks": MagicMock()}):
            import sys
            sys.modules["openml"].study.get_suite.return_value = suite
            sys.modules["openml"].tasks.get_task.side_effect = RuntimeError("fail")
            result = bmd.download_suite("cc18")
        assert result == []

    def test_unknown_suite_key_raises(self, bmd) -> None:
        with pytest.raises(KeyError):
            bmd.download_suite("unknown_suite")


# ── TestFetchDataset ──────────────────────────────────────────────────────────


class TestFetchDataset:
    def _make_openml_task(self, task_type_id=1) -> MagicMock:
        """Build a minimal mock of an OpenML task."""
        import sys
        openml_mock = MagicMock()
        openml_mock.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1
        patch.dict("sys.modules", {"openml": openml_mock, "openml.tasks": openml_mock.tasks}).start()

        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0], name="target")
        task = MagicMock()
        task.get_X_and_y.return_value = (X, y)
        task.task_type_id = task_type_id
        return task

    def test_returns_three_tuple(self, bmd) -> None:
        task = self._make_openml_task()
        result = bmd.fetch_dataset(task)
        assert len(result) == 3

    def test_x_is_dataframe(self, bmd) -> None:
        task = self._make_openml_task()
        X, _, _ = bmd.fetch_dataset(task)
        assert isinstance(X, pd.DataFrame)

    def test_y_is_series(self, bmd) -> None:
        task = self._make_openml_task()
        _, y, _ = bmd.fetch_dataset(task)
        assert isinstance(y, pd.Series)

    def test_classification_task_type_detected(self, bmd) -> None:
        task = self._make_openml_task(task_type_id=1)
        _, _, task_type = bmd.fetch_dataset(task)
        assert task_type == "classification"

    def test_regression_task_type_detected(self, bmd) -> None:
        task = self._make_openml_task(task_type_id=2)
        _, _, task_type = bmd.fetch_dataset(task)
        assert task_type == "regression"

    def test_exception_propagates(self, bmd) -> None:
        task = MagicMock()
        task.get_X_and_y.side_effect = RuntimeError("API error")
        task.task_type_id = 1
        with pytest.raises(RuntimeError, match="API error"):
            bmd.fetch_dataset(task)


# ── TestRunBaselineModels ─────────────────────────────────────────────────────


class TestRunBaselineModels:
    def test_returns_dict(self, bmd, sample_classification_dataset) -> None:
        X, y = sample_classification_dataset
        result = bmd.run_baseline_models(X, y, "classification", len(X))
        assert isinstance(result, dict)

    def test_each_value_has_mean_and_std(self, bmd, sample_classification_dataset) -> None:
        X, y = sample_classification_dataset
        result = bmd.run_baseline_models(X, y, "classification", len(X))
        for stats in result.values():
            assert "mean" in stats and "std" in stats

    def test_five_classification_models(self, bmd, sample_classification_dataset) -> None:
        X, y = sample_classification_dataset
        result = bmd.run_baseline_models(X, y, "classification", len(X))
        assert len(result) == 5

    def test_five_regression_models(self, bmd, sample_regression_dataset) -> None:
        X, y = sample_regression_dataset
        result = bmd.run_baseline_models(X, y, "regression", len(X))
        assert len(result) == 5

    def test_svc_skipped_for_large_dataset(self, bmd, sample_classification_dataset) -> None:
        X, y = sample_classification_dataset
        result = bmd.run_baseline_models(X, y, "classification", n_samples=10_001)
        assert math.isnan(result["svc_rbf"]["mean"])

    def test_svr_skipped_for_large_dataset(self, bmd, sample_regression_dataset) -> None:
        X, y = sample_regression_dataset
        result = bmd.run_baseline_models(X, y, "regression", n_samples=10_001)
        assert math.isnan(result["svr_rbf"]["mean"])

    def test_svc_present_for_small_dataset(self, bmd, sample_classification_dataset) -> None:
        X, y = sample_classification_dataset
        result = bmd.run_baseline_models(X, y, "classification", n_samples=100)
        assert not math.isnan(result["svc_rbf"]["mean"])

    def test_model_failure_returns_nan_not_exception(self, bmd, sample_classification_dataset, monkeypatch) -> None:
        X, y = sample_classification_dataset
        call_count = 0

        original_cvs = bmd.cross_val_score

        def patched_cvs(estimator, *a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("boom")
            return original_cvs(estimator, *a, **kw)

        monkeypatch.setattr(bmd, "cross_val_score", patched_cvs)
        result = bmd.run_baseline_models(X, y, "classification", len(X))
        first_key = list(result.keys())[0]
        assert math.isnan(result[first_key]["mean"])
        non_nan = [k for k, v in result.items() if not math.isnan(v["mean"])]
        assert len(non_nan) > 0

    def test_nan_labels_handled_gracefully(self, bmd, sample_classification_dataset) -> None:
        X, y = sample_classification_dataset
        y_with_nan = y.copy().astype(float)
        y_with_nan.iloc[:5] = float("nan")
        result = bmd.run_baseline_models(X, y_with_nan, "classification", len(X))
        assert all("mean" in v for v in result.values())

    def test_mean_and_std_finite_for_valid_data(self, bmd, sample_classification_dataset) -> None:
        X, y = sample_classification_dataset
        result = bmd.run_baseline_models(X, y, "classification", len(X))
        for name, stats in result.items():
            if name != "svc_rbf":
                assert math.isfinite(stats["mean"]), f"{name} mean is not finite"


# ── TestStoreTask ─────────────────────────────────────────────────────────────


class TestStoreTask:
    def _make_openml_task(self, task_id: int = 123) -> MagicMock:
        task = MagicMock()
        task.task_id = task_id
        task.get_dataset.return_value.name = "test_dataset"
        return task

    def _make_task_schema(self, task_id=None):
        from uuid import uuid4
        from types import SimpleNamespace
        from datetime import datetime, timezone
        tid = task_id or uuid4()
        return SimpleNamespace(
            task_id=tid,
            name="test_dataset",
            task_type="classification",
            domain="tabular",
            n_samples=100,
            n_features=4,
            n_classes=3,
            dataset_uri=f"openml://task/123",
            metadata={"openml_task_id": 123},
            embedding_id=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    async def test_returns_uuid(self, bmd, mock_session) -> None:
        import uuid
        task_schema = self._make_task_schema()
        with patch.object(bmd, "TaskRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.create = AsyncMock(return_value=task_schema)
            MockRepo.return_value = mock_repo
            X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
            y = pd.Series([0, 1])
            result = await bmd.store_task(mock_session, self._make_openml_task(), X, y, "classification")
        assert isinstance(result, type(task_schema.task_id))

    async def test_n_samples_from_dataframe_len(self, bmd, mock_session) -> None:
        from unittest.mock import AsyncMock as AM
        task_schema = self._make_task_schema()
        captured = {}
        async def capture_create(task_create):
            captured["tc"] = task_create
            return task_schema
        with patch.object(bmd, "TaskRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.create = capture_create
            MockRepo.return_value = mock_repo
            X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
            y = pd.Series([0, 1, 2])
            await bmd.store_task(mock_session, self._make_openml_task(), X, y, "classification")
        assert captured["tc"].n_samples == 3

    async def test_n_features_from_dataframe_columns(self, bmd, mock_session) -> None:
        task_schema = self._make_task_schema()
        captured = {}
        async def capture_create(task_create):
            captured["tc"] = task_create
            return task_schema
        with patch.object(bmd, "TaskRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.create = capture_create
            MockRepo.return_value = mock_repo
            X = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
            y = pd.Series([0])
            await bmd.store_task(mock_session, self._make_openml_task(), X, y, "classification")
        assert captured["tc"].n_features == 3

    async def test_dataset_uri_contains_task_id(self, bmd, mock_session) -> None:
        task_schema = self._make_task_schema()
        captured = {}
        async def capture_create(task_create):
            captured["tc"] = task_create
            return task_schema
        with patch.object(bmd, "TaskRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.create = capture_create
            MockRepo.return_value = mock_repo
            X = pd.DataFrame({"a": [1.0]})
            y = pd.Series([0])
            await bmd.store_task(mock_session, self._make_openml_task(task_id=999), X, y, "classification")
        assert "999" in captured["tc"].dataset_uri

    async def test_metadata_includes_openml_task_id(self, bmd, mock_session) -> None:
        task_schema = self._make_task_schema()
        captured = {}
        async def capture_create(task_create):
            captured["tc"] = task_create
            return task_schema
        with patch.object(bmd, "TaskRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.create = capture_create
            MockRepo.return_value = mock_repo
            X = pd.DataFrame({"a": [1.0]})
            y = pd.Series([0])
            await bmd.store_task(mock_session, self._make_openml_task(task_id=42), X, y, "classification")
        assert "openml_task_id" in captured["tc"].metadata

    async def test_n_classes_none_for_regression(self, bmd, mock_session) -> None:
        task_schema = self._make_task_schema()
        captured = {}
        async def capture_create(task_create):
            captured["tc"] = task_create
            return task_schema
        with patch.object(bmd, "TaskRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.create = capture_create
            MockRepo.return_value = mock_repo
            X = pd.DataFrame({"a": [1.0, 2.0]})
            y = pd.Series([1.5, 2.5])
            await bmd.store_task(mock_session, self._make_openml_task(), X, y, "regression")
        assert captured["tc"].n_classes is None


# ── TestStoreExperiments ──────────────────────────────────────────────────────


class TestStoreExperiments:
    def _model_results(self, include_nan: bool = False) -> dict:
        results = {
            "logistic_regression": {"mean": 0.8, "std": 0.02},
            "random_forest": {"mean": 0.85, "std": 0.01},
        }
        if include_nan:
            results["bad_model"] = {"mean": float("nan"), "std": float("nan")}
        return results

    def _setup_repos(self, bmd, mock_session):
        """Patch TaskRepository, ExperimentRepository, PerformanceRepository, ModelORM."""
        from uuid import uuid4
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from unittest.mock import AsyncMock as AM

        exp_id = uuid4()
        exp_schema = SimpleNamespace(
            experiment_id=exp_id,
            task_id=uuid4(),
            model_id=uuid4(),
            status="pending",
            mlflow_run_id=None,
            started_at=None,
            completed_at=None,
        )

        exp_repo = MagicMock()
        exp_repo.create = AM(return_value=exp_schema)
        exp_repo.mark_complete = AM(return_value=None)
        exp_repo.update_status = AM(return_value=None)

        perf_repo = MagicMock()
        metric_point = SimpleNamespace(name="cv_mean", value=0.8, step=None, is_final=True)
        perf_repo.log_metric = AM(return_value=metric_point)

        bmd.ExperimentRepository = MagicMock(return_value=exp_repo)
        bmd.PerformanceRepository = MagicMock(return_value=perf_repo)
        bmd.ModelORM = MagicMock()

        return exp_repo, perf_repo

    async def test_returns_count_of_stored_experiments(self, bmd, mock_session) -> None:
        with patch.object(bmd, "ExperimentRepository"), \
             patch.object(bmd, "PerformanceRepository"), \
             patch.object(bmd, "ModelORM"):
            exp_repo, perf_repo = self._setup_repos(bmd, mock_session)
            task_id = __import__("uuid").uuid4()
            count = await bmd.store_experiments(
                mock_session, task_id, self._model_results(), "classification"
            )
        assert count == 2

    async def test_nan_model_result_not_stored(self, bmd, mock_session) -> None:
        with patch.object(bmd, "ExperimentRepository"), \
             patch.object(bmd, "PerformanceRepository"), \
             patch.object(bmd, "ModelORM"):
            exp_repo, perf_repo = self._setup_repos(bmd, mock_session)
            task_id = __import__("uuid").uuid4()
            count = await bmd.store_experiments(
                mock_session, task_id, self._model_results(include_nan=True), "classification"
            )
        assert count == 2

    async def test_mark_complete_called_per_model(self, bmd, mock_session) -> None:
        with patch.object(bmd, "ExperimentRepository"), \
             patch.object(bmd, "PerformanceRepository"), \
             patch.object(bmd, "ModelORM"):
            exp_repo, perf_repo = self._setup_repos(bmd, mock_session)
            task_id = __import__("uuid").uuid4()
            await bmd.store_experiments(
                mock_session, task_id, self._model_results(), "classification"
            )
        assert exp_repo.mark_complete.await_count == 2

    async def test_metrics_logged_mean_and_std(self, bmd, mock_session) -> None:
        with patch.object(bmd, "ExperimentRepository"), \
             patch.object(bmd, "PerformanceRepository"), \
             patch.object(bmd, "ModelORM"):
            exp_repo, perf_repo = self._setup_repos(bmd, mock_session)
            task_id = __import__("uuid").uuid4()
            await bmd.store_experiments(
                mock_session, task_id, self._model_results(), "classification"
            )
        assert perf_repo.log_metric.await_count == 4  # 2 models × 2 metrics

    async def test_zero_experiments_if_all_nan(self, bmd, mock_session) -> None:
        with patch.object(bmd, "ExperimentRepository"), \
             patch.object(bmd, "PerformanceRepository"), \
             patch.object(bmd, "ModelORM"):
            exp_repo, _ = self._setup_repos(bmd, mock_session)
            task_id = __import__("uuid").uuid4()
            all_nan = {"m1": {"mean": float("nan"), "std": float("nan")}}
            count = await bmd.store_experiments(
                mock_session, task_id, all_nan, "classification"
            )
        assert count == 0


# ── TestStoreEmbedding ────────────────────────────────────────────────────────


class TestStoreEmbedding:
    def _make_deps(self):
        from uuid import uuid4
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from unittest.mock import AsyncMock as AM, MagicMock

        emb_id = uuid4()
        emb_schema = SimpleNamespace(
            embedding_id=emb_id,
            task_id=uuid4(),
            embedding_type="statistical",
            embedding_vector=[0.0] * 25,
            dimension=25,
            model_version="v1",
            created_at=datetime.now(timezone.utc),
        )
        embedder = MagicMock()
        embedder.embed.return_value = np.zeros(25)
        faiss_index = MagicMock()
        return embedder, faiss_index, emb_schema

    async def test_returns_ndarray(self, bmd, mock_session) -> None:
        embedder, faiss_index, emb_schema = self._make_deps()
        with patch.object(bmd, "EmbeddingRepository") as MockEmbRepo, \
             patch.object(bmd, "TaskRepository") as MockTaskRepo:
            mock_emb_repo = MagicMock()
            mock_emb_repo.create = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(return_value=emb_schema)
            MockEmbRepo.return_value = mock_emb_repo
            mock_task_repo = MagicMock()
            mock_task_repo.update_embedding = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(return_value=None)
            MockTaskRepo.return_value = mock_task_repo
            X = pd.DataFrame({"a": [1.0]})
            y = pd.Series([0])
            result = await bmd.store_embedding(mock_session, embedder, faiss_index, emb_schema.task_id, X, y)
        assert isinstance(result, np.ndarray)

    async def test_ndarray_has_correct_shape(self, bmd, mock_session) -> None:
        embedder, faiss_index, emb_schema = self._make_deps()
        embedder.embed.return_value = np.zeros(25)
        from unittest.mock import AsyncMock as AM
        with patch.object(bmd, "EmbeddingRepository") as MockEmbRepo, \
             patch.object(bmd, "TaskRepository") as MockTaskRepo:
            mock_emb_repo = MagicMock()
            mock_emb_repo.create = AM(return_value=emb_schema)
            MockEmbRepo.return_value = mock_emb_repo
            mock_task_repo = MagicMock()
            mock_task_repo.update_embedding = AM(return_value=None)
            MockTaskRepo.return_value = mock_task_repo
            X = pd.DataFrame({"a": [1.0]})
            y = pd.Series([0])
            result = await bmd.store_embedding(mock_session, embedder, faiss_index, emb_schema.task_id, X, y)
        assert result.shape == (25,)

    async def test_faiss_index_add_called_with_str_task_id(self, bmd, mock_session) -> None:
        embedder, faiss_index, emb_schema = self._make_deps()
        from unittest.mock import AsyncMock as AM
        with patch.object(bmd, "EmbeddingRepository") as MockEmbRepo, \
             patch.object(bmd, "TaskRepository") as MockTaskRepo:
            mock_emb_repo = MagicMock()
            mock_emb_repo.create = AM(return_value=emb_schema)
            MockEmbRepo.return_value = mock_emb_repo
            mock_task_repo = MagicMock()
            mock_task_repo.update_embedding = AM(return_value=None)
            MockTaskRepo.return_value = mock_task_repo
            X = pd.DataFrame({"a": [1.0]})
            y = pd.Series([0])
            await bmd.store_embedding(mock_session, embedder, faiss_index, emb_schema.task_id, X, y)
        faiss_index.add.assert_called_once()
        first_arg = faiss_index.add.call_args.args[0]
        assert isinstance(first_arg, str)

    async def test_task_repository_update_embedding_called(self, bmd, mock_session) -> None:
        embedder, faiss_index, emb_schema = self._make_deps()
        from unittest.mock import AsyncMock as AM
        with patch.object(bmd, "EmbeddingRepository") as MockEmbRepo, \
             patch.object(bmd, "TaskRepository") as MockTaskRepo:
            mock_emb_repo = MagicMock()
            mock_emb_repo.create = AM(return_value=emb_schema)
            MockEmbRepo.return_value = mock_emb_repo
            mock_task_repo = MagicMock()
            mock_task_repo.update_embedding = AM(return_value=None)
            MockTaskRepo.return_value = mock_task_repo
            X = pd.DataFrame({"a": [1.0]})
            y = pd.Series([0])
            await bmd.store_embedding(mock_session, embedder, faiss_index, emb_schema.task_id, X, y)
        mock_task_repo.update_embedding.assert_awaited_once()

    async def test_embedding_repository_create_called(self, bmd, mock_session) -> None:
        embedder, faiss_index, emb_schema = self._make_deps()
        from unittest.mock import AsyncMock as AM
        with patch.object(bmd, "EmbeddingRepository") as MockEmbRepo, \
             patch.object(bmd, "TaskRepository") as MockTaskRepo:
            mock_emb_repo = MagicMock()
            mock_emb_repo.create = AM(return_value=emb_schema)
            MockEmbRepo.return_value = mock_emb_repo
            mock_task_repo = MagicMock()
            mock_task_repo.update_embedding = AM(return_value=None)
            MockTaskRepo.return_value = mock_task_repo
            X = pd.DataFrame({"a": [1.0]})
            y = pd.Series([0])
            await bmd.store_embedding(mock_session, embedder, faiss_index, emb_schema.task_id, X, y)
        mock_emb_repo.create.assert_awaited_once()


# ── TestBootstrapOrchestration ────────────────────────────────────────────────


class TestBootstrapOrchestration:
    def _make_args(self, dry_run: bool = False, suites=None, max_tasks=None) -> SimpleNamespace:
        return SimpleNamespace(
            max_tasks=max_tasks,
            output_dir="/tmp/test_bootstrap_output",
            db_url="postgresql+asyncpg://x:y@localhost/test",
            dry_run=dry_run,
            suites=suites or ["cc18"],
        )

    def _make_task_obj(self, task_id: int = 1) -> MagicMock:
        t = MagicMock()
        t.task_id = task_id
        return t

    def _model_results(self) -> dict:
        return {
            "logistic_regression": {"mean": 0.8, "std": 0.02},
            "random_forest": {"mean": 0.85, "std": 0.01},
            "xgboost": {"mean": 0.82, "std": 0.01},
            "svc_rbf": {"mean": 0.78, "std": 0.03},
            "knn": {"mean": 0.75, "std": 0.04},
        }

    async def test_dry_run_returns_zero_counts(self, bmd) -> None:
        args = self._make_args(dry_run=True)
        X = pd.DataFrame({"a": [1.0]})
        y = pd.Series([0])
        with patch.object(bmd, "download_suite", return_value=[self._make_task_obj()]), \
             patch.object(bmd, "fetch_dataset", return_value=(X, y, "classification")), \
             patch.object(bmd, "run_baseline_models", return_value=self._model_results()), \
             patch.object(bmd, "get_engine"), \
             patch.object(bmd, "get_session"):
            result = await bmd._bootstrap_async(args)
        assert result["tasks"] == 0
        assert result["experiments"] == 0

    async def test_returns_task_and_experiment_counts(self, bmd, tmp_path) -> None:
        from uuid import uuid4
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from unittest.mock import AsyncMock as AM
        import contextlib

        args = self._make_args(suites=["cc18"])
        args.output_dir = str(tmp_path)
        task_objs = [self._make_task_obj(i) for i in range(2)]
        X = pd.DataFrame({"a": [1.0, 2.0]})
        y = pd.Series([0, 1])

        task_id = uuid4()
        task_schema = SimpleNamespace(
            task_id=task_id, name="t", task_type="classification", domain="tabular",
            n_samples=2, n_features=1, n_classes=2, dataset_uri="openml://task/1",
            metadata={}, embedding_id=None,
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        emb_schema = SimpleNamespace(
            embedding_id=uuid4(), task_id=task_id,
            embedding_type="statistical", embedding_vector=[0.0]*25,
            dimension=25, model_version="v1", created_at=datetime.now(timezone.utc),
        )
        exp_schema = SimpleNamespace(
            experiment_id=uuid4(), task_id=task_id, model_id=uuid4(),
            status="pending", mlflow_run_id=None, started_at=None, completed_at=None,
        )
        metric_schema = SimpleNamespace(name="cv_mean", value=0.8, step=None, is_final=True)

        with patch.object(bmd, "download_suite", return_value=task_objs), \
             patch.object(bmd, "fetch_dataset", return_value=(X, y, "classification")), \
             patch.object(bmd, "run_baseline_models", return_value=self._model_results()), \
             patch.object(bmd, "get_engine"), \
             patch.object(bmd, "TaskRepository") as MockTaskRepo, \
             patch.object(bmd, "ExperimentRepository") as MockExpRepo, \
             patch.object(bmd, "PerformanceRepository") as MockPerfRepo, \
             patch.object(bmd, "EmbeddingRepository") as MockEmbRepo, \
             patch.object(bmd, "ModelORM"), \
             patch.object(bmd, "StatisticalEmbedder") as MockEmbedder, \
             patch.object(bmd, "FaissIndex") as MockFaiss, \
             patch.object(bmd, "get_session") as mock_get_session:

            mock_task_repo = MagicMock()
            mock_task_repo.create = AM(return_value=task_schema)
            mock_task_repo.update_embedding = AM(return_value=None)
            MockTaskRepo.return_value = mock_task_repo

            mock_exp_repo = MagicMock()
            mock_exp_repo.create = AM(return_value=exp_schema)
            mock_exp_repo.mark_complete = AM(return_value=None)
            MockExpRepo.return_value = mock_exp_repo

            mock_perf_repo = MagicMock()
            mock_perf_repo.log_metric = AM(return_value=metric_schema)
            MockPerfRepo.return_value = mock_perf_repo

            mock_emb_repo = MagicMock()
            mock_emb_repo.create = AM(return_value=emb_schema)
            MockEmbRepo.return_value = mock_emb_repo

            mock_embedder_instance = MagicMock()
            mock_embedder_instance.embed.return_value = np.zeros(25)
            mock_embedder_instance.embedding_dim = 25
            MockEmbedder.return_value = mock_embedder_instance

            mock_faiss_instance = MagicMock()
            MockFaiss.return_value = mock_faiss_instance

            @contextlib.asynccontextmanager
            async def fake_session(engine):
                mock_s = MagicMock()
                mock_s.add = MagicMock()
                mock_s.flush = AM(return_value=None)
                yield mock_s

            mock_get_session.side_effect = fake_session

            result = await bmd._bootstrap_async(args)

        assert result["tasks"] == 2

    async def test_fetch_dataset_failure_skips_task(self, bmd, tmp_path) -> None:
        args = self._make_args()
        args.output_dir = str(tmp_path)
        task_objs = [self._make_task_obj(i) for i in range(3)]

        call_count = 0
        def fetch_side_effect(task_obj):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("fetch error")
            X = pd.DataFrame({"a": [1.0]})
            y = pd.Series([0])
            return X, y, "classification"

        with patch.object(bmd, "download_suite", return_value=task_objs), \
             patch.object(bmd, "fetch_dataset", side_effect=fetch_side_effect), \
             patch.object(bmd, "run_baseline_models", return_value={}), \
             patch.object(bmd, "get_engine"), \
             patch.object(bmd, "get_session"), \
             patch.object(bmd, "store_task", new=__import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(side_effect=RuntimeError("should not reach"))):
            result = await bmd._bootstrap_async(args)

        assert result["tasks"] == 0  # dry_run=False but no successful stores in this test

    async def test_faiss_saved_when_not_dry_run(self, bmd, tmp_path) -> None:
        from unittest.mock import AsyncMock as AM
        import contextlib

        args = self._make_args(dry_run=False)
        args.output_dir = str(tmp_path)

        with patch.object(bmd, "download_suite", return_value=[]), \
             patch.object(bmd, "get_engine"), \
             patch.object(bmd, "StatisticalEmbedder") as MockEmbedder, \
             patch.object(bmd, "FaissIndex") as MockFaiss, \
             patch.object(bmd, "get_session"):
            mock_embedder_instance = MagicMock()
            mock_embedder_instance.embedding_dim = 25
            MockEmbedder.return_value = mock_embedder_instance
            mock_faiss_instance = MagicMock()
            MockFaiss.return_value = mock_faiss_instance

            await bmd._bootstrap_async(args)

        mock_faiss_instance.save.assert_called_once()

    async def test_faiss_not_saved_when_dry_run(self, bmd, tmp_path) -> None:
        args = self._make_args(dry_run=True)
        args.output_dir = str(tmp_path)

        with patch.object(bmd, "download_suite", return_value=[]), \
             patch.object(bmd, "get_engine"), \
             patch.object(bmd, "StatisticalEmbedder") as MockEmbedder, \
             patch.object(bmd, "FaissIndex") as MockFaiss, \
             patch.object(bmd, "get_session"):
            mock_embedder_instance = MagicMock()
            mock_embedder_instance.embedding_dim = 25
            MockEmbedder.return_value = mock_embedder_instance
            mock_faiss_instance = MagicMock()
            MockFaiss.return_value = mock_faiss_instance

            await bmd._bootstrap_async(args)

        mock_faiss_instance.save.assert_not_called()

    async def test_multiple_suites_processed(self, bmd, tmp_path) -> None:
        args = self._make_args(dry_run=True, suites=["cc18", "ctr23"])
        args.output_dir = str(tmp_path)

        download_calls = []
        def track_download(suite_name, max_tasks=None):
            download_calls.append(suite_name)
            return []

        with patch.object(bmd, "download_suite", side_effect=track_download), \
             patch.object(bmd, "get_engine"), \
             patch.object(bmd, "StatisticalEmbedder") as MockEmbedder, \
             patch.object(bmd, "FaissIndex") as MockFaiss, \
             patch.object(bmd, "get_session"):
            mock_embedder_instance = MagicMock()
            mock_embedder_instance.embedding_dim = 25
            MockEmbedder.return_value = mock_embedder_instance
            MockFaiss.return_value = MagicMock()

            await bmd._bootstrap_async(args)

        assert "cc18" in download_calls
        assert "ctr23" in download_calls
