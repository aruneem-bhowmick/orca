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
