"""Tests for scripts/bootstrap_meta_dataset.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


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
