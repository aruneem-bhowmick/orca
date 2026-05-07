"""Tests for scripts/bootstrap_meta_dataset.py (stub stage)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


# ── parse_args defaults ───────────────────────────────────────────────────────

def test_parse_args_suite_id_default(bmd) -> None:
    with patch("sys.argv", ["prog"]):
        args = bmd.parse_args()
    assert args.suite_id == 271


def test_parse_args_max_tasks_default_is_none(bmd) -> None:
    with patch("sys.argv", ["prog"]):
        args = bmd.parse_args()
    assert args.max_tasks is None


def test_parse_args_data_dir_default(bmd) -> None:
    with patch("sys.argv", ["prog"]):
        args = bmd.parse_args()
    assert args.data_dir == "./data/openml"


def test_parse_args_registry_url_default(bmd) -> None:
    with patch("sys.argv", ["prog"]):
        args = bmd.parse_args()
    assert "localhost:5432" in args.registry_url
    assert "orca_registry" in args.registry_url


def test_parse_args_dry_run_default_is_false(bmd) -> None:
    with patch("sys.argv", ["prog"]):
        args = bmd.parse_args()
    assert args.dry_run is False


# ── parse_args custom values ──────────────────────────────────────────────────

def test_parse_args_custom_suite_id(bmd) -> None:
    with patch("sys.argv", ["prog", "--suite-id", "99"]):
        args = bmd.parse_args()
    assert args.suite_id == 99


def test_parse_args_custom_max_tasks(bmd) -> None:
    with patch("sys.argv", ["prog", "--max-tasks", "10"]):
        args = bmd.parse_args()
    assert args.max_tasks == 10


def test_parse_args_custom_data_dir(bmd) -> None:
    with patch("sys.argv", ["prog", "--data-dir", "/tmp/data"]):
        args = bmd.parse_args()
    assert args.data_dir == "/tmp/data"


def test_parse_args_dry_run_flag(bmd) -> None:
    with patch("sys.argv", ["prog", "--dry-run"]):
        args = bmd.parse_args()
    assert args.dry_run is True


def test_parse_args_custom_registry_url(bmd) -> None:
    url = "postgresql://user:pass@remotehost:5432/mydb"
    with patch("sys.argv", ["prog", "--registry-url", url]):
        args = bmd.parse_args()
    assert args.registry_url == url


# ── main() behaviour ──────────────────────────────────────────────────────────

def test_main_returns_zero(bmd) -> None:
    with patch("sys.argv", ["prog"]):
        code = bmd.main()
    assert code == 0


def test_main_prints_suite_id(bmd, capsys) -> None:
    with patch("sys.argv", ["prog"]):
        bmd.main()
    assert "271" in capsys.readouterr().out


def test_main_prints_stub_notice(bmd, capsys) -> None:
    with patch("sys.argv", ["prog"]):
        bmd.main()
    assert "stub" in capsys.readouterr().out.lower()


def test_main_prints_separator_line(bmd, capsys) -> None:
    with patch("sys.argv", ["prog"]):
        bmd.main()
    assert "=" * 10 in capsys.readouterr().out


def test_main_reflects_custom_suite_id(bmd, capsys) -> None:
    with patch("sys.argv", ["prog", "--suite-id", "42"]):
        bmd.main()
    assert "42" in capsys.readouterr().out


def test_main_reflects_dry_run_true(bmd, capsys) -> None:
    with patch("sys.argv", ["prog", "--dry-run"]):
        bmd.main()
    assert "True" in capsys.readouterr().out


def test_main_max_tasks_none_shows_all(bmd, capsys) -> None:
    with patch("sys.argv", ["prog"]):
        bmd.main()
    assert "all" in capsys.readouterr().out


def test_main_max_tasks_int_shown(bmd, capsys) -> None:
    with patch("sys.argv", ["prog", "--max-tasks", "50"]):
        bmd.main()
    assert "50" in capsys.readouterr().out


# ── module structure ──────────────────────────────────────────────────────────

def test_module_has_docstring(bmd) -> None:
    assert bmd.__doc__ is not None and bmd.__doc__.strip()


def test_module_docstring_mentions_openml(bmd) -> None:
    assert "openml" in bmd.__doc__.lower()


def test_script_file_exists(repo_root: Path) -> None:
    assert (repo_root / "scripts" / "bootstrap_meta_dataset.py").is_file()


def test_script_is_executable_stub(bmd) -> None:
    assert callable(bmd.parse_args)
    assert callable(bmd.main)
