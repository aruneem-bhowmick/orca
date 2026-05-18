"""Unit tests for scripts/init_prefect.py work-pool creation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest


@pytest.fixture(scope="module")
def init_prefect(repo_root: Path) -> ModuleType:
    script_path = repo_root / "scripts" / "init_prefect.py"
    spec = importlib.util.spec_from_file_location("init_prefect", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("init_prefect", module)
    spec.loader.exec_module(module)
    return module


class TestCreateOrcalabPool:
    def test_calls_subprocess_run_once(self, init_prefect: ModuleType) -> None:
        with patch("subprocess.run") as mock_run:
            init_prefect.create_orcalab_pool()
        mock_run.assert_called_once()

    def test_uses_prefect_command(self, init_prefect: ModuleType) -> None:
        with patch("subprocess.run") as mock_run:
            init_prefect.create_orcalab_pool()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "prefect"

    def test_creates_work_pool(self, init_prefect: ModuleType) -> None:
        with patch("subprocess.run") as mock_run:
            init_prefect.create_orcalab_pool()
        cmd = mock_run.call_args[0][0]
        assert "work-pool" in cmd
        assert "create" in cmd

    def test_pool_name_is_orcalab_pool(self, init_prefect: ModuleType) -> None:
        with patch("subprocess.run") as mock_run:
            init_prefect.create_orcalab_pool()
        cmd = mock_run.call_args[0][0]
        assert "orcalab-pool" in cmd

    def test_type_is_process(self, init_prefect: ModuleType) -> None:
        with patch("subprocess.run") as mock_run:
            init_prefect.create_orcalab_pool()
        cmd = mock_run.call_args[0][0]
        assert "--type" in cmd
        type_index = cmd.index("--type")
        assert cmd[type_index + 1] == "process"

    def test_check_is_true(self, init_prefect: ModuleType) -> None:
        with patch("subprocess.run") as mock_run:
            init_prefect.create_orcalab_pool()
        kwargs = mock_run.call_args[1]
        assert kwargs.get("check") is True
