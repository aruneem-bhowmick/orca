"""Tests for the OrcaMind CLI — all six commands plus --version and --help."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from orcamind.cli import app

runner = CliRunner()

_SIMPLE_CONFIG = """\
paths:
  model_dir: models
mlflow:
  experiment_name: orcamind
  tracking_uri: http://localhost:5000
seed: 42
device: cpu
"""


def _make_csv(path: Path) -> Path:
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "y": [0, 1, 0]}).to_csv(
        path, index=False
    )
    return path


# ── --version ─────────────────────────────────────────────────────────────────


class TestVersion:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_output_contains_version_number(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert "1.0.0" in result.output

    def test_output_contains_orcamind(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert "OrcaMind" in result.output


# ── --help ────────────────────────────────────────────────────────────────────


class TestHelp:
    def test_root_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        "cmd", ["init", "train", "embed", "recommend", "serve", "dashboard"]
    )
    def test_root_help_lists_all_commands(self, cmd: str) -> None:
        result = runner.invoke(app, ["--help"])
        assert cmd in result.output

    @pytest.mark.parametrize(
        "cmd", ["init", "train", "embed", "recommend", "serve", "dashboard"]
    )
    def test_command_help_exits_zero(self, cmd: str) -> None:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0

    def test_train_help_shows_epochs_flag(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert "--epochs" in result.output

    def test_train_help_shows_device_flag(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert "--device" in result.output

    def test_serve_help_shows_reload_flag(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert "--reload" in result.output

    def test_recommend_help_shows_api_url_flag(self) -> None:
        result = runner.invoke(app, ["recommend", "--help"])
        assert "--api-url" in result.output


# ── init ──────────────────────────────────────────────────────────────────────


class TestInit:
    def test_exits_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            result = runner.invoke(app, ["init"])
        assert result.exit_code == 0

    def test_creates_data_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            runner.invoke(app, ["init"])
        assert (tmp_path / "data").is_dir()

    def test_creates_models_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            runner.invoke(app, ["init"])
        assert (tmp_path / "models").is_dir()

    def test_creates_logs_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            runner.invoke(app, ["init"])
        assert (tmp_path / "logs").is_dir()

    def test_writes_default_config_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            runner.invoke(app, ["init"])
        assert (tmp_path / "config" / "config.yaml").exists()

    def test_config_contains_mlflow_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            runner.invoke(app, ["init"])
        content = (tmp_path / "config" / "config.yaml").read_text()
        assert "mlflow" in content

    def test_skips_existing_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "config.yaml").write_text("seed: 99\n")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            result = runner.invoke(app, ["init"])
        assert "already exists" in result.output

    def test_outputs_workspace_initialized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            result = runner.invoke(app, ["init"])
        assert "initialized" in result.output.lower()

    def test_mlflow_error_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_experiment", side_effect=Exception("no mlflow server")):
            result = runner.invoke(app, ["init"])
        assert result.exit_code == 0

    def test_mlflow_error_logs_skip_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_experiment", side_effect=Exception("no mlflow server")):
            result = runner.invoke(app, ["init"])
        assert "skipped" in result.output.lower()

    def test_outputs_created_directory_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            result = runner.invoke(app, ["init"])
        assert "data" in result.output
        assert "models" in result.output
        assert "logs" in result.output


# ── serve ─────────────────────────────────────────────────────────────────────


class TestServe:
    def test_exits_zero(self) -> None:
        with patch("uvicorn.run"):
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0

    def test_calls_uvicorn_run_once(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["serve"])
        mock_run.assert_called_once()

    def test_default_host_in_output(self) -> None:
        with patch("uvicorn.run"):
            result = runner.invoke(app, ["serve"])
        assert "0.0.0.0" in result.output

    def test_default_port_in_output(self) -> None:
        with patch("uvicorn.run"):
            result = runner.invoke(app, ["serve"])
        assert "8000" in result.output

    def test_passes_host_to_uvicorn(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["serve", "--host", "127.0.0.1"])
        assert mock_run.call_args.kwargs["host"] == "127.0.0.1"

    def test_passes_port_to_uvicorn(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["serve", "--port", "9090"])
        assert mock_run.call_args.kwargs["port"] == 9090

    def test_short_port_flag(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["serve", "-p", "7777"])
        assert mock_run.call_args.kwargs["port"] == 7777

    def test_reload_true_when_flag_given(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["serve", "--reload"])
        assert mock_run.call_args.kwargs["reload"] is True

    def test_reload_false_by_default(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["serve"])
        assert mock_run.call_args.kwargs["reload"] is False

    def test_custom_host_and_port_in_output(self) -> None:
        with patch("uvicorn.run"):
            result = runner.invoke(app, ["serve", "--host", "10.0.0.1", "--port", "9999"])
        assert "10.0.0.1" in result.output
        assert "9999" in result.output


# ── dashboard ─────────────────────────────────────────────────────────────────


class TestDashboard:
    def test_exits_zero(self) -> None:
        with patch("subprocess.run"):
            result = runner.invoke(app, ["dashboard"])
        assert result.exit_code == 0

    def test_calls_subprocess_run_once(self) -> None:
        with patch("subprocess.run") as mock_run:
            runner.invoke(app, ["dashboard"])
        mock_run.assert_called_once()

    def test_command_includes_streamlit(self) -> None:
        with patch("subprocess.run") as mock_run:
            runner.invoke(app, ["dashboard"])
        cmd = mock_run.call_args.args[0]
        assert "streamlit" in cmd

    def test_command_includes_run_subcommand(self) -> None:
        with patch("subprocess.run") as mock_run:
            runner.invoke(app, ["dashboard"])
        cmd = mock_run.call_args.args[0]
        assert "run" in cmd

    def test_default_port_8501(self) -> None:
        with patch("subprocess.run") as mock_run:
            runner.invoke(app, ["dashboard"])
        assert "8501" in str(mock_run.call_args.args[0])

    def test_custom_port(self) -> None:
        with patch("subprocess.run") as mock_run:
            runner.invoke(app, ["dashboard", "--port", "9999"])
        assert "9999" in str(mock_run.call_args.args[0])

    def test_short_port_flag(self) -> None:
        with patch("subprocess.run") as mock_run:
            runner.invoke(app, ["dashboard", "-p", "8888"])
        assert "8888" in str(mock_run.call_args.args[0])

    def test_outputs_starting_message(self) -> None:
        with patch("subprocess.run"):
            result = runner.invoke(app, ["dashboard"])
        assert "dashboard" in result.output.lower()

    def test_check_true_passed_to_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run:
            runner.invoke(app, ["dashboard"])
        assert mock_run.call_args.kwargs.get("check") is True
