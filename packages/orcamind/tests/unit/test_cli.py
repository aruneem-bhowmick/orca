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
        assert "127.0.0.1" in result.output

    def test_default_port_in_output(self) -> None:
        with patch("uvicorn.run"):
            result = runner.invoke(app, ["serve"])
        assert "8000" in result.output

    def test_passes_host_to_uvicorn(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["serve", "--host", "0.0.0.0"])
        assert mock_run.call_args.kwargs["host"] == "0.0.0.0"

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


# ── embed ─────────────────────────────────────────────────────────────────────


_STAT_VEC = np.zeros(25, dtype=np.float64)
_NEURAL_VEC = np.zeros(64, dtype=np.float32)


class TestEmbed:
    def test_missing_file_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["embed", "nonexistent.csv"])
        assert result.exit_code != 0

    def test_no_args_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["embed"])
        assert result.exit_code != 0

    def test_exits_zero_on_success(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            result = runner.invoke(app, ["embed", str(csv_file)])
        assert result.exit_code == 0

    def test_output_is_valid_json(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            result = runner.invoke(app, ["embed", str(csv_file)])
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)

    def test_output_contains_statistical_key(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            result = runner.invoke(app, ["embed", str(csv_file)])
        parsed = json.loads(result.output)
        assert "statistical" in parsed

    def test_output_contains_neural_key(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            result = runner.invoke(app, ["embed", str(csv_file)])
        parsed = json.loads(result.output)
        assert "neural" in parsed

    def test_output_contains_dataset_key(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            result = runner.invoke(app, ["embed", str(csv_file)])
        parsed = json.loads(result.output)
        assert parsed["dataset"] == str(csv_file)

    def test_statistical_vector_has_25_dimensions(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        stat_vec = np.ones(25, dtype=np.float64)
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=stat_vec), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            result = runner.invoke(app, ["embed", str(csv_file)])
        assert len(json.loads(result.output)["statistical"]) == 25

    def test_neural_vector_has_64_dimensions(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        neural_vec = np.ones(64, dtype=np.float32)
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=neural_vec):
            result = runner.invoke(app, ["embed", str(csv_file)])
        assert len(json.loads(result.output)["neural"]) == 64

    def test_saves_json_to_output_file(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        out_file = tmp_path / "embedding.json"
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            result = runner.invoke(app, ["embed", str(csv_file), "--output", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()

    def test_output_file_contains_both_keys(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        out_file = tmp_path / "embedding.json"
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            runner.invoke(app, ["embed", str(csv_file), "--output", str(out_file)])
        content = json.loads(out_file.read_text())
        assert "statistical" in content and "neural" in content

    def test_short_output_flag(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        out_file = tmp_path / "out.json"
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            runner.invoke(app, ["embed", str(csv_file), "-o", str(out_file)])
        assert out_file.exists()

    def test_calls_statistical_embedder(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC) as mock_stat, \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC):
            runner.invoke(app, ["embed", str(csv_file)])
        mock_stat.assert_called_once()

    def test_calls_neural_embedder(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("orcamind.embedders.neural.NeuralEmbedder.embed", return_value=_NEURAL_VEC) as mock_neural:
            runner.invoke(app, ["embed", str(csv_file)])
        mock_neural.assert_called_once()


# ── train ─────────────────────────────────────────────────────────────────────

try:
    import pytorch_lightning as _pl  # noqa: F401

    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False


def _write_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.yaml"
    config_file.write_text(_SIMPLE_CONFIG)
    return config_file


@pytest.mark.skipif(not _PL_AVAILABLE, reason="pytorch_lightning not installed")
class TestTrain:
    def test_missing_config_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["train", "--config", "no_such_file.yaml"])
        assert result.exit_code != 0

    def test_exits_zero_with_valid_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = _write_config(tmp_path)
        mock_trainer = MagicMock()
        with patch("pytorch_lightning.Trainer", return_value=mock_trainer), \
             patch("torch.save"):
            result = runner.invoke(app, ["train", "--config", str(cfg_file), "--epochs", "1"])
        assert result.exit_code == 0

    def test_outputs_config_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = _write_config(tmp_path)
        mock_trainer = MagicMock()
        with patch("pytorch_lightning.Trainer", return_value=mock_trainer), \
             patch("torch.save"):
            result = runner.invoke(app, ["train", "--config", str(cfg_file)])
        assert str(cfg_file) in result.output

    def test_outputs_epoch_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = _write_config(tmp_path)
        mock_trainer = MagicMock()
        with patch("pytorch_lightning.Trainer", return_value=mock_trainer), \
             patch("torch.save"):
            result = runner.invoke(app, ["train", "--config", str(cfg_file), "--epochs", "7"])
        assert "7" in result.output

    def test_passes_epochs_to_trainer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = _write_config(tmp_path)
        mock_trainer = MagicMock()
        with patch("pytorch_lightning.Trainer", return_value=mock_trainer) as mock_cls, \
             patch("torch.save"):
            runner.invoke(app, ["train", "--config", str(cfg_file), "--epochs", "5"])
        assert mock_cls.call_args.kwargs["max_epochs"] == 5

    def test_calls_trainer_fit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = _write_config(tmp_path)
        mock_trainer = MagicMock()
        with patch("pytorch_lightning.Trainer", return_value=mock_trainer), \
             patch("torch.save"):
            runner.invoke(app, ["train", "--config", str(cfg_file), "--epochs", "1"])
        mock_trainer.fit.assert_called_once()

    def test_calls_torch_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = _write_config(tmp_path)
        mock_trainer = MagicMock()
        with patch("pytorch_lightning.Trainer", return_value=mock_trainer), \
             patch("torch.save") as mock_save:
            runner.invoke(app, ["train", "--config", str(cfg_file), "--epochs", "1"])
        mock_save.assert_called_once()

    def test_outputs_checkpoint_filename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = _write_config(tmp_path)
        mock_trainer = MagicMock()
        with patch("pytorch_lightning.Trainer", return_value=mock_trainer), \
             patch("torch.save"):
            result = runner.invoke(app, ["train", "--config", str(cfg_file), "--epochs", "1"])
        assert "orcamind_final.pt" in result.output

    def test_short_config_flag(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert "-c" in result.output

    def test_short_epochs_flag(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert "-e" in result.output

    def test_accepts_device_cpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = _write_config(tmp_path)
        mock_trainer = MagicMock()
        with patch("pytorch_lightning.Trainer", return_value=mock_trainer), \
             patch("torch.save"):
            result = runner.invoke(app, ["train", "--config", str(cfg_file), "--device", "cpu"])
        assert result.exit_code == 0

    def test_device_shown_in_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_file = _write_config(tmp_path)
        mock_trainer = MagicMock()
        with patch("pytorch_lightning.Trainer", return_value=mock_trainer), \
             patch("torch.save"):
            result = runner.invoke(app, ["train", "--config", str(cfg_file), "--device", "cpu"])
        assert "cpu" in result.output


# ── recommend ─────────────────────────────────────────────────────────────────


_SAMPLE_RECS = [
    {"model_name": "RandomForest", "confidence": 0.9, "algorithm": "nn"},
    {"model_name": "XGBoost", "confidence": 0.8, "algorithm": "nn"},
    {"model_name": "SVM", "confidence": 0.7, "algorithm": "nn"},
]


def _mock_http_client(response: MagicMock) -> MagicMock:
    """Build a mock httpx.Client context manager that returns *response* from .post()."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.post = MagicMock(return_value=response)
    return client


def _ok_response(data: object) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


class TestRecommend:
    def test_missing_file_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["recommend", "nonexistent.csv"])
        assert result.exit_code != 0

    def test_no_args_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["recommend"])
        assert result.exit_code != 0

    def test_exits_zero_on_success(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        client = _mock_http_client(_ok_response(_SAMPLE_RECS))
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["recommend", str(csv_file)])
        assert result.exit_code == 0

    def test_calls_recommend_model_endpoint(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        client = _mock_http_client(_ok_response(_SAMPLE_RECS))
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            runner.invoke(app, ["recommend", str(csv_file)])
        call_url = client.post.call_args.args[0]
        assert "/api/v1/recommend-model" in call_url

    def test_sends_embedding_in_payload(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        client = _mock_http_client(_ok_response(_SAMPLE_RECS))
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            runner.invoke(app, ["recommend", str(csv_file)])
        payload = client.post.call_args.kwargs["json"]
        assert "task_embedding" in payload

    def test_sends_top_k_in_payload(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        client = _mock_http_client(_ok_response(_SAMPLE_RECS))
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            runner.invoke(app, ["recommend", str(csv_file), "--top-k", "5"])
        payload = client.post.call_args.kwargs["json"]
        assert payload["top_k"] == 5

    def test_default_top_k_is_3(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        client = _mock_http_client(_ok_response(_SAMPLE_RECS))
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            runner.invoke(app, ["recommend", str(csv_file)])
        payload = client.post.call_args.kwargs["json"]
        assert payload["top_k"] == 3

    def test_custom_api_url_used_in_request(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        client = _mock_http_client(_ok_response(_SAMPLE_RECS))
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            runner.invoke(app, ["recommend", str(csv_file), "--api-url", "http://myserver:9000"])
        call_url = client.post.call_args.args[0]
        assert "myserver:9000" in call_url

    def test_http_error_exits_nonzero(self, tmp_path: Path) -> None:
        import httpx

        csv_file = _make_csv(tmp_path / "data.csv")
        client = _mock_http_client(MagicMock())
        client.post.side_effect = httpx.HTTPError("connection refused")
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["recommend", str(csv_file)])
        assert result.exit_code != 0

    def test_short_top_k_flag(self) -> None:
        result = runner.invoke(app, ["recommend", "--help"])
        assert "-k" in result.output

    def test_api_url_flag_in_help(self) -> None:
        result = runner.invoke(app, ["recommend", "--help"])
        assert "--api-url" in result.output

    def test_output_contains_top_k_in_title(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        client = _mock_http_client(_ok_response(_SAMPLE_RECS))
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["recommend", str(csv_file), "--top-k", "3"])
        assert "3" in result.output

    def test_list_response_handled(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        client = _mock_http_client(_ok_response(_SAMPLE_RECS))
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["recommend", str(csv_file)])
        assert result.exit_code == 0

    def test_dict_response_with_recommendations_key(self, tmp_path: Path) -> None:
        csv_file = _make_csv(tmp_path / "data.csv")
        wrapped = {"recommendations": _SAMPLE_RECS}
        client = _mock_http_client(_ok_response(wrapped))
        with patch("orcamind.embedders.statistical.StatisticalEmbedder.embed", return_value=_STAT_VEC), \
             patch("httpx.Client", return_value=client):
            result = runner.invoke(app, ["recommend", str(csv_file)])
        assert result.exit_code == 0
