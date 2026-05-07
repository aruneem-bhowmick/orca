"""Tests for the orcamind Typer CLI (scaffold-stage stub commands)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from orcamind.cli import app

runner = CliRunner()


# ── App object ────────────────────────────────────────────────────────────────

def test_app_object_exists() -> None:
    assert app is not None


def test_app_name() -> None:
    assert app.info.name == "orcamind"


# ── Global help ───────────────────────────────────────────────────────────────

def test_help_exits_cleanly() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("cmd", ["train", "serve", "embed", "recommend"])
def test_help_lists_command(cmd: str) -> None:
    result = runner.invoke(app, ["--help"])
    assert cmd in result.output, f"Command '{cmd}' missing from --help output"


# ── train ─────────────────────────────────────────────────────────────────────

def test_train_exits_cleanly() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0


def test_train_shows_stub_notice() -> None:
    result = runner.invoke(app, ["train"])
    assert "not yet implemented" in result.output


def test_train_shows_default_config_path() -> None:
    result = runner.invoke(app, ["train"])
    assert "config/config.yaml" in result.output


def test_train_accepts_custom_config_flag() -> None:
    result = runner.invoke(app, ["train", "--config", "custom/path.yaml"])
    assert result.exit_code == 0
    assert "custom/path.yaml" in result.output


def test_train_accepts_short_config_flag() -> None:
    result = runner.invoke(app, ["train", "-c", "alt.yaml"])
    assert result.exit_code == 0
    assert "alt.yaml" in result.output


def test_train_help_describes_command() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "meta-training" in result.output.lower()


# ── serve ─────────────────────────────────────────────────────────────────────

def test_serve_exits_cleanly() -> None:
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0


def test_serve_shows_stub_notice() -> None:
    result = runner.invoke(app, ["serve"])
    assert "not yet implemented" in result.output


def test_serve_shows_default_host() -> None:
    result = runner.invoke(app, ["serve"])
    assert "0.0.0.0" in result.output


def test_serve_shows_default_port() -> None:
    result = runner.invoke(app, ["serve"])
    assert "8000" in result.output


def test_serve_custom_port() -> None:
    result = runner.invoke(app, ["serve", "--port", "9999"])
    assert result.exit_code == 0
    assert "9999" in result.output


def test_serve_custom_port_short_flag() -> None:
    result = runner.invoke(app, ["serve", "-p", "7777"])
    assert result.exit_code == 0
    assert "7777" in result.output


def test_serve_custom_host() -> None:
    result = runner.invoke(app, ["serve", "--host", "127.0.0.1"])
    assert result.exit_code == 0
    assert "127.0.0.1" in result.output


# ── embed ─────────────────────────────────────────────────────────────────────

def test_embed_without_args_fails() -> None:
    result = runner.invoke(app, ["embed"])
    assert result.exit_code != 0


def test_embed_with_dataset_exits_cleanly() -> None:
    result = runner.invoke(app, ["embed", "data.csv"])
    assert result.exit_code == 0


def test_embed_shows_dataset_path() -> None:
    result = runner.invoke(app, ["embed", "my_data.csv"])
    assert "my_data.csv" in result.output


def test_embed_shows_stub_notice() -> None:
    result = runner.invoke(app, ["embed", "data.csv"])
    assert "not yet implemented" in result.output


def test_embed_default_output_file() -> None:
    result = runner.invoke(app, ["embed", "data.csv"])
    assert "embedding.npy" in result.output


def test_embed_custom_output_flag() -> None:
    result = runner.invoke(app, ["embed", "data.csv", "--output", "result.npy"])
    assert result.exit_code == 0
    assert "result.npy" in result.output


def test_embed_custom_output_short_flag() -> None:
    result = runner.invoke(app, ["embed", "data.csv", "-o", "out.npy"])
    assert result.exit_code == 0
    assert "out.npy" in result.output


# ── recommend ─────────────────────────────────────────────────────────────────

def test_recommend_without_args_fails() -> None:
    result = runner.invoke(app, ["recommend"])
    assert result.exit_code != 0


def test_recommend_with_dataset_exits_cleanly() -> None:
    result = runner.invoke(app, ["recommend", "data.csv"])
    assert result.exit_code == 0


def test_recommend_shows_dataset_path() -> None:
    result = runner.invoke(app, ["recommend", "my_data.csv"])
    assert "my_data.csv" in result.output


def test_recommend_shows_stub_notice() -> None:
    result = runner.invoke(app, ["recommend", "data.csv"])
    assert "not yet implemented" in result.output


def test_recommend_default_top_k_is_5() -> None:
    result = runner.invoke(app, ["recommend", "data.csv"])
    assert "5" in result.output


def test_recommend_custom_top_k() -> None:
    result = runner.invoke(app, ["recommend", "data.csv", "--top-k", "10"])
    assert result.exit_code == 0
    assert "10" in result.output


def test_recommend_custom_top_k_short_flag() -> None:
    result = runner.invoke(app, ["recommend", "data.csv", "-k", "3"])
    assert result.exit_code == 0
    assert "3" in result.output
