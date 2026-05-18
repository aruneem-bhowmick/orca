"""Smoke tests for the orcanet CLI entry point."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from orcanet.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    """Provide an isolated CLI test runner per test."""
    return CliRunner()


def test_version_command(runner: CliRunner) -> None:
    """version command prints the package version."""
    from orcanet import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_exits_cleanly(runner: CliRunner) -> None:
    """Invoking with --help exits 0 and mentions the app description."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "OrcaNet" in result.output


def test_serve_help(runner: CliRunner) -> None:
    """serve --help shows host, port, and reload options."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--reload" in result.output


def test_no_args_shows_help(runner: CliRunner) -> None:
    """Running with no arguments exits 0 and shows help (no_args_is_help=True)."""
    result = runner.invoke(app, [])
    # typer no_args_is_help=True exits with code 0 and prints help text
    assert result.exit_code == 0
    assert "version" in result.output
    assert "serve" in result.output
