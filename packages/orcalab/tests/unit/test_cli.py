"""Tests for the OrcaLab CLI stub — all four commands plus --version and --help."""

from __future__ import annotations

from typer.testing import CliRunner

from orcalab.cli import app

runner = CliRunner()


class TestVersion:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_output_contains_version_number(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert "0.1.0" in result.output

    def test_output_contains_orcalab(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert "OrcaLab" in result.output


class TestHelp:
    def test_root_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_root_help_lists_init(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "init" in result.output

    def test_root_help_lists_sweep(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "sweep" in result.output

    def test_root_help_lists_serve(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "serve" in result.output

    def test_root_help_lists_dashboard(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "dashboard" in result.output

    def test_sweep_help_shows_n_trials_flag(self) -> None:
        result = runner.invoke(app, ["sweep", "--help"])
        assert "--n-trials" in result.output

    def test_sweep_help_shows_strategy_flag(self) -> None:
        result = runner.invoke(app, ["sweep", "--help"])
        assert "--strategy" in result.output

    def test_serve_help_shows_port_flag(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert "--port" in result.output

    def test_serve_help_shows_reload_flag(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert "--reload" in result.output


class TestInit:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0

    def test_output_contains_not_yet_implemented(self) -> None:
        result = runner.invoke(app, ["init"])
        assert "not yet implemented" in result.output


class TestSweep:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["sweep", "task-123"])
        assert result.exit_code == 0

    def test_no_task_id_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["sweep"])
        assert result.exit_code != 0

    def test_output_contains_not_yet_implemented(self) -> None:
        result = runner.invoke(app, ["sweep", "task-123"])
        assert "not yet implemented" in result.output

    def test_accepts_n_trials_flag(self) -> None:
        result = runner.invoke(app, ["sweep", "task-123", "--n-trials", "50"])
        assert result.exit_code == 0

    def test_accepts_strategy_flag(self) -> None:
        result = runner.invoke(app, ["sweep", "task-123", "--strategy", "cma"])
        assert result.exit_code == 0


class TestServe:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0

    def test_output_contains_not_yet_implemented(self) -> None:
        result = runner.invoke(app, ["serve"])
        assert "not yet implemented" in result.output


class TestDashboard:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["dashboard"])
        assert result.exit_code == 0

    def test_output_contains_not_yet_implemented(self) -> None:
        result = runner.invoke(app, ["dashboard"])
        assert "not yet implemented" in result.output
