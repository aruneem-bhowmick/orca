"""OrcaLab CLI entry point."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="orcalab",
    help="OrcaLab — hyperparameter search and experiment orchestration CLI",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo("OrcaLab v0.1.0")
        raise typer.Exit()


@app.callback()
def callback(
    version: bool = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """OrcaLab — hyperparameter search and experiment orchestration CLI."""


@app.command()
def init() -> None:
    """Initialise OrcaLab workspace directories and default config."""
    typer.echo("[orcalab] init: not yet implemented")


@app.command()
def sweep(
    task_id: str = typer.Argument(..., help="OrcaMind task ID to sweep."),
    n_trials: int = typer.Option(100, "--n-trials", "-n", help="Number of Optuna trials."),
    strategy: str = typer.Option(
        "tpe", "--strategy", "-s", help="Search strategy (tpe, cma, random)."
    ),
) -> None:
    """Run a hyperparameter sweep for the given task."""
    typer.echo("[orcalab] sweep: not yet implemented")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8001, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
) -> None:
    """Start the OrcaLab FastAPI service with Uvicorn."""
    typer.echo("[orcalab] serve: not yet implemented")


@app.command()
def dashboard(
    port: int = typer.Option(8502, "--port", "-p", help="Streamlit server port."),
) -> None:
    """Launch the OrcaLab Streamlit dashboard."""
    typer.echo("[orcalab] dashboard: not yet implemented")


if __name__ == "__main__":
    app()
