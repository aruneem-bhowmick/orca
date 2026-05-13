"""OrcaMind CLI entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="orcamind",
    help="OrcaMind — meta-learning engine CLI",
    no_args_is_help=True,
)

_DEFAULT_CONFIG_YAML = """\
defaults:
  - model: maml
  - dataset: openml
  - optimizer: adam

paths:
  data_dir: ${oc.env:ORCA_DATA_DIR,./data}
  model_dir: ${oc.env:ORCA_MODEL_DIR,./models}
  log_dir: ${oc.env:ORCA_LOG_DIR,./logs}
mlflow:
  tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,http://localhost:5000}
  experiment_name: orcamind
seed: 42
device: cuda
"""


def _version_callback(value: bool) -> None:
    if value:
        typer.echo("OrcaMind v1.0.0")
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
    """OrcaMind — meta-learning engine CLI."""


@app.command()
def init() -> None:
    """Initialise workspace directories, default config, and MLflow experiment."""
    typer.echo("[orcamind] init command not yet implemented — scaffold only")


@app.command()
def train(
    config: str = typer.Option(
        "config/config.yaml", "--config", "-c", help="Path to OmegaConf/Hydra config file."
    ),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of meta-training epochs."),
    device: str = typer.Option("cpu", "--device", "-d", help="Training device: cpu or cuda."),
) -> None:
    """Launch a meta-training run using the configured meta-learner and task sampler."""
    typer.echo(f"[orcamind] Starting training with config: {config}")
    typer.echo("[orcamind] train command not yet implemented — scaffold only")


@app.command()
def embed(
    dataset_path: str = typer.Argument(..., help="Path to a CSV or Parquet dataset file."),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save embedding JSON to this file."
    ),
) -> None:
    """Compute a task embedding for a dataset using StatisticalEmbedder and NeuralEmbedder."""
    typer.echo(f"[orcamind] Embedding dataset: {dataset_path}")
    typer.echo("[orcamind] embed command not yet implemented — scaffold only")


@app.command()
def recommend(
    dataset_path: str = typer.Argument(..., help="Path to a CSV or Parquet dataset file."),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of model recommendations."),
    api_url: str = typer.Option(
        "http://localhost:8000", "--api-url", help="OrcaMind API base URL."
    ),
) -> None:
    """Recommend model configurations for a dataset via the OrcaMind API."""
    typer.echo(f"[orcamind] Recommending top-{top_k} models for: {dataset_path}")
    typer.echo("[orcamind] recommend command not yet implemented — scaffold only")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
) -> None:
    """Start the OrcaMind FastAPI service with Uvicorn."""
    typer.echo(f"[orcamind] Starting API server on {host}:{port}")
    typer.echo("[orcamind] serve command not yet implemented — scaffold only")


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit server port."),
) -> None:
    """Launch the OrcaMind Streamlit dashboard."""
    typer.echo(f"[orcamind] Starting dashboard on port {port}")
    typer.echo("[orcamind] dashboard command not yet implemented — scaffold only")


if __name__ == "__main__":
    app()
