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
    for directory in ("data", "models", "logs"):
        Path(directory).mkdir(parents=True, exist_ok=True)
        typer.echo(f"[orcamind] Created directory: {directory}/")

    config_path = Path("config/config.yaml")
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_DEFAULT_CONFIG_YAML)
        typer.echo(f"[orcamind] Wrote default config: {config_path}")
    else:
        typer.echo(f"[orcamind] Config already exists: {config_path}")

    try:
        import mlflow
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(config_path))
        experiment_name = cfg.mlflow.experiment_name
        tracking_uri = cfg.mlflow.tracking_uri
        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(str(experiment_name))
        typer.echo(f"[orcamind] MLflow experiment ready: {experiment_name}")
    except Exception as exc:
        typer.echo(f"[orcamind] MLflow init skipped: {exc}", err=True)

    typer.echo("[orcamind] Workspace initialized.")


@app.command()
def train(
    config: str = typer.Option(
        "config/config.yaml", "--config", "-c", help="Path to OmegaConf/Hydra config file."
    ),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of meta-training epochs."),
    device: str = typer.Option("cpu", "--device", "-d", help="Training device: cpu or cuda."),
) -> None:
    """Launch a meta-training run using the configured meta-learner and task sampler."""
    typer.echo(f"[orcamind] Loading config: {config}")
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config)
    except Exception as exc:
        typer.echo(f"[orcamind] Failed to load config: {exc}", err=True)
        raise typer.Exit(code=1)

    import torch
    import torch.nn as nn

    from orcamind.core.base import Task
    from orcamind.core.maml import MAML
    from orcamind.training.meta_trainer import MetaTrainer
    from orcamind.training.task_sampler import UniformTaskSampler

    typer.echo(f"[orcamind] Device: {device} | Epochs: {epochs}")

    base_model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
    meta_learner = MAML(model=base_model, inner_lr=0.01, outer_lr=0.001, inner_steps=5)
    sampler = UniformTaskSampler()
    seed_task = Task(
        support_x=torch.zeros(5, 10),
        support_y=torch.zeros(5, dtype=torch.long),
        query_x=torch.zeros(5, 10),
        query_y=torch.zeros(5, dtype=torch.long),
    )

    trainer_module = MetaTrainer(
        meta_learner=meta_learner,
        sampler=sampler,
        task_pool=[seed_task],
        batch_size=1,
    )

    accelerator = "gpu" if device == "cuda" and torch.cuda.is_available() else "cpu"
    pl_trainer = MetaTrainer.make_trainer(
        max_epochs=epochs,
        enable_checkpointing=False,
        logger=False,
        accelerator=accelerator,
        enable_progress_bar=False,
    )

    typer.echo("[orcamind] Starting meta-training…")
    pl_trainer.fit(trainer_module)

    try:
        model_dir = Path(str(OmegaConf.select(cfg, "paths.model_dir", default="models")))
    except Exception:
        model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "orcamind_final.pt"
    torch.save(meta_learner.model.state_dict(), str(checkpoint_path))
    typer.echo(f"[orcamind] Checkpoint saved: {checkpoint_path}")


@app.command()
def embed(
    dataset_path: str = typer.Argument(..., help="Path to a CSV or Parquet dataset file."),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save embedding JSON to this file."
    ),
) -> None:
    """Compute a task embedding for a dataset using StatisticalEmbedder and NeuralEmbedder."""
    import pandas as pd

    from orcamind.embedders.neural import NeuralEmbedder
    from orcamind.embedders.statistical import StatisticalEmbedder

    path = Path(dataset_path)
    if not path.exists():
        typer.echo(f"[orcamind] Dataset not found: {dataset_path}", err=True)
        raise typer.Exit(code=1)

    try:
        df = (
            pd.read_parquet(path)
            if path.suffix.lower() in {".parquet", ".pq"}
            else pd.read_csv(path)
        )
    except Exception as exc:
        typer.echo(f"[orcamind] Failed to load dataset: {exc}", err=True)
        raise typer.Exit(code=1)

    stat_embedder = StatisticalEmbedder()
    stat_vec = stat_embedder.embed(df)

    neural_embedder = NeuralEmbedder(input_dim=stat_embedder.embedding_dim)
    neural_vec = neural_embedder.embed(df)

    result = {
        "dataset": dataset_path,
        "statistical": stat_vec.tolist(),
        "neural": neural_vec.tolist(),
    }

    output_json = json.dumps(result, indent=2)
    typer.echo(output_json)

    if output:
        Path(output).write_text(output_json)
        typer.echo(f"[orcamind] Embedding saved: {output}", err=True)


@app.command()
def recommend(
    dataset_path: str = typer.Argument(..., help="Path to a CSV or Parquet dataset file."),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of model recommendations."),
    api_url: str = typer.Option(
        "http://localhost:8000", "--api-url", help="OrcaMind API base URL."
    ),
) -> None:
    """Recommend model configurations for a dataset via the OrcaMind API."""
    import httpx
    import pandas as pd
    from rich.console import Console
    from rich.table import Table

    from orcamind.embedders.statistical import StatisticalEmbedder

    path = Path(dataset_path)
    if not path.exists():
        typer.echo(f"[orcamind] Dataset not found: {dataset_path}", err=True)
        raise typer.Exit(code=1)

    try:
        df = (
            pd.read_parquet(path)
            if path.suffix.lower() in {".parquet", ".pq"}
            else pd.read_csv(path)
        )
    except Exception as exc:
        typer.echo(f"[orcamind] Failed to load dataset: {exc}", err=True)
        raise typer.Exit(code=1)

    embedding = StatisticalEmbedder().embed(df).tolist()
    payload = {"task_embedding": embedding, "top_k": top_k}

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{api_url}/api/v1/recommend-model", json=payload)
            resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        typer.echo(f"[orcamind] API error: {exc}", err=True)
        raise typer.Exit(code=1)

    recommendations = data if isinstance(data, list) else data.get("recommendations", [])
    console = Console()
    table = Table(
        title=f"Top-{top_k} Model Recommendations",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", style="bold cyan", width=6)
    table.add_column("Model", style="bold")
    table.add_column("Confidence", style="green")
    table.add_column("Algorithm", style="yellow")

    for rank, rec in enumerate(recommendations[:top_k], start=1):
        table.add_row(
            str(rank),
            rec.get("model_name", "—"),
            f"{rec.get('confidence', 0.0):.3f}",
            rec.get("algorithm", "—"),
        )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
) -> None:
    """Start the OrcaMind FastAPI service with Uvicorn."""
    import uvicorn

    typer.echo(f"[orcamind] Starting API server on {host}:{port}")
    uvicorn.run(
        "orcamind.api.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit server port."),
) -> None:
    """Launch the OrcaMind Streamlit dashboard."""
    dashboard_script = Path(__file__).parent / "dashboard" / "app.py"
    typer.echo(f"[orcamind] Starting dashboard on port {port}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_script),
            "--server.port",
            str(port),
        ],
        check=True,
    )


if __name__ == "__main__":
    app()
