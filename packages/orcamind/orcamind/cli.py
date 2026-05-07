"""OrcaMind CLI entry point."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="orcamind",
    help="OrcaMind — meta-learning engine CLI",
    no_args_is_help=True,
)


@app.command()
def train(
    config: str = typer.Option(
        "config/config.yaml", "--config", "-c", help="Path to Hydra config"
    ),
    overrides: list[str] = typer.Argument(default=None, help="Hydra overrides (key=value)"),
) -> None:
    """Launch a meta-training run."""
    typer.echo(f"[orcamind] Starting training with config: {config}")
    typer.echo("[orcamind] train command not yet implemented — scaffold only")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
) -> None:
    """Start the OrcaMind FastAPI service."""
    typer.echo(f"[orcamind] Starting API server on {host}:{port}")
    typer.echo("[orcamind] serve command not yet implemented — scaffold only")


@app.command()
def embed(
    dataset: str = typer.Argument(..., help="Path to dataset CSV"),
    output: str = typer.Option("embedding.npy", "--output", "-o", help="Output .npy file"),
) -> None:
    """Compute and save a task embedding for a dataset."""
    typer.echo(f"[orcamind] Embedding dataset: {dataset} → {output}")
    typer.echo("[orcamind] embed command not yet implemented — scaffold only")


@app.command()
def recommend(
    dataset: str = typer.Argument(..., help="Path to dataset CSV"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of recommendations"),
) -> None:
    """Recommend model configurations for a dataset."""
    typer.echo(f"[orcamind] Recommending top-{top_k} models for: {dataset}")
    typer.echo("[orcamind] recommend command not yet implemented — scaffold only")


if __name__ == "__main__":
    app()
