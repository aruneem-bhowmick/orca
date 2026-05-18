"""OrcaNet CLI entry point."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="orcanet",
    help="OrcaNet — cross-domain knowledge transfer agent.",
    no_args_is_help=True,
)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to."),  # noqa: S104 — container-only; network namespace isolates the service
    port: int = typer.Option(8002, help="Port to listen on."),
    reload: bool = typer.Option(False, help="Enable auto-reload for development."),
) -> None:
    """Start the OrcaNet FastAPI server."""
    import uvicorn

    uvicorn.run(
        "orcanet.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def version() -> None:
    """Print the OrcaNet version."""
    from orcanet import __version__

    typer.echo(f"OrcaNet {__version__}")


if __name__ == "__main__":
    app()
