import sys
import typing as t
import importlib
from dataclasses import dataclass

import typer

app = typer.Typer(add_completion=False, help="StateSet Agents CLI")


def _echo(s: str) -> None:
    typer.echo(s)


@app.callback()
def main_callback() -> None:
    """StateSet Agents command-line interface."""
    # No-op. Subcommands implement functionality.
    return None


@app.command()
def version() -> None:
    """Show installed version and basic environment info."""
    try:
        from stateset_agents import __version__
    except Exception:
        __version__ = "unknown"

    _echo(f"stateset-agents version: {__version__}")
    _echo(f"python: {sys.version.split()[0]}")


@app.command()
def train(
    config: t.Optional[str] = typer.Option(
        None,
        help="Path to a training config file (optional).",
    ),
    episodes: int = typer.Option(
        0,
        help="Number of episodes (optional hint for examples).",
    ),
    dry_run: bool = typer.Option(
        True,
        help="If true, only validates environment and shows guidance.",
    ),
) -> None:
    """Guide or launch training (scaffold)."""
    # Soft-import training to avoid heavy deps errors.
    try:
        import stateset_agents.training as training
    except Exception as e:
        _echo("Training modules unavailable. Install dev/extras and try again.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2)

    if dry_run:
        _echo("Dry-run: environment looks OK. To run a full example:")
        _echo("  python examples/quick_start.py")
        _echo("Or use TRL GRPO (if installed):")
        _echo("  python examples/train_with_trl_grpo.py")
        return

    _echo("Direct trainer wiring is not yet exposed via CLI.")
    _echo("Please use examples or programmatic APIs:")
    _echo("  from stateset_agents.training import MultiTurnGRPOTrainer, TrainingConfig")
    raise typer.Exit(code=0)


@app.command()
def evaluate(
    dataset: t.Optional[str] = typer.Option(None, help="Evaluation dataset path (optional)."),
    dry_run: bool = typer.Option(True, help="Validate environment only."),
) -> None:
    """Evaluate a trained agent (scaffold)."""
    try:
        import stateset_agents.training as _  # noqa: F401
    except Exception as e:
        _echo("Evaluation components unavailable. Install dev/extras and try again.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2)

    _echo("Evaluation CLI not fully wired. See examples and docs.")
    raise typer.Exit(code=0)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8001, help="Bind port"),
) -> None:
    """Run the Ultimate GRPO Service (if FastAPI/uvicorn available)."""
    try:
        # Import via package namespace to resolve relative imports in module.
        svc = importlib.import_module("stateset_agents.api.ultimate_grpo_service")
    except Exception as e:
        _echo("API service unavailable. Install 'api' extras (fastapi, uvicorn).")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2)

    # If module exposes 'app' and FastAPI is present, run uvicorn.
    app_obj = getattr(svc, "app", None)
    if app_obj is None:
        _echo("Service module loaded but FastAPI not available.")
        raise typer.Exit(code=2)

    try:
        import uvicorn  # type: ignore
    except Exception:
        _echo("uvicorn not installed. Try: pip install 'stateset-agents[api]'")
        raise typer.Exit(code=2)

    _echo("Starting StateSet Agents service...")
    uvicorn.run(app_obj, host=host, port=port, log_level="info")


def run() -> None:
    app()


if __name__ == "__main__":
    run()

