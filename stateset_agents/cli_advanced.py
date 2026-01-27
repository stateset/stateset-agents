"""
Advanced CLI commands for StateSet Agents.

Provides debugging, profiling, validation, and REPL support.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.tree import Tree

console = Console()
app = typer.Typer(help="Advanced StateSet Agents CLI commands")


@app.command()
def debug(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
    agent_path: Optional[str] = typer.Option(
        None, "--agent", "-a", help="Path to saved agent"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Launch interactive debug REPL for agents."""
    try:
        from IPython import embed
        console.print("[bold green]🚀 Starting StateSet Agents Debug REPL[/bold green]")
        console.print("Available variables:")
        console.print("  - console: Rich console")
        console.print("  - debug_config: Loaded config (if provided)")
        console.print("  - agent: Loaded agent (if provided)")
        console.print("")

        # Load config if provided
        debug_config = None
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                debug_config = json.load(f)
            console.print(f"[blue]Loaded config from {config_path}[/blue]")

        # Load agent if provided
        agent = None
        if agent_path and Path(agent_path).exists():
            console.print(f"[blue]Loading agent from {agent_path}...[/blue]")
            # Agent loading logic here

        # Create namespace for REPL
        namespace = {
            "console": console,
            "debug_config": debug_config,
            "agent": agent,
        }

        embed(using=False, colors="neutral", banner1="", user_ns=namespace)

    except ImportError:
        console.print("[red]IPython not installed. Install with 'pip install ipython'[/red]")
        sys.exit(1)


@app.command()
def profile(
    trainer: str = typer.Option("gspo", "--trainer", "-t", help="Trainer type to profile"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
    duration: int = typer.Option(60, "--duration", "-d", help="Profiling duration in seconds"),
    memory: bool = typer.Option(True, "--memory/--no-memory", help="Profile memory usage"),
):
    """Profile training performance."""
    console.print(f"[bold green]🔍 Profiling {trainer} trainer[/bold green]")

    import cProfile
    import io
    import pstats
    from datetime import datetime

    # Create profiler
    profiler = cProfile.Profile()

    # Run profiling
    console.print(f"[blue]Profiling for {duration} seconds...[/blue]")
    profiler.enable()

    # Simulate or run actual training
    # In real implementation, this would run training
    import time
    time.sleep(min(duration, 10))  # Cap at 10s for demo

    profiler.disable()

    # Generate report
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)

    console.print(Panel(s.getvalue(), title="Profile Results"))

    # Memory profiling
    if memory:
        try:
            import psutil
            import torch

            process = psutil.Process()
            mem_info = process.memory_info()

            table = Table(title="Memory Usage")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("RSS", f"{mem_info.rss / 1024 / 1024:.1f} MB")
            table.add_row("VMS", f"{mem_info.vms / 1024 / 1024:.1f} MB")

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    table.add_row(f"GPU {i} Allocated", f"{allocated:.2f} GB")
                    table.add_row(f"GPU {i} Reserved", f"{reserved:.2f} GB")

            console.print(table)
        except ImportError:
            console.print("[yellow]psutil not installed. Install with 'pip install psutil'[/yellow]")


@app.command()
def doctor(
    training: bool = typer.Option(
        False, "--training", help="Check training readiness"
    ),
    fix: bool = typer.Option(
        False, "--fix", help="Attempt to fix issues automatically"
    ),
):
    """Run diagnostics and check system health."""
    console.print("[bold green]🔧 Running StateSet Agents Doctor[/bold green]")

    from stateset_agents.config.validation import PreFlightChecker

    # System requirements check
    console.print("\n[bold]System Requirements:[/bold]")
    result = PreFlightChecker.check_system_requirements()

    table = Table()
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Message", style="white")

    # Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    table.add_row("Python Version", "[green]✓" if sys.version_info >= (3, 8) else "[red]✗", py_version)

    # CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            table.add_row("CUDA", f"[green]✓", f"{torch.cuda.device_count()} GPU(s)")
        else:
            table.add_row("CUDA", "[yellow]⚠", "Not available (CPU only)")
    except ImportError:
        table.add_row("PyTorch", "[red]✗", "Not installed")

    # Optional dependencies
    for pkg, name in [
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("wandb", "Weights & Biases"),
        ("vllm", "vLLM"),
    ]:
        try:
            __import__(pkg)
            table.add_row(name, "[green]✓", "Installed")
        except ImportError:
            table.add_row(name, "[yellow]⚠", "Not installed")

    console.print(table)

    # Training readiness
    if training:
        console.print("\n[bold]Training Readiness:[/bold]")
        console.print("[yellow]Configure with --config to run full check[/yellow]")

    # Warnings and errors
    if result.errors:
        console.print("\n[red]Errors found:[/red]")
        for error in result.errors:
            console.print(f"  [red]✗[/red] {error.field}: {error.message}")

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  [yellow]⚠[/yellow] {warning.field}: {warning.message}")

    if not result.errors and not result.warnings:
        console.print("\n[bold green]✓ All checks passed![/bold green]")


@app.command()
def validate(
    config_path: str = typer.Argument(..., help="Path to config file"),
    config_type: str = typer.Option(
        "training",
        "--type",
        "-t",
        help="Config type (training, agent, environment)"
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Enable strict validation"
    ),
):
    """Validate configuration file."""
    console.print(f"[bold green]📋 Validating {config_type} config[/bold green]")

    # Load config
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    with open(path) as f:
        if path.suffix == ".json":
            config = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                config = yaml.safe_load(f)
            except ImportError:
                console.print("[red]PyYAML not installed. Install with 'pip install pyyaml'[/red]")
                raise typer.Exit(1)
        else:
            console.print(f"[red]Unsupported config format: {path.suffix}[/red]")
            raise typer.Exit(1)

    # Validate based on type
    from stateset_agents.config.validation import ConfigValidator

    validator = ConfigValidator()

    if config_type == "training":
        result = validator.validate_training_config(config, strict=strict)
    elif config_type == "agent":
        result = validator.validate_agent_config(config)
    elif config_type == "environment":
        result = validator.validate_environment_config(config)
    else:
        console.print(f"[red]Unknown config type: {config_type}[/red]")
        raise typer.Exit(1)

    # Display results
    if result.is_valid:
        console.print("[bold green]✓ Configuration is valid[/bold green]")
    else:
        console.print("[bold red]✗ Configuration has errors[/bold red]")

    if result.errors:
        table = Table(title="Errors")
        table.add_column("Field", style="red")
        table.add_column("Message", style="white")
        for error in result.errors:
            table.add_row(error.field, error.message)
        console.print(table)

    if result.warnings:
        table = Table(title="Warnings")
        table.add_column("Field", style="yellow")
        table.add_column("Message", style="white")
        for warning in result.warnings:
            table.add_row(warning.field, warning.message)
        console.print(table)

    raise typer.Exit(1 if result.errors else 0)


@app.command()
def progress(
    checkpoint_dir: str = typer.Argument(..., help="Directory with checkpoints"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for updates"),
    interval: int = typer.Option(5, "--interval", "-i", help="Watch interval in seconds"),
):
    """View training progress from checkpoints."""
    import time

    def display_progress(path: Path):
        """Display progress from checkpoint directory."""
        checkpoints = sorted(path.glob("checkpoint-*"))

        if not checkpoints:
            console.print("[yellow]No checkpoints found[/yellow]")
            return

        table = Table(title="Training Progress")
        table.add_column("Checkpoint", style="cyan")
        table.add_column("Global Step", style="green")
        table.add_column("Modified", style="white")

        for ckpt in checkpoints:
            # Try to load metadata
            metadata_file = ckpt / "trainer_state.json"
            global_step = "?"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        state = json.load(f)
                        global_step = str(state.get("global_step", "?"))
                except Exception:
                    pass

            mtime = ckpt.stat().st_mtime
            from datetime import datetime
            modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            table.add_row(ckpt.name, global_step, modified)

        console.print(table)
        console.print(f"\nTotal checkpoints: {len(checkpoints)}")

    path = Path(checkpoint_dir)

    if watch:
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                display_progress(path)
                time.sleep(interval)
                console.clear()
    else:
        display_progress(path)


@app.command()
def tree(
    checkpoint_dir: str = typer.Argument(..., help="Directory with checkpoints"),
    max_depth: int = typer.Option(3, "--depth", "-d", help="Maximum depth"),
):
    """Display checkpoint directory tree."""
    console.print("[bold green]🌳 Checkpoint Tree[/bold green]")

    path = Path(checkpoint_dir)
    if not path.exists():
        console.print(f"[red]Directory not found: {checkpoint_dir}[/red]")
        raise typer.Exit(1)

    def build_tree(p: Path, tree: Tree, depth: int = 0):
        if depth >= max_depth:
            return

        for item in sorted(p.iterdir()):
            if item.is_dir():
                subtree = tree.add(f"[bold]{item.name}[/bold]")
                build_tree(item, subtree, depth + 1)
            else:
                size = item.stat().st_size / 1024 / 1024
                tree.add(f"{item.name} ({size:.1f} MB)")

    tree_root = Tree(f"[bold]{path.name}[/bold]")
    build_tree(path, tree_root)
    console.print(tree_root)


def run_advanced_cli():
    """Entry point for advanced CLI."""
    app()
