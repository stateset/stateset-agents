"""Benchmark subcommand group subcommands for the StateSet Agents CLI.

Split out of stateset_agents/cli.py. Each command attaches to the parent
Typer app exported by cli; helpers _echo, _load_config, etc. are
re-bound locally for readability. Helpers that tests patch on
stateset_agents.cli (_collect_dependency_status, _collect_import_status)
are looked up via late binding through the _cli module reference so the
patches still propagate.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app

_echo = _cli._echo


benchmark_app = typer.Typer(
    add_completion=False,
    help="Run and aggregate Phase 0 / whitepaper-v1 benchmarks.",
)


@benchmark_app.command("smoke")
def benchmark_smoke() -> None:
    """Quick end-to-end smoke test of the GSM8K benchmark pipeline (no training).

    Verifies that the dataset loads, answers parse, seeds initialize, and the
    runner is importable. Takes about 6 seconds; needs no GPU.
    """
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_phase0_benchmark.py"
    if not script.exists():
        _echo(f"Benchmark script not found at {script}", err=True)
        raise typer.Exit(code=1)

    output_path = Path(tempfile.gettempdir()) / "stateset_smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--trainer",
            "gspo",
            "--smoke-test",
            "--output",
            str(output_path),
        ],
        check=False,
    )
    raise typer.Exit(code=result.returncode)


@benchmark_app.command("phase0")
def benchmark_phase0(
    trainer: str = typer.Option(
        "gspo", "--trainer", "-t", help="Trainer to benchmark: grpo, gspo, dapo."
    ),
    model: str = typer.Option("Qwen/Qwen3.5-0.8B", "--model", "-m"),
    seed: int = typer.Option(42, "--seed", "-s"),
    output: str = typer.Option(
        "benchmark_results/whitepaper_v1/run.json",
        "--output",
        "-o",
        help="Path for the JSON result file.",
    ),
    num_train_examples: int = typer.Option(200, "--num-train-examples"),
    num_eval_examples: int = typer.Option(100, "--num-eval-examples"),
) -> None:
    """Run a single Phase 0 benchmark and emit a schema-compliant JSON result.

    The result file conforms to benchmark_results/SCHEMA.md and is suitable
    for aggregation via `stateset-agents benchmark aggregate`.
    """
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_phase0_benchmark.py"
    if not script.exists():
        _echo(f"Benchmark script not found at {script}", err=True)
        raise typer.Exit(code=1)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--trainer",
            trainer,
            "--model",
            model,
            "--seed",
            str(seed),
            "--output",
            output,
            "--num-train-examples",
            str(num_train_examples),
            "--num-eval-examples",
            str(num_eval_examples),
        ],
        check=False,
    )
    raise typer.Exit(code=result.returncode)


@benchmark_app.command("plot")
def benchmark_plot(
    results_dir: str = typer.Option(
        "benchmark_results/whitepaper_v1",
        "--results-dir",
        "-d",
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o"),
    no_matplotlib: bool = typer.Option(
        False,
        "--no-matplotlib",
        help="Skip PNG figures; emit only text_plots.md.",
    ),
) -> None:
    """Generate publication figures from aggregated benchmark results.

    Reads ``summary.csv`` from the results directory and writes two PNGs plus
    a text-table fallback. Run ``aggregate`` first to produce the CSV.
    """
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_phase0_results.py"
    if not script.exists():
        _echo(f"Plot script not found at {script}", err=True)
        raise typer.Exit(code=1)

    cmd = [sys.executable, str(script), "--results-dir", results_dir]
    if output_dir:
        cmd += ["--output-dir", output_dir]
    if no_matplotlib:
        cmd += ["--no-matplotlib"]
    result = subprocess.run(cmd, check=False)
    raise typer.Exit(code=result.returncode)


@benchmark_app.command("aggregate")
def benchmark_aggregate(
    results_dir: str = typer.Option(
        "benchmark_results/whitepaper_v1",
        "--results-dir",
        "-d",
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit non-zero if any (trainer, model) group fails publication gates.",
    ),
) -> None:
    """Aggregate all *.json results in a directory into summary.md + summary.csv.

    The publication gates (3 seeds, σ < 0.1, +0.03 improvement) are defined in
    benchmark_results/SCHEMA.md. Use --strict to fail CI on any gate violation.
    """
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "aggregate_phase0_results.py"
    )
    if not script.exists():
        _echo(f"Aggregate script not found at {script}", err=True)
        raise typer.Exit(code=1)

    cmd = [sys.executable, str(script), "--results-dir", results_dir]
    if output_dir:
        cmd += ["--output-dir", output_dir]
    if strict:
        cmd += ["--strict"]
    result = subprocess.run(cmd, check=False)
    raise typer.Exit(code=result.returncode)


app.add_typer(benchmark_app, name="benchmark")
