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
from stateset_agents.training.river_progress import DEFAULT_ZERO_UPDATE_PATIENCE

_echo = _cli._echo


benchmark_app = typer.Typer(
    add_completion=False,
    help="Run benchmarks and compare held-out agent outcomes.",
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


@benchmark_app.command("audit-refund-traces")
def audit_refund_traces_command(
    report: str = typer.Option(..., "--report", help="Native test_results.json."),
    cases: str = typer.Option(..., "--cases", help="The run's saved test.json."),
    output: str = typer.Option("trace_audit.json", "--output", "-o"),
) -> None:
    """Replay saved refund actions offline; exit 1 for inconsistent outcomes."""
    import asyncio
    import json
    from pathlib import Path
    from typing import Any

    from stateset_agents.evaluation.refund_trace import audit_refund_traces
    from stateset_agents.remote.river_rl import atomic_json

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key in trace audit input")
            value[key] = item
        return value

    try:
        destination = Path(output)
        if destination.resolve() in {Path(report).resolve(), Path(cases).resolve()}:
            raise ValueError("Trace audit output must not overwrite input evidence")
        result = asyncio.run(
            audit_refund_traces(
                json.loads(Path(report).read_text(), object_pairs_hook=unique_object),
                json.loads(Path(cases).read_text(), object_pairs_hook=unique_object),
            )
        )
        result["sources"] = {"report": report, "cases": cases}
        atomic_json(destination, result)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        _echo(f"Invalid trace audit: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    status = "PASS" if result["passed"] else "FAIL"
    _echo(
        f"{status}: {len(result['checked'])} cases replayed; {len(result['issues'])} issues. Audit: {output}"
    )
    _echo(
        "No provider requests. Replay does not verify token use, latency, or model execution."
    )
    if not result["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("compare-agents")
def compare_agents(
    baseline: list[str] = typer.Option(
        ..., "--baseline", help="Baseline test_results.json; repeat for each seed."
    ),
    candidate: list[str] = typer.Option(
        ..., "--candidate", help="Candidate test_results.json; repeat for each seed."
    ),
    output: str = typer.Option("agent_comparison.json", "--output", "-o"),
    min_seeds: int = typer.Option(3, "--min-seeds", min=1),
    min_gain: float = typer.Option(0.03, "--min-gain", min=0, max=1),
    require_significance: bool = typer.Option(
        False,
        "--require-significance",
        help="Require the exact seed sign test to pass.",
    ),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level in (0, 1)."),
    comparisons: int = typer.Option(
        1,
        "--comparisons",
        min=1,
        help="Number of planned comparisons for Bonferroni correction.",
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit 1 when improvement gates fail."
    ),
) -> None:
    """Compare matched held-out agent runs without contacting a model provider."""
    import json
    from pathlib import Path

    from stateset_agents.evaluation.agent_runs import compare_runs
    from stateset_agents.remote.river_rl import atomic_json

    try:
        destination = Path(output)
        if destination.resolve() in {Path(p).resolve() for p in baseline + candidate}:
            raise ValueError("Comparison output must not overwrite input evidence")
        before = [json.loads(Path(p).read_text()) for p in baseline]
        after = [json.loads(Path(p).read_text()) for p in candidate]
        if any(not isinstance(r, dict) for r in before + after):
            raise ValueError("Each report must be a JSON object")
        report = compare_runs(
            before,
            after,
            min_seeds=min_seeds,
            min_gain=min_gain,
            require_significance=require_significance,
            alpha=alpha,
            comparisons=comparisons,
        )
        report["sources"] = {"baseline": baseline, "candidate": candidate}
        atomic_json(destination, report)
    except (OSError, ValueError) as exc:
        _echo(f"Invalid comparison: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    status = "PASS" if report["passed"] else "FAIL"
    _echo(
        f"{status}: mean success gain {report['mean_success_gain']:+.1%}; report: {output}"
    )
    if strict and not report["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("compare-agent-study")
def compare_agent_study(
    base: list[str] = typer.Option(
        ..., "--base", help="Base test report; repeat for each seed."
    ),
    sft: list[str] = typer.Option(
        ..., "--sft", help="Reference SFT test report; repeat for each seed."
    ),
    rejection_sft: list[str] = typer.Option(
        ..., "--rejection-sft", help="Rejection SFT test report; repeat for each seed."
    ),
    rl: list[str] = typer.Option(
        ..., "--rl", help="RL test report; repeat for each seed."
    ),
    output: str = typer.Option("agent_study.json", "--output", "-o"),
    min_seeds: int = typer.Option(6, "--min-seeds", min=1),
    min_gain: float = typer.Option(0.03, "--min-gain", min=0, max=1),
    alpha: float = typer.Option(0.05, "--alpha"),
    strict: bool = typer.Option(
        False, "--strict", help="Exit 1 when study gates fail."
    ),
) -> None:
    """Compare RL against all three baselines with corrected statistical gates."""
    import json
    from pathlib import Path

    from stateset_agents.evaluation.agent_runs import compare_study
    from stateset_agents.remote.river_rl import atomic_json

    sources = {"base": base, "sft": sft, "rejection_sft": rejection_sft, "rl": rl}
    try:
        destination = Path(output)
        if destination.resolve() in {
            Path(p).resolve() for paths in sources.values() for p in paths
        }:
            raise ValueError("Study output must not overwrite input evidence")
        arms = {
            name: [json.loads(Path(p).read_text()) for p in paths]
            for name, paths in sources.items()
        }
        if any(not isinstance(r, dict) for reports in arms.values() for r in reports):
            raise ValueError("Each report must be a JSON object")
        report = compare_study(
            arms, min_seeds=min_seeds, min_gain=min_gain, alpha=alpha
        )
        report["sources"] = sources
        atomic_json(destination, report)
    except (OSError, ValueError) as exc:
        _echo(f"Invalid study: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    status = "PASS" if report["passed"] else "FAIL"
    _echo(f"{status}: four-arm learning-evidence gates; report: {output}")
    if strict and not report["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("plan-agent-study")
def plan_agent_study(
    output: str = typer.Option(..., "--output", "-o", help="Fresh study directory."),
    base_model: str = typer.Option("Qwen/Qwen3.5-9B", "--base-model"),
    seeds: str = typer.Option(
        "42,43,44,45,46,47",
        "--seeds",
        help="At least six unique comma-separated seeds.",
    ),
    train_count: int = typer.Option(256, "--train-count"),
    validation_count: int = typer.Option(64, "--validation-count"),
    test_count: int = typer.Option(128, "--test-count"),
    steps: int = typer.Option(20, "--steps"),
    concurrency: int = typer.Option(8, "--concurrency"),
    sft_epochs: int = typer.Option(3, "--sft-epochs"),
    sft_learning_rate: float = typer.Option(2e-5, "--sft-learning-rate"),
    rl_learning_rate: float = typer.Option(1e-5, "--rl-learning-rate"),
    rollout_token_budget: int | None = typer.Option(
        None,
        "--rollout-token-budget",
        help="Per-run conservative rollout output-token admission cap; not a billing cap.",
    ),
    zero_update_patience: int = typer.Option(
        DEFAULT_ZERO_UPDATE_PATIENCE,
        "--zero-update-patience",
        help="Stop RL after consecutive observed skipped optimizer updates; 0 disables.",
    ),
) -> None:
    """Freeze a four-arm campaign with commands and split hashes; runs no jobs."""
    from pathlib import Path

    from stateset_agents.evaluation.agent_study import StudyConfig, prepare_study

    try:
        config = StudyConfig(
            base_model=base_model,
            seeds=tuple(int(s.strip()) for s in seeds.split(",")),
            train_count=train_count,
            validation_count=validation_count,
            test_count=test_count,
            steps=steps,
            concurrency=concurrency,
            sft_epochs=sft_epochs,
            sft_learning_rate=sft_learning_rate,
            rl_learning_rate=rl_learning_rate,
            rollout_token_budget=rollout_token_budget,
            zero_update_patience=zero_update_patience,
        )
        plan = prepare_study(Path(output), config)
    except (OSError, ValueError) as exc:
        _echo(f"Cannot plan study: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    _echo(
        f"Prepared {len(plan['stages'])} stages: {output}/study_plan.json. No jobs executed; paid stages require a provider budget."
    )


@benchmark_app.command("prepare-agent-study-data")
def prepare_agent_study_data(
    study_dir: str = typer.Option(..., "--study-dir"),
) -> None:
    """Prepare every planned dataset offline, preserving existing evidence."""
    import asyncio
    from pathlib import Path

    from stateset_agents.evaluation.study_preflight import prepare_study_data
    from stateset_agents.remote.river_rl import atomic_json

    directory = Path(study_dir)
    try:
        report = asyncio.run(prepare_study_data(directory))
        atomic_json(directory / "data_preparation.json", report)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        _echo(f"Cannot prepare study data: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    created = sum(row["status"] == "created" for row in report["datasets"])
    reused = len(report["datasets"]) - created
    _echo(f"Study datasets: {created} created, {reused} reused. No provider requests.")


@benchmark_app.command("preflight-agent-study")
def preflight_agent_study(
    study_dir: str = typer.Option(..., "--study-dir"),
    strict: bool = typer.Option(
        False, "--strict", help="Exit 1 for unmet local prerequisites."
    ),
) -> None:
    """Check local data, Python, SDK, and credential presence without provider calls."""
    from pathlib import Path

    from stateset_agents.evaluation.study_preflight import preflight_study
    from stateset_agents.remote.river_rl import atomic_json

    directory = Path(study_dir)
    try:
        report = preflight_study(directory)
        atomic_json(directory / "study_preflight.json", report)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        _echo(f"Cannot inspect study: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    status = "PASS" if report["local_checks_passed"] else "FAIL"
    _echo(
        f"Local prerequisites: {status}. Report: {directory / 'study_preflight.json'}"
    )
    for issue in report["issues"]:
        _echo(f"- {issue}")
    _echo(
        "Provider access, model availability and costs remain unverified. No jobs executed."
    )
    if strict and not report["local_checks_passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("audit-agent-study")
def audit_agent_study(
    study_dir: str = typer.Option(
        ...,
        "--study-dir",
        help="Directory containing study_plan.json and run artifacts.",
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit 1 when evidence is incomplete or gates fail."
    ),
) -> None:
    """Audit planned run provenance, validation selection and all three comparisons."""
    import asyncio
    from pathlib import Path

    from stateset_agents.evaluation.agent_study import audit_study
    from stateset_agents.remote.river_rl import atomic_json

    try:
        directory = Path(study_dir)
        report = asyncio.run(audit_study(directory))
        atomic_json(directory / "study_audit.json", report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _echo(f"Invalid study: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    status = "PASS" if report["passed"] else "FAIL"
    _echo(
        f"{status}: {len(report['issues'])} provenance issues; report: {study_dir}/study_audit.json"
    )
    if strict and not report["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("prepare-refund-data")
def prepare_refund_data_command(
    output: str = typer.Option(..., "--output", "-o", help="Fresh dataset directory."),
    seed: int = typer.Option(42, "--seed", min=0),
    train_count: int = typer.Option(256, "--train-count", min=1),
    validation_count: int = typer.Option(64, "--validation-count", min=1),
    test_count: int = typer.Option(128, "--test-count", min=1),
) -> None:
    """Export verified reference-policy SFT chats and held-out refund cases offline."""
    import asyncio
    from pathlib import Path

    from stateset_agents.data.refund_demonstrations import prepare_refund_data

    try:
        manifest = asyncio.run(
            prepare_refund_data(
                Path(output),
                seed=seed,
                train_count=train_count,
                validation_count=validation_count,
                test_count=test_count,
            )
        )
    except (OSError, ValueError) as exc:
        _echo(f"Cannot prepare refund data: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    _echo(
        f"Prepared {manifest['selected']} verified reference demonstrations: {output}/train.jsonl"
    )


@benchmark_app.command("filter-refund-data")
def filter_refund_data_command(
    data_dir: str = typer.Option(
        ..., "--data-dir", help="Prepared reference dataset directory."
    ),
    candidates: str = typer.Option(
        ..., "--candidates", help="training_candidates.json from a collection run."
    ),
    output: str = typer.Option(
        ..., "--output", "-o", help="Fresh filtered dataset directory."
    ),
) -> None:
    """Replay model candidates; export only successful training-case conversations."""
    import asyncio
    from pathlib import Path

    from stateset_agents.data.refund_demonstrations import filter_refund_candidates

    try:
        manifest = asyncio.run(
            filter_refund_candidates(Path(data_dir), Path(candidates), Path(output))
        )
    except (OSError, ValueError) as exc:
        _echo(f"Cannot filter refund data: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    _echo(
        f"Selected {manifest['selected']} cases; rejected {manifest['rejected']} candidates. Audit: {output}/replay_audit.json"
    )
    if not manifest["selected"]:
        raise typer.Exit(code=1)
