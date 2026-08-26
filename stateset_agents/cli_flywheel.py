"""``stateset-agents flywheel`` — the improvement loop, unattended.

Harvest the current generation's rare successes (best-of-N against
objective checks), train the next generation on nothing but those, measure
it, and repeat — until the score plateaus, the budget would be exceeded, or
a harvest comes back dry. The methodology is ``docs/FLYWHEEL_HEADROOM.md``
(2/12 → 10/12 for $3.32); this command is that experiment as a product.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app
from stateset_agents.core.errors import StateSetError
from stateset_agents.remote.registry import available_providers, get_executor

_echo = _cli._echo


def _load_specs(path: Path, label: str) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise typer.BadParameter(f"cannot read {label} file: {exc}") from exc
    except ValueError as exc:
        raise typer.BadParameter(f"{label} file is not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise typer.BadParameter(f"{label} file must be a non-empty JSON list")
    return data


@app.command()
def flywheel(
    base_model: str = typer.Option(
        ..., help="Hugging Face base model every generation is LoRA-tuned from."
    ),
    harvest_prompts: Path = typer.Option(
        ...,
        help=(
            "JSON file: list of {prompt, expect, forbid} specs sampled "
            "during harvest. The checks define success; they are mandatory."
        ),
    ),
    eval_prompts: Path = typer.Option(
        ...,
        help=(
            "JSON file: list of {prompt, expect, forbid} specs that score "
            "each generation. Keep disjoint from the harvest prompts."
        ),
    ),
    output_root: Path = typer.Option(
        Path("outputs/flywheel"), help="Where generations and the report land."
    ),
    initial_adapter: Path | None = typer.Option(
        None, help="Existing adapter to start from (defaults to the bare base)."
    ),
    teacher_base_model: str | None = typer.Option(
        None,
        help="Distillation: a FIXED teacher model does the harvesting while "
        "the student (--base-model) trains on its successes. Rent wisdom, "
        "deploy cheap.",
    ),
    teacher_adapter: Path | None = typer.Option(
        None, help="The teacher's adapter (checkpoint pointer dir for River)."
    ),
    generations: int = typer.Option(3, help="Maximum NEW generations to train."),
    best_of: int = typer.Option(8, help="Samples per harvest prompt."),
    temperature: float = typer.Option(0.9, help="Harvest sampling temperature."),
    target_harvest_rate: float | None = typer.Option(
        None,
        help="The rarity controller: probe a few prompts at several "
        "temperatures each generation and harvest at the one whose pass "
        "rate lands nearest this target (the measured operating window is "
        "~0.6). Overrides --temperature per generation.",
    ),
    max_cost: float | None = typer.Option(
        None,
        help=(
            "Hard dollar ceiling for the WHOLE run; each rental is refused "
            "if its worst case would break what remains."
        ),
    ),
    provider: str = typer.Option(
        "runpod", help=f"One of: {', '.join(available_providers())}."
    ),
    gpu: str | None = typer.Option(
        None, help="GPU type, in the provider's own vocabulary."
    ),
    container_disk_gb: int | None = typer.Option(
        None, help="Container disk per pod (~2.5x the checkpoint size)."
    ),
    num_epochs: int = typer.Option(3, help="Training epochs per generation."),
    algorithm: str = typer.Option(
        "sft",
        help=(
            "sft (default): rejection-sampling flywheel — imitate the "
            "winners. cispo or importance_sampling: GRPO-style RL on River "
            "— train on EVERY sample, gradient-weighted by graded reward "
            "(refusal violations punished, not just filtered). RL requires "
            "--provider river."
        ),
    ),
    rounds: int = typer.Option(
        4, help="RL only: sample->grade->train_step rounds in one session."
    ),
    repeats: int = typer.Option(
        1,
        help=(
            "Run the whole loop this many times and report the score "
            "distribution (min/mean/max). The budget is shared across "
            "repeats. Two live runs scored 7/12 and 11/12 — one run "
            "misstates the mechanism."
        ),
    ),
    dry_run: bool = typer.Option(
        False, help="Print each job's plan without renting anything."
    ),
) -> None:
    """Run the self-improvement loop until it stops earning its cost."""
    from stateset_agents.flywheel import (
        FlywheelConfig,
        run_flywheel,
        run_flywheel_repeats,
    )

    config = FlywheelConfig(
        base_model=base_model,
        harvest_prompts=_load_specs(harvest_prompts, "--harvest-prompts"),
        eval_prompts=_load_specs(eval_prompts, "--eval-prompts"),
        output_root=output_root,
        initial_adapter=initial_adapter,
        teacher_base_model=teacher_base_model,
        teacher_adapter=teacher_adapter,
        generations=generations,
        best_of=best_of,
        temperature=temperature,
        target_harvest_rate=target_harvest_rate,
        max_cost_usd=max_cost,
        gpu=gpu,
        container_disk_gb=container_disk_gb,
        num_epochs=num_epochs,
        dry_run=dry_run,
    )
    try:
        executor = get_executor(provider)
    except StateSetError as exc:
        _echo(str(exc))
        raise typer.Exit(code=1) from exc

    provider_name = provider.strip().lower()
    requested_kind = "rl" if algorithm != "sft" else "harvest"
    if not executor.supports(requested_kind):
        supported = ", ".join(sorted(executor.supported_job_kinds))
        _echo(
            f"Provider {provider_name!r} cannot run the {algorithm} flywheel "
            f"({requested_kind} jobs); supported modes: {supported}."
        )
        raise typer.Exit(code=2)

    if algorithm != "sft":
        if provider_name != "river":
            _echo("--algorithm requires --provider river (zero-infra RL).")
            raise typer.Exit(code=2)
        from stateset_agents.remote.job import RemoteJobSpec

        spec = RemoteJobSpec(
            dataset=harvest_prompts,
            base_model=base_model,
            output_dir=output_root,
            job_kind="rl",
            lora_r=16,
            learning_rate=4e-5,
            harvest={
                "adapter_dir": str(initial_adapter) if initial_adapter else None,
                "best_of": best_of,
                "temperature": temperature,
                "rounds": rounds,
                "loss_fn": algorithm,
            },
            eval_prompts=list[str | dict](_load_specs(eval_prompts, "--eval-prompts")),
            dry_run=dry_run,
        )
        _echo(
            f"RL flywheel ({algorithm}): {base_model} on river, "
            f"{rounds} round(s) x best-of {best_of}"
        )
        try:
            result = executor.wait(executor.submit(spec))
        except StateSetError as exc:
            _echo(str(exc))
            raise typer.Exit(code=1) from exc
        for line in result.logs:
            _echo(f"  {line}")
        if not result.succeeded:
            _echo("RL run failed.", err=True)
            raise typer.Exit(code=1)
        _echo(f"Report: {output_root / 'rl_report.json'}")
        return

    _echo(
        f"Flywheel: {base_model} on {provider}, up to {generations} "
        f"generation(s)"
        + (f", ceiling ${max_cost:.2f}" if max_cost is not None else "")
        + ("  [dry run]" if dry_run else "")
    )
    try:
        if repeats > 1:
            aggregate = run_flywheel_repeats(config, executor, repeats)
            _echo("")
            for run in aggregate["runs"]:
                if run.get("skipped"):
                    _echo(f"  run {run['run']}: SKIPPED — {run['skipped']}")
                else:
                    _echo(
                        f"  run {run['run']}: best {run['best_eval_passed']} "
                        f"({run['stop_reason']}, "
                        f"${run['cost_usd'] or 0:.2f})"
                    )
            _echo(
                f"Distribution over {aggregate['completed']} run(s): "
                f"min {aggregate['min']}  mean {aggregate['mean']}  "
                f"max {aggregate['max']}"
            )
            _echo(f"Total: ${aggregate['total_cost_usd']:.2f}")
            _echo(f"Report: {output_root / 'flywheel_repeats_report.json'}")
            return
        report = run_flywheel(config, executor)
    except StateSetError as exc:
        _echo(str(exc))
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        _echo(str(exc))
        raise typer.Exit(code=1) from exc

    _echo("")
    _echo(f"Stopped: {report['stop_reason']}")
    for row in report["generations"]:
        score = (
            f"{row['eval_passed']}/{row['eval_total']}"
            if row["eval_passed"] is not None
            else "—"
        )
        cost = f"${row['cost_usd']:.2f}" if row["cost_usd"] is not None else "$?"
        _echo(
            f"  gen {row['generation']}: harvested "
            f"{row['harvest_kept']}/{row['harvest_samples']}, eval {score}, "
            f"{cost}" + (f"  [{row['stopped']}]" if row["stopped"] else "")
        )
    _echo(
        f"Total: ${report['total_cost_usd']:.2f}"
        + (
            f" (+{report['unpriced_jobs']} unpriced job(s))"
            if report["unpriced_jobs"]
            else ""
        )
    )
    if report["final_adapter"]:
        _echo(f"Final adapter: {report['final_adapter']}")
        _echo(
            "Serve it:  stateset-agents serve-remote "
            f"--base-model {base_model} --adapter {report['final_adapter']}"
        )
    _echo(f"Report: {output_root / 'flywheel_report.json'}")
