"""``stateset-agents flywheel`` — the improvement loop, unattended.

Harvest the current generation's rare successes (best-of-N against
objective checks), train the next generation on nothing but those, measure
it, and repeat — until the score plateaus, the budget would be exceeded, or
a harvest comes back dry. The methodology is ``docs/FLYWHEEL_HEADROOM.md``
(2/12 → 10/12 for $3.32); this command is that experiment as a product.
"""

from __future__ import annotations

import json
import math
import statistics
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
    ctx: typer.Context,
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
    temperature: float | None = typer.Option(
        None, help="Sampling temperature (SFT: 0.9; RL: 1.0)."
    ),
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
    learning_rate: float = typer.Option(4e-5, help="RL optimizer learning rate."),
    lora_r: int = typer.Option(16, help="RL LoRA rank."),
    seed: int = typer.Option(0, help="RL seed; incremented for each repeat."),
    top_p: float = typer.Option(1.0, help="RL nucleus sampling probability."),
    max_new_tokens: int = typer.Option(300, help="RL generation limit per turn."),
    eval_max_new_tokens: int = typer.Option(
        300, help="RL validation generation limit."
    ),
    normalization: str = typer.Option(
        "token", help="RL weighting: token, sequence, or sum."
    ),
    microbatch_size: int = typer.Option(
        8, help="RL transport batch size; one optimizer update per round."
    ),
    truncation: str = typer.Option(
        "drop_group", help="RL incomplete groups: drop_group or error."
    ),
    eps_max: float = typer.Option(6.0, help="CISPO importance-ratio cap."),
    clip_low: float = typer.Option(0.2, help="PPO lower clipping distance."),
    clip_high: float = typer.Option(0.2, help="PPO upper clipping distance."),
    grad_clip_norm: float = typer.Option(1.0, help="RL gradient norm limit."),
    max_generated_tokens: int | None = typer.Option(
        None,
        help="Hard generated-token ceiling across RL repeats, including validation and retries.",
    ),
    resume: bool = typer.Option(
        False, help="Resume one RL run from committed optimizer state."
    ),
    reward: str | None = typer.Option(
        None, help="StateSet domain reward for RL training and validation."
    ),
    reward_threshold: float = typer.Option(0.7, help="Pass threshold for --reward."),
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

    if algorithm not in {"sft", "cispo", "ppo", "importance_sampling"}:
        raise typer.BadParameter(
            "algorithm must be sft, cispo, ppo, or importance_sampling"
        )
    if repeats < 1:
        raise typer.BadParameter("repeats must be positive")
    if algorithm == "sft":
        for name in (
            "rounds",
            "learning_rate",
            "lora_r",
            "seed",
            "top_p",
            "max_new_tokens",
            "eval_max_new_tokens",
            "normalization",
            "microbatch_size",
            "truncation",
            "eps_max",
            "clip_low",
            "clip_high",
            "grad_clip_norm",
            "max_generated_tokens",
            "resume",
            "reward",
            "reward_threshold",
        ):
            source = ctx.get_parameter_source(name)
            if source is not None and source.name == "COMMANDLINE":
                raise typer.BadParameter(f"--{name.replace('_', '-')} is RL-only")
    temperature = (
        temperature if temperature is not None else (0.9 if algorithm == "sft" else 1.0)
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
        from stateset_agents.remote.river_rl import (
            RiverRLConfig,
            atomic_json,
            reward_function_scorer,
        )

        for name in (
            "generations",
            "teacher_base_model",
            "teacher_adapter",
            "target_harvest_rate",
            "gpu",
            "container_disk_gb",
            "num_epochs",
        ):
            source = ctx.get_parameter_source(name)
            if source is not None and source.name == "COMMANDLINE":
                raise typer.BadParameter(
                    f"--{name.replace('_', '-')} is SFT-only; RL uses --rounds"
                )
        if max_cost is not None:
            raise typer.BadParameter(
                "River has no enforceable dollar quote; use --max-generated-tokens"
            )
        if resume and repeats != 1:
            raise typer.BadParameter(
                "Resume each repeat separately using its run directory and seed"
            )
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise typer.BadParameter("learning-rate must be finite and positive")
        if max_generated_tokens is not None and max_generated_tokens < repeats:
            raise typer.BadParameter(
                "max-generated-tokens must allow at least one token per repeat"
            )
        if reward:
            from stateset_agents.remote.river import RiverExecutor
            from stateset_agents.training.sft import _create_domain_reward

            executor = RiverExecutor(
                rl_scorer=reward_function_scorer(
                    _create_domain_reward(reward), threshold=reward_threshold
                ),
                rl_scorer_id=f"domain:{reward}:threshold={reward_threshold}:v1",
            )
        elif ctx.get_parameter_source("reward_threshold").name == "COMMANDLINE":
            raise typer.BadParameter("--reward-threshold requires --reward")
        runs = []
        try:
            for repeat in range(repeats):
                run_output = (
                    output_root
                    if repeats == 1
                    else output_root / f"run-{repeat + 1:03d}"
                )
                rl_config = RiverRLConfig(
                    rounds=rounds,
                    best_of=best_of,
                    seed=seed + repeat,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    loss_fn=algorithm,
                    normalization=normalization,
                    truncation=truncation,
                    microbatch_size=microbatch_size,
                    eps_max=eps_max,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    grad_clip_norm=grad_clip_norm,
                    max_generated_tokens=(
                        None
                        if max_generated_tokens is None
                        else max_generated_tokens // repeats
                    ),
                )
                from dataclasses import asdict

                spec = RemoteJobSpec(
                    dataset=harvest_prompts,
                    base_model=base_model,
                    output_dir=run_output,
                    job_kind="rl",
                    lora_r=lora_r,
                    learning_rate=learning_rate,
                    harvest={
                        **asdict(rl_config),
                        "adapter_dir": (
                            str(initial_adapter) if initial_adapter else None
                        ),
                    },
                    eval_prompts=list[str | dict](
                        _load_specs(eval_prompts, "--eval-prompts")
                    ),
                    eval_max_new_tokens=eval_max_new_tokens,
                    dry_run=dry_run,
                    resume=resume,
                )
                _echo(
                    f"RL run {repeat + 1}/{repeats}: {algorithm}, seed {seed + repeat}, {rounds} rounds"
                )
                result = executor.wait(executor.submit(spec))
                for line in result.logs:
                    _echo(f"  {line}")
                if not result.succeeded:
                    raise RuntimeError("RL run failed")
                report = json.loads((run_output / "rl_report.json").read_text())
                best = report.get("best_eval") or {}
                runs.append(
                    {
                        "seed": seed + repeat,
                        "output_dir": str(run_output),
                        "passed": best.get("passed"),
                        "total": best.get("total"),
                        "best_round": report.get("best_round"),
                    }
                )
                if repeats > 1:
                    rates = [r["passed"] / r["total"] for r in runs if r["total"]]
                    atomic_json(
                        output_root / "rl_repeats_report.json",
                        {
                            "runs": runs,
                            "requested": repeats,
                            "completed": len(runs),
                            "dry_run": dry_run,
                            "pass_rate_mean": statistics.mean(rates) if rates else None,
                            "pass_rate_std": (
                                statistics.stdev(rates) if len(rates) > 1 else None
                            ),
                            "pass_rate_min": min(rates) if rates else None,
                            "pass_rate_max": max(rates) if rates else None,
                        },
                    )
        except (StateSetError, ValueError, RuntimeError) as exc:
            _echo(str(exc))
            raise typer.Exit(code=1) from exc
        _echo(
            f"Report: {output_root / ('rl_report.json' if repeats == 1 else 'rl_repeats_report.json')}"
        )
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
