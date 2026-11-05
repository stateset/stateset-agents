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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app
from stateset_agents.core.errors import StateSetError
from stateset_agents.remote.registry import available_providers, get_executor

if TYPE_CHECKING:
    from stateset_agents.flywheel import FlywheelConfig
    from stateset_agents.remote.executor import RemoteExecutor

_echo = _cli._echo

#: Options that only the RL (River) loop reads.
_RL_ONLY = (
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
)
#: Options that only the SFT harvest loop reads.
_SFT_ONLY = (
    "generations",
    "teacher_base_model",
    "teacher_adapter",
    "target_harvest_rate",
    "gpu",
    "container_disk_gb",
    "num_epochs",
)
#: Loss knobs each RL loss reads (RiverRLConfig.loss_kwargs).
_LOSS_KNOBS: dict[str, set[str]] = {
    "cispo": {"eps_max"},
    "ppo": {"clip_low", "clip_high"},
    "importance_sampling": set(),
}


def _given_on_command_line(ctx: typer.Context, name: str) -> bool:
    """Return whether the user passed parameter ``name`` explicitly."""
    source = ctx.get_parameter_source(name)
    return source is not None and source.name == "COMMANDLINE"


def _load_specs(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise typer.BadParameter(f"cannot read {label} file: {exc}") from exc
    except ValueError as exc:
        raise typer.BadParameter(f"{label} file is not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise typer.BadParameter(f"{label} file must be a non-empty JSON list")
    from stateset_agents.training.sft import has_eval_assertions

    unchecked = [
        index
        for index, spec in enumerate(data)
        if not isinstance(spec, dict)
        or not (
            isinstance(spec.get("prompt"), str) or isinstance(spec.get("turns"), list)
        )
        or not has_eval_assertions(spec)
    ]
    if unchecked:
        raise typer.BadParameter(
            f"{label} spec(s) at index {unchecked[:5]} need a prompt (or turns) "
            "and expect, forbid or turn_expect checks; a spec without checks "
            "passes trivially"
        )
    return data


@dataclass(frozen=True)
class FlywheelOptions:
    """The ``flywheel`` command's parameters, in signature order (a test pins
    the two together)."""

    base_model: str
    harvest_prompts: Path
    eval_prompts: Path
    output_root: Path
    initial_adapter: Path | None
    teacher_base_model: str | None
    teacher_adapter: Path | None
    generations: int
    best_of: int
    temperature: float | None
    target_harvest_rate: float | None
    max_cost: float | None
    provider: str
    gpu: str | None
    container_disk_gb: int | None
    num_epochs: int
    algorithm: str
    rounds: int
    repeats: int
    learning_rate: float
    lora_r: int
    seed: int
    top_p: float
    max_new_tokens: int
    eval_max_new_tokens: int
    normalization: str
    microbatch_size: int
    truncation: str
    eps_max: float
    clip_low: float
    clip_high: float
    grad_clip_norm: float
    max_generated_tokens: int | None
    resume: bool
    reward: str | None
    reward_threshold: float
    dry_run: bool

    @property
    def sampling_temperature(self) -> float:
        """``--temperature``, else 0.9 for SFT harvesting and 1.0 for RL."""
        if self.temperature is not None:
            return self.temperature
        return 0.9 if self.algorithm == "sft" else 1.0


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
    arguments = dict(locals())  # exactly the parameters, at this point
    del arguments["ctx"]
    _run_flywheel_command(ctx, FlywheelOptions(**arguments))


def _flag(name: str) -> str:
    """The command-line spelling of parameter ``name``."""
    return "--" + name.replace("_", "-")


def _check_mode_flags(ctx: typer.Context, options: FlywheelOptions) -> None:
    """Reject an unknown algorithm, a non-positive repeat count and RL-only
    flags given to the SFT loop."""
    if options.algorithm not in {"sft", "cispo", "ppo", "importance_sampling"}:
        raise typer.BadParameter(
            "algorithm must be sft, cispo, ppo, or importance_sampling"
        )
    if options.repeats < 1:
        raise typer.BadParameter("repeats must be positive")
    if options.algorithm == "sft":
        for name in _RL_ONLY:
            if _given_on_command_line(ctx, name):
                raise typer.BadParameter(f"{_flag(name)} is RL-only")


def _flywheel_config(options: FlywheelOptions) -> FlywheelConfig:
    """Load and check both spec files and build the SFT loop's config."""
    from stateset_agents.flywheel import FlywheelConfig

    harvest_specs = _load_specs(options.harvest_prompts, "--harvest-prompts")
    eval_specs = _load_specs(options.eval_prompts, "--eval-prompts")
    try:
        return FlywheelConfig(
            base_model=options.base_model,
            harvest_prompts=harvest_specs,
            eval_prompts=eval_specs,
            output_root=options.output_root,
            initial_adapter=options.initial_adapter,
            teacher_base_model=options.teacher_base_model,
            teacher_adapter=options.teacher_adapter,
            generations=options.generations,
            best_of=options.best_of,
            temperature=options.sampling_temperature,
            target_harvest_rate=options.target_harvest_rate,
            max_cost_usd=options.max_cost,
            gpu=options.gpu,
            container_disk_gb=options.container_disk_gb,
            num_epochs=options.num_epochs,
            dry_run=options.dry_run,
        )
    except ValueError as exc:
        raise typer.BadParameter(f"--eval-prompts: {exc}") from exc


def _require_executor(options: FlywheelOptions) -> RemoteExecutor:
    """The provider's executor, which must run this algorithm's job kind."""
    try:
        executor = get_executor(options.provider)
    except StateSetError as exc:
        _echo(str(exc))
        raise typer.Exit(code=1) from exc
    requested_kind = "rl" if options.algorithm != "sft" else "harvest"
    if not executor.supports(requested_kind):
        supported = ", ".join(sorted(executor.supported_job_kinds))
        _echo(
            f"Provider {options.provider.strip().lower()!r} cannot run the "
            f"{options.algorithm} flywheel ({requested_kind} jobs); supported "
            f"modes: {supported}."
        )
        raise typer.Exit(code=2)
    return executor


def _run_flywheel_command(ctx: typer.Context, options: FlywheelOptions) -> None:
    """Validate the flags, then run the SFT harvest loop or River RL."""
    _check_mode_flags(ctx, options)
    config = _flywheel_config(options)
    executor = _require_executor(options)
    if options.algorithm != "sft":
        _run_rl(ctx, options, executor)
        return
    _run_sft(options, config, executor)


def _check_rl_flags(ctx: typer.Context, options: FlywheelOptions) -> None:
    """Reject flags the chosen RL loss or River cannot honour."""
    # Each loss reads only its own knobs (RiverRLConfig.loss_kwargs); an
    # explicit knob for a different loss would be silently dropped.
    loss_knobs = _LOSS_KNOBS[options.algorithm]
    for name in ("eps_max", "clip_low", "clip_high"):
        if name not in loss_knobs and _given_on_command_line(ctx, name):
            raise typer.BadParameter(
                f"{_flag(name)} does not apply to --algorithm {options.algorithm}"
            )
    if options.provider.strip().lower() != "river":
        _echo("--algorithm requires --provider river (zero-infra RL).")
        raise typer.Exit(code=2)
    for name in _SFT_ONLY:
        if _given_on_command_line(ctx, name):
            raise typer.BadParameter(f"{_flag(name)} is SFT-only; RL uses --rounds")
    if options.max_cost is not None:
        raise typer.BadParameter(
            "River has no enforceable dollar quote; use --max-generated-tokens"
        )
    if options.resume and options.repeats != 1:
        raise typer.BadParameter(
            "Resume each repeat separately using its run directory and seed"
        )
    if not math.isfinite(options.learning_rate) or options.learning_rate <= 0:
        raise typer.BadParameter("learning-rate must be finite and positive")
    if (
        options.max_generated_tokens is not None
        and options.max_generated_tokens < options.repeats
    ):
        raise typer.BadParameter(
            "max-generated-tokens must allow at least one token per repeat"
        )


def _rl_executor(
    ctx: typer.Context, options: FlywheelOptions, executor: RemoteExecutor
) -> RemoteExecutor:
    """River with the ``--reward`` domain scorer, else ``executor`` unchanged."""
    if options.reward:
        from stateset_agents.remote.river import RiverExecutor
        from stateset_agents.remote.river_rl import reward_function_scorer
        from stateset_agents.training.sft import _create_domain_reward

        threshold = options.reward_threshold
        return RiverExecutor(
            rl_scorer=reward_function_scorer(
                _create_domain_reward(options.reward), threshold=threshold
            ),
            rl_scorer_id=f"domain:{options.reward}:threshold={threshold}:v1",
        )
    if _given_on_command_line(ctx, "reward_threshold"):
        raise typer.BadParameter("--reward-threshold requires --reward")
    return executor


def _run_rl(
    ctx: typer.Context, options: FlywheelOptions, executor: RemoteExecutor
) -> None:
    """River RL: ``--repeats`` independent runs, each its own seed and budget."""
    _check_rl_flags(ctx, options)
    executor = _rl_executor(ctx, options, executor)
    runs: list[dict[str, Any]] = []
    try:
        for repeat in range(options.repeats):
            runs.append(_rl_run(options, executor, repeat))
            if options.repeats > 1:
                _write_rl_repeats_report(options, runs)
    except (StateSetError, ValueError, RuntimeError) as exc:
        _echo(str(exc))
        raise typer.Exit(code=1) from exc
    report = "rl_report.json" if options.repeats == 1 else "rl_repeats_report.json"
    _echo(f"Report: {options.output_root / report}")


def _rl_run(
    options: FlywheelOptions, executor: RemoteExecutor, repeat: int
) -> dict[str, Any]:
    """Submit RL repeat ``repeat`` and summarise its best validation."""
    from stateset_agents.remote.job import RemoteJobSpec
    from stateset_agents.remote.river_rl import RiverRLConfig

    repeats, seed = options.repeats, options.seed + repeat
    run_output = (
        options.output_root
        if repeats == 1
        else options.output_root / f"run-{repeat + 1:03d}"
    )
    rl_config = RiverRLConfig(
        rounds=options.rounds,
        best_of=options.best_of,
        seed=seed,
        temperature=options.sampling_temperature,
        top_p=options.top_p,
        max_new_tokens=options.max_new_tokens,
        loss_fn=options.algorithm,
        normalization=options.normalization,
        truncation=options.truncation,
        microbatch_size=options.microbatch_size,
        eps_max=options.eps_max,
        clip_low=options.clip_low,
        clip_high=options.clip_high,
        grad_clip_norm=options.grad_clip_norm,
        max_generated_tokens=(
            None
            if options.max_generated_tokens is None
            else options.max_generated_tokens // repeats
        ),
    )
    spec = RemoteJobSpec(
        dataset=options.harvest_prompts,
        base_model=options.base_model,
        output_dir=run_output,
        job_kind="rl",
        lora_r=options.lora_r,
        learning_rate=options.learning_rate,
        harvest={
            **asdict(rl_config),
            "adapter_dir": (
                str(options.initial_adapter) if options.initial_adapter else None
            ),
        },
        eval_prompts=list[str | dict](
            _load_specs(options.eval_prompts, "--eval-prompts")
        ),
        eval_max_new_tokens=options.eval_max_new_tokens,
        dry_run=options.dry_run,
        resume=options.resume,
    )
    _echo(
        f"RL run {repeat + 1}/{repeats}: {options.algorithm}, seed {seed}, "
        f"{options.rounds} rounds"
    )
    result = executor.wait(executor.submit(spec))
    for line in result.logs:
        _echo(f"  {line}")
    if not result.succeeded:
        raise RuntimeError("RL run failed")
    report = json.loads((run_output / "rl_report.json").read_text())
    best = report.get("best_eval") or {}
    return {
        "seed": seed,
        "output_dir": str(run_output),
        "passed": best.get("passed"),
        "total": best.get("total"),
        "best_round": report.get("best_round"),
    }


def _write_rl_repeats_report(
    options: FlywheelOptions, runs: list[dict[str, Any]]
) -> None:
    """Rewrite the pass-rate distribution after every completed repeat."""
    from stateset_agents.remote.river_rl import atomic_json

    rates = [r["passed"] / r["total"] for r in runs if r["total"]]
    atomic_json(
        options.output_root / "rl_repeats_report.json",
        {
            "runs": runs,
            "requested": options.repeats,
            "completed": len(runs),
            "dry_run": options.dry_run,
            "pass_rate_mean": statistics.mean(rates) if rates else None,
            "pass_rate_std": (statistics.stdev(rates) if len(rates) > 1 else None),
            "pass_rate_min": min(rates) if rates else None,
            "pass_rate_max": max(rates) if rates else None,
        },
    )


def _run_sft(
    options: FlywheelOptions, config: FlywheelConfig, executor: RemoteExecutor
) -> None:
    """The rejection-sampling SFT loop, once or ``--repeats`` times."""
    from stateset_agents.flywheel import run_flywheel, run_flywheel_repeats

    _echo(
        f"Flywheel: {options.base_model} on {options.provider}, up to "
        f"{options.generations} generation(s)"
        + (f", ceiling ${options.max_cost:.2f}" if options.max_cost is not None else "")
        + ("  [dry run]" if options.dry_run else "")
    )
    try:
        if options.repeats > 1:
            aggregate = run_flywheel_repeats(config, executor, options.repeats)
            _echo_repeats(aggregate, options.output_root)
            return
        report = run_flywheel(config, executor)
    except StateSetError as exc:
        _echo(str(exc))
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        _echo(str(exc))
        raise typer.Exit(code=1) from exc
    _echo_flywheel_report(report, options)


def _echo_repeats(aggregate: dict[str, Any], output_root: Path) -> None:
    """Print each repeat's outcome and the score distribution."""
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


def _echo_flywheel_report(report: dict[str, Any], options: FlywheelOptions) -> None:
    """Print one run's generations, cost and the final adapter."""
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
            f"--base-model {options.base_model} --adapter {report['final_adapter']}"
        )
    _echo(f"Report: {options.output_root / 'flywheel_report.json'}")
