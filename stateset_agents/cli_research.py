"""Autonomous research + fine-tune subcommands for the StateSet Agents CLI.

Split out of stateset_agents/cli.py. Each command attaches to the parent
Typer app exported by cli; helpers _echo, _load_config, etc. are
re-bound locally for readability. Helpers that tests patch on
stateset_agents.cli (_collect_dependency_status, _collect_import_status)
are looked up via late binding through the _cli module reference so the
patches still propagate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import (
    CLI_IMPORT_EXCEPTIONS,
    CLI_TRAIN_EXCEPTIONS,
    app,
)

_echo = _cli._echo
_load_config = _cli._load_config
_coerce_positive_int = _cli._coerce_positive_int


@app.command("auto-research")
def auto_research(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to auto-research config file (YAML/JSON).",
    ),
    max_experiments: int = typer.Option(
        0,
        "--max-experiments",
        "-n",
        help="Maximum experiments to run (0 = unlimited).",
    ),
    time_budget: int = typer.Option(
        300,
        "--time-budget",
        "-t",
        help="Wall-clock seconds per experiment.",
    ),
    proposer: str = typer.Option(
        "perturbation",
        "--proposer",
        "-p",
        help="Proposer strategy: perturbation, smart, adaptive, random, grid, bayesian, llm.",
    ),
    algorithm: str = typer.Option(
        "gspo",
        "--algorithm",
        "-a",
        help="Training algorithm: gspo, grpo, dapo, vapo.",
    ),
    output_dir: str = typer.Option(
        "./auto_research_results",
        "--output-dir",
        "-o",
        help="Directory for results and checkpoints.",
    ),
    search_space: str = typer.Option(
        "grpo",
        "--search-space",
        "-s",
        help="Search space: grpo, auto_research, quick, reward, model, multi_algorithm, full.",
    ),
    improvement_patience: int = typer.Option(
        0,
        "--improvement-patience",
        help="Stop after this many consecutive non-improvements (0 = disabled).",
    ),
    max_wall_clock: int = typer.Option(
        0,
        "--max-wall-clock",
        help="Total wall-clock budget in seconds (0 = unlimited).",
    ),
    wandb: bool = typer.Option(
        False,
        "--wandb",
        help="Log experiments to Weights & Biases.",
    ),
    wandb_project: str = typer.Option(
        "auto-research",
        "--wandb-project",
        help="W&B project name.",
    ),
    stub: bool = typer.Option(
        False,
        "--stub",
        help="Run with stub model for testing the loop without GPU.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and show plan without running.",
    ),
) -> None:
    """Run the autonomous research loop to optimize agent training.

    The loop autonomously proposes experiments, trains with a time budget,
    evaluates on held-out scenarios, and keeps only improvements.

    Resumes automatically if a previous run exists in the output directory.
    """
    from stateset_agents.training.auto_research.config import AutoResearchConfig

    # Load from config file if provided, then override with CLI args
    if config:
        try:
            ar_config = AutoResearchConfig.from_file(config)
        except (ImportError, ValueError) as exc:
            # Fallback: load as generic dict for non-YAML/JSON
            _echo(f"Warning: {exc}. Falling back to manual config parsing.")
            ar_config = AutoResearchConfig()
    else:
        ar_config = AutoResearchConfig()

    # CLI args override file config (only if explicitly provided / non-default)
    if time_budget != 300:
        ar_config.time_budget = time_budget
    if max_experiments != 0:
        ar_config.max_experiments = max_experiments
    if max_wall_clock != 0:
        ar_config.max_wall_clock = max_wall_clock
    if proposer != "perturbation":
        ar_config.proposer = proposer
    if algorithm != "gspo":
        ar_config.trainer_algorithm = algorithm
    if output_dir != "./auto_research_results":
        ar_config.output_dir = output_dir
    if search_space != "grpo":
        ar_config.search_space_name = search_space
    if improvement_patience != 0:
        ar_config.improvement_patience = improvement_patience
    if wandb:
        ar_config.log_to_wandb = True
    if wandb_project != "auto-research":
        ar_config.wandb_project = wandb_project

    warnings = ar_config.validate()
    for w in warnings:
        _echo(f"Warning: {w}")

    if dry_run:
        _echo("Dry-run: auto-research configuration validated.")
        _echo(f"  Proposer:        {ar_config.proposer}")
        _echo(f"  Algorithm:       {ar_config.trainer_algorithm}")
        _echo(f"  Search space:    {ar_config.search_space_name}")
        _echo(f"  Time budget:     {ar_config.time_budget}s")
        _echo(f"  Max experiments: {ar_config.max_experiments or 'unlimited'}")
        _echo(f"  Max wall clock:  {ar_config.max_wall_clock or 'unlimited'}s")
        _echo(f"  Output dir:      {ar_config.output_dir}")
        _echo(f"  W&B logging:     {ar_config.log_to_wandb}")

        # Show available search spaces
        try:
            from stateset_agents.training.auto_research.search_spaces import (
                list_auto_research_search_spaces,
            )
            from stateset_agents.training.hpo.search_spaces import (
                list_available_search_spaces,
            )

            ar_spaces = list_auto_research_search_spaces()
            hpo_spaces = list_available_search_spaces()
            _echo(
                f"  Available search spaces: {', '.join(sorted(set(ar_spaces + hpo_spaces)))}"
            )
        except Exception:
            pass

        # Check if resumable
        from pathlib import Path

        jsonl = Path(ar_config.output_dir) / "experiments.jsonl"
        if jsonl.exists():
            count = sum(1 for line in jsonl.open() if line.strip())
            _echo(f"  Resume:          yes ({count} previous experiments found)")
        else:
            _echo("  Resume:          no (fresh run)")

        return

    # Set up agent, environment, reward
    try:
        from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
        from stateset_agents.core.environment import ConversationEnvironment
        from stateset_agents.core.reward import (
            CompositeReward,
            HelpfulnessReward,
            SafetyReward,
        )
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo(f"Core modules unavailable: {e}")
        raise typer.Exit(code=2) from e

    # Load agent/env config from the file if provided
    file_cfg = _load_config(config) if config else {}
    ar_section = file_cfg.get("auto_research", file_cfg)
    agent_cfg = ar_section.get("agent", {}) if isinstance(ar_section, dict) else {}
    env_cfg = ar_section.get("environment", {}) if isinstance(ar_section, dict) else {}

    ac = AgentConfig(
        model_name=agent_cfg.get("model_name", "stub://demo" if stub else "gpt2"),
        max_new_tokens=agent_cfg.get("max_new_tokens", 64),
        temperature=agent_cfg.get("temperature", 0.7),
        use_stub_model=agent_cfg.get("use_stub_model", stub),
        stub_responses=(
            agent_cfg.get(
                "stub_responses",
                ["Stub response for auto-research testing."],
            )
            if stub
            else None
        ),
    )

    scenarios = env_cfg.get("scenarios") or [
        {
            "topic": "general_help",
            "context": "Demo scenario for auto-research",
            "user_responses": ["Thanks, tell me more.", "Interesting, go on."],
        }
    ]
    eval_scenarios = env_cfg.get("eval_scenarios", scenarios)

    environment = ConversationEnvironment(
        scenarios=scenarios,
        max_turns=env_cfg.get("max_turns", 8),
    )

    reward_fn = CompositeReward(
        [
            HelpfulnessReward(weight=0.6),
            SafetyReward(weight=0.4),
        ]
    )

    import asyncio

    from stateset_agents.training.auto_research.experiment_loop import run_auto_research

    async def _run() -> None:
        agent = MultiTurnAgent(ac)
        await agent.initialize()

        tracker = await run_auto_research(
            agent=agent,
            environment=environment,
            eval_scenarios=eval_scenarios,
            reward_fn=reward_fn,
            config=ar_config,
        )

        _echo(f"Done. Results saved to {ar_config.output_dir}")
        if tracker.best_record:
            _echo(
                f"Best {ar_config.objective_metric}: "
                f"{tracker.best_value:.6f} "
                f"(experiment {tracker.best_record.experiment_id})"
            )

    try:
        asyncio.run(_run())
    except CLI_TRAIN_EXCEPTIONS as e:
        _echo(f"Auto-research failed: {e}")
        raise typer.Exit(code=2) from e


@app.command("fine-tune")
def fine_tune(
    curated: str = typer.Argument(
        ...,
        help="Path to a curated JSONL (from `stateset-agents grade-batch ... CURATED=...`).",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen3.5-0.8B",
        "--base-model",
        "-m",
        help="HF base model to fine-tune.",
    ),
    output_dir: str = typer.Option(
        "outputs/sft_v1",
        "--output-dir",
        "-o",
        help="Where the LoRA adapter is saved.",
    ),
    min_score: float = typer.Option(
        0.7,
        "--min-score",
        help="Drop curated examples below this score before SFT.",
    ),
    num_epochs: int = typer.Option(
        3,
        "--num-epochs",
        "-e",
        help="Training epochs.",
    ),
    lora_r: int = typer.Option(
        16,
        "--lora-r",
        help="LoRA rank.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the training plan without running it (forced when no GPU).",
    ),
) -> None:
    """Fine-tune from a curated JSONL in one command.

    Convenience wrapper around `make full-loop` — runs prepare_sft_dataset.py
    then sft_from_curated.py. Produces a LoRA adapter consumable by
    `stateset-agents chat --checkpoint` and `stateset-agents serve --checkpoint`.

    Examples:

        stateset-agents fine-tune curated.jsonl
        stateset-agents fine-tune curated.jsonl --base-model Qwen/Qwen3.5-0.8B --num-epochs 5
        stateset-agents fine-tune curated.jsonl --dry-run
    """
    import subprocess
    import tempfile

    curated_path = Path(curated)
    if not curated_path.exists():
        print(f"Curated file not found: {curated}", file=sys.stderr)
        raise typer.Exit(code=2)

    script_dir = Path(__file__).resolve().parents[1] / "scripts"

    with tempfile.TemporaryDirectory() as tmp:
        sft_jsonl = Path(tmp) / "sft_train.jsonl"

        _echo("▶ Step 1/2: prepare-sft (curated → chat format)")
        result = subprocess.run(
            [
                sys.executable,
                str(script_dir / "prepare_sft_dataset.py"),
                "--input",
                str(curated_path),
                "--format",
                "chat",
                "--output",
                str(sft_jsonl),
                "--min-score",
                str(min_score),
                "--dedup",
                "--stats",
            ],
            check=False,
        )
        if result.returncode != 0:
            raise typer.Exit(code=result.returncode)

        _echo("")
        _echo("▶ Step 2/2: sft-from-curated (chat format → trained adapter)")
        cmd = [
            sys.executable,
            str(script_dir / "sft_from_curated.py"),
            "--dataset",
            str(sft_jsonl),
            "--base-model",
            base_model,
            "--output-dir",
            output_dir,
            "--num-epochs",
            str(num_epochs),
            "--lora-r",
            str(lora_r),
        ]
        if dry_run:
            cmd.append("--dry-run")
        result = subprocess.run(cmd, check=False)
        raise typer.Exit(code=result.returncode)
