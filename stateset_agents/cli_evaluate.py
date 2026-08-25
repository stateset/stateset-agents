"""The ``evaluate`` subcommand for the StateSet Agents CLI.

Split out of ``stateset_agents/cli.py``. The command attaches to the parent
Typer app exported by ``cli``; helpers such as ``_echo`` are re-bound locally
for readability, following the sibling ``cli_chat`` / ``cli_train`` pattern.

Two modes share the command: a single ``--message`` round-trip against a
checkpoint, and a ``--scenarios`` batch scored by a named reward function that
renders a markdown report. The helpers below carve the batch path into its
distinct steps (load rows, build reward, run, render).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app

_echo = _cli._echo
CLI_IMPORT_EXCEPTIONS = _cli.CLI_IMPORT_EXCEPTIONS
CLI_TRAIN_EXCEPTIONS = _cli.CLI_TRAIN_EXCEPTIONS

#: Reward names accepted by ``--reward`` in batch mode.
REWARD_CHOICES = ("gsm8k", "customer_support", "tool_calling")


def _echo_dry_run_plan(
    checkpoint: str | None,
    scenarios: str | None,
    reward: str | None,
    output: str | None,
    message: str,
) -> None:
    """Print what an evaluation *would* do, without loading anything."""
    _echo("Dry-run: evaluation was not executed.")
    if checkpoint:
        _echo(f"Checkpoint: {checkpoint}")
    if scenarios:
        _echo(f"Scenarios: {scenarios}")
        _echo(f"Reward: {reward}")
        _echo(f"Output: {output or '(stdout)'}")
    else:
        _echo(f"Message: {message}")


def _validate_batch_args(scenarios: str | None, reward: str | None) -> None:
    """Reject a ``--scenarios`` run with a missing or unknown ``--reward``."""
    if not scenarios:
        return
    if not reward:
        print("--reward is required with --scenarios.", file=sys.stderr)
        raise typer.Exit(code=2)
    if reward not in set(REWARD_CHOICES):
        print(
            f"Unknown reward: {reward!r}. Options: gsm8k, customer_support, tool_calling.",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)


def _require_checkpoint(checkpoint: str | None) -> Path:
    """Resolve ``--checkpoint`` to an existing directory or exit."""
    if not checkpoint:
        _echo("checkpoint is required unless --dry-run is used.")
        raise typer.Exit(code=2)

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        _echo(f"Checkpoint not found: {checkpoint}")
        raise typer.Exit(code=2)
    return ckpt_path


def _import_checkpoint_loader() -> Any:
    """Import ``load_agent_from_checkpoint`` lazily (it pulls in torch)."""
    try:
        from stateset_agents.core.agent import load_agent_from_checkpoint
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo(f"Failed to import loader: {e}")
        raise typer.Exit(code=2) from e
    return load_agent_from_checkpoint


def _load_scenario_rows(scenarios: str) -> list[dict[str, Any]]:
    """Read a JSONL scenario file into a non-empty list of rows."""
    scenarios_path = Path(scenarios)
    if not scenarios_path.exists():
        print(f"Scenarios file not found: {scenarios}", file=sys.stderr)
        raise typer.Exit(code=2)

    rows = [
        json.loads(line)
        for line in scenarios_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        print("No scenarios loaded.", file=sys.stderr)
        raise typer.Exit(code=2)
    return rows


def _build_reward_fn(reward: str | None) -> Any:
    """Instantiate the reward function named by ``--reward``."""
    if reward == "gsm8k":
        from stateset_agents.data.gsm8k import GSM8KReward

        return GSM8KReward()
    if reward == "customer_support":
        from stateset_agents.data.customer_support_bench import SupportRewardComposite

        return SupportRewardComposite()
    from stateset_agents.data.tool_calling_bench import ToolCallReward

    return ToolCallReward()


async def _score_scenarios(
    load_agent_from_checkpoint: Any,
    checkpoint: str,
    rows: list[dict[str, Any]],
    reward_fn: Any,
) -> list[dict[str, Any]]:
    """Run every scenario through the agent and score it."""
    from stateset_agents.core.trajectory import ConversationTurn

    agent = await load_agent_from_checkpoint(checkpoint, load_model=True)
    results = []
    for row in rows:
        query = row.get("user_query") or row.get("question") or row.get("prompt") or ""
        response = await agent.generate_response([{"role": "user", "content": query}])
        turns = [ConversationTurn(role="assistant", content=response)]
        result = await reward_fn.compute_reward(turns, context=row)
        results.append(
            {
                "query": query,
                "response": response,
                "score": float(result.score),
            }
        )
    return results


def _render_batch_report(
    results: list[dict[str, Any]],
    reward: str | None,
    checkpoint: str,
    threshold: float,
) -> str:
    """Render the batch scores as the markdown report written to ``--output``."""
    scores = [r["score"] for r in results]
    mean = sum(scores) / len(scores) if scores else 0.0
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    n_pass = sum(1 for s in scores if s >= threshold)

    lines = [
        f"# Batch evaluation — `{reward}`",
        "",
        f"**Checkpoint:** `{checkpoint}`",
        f"**Scenarios:** {len(results)}",
        f"**Mean score:** {mean:.3f} ± {std:.3f}",
        f"**Pass rate (≥ {threshold}):** {n_pass}/{len(results)} ({100 * n_pass / len(results):.1f}%)",
        "",
        "| # | Score | Query | Response (head) |",
        "|---|-------|-------|-----------------|",
    ]
    for i, r in enumerate(results):
        marker = (
            "✅" if r["score"] >= threshold else ("⚠️ " if r["score"] >= 0.1 else "❌")
        )
        q_preview = r["query"][:50].replace("|", "\\|").replace("\n", " ")
        r_preview = r["response"][:50].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | {marker} {r['score']:.3f} | {q_preview} | {r_preview} |")

    return "\n".join(lines) + "\n"


def _emit_report(md: str, output: str | None) -> None:
    """Write the markdown report to ``--output``, or to stdout when unset."""
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        _echo(f"Wrote batch eval report → {output}")
    else:
        print(md)


def _run_batch_mode(
    load_agent_from_checkpoint: Any,
    checkpoint: str,
    scenarios: str,
    reward: str | None,
    output: str | None,
    threshold: float,
) -> None:
    """Score a whole scenario file and emit the markdown report."""
    import asyncio

    rows = _load_scenario_rows(scenarios)
    reward_fn = _build_reward_fn(reward)

    try:
        results = asyncio.run(
            _score_scenarios(load_agent_from_checkpoint, checkpoint, rows, reward_fn)
        )
    except CLI_TRAIN_EXCEPTIONS as e:
        _echo(f"Batch evaluation failed: {e}")
        raise typer.Exit(code=2) from e

    _emit_report(_render_batch_report(results, reward, checkpoint, threshold), output)


def _run_single_message(
    load_agent_from_checkpoint: Any, checkpoint: str, message: str
) -> None:
    """Send one message to the checkpointed agent and print the reply."""
    import asyncio

    async def _run() -> str:
        agent = await load_agent_from_checkpoint(checkpoint, load_model=True)
        resp = await agent.generate_response([{"role": "user", "content": message}])
        return str(resp)

    try:
        resp = asyncio.run(_run())
        _echo(f"Response: {resp}")
    except CLI_TRAIN_EXCEPTIONS as e:
        _echo(f"Evaluation failed: {e}")
        raise typer.Exit(code=2) from e


@app.command()
def evaluate(
    checkpoint: str | None = typer.Option(
        None, "--checkpoint", help="Path to a saved checkpoint directory"
    ),
    message: str = typer.Option(
        "Hello", help="Single message to evaluate (ignored when --scenarios is set)"
    ),
    scenarios: str | None = typer.Option(
        None,
        "--scenarios",
        help='JSONL of scenarios for batch mode. Each line: {"user_query": ..., <reward-specific context>}.',
    ),
    reward: str | None = typer.Option(
        None,
        "--reward",
        help="Reward name for batch mode: gsm8k, customer_support, tool_calling.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the batch-mode markdown report to this path (default: stdout).",
    ),
    threshold: float = typer.Option(
        0.7,
        "--threshold",
        help="Pass/fail threshold for the markdown summary (just informational).",
    ),
    dry_run: bool = typer.Option(
        False, help="Show evaluation plan without loading checkpoint."
    ),
) -> None:
    """Evaluate a saved checkpoint — single message or batch with reward.

    Single-message mode (the original behavior):

        stateset-agents evaluate --checkpoint outputs/v1 --message "Hello"

    Batch mode — score every scenario in a JSONL against a reward function:

        stateset-agents evaluate --checkpoint outputs/v1 \\
            --scenarios eval_set.jsonl --reward customer_support \\
            --output eval_report.md

    The batch markdown report shows mean score, perfect/zero counts, and
    a per-scenario table. Same shape as ``grade_transcript`` output so the
    two reports compose naturally.
    """
    if dry_run:
        _echo_dry_run_plan(checkpoint, scenarios, reward, output, message)
        return

    # Argument validation before filesystem checks — gives clearer errors.
    _validate_batch_args(scenarios, reward)
    _require_checkpoint(checkpoint)
    assert checkpoint is not None  # _require_checkpoint exits when it is not
    load_agent_from_checkpoint = _import_checkpoint_loader()

    if scenarios:
        _run_batch_mode(
            load_agent_from_checkpoint,
            checkpoint,
            scenarios,
            reward,
            output,
            threshold,
        )
        return

    # Single-message mode (preserved behavior).
    _run_single_message(load_agent_from_checkpoint, checkpoint, message)
