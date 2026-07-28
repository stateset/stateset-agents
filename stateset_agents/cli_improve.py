"""``stateset-agents improve`` — the grade -> curate -> retrain loop, one command.

Thin orchestrator over existing, tested pieces:

* ``stateset_agents.data.trajectory_ingest`` — converts OpenAI/LangChain logs
  into transcript JSONLs (same shape ``chat --history`` writes).
* ``scripts/grade_transcript.py`` — scores every assistant turn with a reward
  function and curates high-scoring (prompt, response) pairs.
* ``scripts/sft_from_curated.py`` / ``examples/finetune_gspo.py`` — the two
  training paths that consume the curated set (referenced, not reimplemented,
  in the generated ``next_steps.md``).

No new grading or curation logic lives here — every phase below imports and
calls the functions those scripts already define and test.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app

_echo = _cli._echo

# `scripts/` isn't a package (no __init__.py) — the existing test suite
# (tests/unit/test_trajectory_ingest.py) already imports grade_transcript.py
# the same way: add scripts/ to sys.path and import it directly. This keeps
# grade_transcript.py as the single source of truth for grading/curation
# logic instead of duplicating it here.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

KNOWN_REWARDS = ("gsm8k", "customer_support", "tool_calling")
JUDGE_REWARD_HINTS = ("judge", "llm", "gpt", "claude", "openai", "anthropic", "ruler")

SUMMARY_FILENAME = "improve_summary.json"
CURATED_FILENAME = "curated.jsonl"
NEXT_STEPS_FILENAME = "next_steps.md"


def _resolve_reward_name(reward: str) -> None:
    """Raise typer.Exit with a clear message for unsupported/judge rewards."""
    if reward in KNOWN_REWARDS:
        return
    lowered = reward.lower()
    if any(hint in lowered for hint in JUDGE_REWARD_HINTS):
        _echo(
            f"Unsupported --reward '{reward}': the `improve` loop runs offline "
            "with rule-based rewards only, so LLM-judge rewards (which require "
            "an API key) are not supported here. Choose one of: "
            f"{', '.join(KNOWN_REWARDS)}.",
            err=True,
        )
    else:
        _echo(
            f"Unknown --reward '{reward}'. Choose one of: {', '.join(KNOWN_REWARDS)}.",
            err=True,
        )
    raise typer.Exit(code=2)


def _ingest_to_transcripts(fmt: str, source: Path, dest_dir: Path) -> list[Path]:
    """Ingest an OpenAI/LangChain log into per-conversation transcript JSONLs.

    Delegates entirely to ``stateset_agents.data.trajectory_ingest`` (the same
    functions ``stateset-agents ingest`` uses).
    """
    from stateset_agents.data.trajectory_ingest import (
        from_langchain_json,
        from_openai_jsonl,
        to_grading_history,
    )

    if fmt == "openai":
        trajectories = from_openai_jsonl(source)
    else:
        trajectories = from_langchain_json(source)

    if not trajectories:
        _echo(f"No conversations found in {source}", err=True)
        raise typer.Exit(code=1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, traj in enumerate(trajectories):
        conv_path = dest_dir / f"conversation_{i}.jsonl"
        with conv_path.open("w", encoding="utf-8") as f:
            for turn in to_grading_history(traj):
                f.write(json.dumps(turn) + "\n")
        written.append(conv_path)
    return written


def _collect_transcript_files(transcripts_dir: Path) -> list[Path]:
    files = sorted(p for p in transcripts_dir.glob("*.jsonl") if p.is_file())
    return files


async def _grade_all(transcript_files: list[Path], reward: Any) -> list[dict[str, Any]]:
    """Grade every transcript file, returning one summary dict per file."""
    import grade_transcript as gt  # local import: sys.path was patched above

    per_transcript: list[dict[str, Any]] = []
    for path in transcript_files:
        turns = gt.load_transcript(path)
        rows = await gt.grade_transcript(turns, [], reward)
        per_transcript.append({"path": path, "turns": turns, "rows": rows})
    return per_transcript


def _build_summary(
    reward_name: str,
    threshold: float,
    per_transcript: list[dict[str, Any]],
    curated_count: int,
    curated_path: Path,
) -> dict[str, Any]:
    all_scores: list[float] = []
    breakdown_totals: dict[str, list[float]] = {}
    transcript_summaries: list[dict[str, Any]] = []

    for entry in per_transcript:
        rows = entry["rows"]
        scores = [float(r["score"]) for r in rows]
        all_scores.extend(scores)
        for row in rows:
            for key, value in (row.get("breakdown") or {}).items():
                breakdown_totals.setdefault(key, []).append(float(value))
        transcript_summaries.append(
            {
                "name": entry["path"].name,
                "assistant_turns": len(rows),
                "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
                "above_threshold": sum(1 for s in scores if s >= threshold),
            }
        )

    mean_score = (sum(all_scores) / len(all_scores)) if all_scores else 0.0
    reward_breakdown = {
        key: (sum(values) / len(values)) if values else 0.0
        for key, values in breakdown_totals.items()
    }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reward": reward_name,
        "threshold": threshold,
        "transcript_count": len(per_transcript),
        "assistant_turn_count": len(all_scores),
        "mean_score": mean_score,
        "above_threshold_count": sum(1 for s in all_scores if s >= threshold),
        "reward_breakdown": reward_breakdown,
        "curated_count": curated_count,
        "curated_path": str(curated_path),
        "transcripts": transcript_summaries,
    }


def _render_next_steps(summary: dict[str, Any], output_dir: Path) -> str:
    curated_path = output_dir / CURATED_FILENAME
    sft_dataset_path = output_dir / "sft_train.jsonl"
    sft_output_dir = output_dir / "sft_v1"

    lines = [
        "# Next steps — train on the curated set",
        "",
        f"`stateset-agents improve` curated **{summary['curated_count']}** "
        f"example(s) (score >= {summary['threshold']}) out of "
        f"{summary['assistant_turn_count']} graded assistant turn(s), "
        f"mean score {summary['mean_score']:.3f}, into:",
        "",
        f"    {curated_path}",
        "",
        "## Option A — supervised fine-tune (fast, CPU-friendly path check)",
        "",
        "```bash",
        "python scripts/prepare_sft_dataset.py \\",
        f"    --input {curated_path} --format chat \\",
        f"    --output {sft_dataset_path} --min-score {summary['threshold']} --dedup",
        "",
        "python scripts/sft_from_curated.py \\",
        f"    --dataset {sft_dataset_path} \\",
        "    --base-model Qwen/Qwen3.5-0.8B \\",
        f"    --output-dir {sft_output_dir} \\",
        "    --num-epochs 3 --lora-r 16",
        "```",
        "",
        "## Option B — RL fine-tune with GSPO on the same signal",
        "",
        "```bash",
        "python examples/finetune_gspo.py \\",
        f"    --reward {summary['reward']} \\",
        f"    --dataset {curated_path}",
        "```",
        "",
        "Then chat with the result and grade again:",
        "",
        "```bash",
        f"stateset-agents chat --checkpoint {sft_output_dir} --history session2.jsonl",
        f"stateset-agents improve run --transcripts session2.jsonl --reward "
        f"{summary['reward']} --output {output_dir}",
        "```",
    ]
    return "\n".join(lines) + "\n"


@app.command("improve")
def improve(
    action: str = typer.Argument(
        "run", help="Subphase: 'run' (grade + curate + next steps) or 'status'."
    ),
    transcripts: str | None = typer.Option(
        None,
        "--transcripts",
        help="For --format transcripts: a directory of transcript JSONL files "
        "(one conversation per file, {'role','content'} per line — the shape "
        "`stateset-agents chat --history` writes). For --format openai/"
        "langchain: the single source log file to ingest first.",
    ),
    reward: str | None = typer.Option(
        None,
        "--reward",
        help="Reward function: gsm8k, customer_support, or tool_calling "
        "(rule-based, no API key required).",
    ),
    output: str = typer.Option(
        ..., "--output", "-o", help="Output directory for curated data + reports."
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", help="Minimum score for curation (default: 0.7)."
    ),
    format: str = typer.Option(
        "transcripts",
        "--format",
        "-f",
        help="Input format: 'transcripts' (already chat-history JSONL), "
        "'openai', or 'langchain' (ingested first via stateset_agents.data."
        "trajectory_ingest).",
    ),
) -> None:
    """Run the grade -> curate -> retrain loop as a single command.

    ``run`` (default):

        stateset-agents improve run --transcripts sessions/ \\
            --reward customer_support --output improved/

        stateset-agents improve run --transcripts logs.jsonl --format openai \\
            --reward customer_support --output improved/

    ``status`` prints the summary from a previous run:

        stateset-agents improve status --output improved/
    """
    output_dir = Path(output)

    if action == "status":
        summary_path = output_dir / SUMMARY_FILENAME
        if not summary_path.exists():
            _echo(
                f"No previous `improve run` found at {summary_path}. "
                "Run `stateset-agents improve run` first.",
                err=True,
            )
            raise typer.Exit(code=1)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _echo(json.dumps(summary, indent=2, sort_keys=True))
        return

    if action != "run":
        _echo(
            f"Unknown improve subcommand: {action!r}. Use 'run' or 'status'.", err=True
        )
        raise typer.Exit(code=2)

    if not transcripts:
        _echo("--transcripts is required.", err=True)
        raise typer.Exit(code=2)
    if not reward:
        _echo("--reward is required.", err=True)
        raise typer.Exit(code=2)

    fmt = format.strip().lower()
    if fmt not in ("transcripts", "openai", "langchain"):
        _echo(
            f"Unsupported --format '{format}'. Choose 'transcripts', 'openai', "
            "or 'langchain'.",
            err=True,
        )
        raise typer.Exit(code=2)

    _resolve_reward_name(reward)

    transcripts_path = Path(transcripts)
    if not transcripts_path.exists():
        _echo(f"--transcripts path not found: {transcripts_path}", err=True)
        raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: ingest (only when the source isn't already transcript JSONL).
    if fmt in ("openai", "langchain"):
        if not transcripts_path.is_file():
            _echo(
                f"--transcripts must be a file when --format is {fmt!r}: "
                f"{transcripts_path}",
                err=True,
            )
            raise typer.Exit(code=2)
        ingest_dir = output_dir / "ingested"
        try:
            transcript_files = _ingest_to_transcripts(fmt, transcripts_path, ingest_dir)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            _echo(f"Failed to ingest {transcripts_path}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        _echo(f"Ingested {len(transcript_files)} conversation(s) -> {ingest_dir}/")
    else:
        if not transcripts_path.is_dir():
            _echo(
                "--transcripts must be a directory when --format is 'transcripts': "
                f"{transcripts_path}",
                err=True,
            )
            raise typer.Exit(code=2)
        transcript_files = _collect_transcript_files(transcripts_path)
        if not transcript_files:
            _echo(f"No .jsonl transcript files found in {transcripts_path}", err=True)
            raise typer.Exit(code=1)

    # Phase 2: grade.
    import grade_transcript as gt  # local import: sys.path was patched above

    reward_fn = gt.get_reward(reward)
    per_transcript = asyncio.run(_grade_all(transcript_files, reward_fn))
    total_turns = sum(len(e["rows"]) for e in per_transcript)
    if total_turns == 0:
        _echo("No assistant turns found across the given transcripts.", err=True)
        raise typer.Exit(code=1)

    # Phase 3 + 4: curate above --threshold into <output>/curated.jsonl.
    curated_path = output_dir / CURATED_FILENAME
    curated_path.unlink(missing_ok=True)  # fresh curation per run
    curated_count = 0
    for entry in per_transcript:
        curated_count += gt.write_curated_examples(
            entry["path"], entry["turns"], entry["rows"], threshold, curated_path
        )

    # Phase 5: summary + next steps.
    summary = _build_summary(
        reward, threshold, per_transcript, curated_count, curated_path
    )
    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    next_steps_path = output_dir / NEXT_STEPS_FILENAME
    next_steps_path.write_text(
        _render_next_steps(summary, output_dir), encoding="utf-8"
    )

    _echo(
        f"Graded {summary['transcript_count']} transcript(s), "
        f"{summary['assistant_turn_count']} assistant turn(s), "
        f"mean score {summary['mean_score']:.3f}."
    )
    _echo(f"Curated {curated_count} example(s) (>= {threshold}) -> {curated_path}")
    _echo(f"Summary  -> {summary_path}")
    _echo(f"Next steps -> {next_steps_path}")
