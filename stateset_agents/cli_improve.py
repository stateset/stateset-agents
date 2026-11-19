"""``stateset-agents improve`` — the grade -> curate -> retrain loop, one command.

Thin orchestrator over existing, tested pieces:

* ``stateset_agents.data.trajectory_ingest`` — converts OpenAI/LangChain logs
  into transcript JSONLs (same shape ``chat --history`` writes).
* ``stateset_agents.data.grade_transcript`` — scores every assistant turn with a reward
  function and curates high-scoring (prompt, response) pairs.
* ``stateset_agents.training.sft`` — the training path that consumes the curated
  set (referenced, not reimplemented, in the generated ``next_steps.md``).

No new grading or curation logic lives here — every phase below imports and
calls the packaged grading and training functions.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app

_echo = _cli._echo

# "nsr" is network-backed but deterministic and rule-based (a symbolic
# verifier, not an LLM judge), so it is allowed where judge rewards are
# refused; it runs fail-closed (unreachable verifier scores 0.0).
KNOWN_REWARDS = ("gsm8k", "customer_support", "tool_calling", "nsr")
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


async def _grade_all(
    transcript_files: list[Path],
    reward: Any,
    reward_name: str,
    context_path: Path | None,
) -> list[dict[str, Any]]:
    """Load aligned reference rows and grade the assistant turns in each file."""
    from stateset_agents.data import grade_transcript as gt

    if (
        context_path is not None
        and not context_path.is_dir()
        and len(transcript_files) != 1
    ):
        raise ImproveUsageError(
            "For multiple transcripts, --context-file must be a directory of matching JSONL files"
        )
    per_transcript: list[dict[str, Any]] = []
    for path in transcript_files:
        turns = gt.load_transcript(path)
        sidecar = (
            context_path / path.name
            if context_path and context_path.is_dir()
            else context_path
        )
        try:
            contexts = gt.load_contexts(
                sidecar, expected_count=sum(t["role"] == "assistant" for t in turns)
            )
        except (ValueError, OSError) as exc:
            raise ImproveDataError(
                f"Invalid grading contexts for {path.name}: {exc}"
            ) from exc
        rows = await gt.grade_transcript(
            turns, contexts, reward, reward_name=reward_name
        )
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
        scores = [float(r["score"]) for r in rows if r["score"] is not None]
        all_scores.extend(scores)
        for row in rows:
            for key, value in (row.get("breakdown") or {}).items():
                if isinstance(value, (int, float)):
                    breakdown_totals.setdefault(key, []).append(float(value))
        transcript_summaries.append(
            {
                "name": entry["path"].name,
                "assistant_turns": len(rows),
                "ungradable_turns": [
                    {
                        "assistant_turn_idx": row["assistant_turn_idx"],
                        "reason": row["explanation"],
                    }
                    for row in rows
                    if row["score"] is None
                ],
                "mean_score": (sum(scores) / len(scores)) if scores else None,
                "above_threshold": sum(1 for s in scores if s >= threshold),
            }
        )

    mean_score = (sum(all_scores) / len(all_scores)) if all_scores else None
    total_turns = sum(row["assistant_turns"] for row in transcript_summaries)
    reward_breakdown = {
        key: (sum(values) / len(values)) if values else 0.0
        for key, values in breakdown_totals.items()
    }

    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reward": reward_name,
        "threshold": threshold,
        "transcript_count": len(per_transcript),
        "assistant_turn_count": total_turns,
        "graded_turn_count": len(all_scores),
        "ungradable_turn_count": total_turns - len(all_scores),
        "mean_score": mean_score,
        "above_threshold_count": sum(1 for s in all_scores if s >= threshold),
        "reward_breakdown": reward_breakdown,
        "curated_count": curated_count,
        "curated_path": str(curated_path),
        "transcripts": transcript_summaries,
    }


def _render_next_steps(summary: dict[str, Any], output_dir: Path) -> str:
    """Render executable installed-package commands with safely quoted paths."""
    curated_path = output_dir / CURATED_FILENAME
    curated = shlex.quote(str(curated_path))
    dataset = shlex.quote(str(output_dir / "sft_train.jsonl"))
    checkpoint = shlex.quote(str(output_dir / "sft_v1"))
    output = shlex.quote(str(output_dir))
    reward = shlex.quote(str(summary["reward"]))
    needs_reference = summary["reward"] in ("gsm8k", "tool_calling")
    round2_reference = " --context-file round2-references/" if needs_reference else ""
    reference_note = (
        " Create matching reference files in `round2-references/` before grading."
        if needs_reference
        else ""
    )
    mean = (
        f"{summary['mean_score']:.3f}"
        if summary["mean_score"] is not None
        else "unavailable"
    )
    preamble = (
        "# Next steps — train on the curated set\n\n"
        f"Curated **{summary['curated_count']}** examples from "
        f"{summary['assistant_turn_count']} assistant turns "
        f"at threshold {summary['threshold']}. Mean score: {mean}.\n\n"
        f"Dataset: `{curated_path}`. Exact conversation duplicates are removed, "
        "so curated_count may be lower than above_threshold_count.\n\n"
    )
    if not summary["curated_count"]:
        if needs_reference and summary.get("ungradable_turn_count"):
            preamble += (
                "Review the reasons in `improve_summary.json` and supply one grading "
                "reference per assistant turn using `--context-file`.\n\n"
            )
        return preamble + (
            "No examples passed curation. Review improve_summary.json and the "
            "source conversations before preparing a training dataset.\n"
        )
    return preamble + f"""## Prepare chat-format training data

These commands work with an installed package; no repository checkout is required.
Review the examples for correctness before training. Heuristic scores alone do
not establish that a response follows your business policy.

```bash
python -m stateset_agents.data.prepare_sft_dataset --input {curated} --format chat --output {dataset} --min-score {summary['threshold']} --dedup
```

## Preview local training

This validates the dataset and prints an SFT plan. It does not train or create a checkpoint.

```bash
python -m stateset_agents.training.sft --dataset {dataset} --base-model Qwen/Qwen3.5-0.8B --output-dir {checkpoint} --num-epochs 3 --lora-r 16 --dry-run
```

To train on a supported local GPU, install `stateset-agents[training]` and rerun
without `--dry-run`. Inspect the training result and held-out evaluation before serving.

## Preview remote training

RunPod has retained live training evidence; see the provider guide for its tested
configurations and limitations. This command only prints a resource plan and
allocates no compute. A live run needs credentials and an explicit cost budget.

```bash
stateset-agents train-remote --provider runpod --dataset {dataset} --base-model Qwen/Qwen3.5-0.8B --output-dir {checkpoint} --plan-only
```

The CLI's compatibility default remains Modal; its live transport is unverified.
Choose the provider explicitly. `--dry-run` on a remote provider can still allocate
compute; `--plan-only` is the non-billable planning command.

## Explore RL configuration

This prints a GSPO preset without loading weights or training. The task argument
labels the configuration; it does not wire a task-specific environment or reward.
See the training guide to connect your scenarios and reward before an RL run.

```bash
python -m stateset_agents.training.gspo_preview --model qwen3.5-0.8b --task {reward}
```

## Evaluate and capture another conversation

After the checkpoint passes your evaluation, capture another conversation.{reference_note}

```bash
mkdir -p round2
stateset-agents chat --checkpoint {checkpoint} --history round2/session.jsonl
stateset-agents improve run --transcripts round2/ --reward {reward}{round2_reference} --output {output}
```
"""


def _load_persona(persona_path: str) -> dict[str, Any]:
    """Load and validate a persona config file.

    The file is JSON: ``{"opener": ["hi, this is"], "signoff": ["best,"]}``.
    Both keys are optional lists of case-insensitive substrings; the opener
    must appear in the first assistant turn, the signoff in the last.
    """
    path = Path(persona_path)
    if not path.exists():
        raise ImproveUsageError(f"--persona file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ImproveUsageError(f"Failed to read --persona {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ImproveUsageError(
            f"--persona {path} must be a JSON object with optional "
            "'opener'/'signoff' string lists."
        )
    unknown = set(data) - {"opener", "signoff"}
    if unknown:
        raise ImproveUsageError(
            f"--persona {path} has unknown key(s): {', '.join(sorted(unknown))}. "
            "Allowed: opener, signoff."
        )
    for key in ("opener", "signoff"):
        value = data.get(key, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ImproveUsageError(
                f"--persona {path}: {key!r} must be a list of strings."
            )
    if not data.get("opener") and not data.get("signoff"):
        raise ImproveUsageError(
            f"--persona {path} must define at least one of 'opener'/'signoff'."
        )
    return data


def _build_reward(
    reward: str,
    persona: dict[str, Any] | None,
    llm_judge: bool,
    log: Any,
) -> Any:
    """Build the grading reward: rule-based, optionally judge-blended.

    With ``llm_judge=True``, the rule-based reward is blended with an LLM
    judge **only if one is configured** (API key resolvable and optional
    dependencies installed) — otherwise this logs and degrades gracefully to
    the pure rule-based reward, keeping the loop offline-safe.
    """
    from stateset_agents.data import grade_transcript as gt

    try:
        reward_fn = gt.get_reward(reward, persona=persona)
    except ValueError as exc:
        raise ImproveUsageError(str(exc)) from exc
    if not llm_judge:
        return reward_fn

    judge = None
    try:
        from stateset_agents.rewards.llm_judge import JudgeConfig, LLMJudge

        config = JudgeConfig()
        if config.api_key:
            judge = LLMJudge(config)
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        log(f"LLM judge unavailable ({exc}); continuing rule-based.")

    if judge is None:
        log(
            "LLM judge requested but not configured (no API key found); "
            "grading with the rule-based reward only."
        )
        return reward_fn

    from stateset_agents.rewards.llm_judge_adapter import LLMJudgeRewardWithFallback

    log("LLM judge active: blending judge scores with the rule-based reward.")
    return LLMJudgeRewardWithFallback(
        judge=judge,
        heuristic=reward_fn,
        require_reference_match=reward in ("gsm8k", "tool_calling"),
    )


class ImproveUsageError(ValueError):
    """Raised by :func:`run_improve` for bad arguments (CLI exit code 2)."""


class ImproveDataError(ValueError):
    """Raised by :func:`run_improve` for data problems (CLI exit code 1)."""


async def run_improve_async(
    *,
    transcripts: str,
    reward: str,
    output: str,
    threshold: float = 0.7,
    format: str = "transcripts",
    persona: str | None = None,
    llm_judge: bool = False,
    context_file: str | None = None,
    echo: Any = None,
) -> dict[str, Any]:
    """Run the grade -> curate -> retrain loop and return the summary dict.

    This is the single implementation of the ``improve run`` orchestration —
    both the ``stateset-agents improve run`` CLI command and the MCP
    ``improve_run`` tool call this function. Raises :class:`ImproveUsageError`
    for bad arguments (CLI maps this to exit code 2) or
    :class:`ImproveDataError` for data problems (CLI maps this to exit code
    1). ``echo`` is an optional ``callable(str) -> None`` used for progress
    messages (defaults to no-op).

    ``persona`` is an optional path to a JSON persona config
    (``{"opener": [...], "signoff": [...]}``) scored as a persona-fidelity
    component (customer_support reward only). ``llm_judge=True`` blends an
    LLM judge into grading when one is configured (API key present),
    degrading gracefully to rule-based grading otherwise.
    """
    _log = echo if echo is not None else (lambda _msg: None)

    if not transcripts:
        raise ImproveUsageError("--transcripts is required.")
    if not reward:
        raise ImproveUsageError("--reward is required.")

    fmt = format.strip().lower()
    if fmt not in ("transcripts", "openai", "langchain"):
        raise ImproveUsageError(
            f"Unsupported --format '{format}'. Choose 'transcripts', 'openai', "
            "or 'langchain'."
        )

    if reward not in KNOWN_REWARDS:
        lowered = reward.lower()
        if any(hint in lowered for hint in JUDGE_REWARD_HINTS):
            raise ImproveUsageError(
                f"Unsupported --reward '{reward}': the `improve` loop runs offline "
                "with rule-based rewards only, so LLM-judge rewards (which require "
                "an API key) are not supported here. Choose one of: "
                f"{', '.join(KNOWN_REWARDS)}."
            )
        raise ImproveUsageError(
            f"Unknown --reward '{reward}'. Choose one of: {', '.join(KNOWN_REWARDS)}."
        )

    persona_config = _load_persona(persona) if persona else None

    transcripts_path = Path(transcripts)
    if not transcripts_path.exists():
        raise ImproveUsageError(f"--transcripts path not found: {transcripts_path}")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: ingest (only when the source isn't already transcript JSONL).
    context_path = Path(context_file) if context_file else None
    if context_path is not None and not context_path.exists():
        raise ImproveDataError(f"Grading context path not found: {context_path}")
    if fmt in ("openai", "langchain"):
        if not transcripts_path.is_file():
            raise ImproveUsageError(
                f"--transcripts must be a file when --format is {fmt!r}: "
                f"{transcripts_path}"
            )
        ingest_dir = output_dir / "ingested"
        try:
            transcript_files = _ingest_to_transcripts(fmt, transcripts_path, ingest_dir)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise ImproveDataError(
                f"Failed to ingest {transcripts_path}: {exc}"
            ) from exc
        _log(f"Ingested {len(transcript_files)} conversation(s) -> {ingest_dir}/")
    else:
        if not transcripts_path.is_dir():
            raise ImproveUsageError(
                "--transcripts must be a directory when --format is 'transcripts': "
                f"{transcripts_path}"
            )
        transcript_files = _collect_transcript_files(transcripts_path)
        if context_path is not None and context_path.is_file():
            transcript_files = [
                p for p in transcript_files if p.resolve() != context_path.resolve()
            ]
        if not transcript_files:
            raise ImproveDataError(
                f"No .jsonl transcript files found in {transcripts_path}"
            )

    # Phase 2: grade.
    from stateset_agents.data import grade_transcript as gt

    reward_fn = _build_reward(reward, persona_config, llm_judge, _log)
    per_transcript = await _grade_all(transcript_files, reward_fn, reward, context_path)
    total_turns = sum(len(e["rows"]) for e in per_transcript)
    if total_turns == 0:
        raise ImproveDataError("No assistant turns found across the given transcripts.")

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

    _log(
        f"Graded {summary['transcript_count']} transcript(s), "
        f"{summary['assistant_turn_count']} assistant turn(s), "
        f"{summary['ungradable_turn_count']} ungradable."
    )
    _log(f"Curated {curated_count} example(s) (>= {threshold}) -> {curated_path}")
    _log(f"Summary  -> {summary_path}")
    _log(f"Next steps -> {next_steps_path}")

    summary["summary_path"] = str(summary_path)
    summary["next_steps_path"] = str(next_steps_path)
    return summary


def run_improve(
    *,
    transcripts: str,
    reward: str,
    output: str,
    threshold: float = 0.7,
    format: str = "transcripts",
    persona: str | None = None,
    llm_judge: bool = False,
    context_file: str | None = None,
    echo: Any = None,
) -> dict[str, Any]:
    """Synchronous CLI wrapper around :func:`run_improve_async`."""
    return asyncio.run(
        run_improve_async(
            transcripts=transcripts,
            reward=reward,
            output=output,
            threshold=threshold,
            format=format,
            persona=persona,
            llm_judge=llm_judge,
            context_file=context_file,
            echo=echo,
        )
    )


def get_improve_status(output: str) -> dict[str, Any]:
    """Return the summary JSON from a previous ``improve run``.

    Raises :class:`ImproveDataError` when no previous run is found at
    ``<output>/improve_summary.json``.
    """
    output_dir = Path(output)
    summary_path = output_dir / SUMMARY_FILENAME
    if not summary_path.exists():
        raise ImproveDataError(
            f"No previous `improve run` found at {summary_path}. "
            "Run `stateset-agents improve run` first."
        )
    result: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    return result


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
    context_file: str | None = typer.Option(
        None,
        "--context-file",
        help="Per-assistant-turn JSONL, or a directory with one matching file per transcript.",
    ),
    persona: str | None = typer.Option(
        None,
        "--persona",
        help="Path to a JSON persona config ({'opener': [...], 'signoff': "
        "[...]}) scored as a persona-fidelity component "
        "(customer_support reward only).",
    ),
    llm_judge: bool = typer.Option(
        False,
        "--llm-judge",
        help="Blend an LLM judge into grading when one is configured "
        "(API key present); degrades gracefully to rule-based grading "
        "otherwise.",
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
    if action == "status":
        try:
            summary = get_improve_status(output)
        except ImproveDataError as exc:
            _echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _echo(json.dumps(summary, indent=2, sort_keys=True))
        return

    if action != "run":
        _echo(
            f"Unknown improve subcommand: {action!r}. Use 'run' or 'status'.", err=True
        )
        raise typer.Exit(code=2)

    try:
        run_improve(
            transcripts=transcripts or "",
            reward=reward or "",
            output=output,
            threshold=threshold,
            format=format,
            persona=persona,
            llm_judge=llm_judge,
            context_file=context_file,
            echo=_echo,
        )
    except ImproveUsageError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except ImproveDataError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
