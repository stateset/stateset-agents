"""StateSet Agents MCP server.

Exposes the framework's grade -> curate -> retrain "improve" loop as MCP
tools so any MCP client (Claude Code/Desktop, other agents) can drive it.

Every tool here is a thin wrapper: it validates inputs, calls the existing,
already-tested module functions (``stateset_agents.cli_improve``,
``stateset_agents.data.trajectory_ingest``, ``scripts/grade_transcript.py``,
``examples/model_presets.py``, ``examples/finetune_gspo.py``), and returns a
structured dict. No grading/curation/training logic is reimplemented here.

v1 scope: no tool starts real GPU training. ``dry_run_finetune`` only ever
runs ``examples/finetune_gspo.py --dry-run`` (stub backend, no model
download, no training).

Usage::

    pip install 'stateset-agents[mcp]'
    stateset-agents mcp --transport stdio

Or run directly::

    python -m stateset_agents.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

MCP_INSTALL_HINT = "pip install stateset-agents[mcp]"


def _require_mcp() -> Any:
    """Lazily import the ``mcp`` package, raising a clear error if missing."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised via subprocess test
        raise ImportError(
            "The 'mcp' package is required for the StateSet Agents MCP server. "
            f"Install it with: {MCP_INSTALL_HINT}"
        ) from exc
    return FastMCP


def _err(exc: Exception) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Tool implementations (plain functions — importable/testable without an
# MCP client, and registered onto the FastMCP instance in create_server()).
# ---------------------------------------------------------------------------


def list_rewards() -> dict[str, Any]:
    """List reward function names supported by the grade/improve loop."""
    from stateset_agents.cli_improve import KNOWN_REWARDS

    return {"rewards": list(KNOWN_REWARDS)}


def ingest_transcripts(
    input_path: str, format: str, output_dir: str
) -> dict[str, Any]:
    """Convert an OpenAI/LangChain conversation log into transcript JSONLs.

    Wraps ``stateset_agents.data.trajectory_ingest`` + the same
    ``to_grading_history`` writing ``stateset-agents ingest`` does. Writes
    one ``<output_dir>/conversation_<N>.jsonl`` file per conversation.
    """
    try:
        fmt = format.strip().lower()
        if fmt not in ("openai", "langchain"):
            return {
                "error": f"Unsupported format '{format}'. Choose 'openai' or 'langchain'."
            }

        source = Path(input_path)
        if not source.exists():
            return {"error": f"Input file not found: {source}"}

        from stateset_agents.data.trajectory_ingest import (
            from_langchain_json,
            from_openai_jsonl,
            to_grading_history,
        )

        trajectories = (
            from_openai_jsonl(source) if fmt == "openai" else from_langchain_json(source)
        )
        if not trajectories:
            return {"error": f"No conversations found in {source}"}

        dest_dir = Path(output_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for i, traj in enumerate(trajectories):
            conv_path = dest_dir / f"conversation_{i}.jsonl"
            with conv_path.open("w", encoding="utf-8") as f:
                for turn in to_grading_history(traj):
                    f.write(json.dumps(turn) + "\n")
            written.append(str(conv_path))

        return {
            "conversation_count": len(trajectories),
            "turn_count": sum(len(t.turns) for t in trajectories),
            "output_dir": str(dest_dir),
            "files": written,
        }
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _err(exc)


def grade_transcript(history_path: str, reward: str) -> dict[str, Any]:
    """Grade a single transcript JSONL, returning mean score + breakdown."""
    try:
        history = Path(history_path)
        if not history.exists():
            return {"error": f"Transcript not found: {history}"}

        _ensure_scripts_on_path()
        import grade_transcript as gt

        try:
            reward_fn = gt.get_reward(reward)
        except ValueError as exc:
            return {"error": str(exc)}

        turns = gt.load_transcript(history)
        rows = asyncio.run(gt.grade_transcript(turns, [], reward_fn))

        scores = [row["score"] for row in rows]
        mean_score = (sum(scores) / len(scores)) if scores else 0.0

        breakdown_totals: dict[str, list[float]] = {}
        for row in rows:
            for key, value in (row.get("breakdown") or {}).items():
                breakdown_totals.setdefault(key, []).append(float(value))
        breakdown = {
            key: (sum(values) / len(values)) if values else 0.0
            for key, values in breakdown_totals.items()
        }

        return {
            "history_path": str(history),
            "reward": reward,
            "assistant_turn_count": len(rows),
            "mean_score": mean_score,
            "breakdown": breakdown,
            "rows": rows,
        }
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _err(exc)


def improve_run(
    transcripts_dir: str,
    reward: str,
    output_dir: str,
    threshold: float = 0.7,
    format: str = "transcripts",
) -> dict[str, Any]:
    """Run the grade -> curate -> next-steps loop. Wraps ``cli_improve.run_improve``."""
    try:
        from stateset_agents.cli_improve import (
            ImproveDataError,
            ImproveUsageError,
            run_improve,
        )

        try:
            return run_improve(
                transcripts=transcripts_dir,
                reward=reward,
                output=output_dir,
                threshold=threshold,
                format=format,
            )
        except (ImproveUsageError, ImproveDataError) as exc:
            return {"error": str(exc)}
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _err(exc)


def improve_status(output_dir: str) -> dict[str, Any]:
    """Return the summary JSON written by a previous ``improve_run``."""
    try:
        from stateset_agents.cli_improve import ImproveDataError, get_improve_status

        try:
            return get_improve_status(output_dir)
        except ImproveDataError as exc:
            return {"error": str(exc)}
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _err(exc)


def list_model_presets() -> dict[str, Any]:
    """List model preset names + key hyperparameter fields."""
    try:
        from examples.model_presets import PRESETS, list_preset_names

        presets = []
        for name in list_preset_names():
            preset = PRESETS[name]
            presets.append(
                {
                    "name": name,
                    "model_id": preset.model_id,
                    "tokenizer_id": preset.tokenizer_id,
                    "max_prompt_length": preset.max_prompt_length,
                    "max_completion_length": preset.max_completion_length,
                    "learning_rate": preset.learning_rate,
                    "num_generations": preset.num_generations,
                    "bf16": preset.bf16,
                    "use_4bit": preset.use_4bit,
                    "use_8bit": preset.use_8bit,
                    "starter_module": preset.starter_module,
                }
            )
        return {"presets": presets}
    except (ValueError, OSError) as exc:
        return _err(exc)


def dry_run_finetune(model_preset: str) -> dict[str, Any]:
    """Run ``examples/finetune_gspo.py --model <preset> --dry-run`` and return its config summary.

    v1 only ever dry-runs (stub backend, no real model weights, no
    training) — this tool never starts GPU training.
    """
    try:
        from examples.model_presets import list_preset_names

        if model_preset not in list_preset_names():
            available = ", ".join(list_preset_names())
            return {
                "error": (
                    f"Unknown model preset {model_preset!r}. "
                    f"Available presets: {available}"
                )
            }

        script = _REPO_ROOT / "examples" / "finetune_gspo.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--model",
                model_preset,
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(_REPO_ROOT),
            timeout=180,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "error": f"dry-run exited with code {proc.returncode}",
                "stderr": proc.stderr[-4000:],
            }

        # preview_payload() prints a single JSON object to stdout.
        stdout = proc.stdout.strip()
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return {
                "error": "Could not parse dry-run output as JSON",
                "stdout": stdout[-4000:],
            }
        return {"model_preset": model_preset, "config": payload}
    except subprocess.TimeoutExpired as exc:
        return _err(exc)
    except (ValueError, OSError) as exc:
        return _err(exc)


def _ensure_scripts_on_path() -> None:
    scripts_dir = str(_SCRIPTS_DIR)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server() -> Any:
    """Build and return a configured ``FastMCP`` server instance.

    Raises ``ImportError`` with an install hint if the ``mcp`` package is
    not installed.
    """
    FastMCP = _require_mcp()  # noqa: N806
    _ensure_scripts_on_path()

    mcp = FastMCP(
        "stateset-agents",
        instructions=(
            "Drives the StateSet Agents framework's improvement loop: "
            "ingest third-party conversation logs, grade transcripts with "
            "rule-based reward functions, curate high-scoring examples, and "
            "preview (dry-run only) fine-tuning configs. No tool starts "
            "real GPU training in this version."
        ),
    )

    mcp.tool()(list_rewards)
    mcp.tool()(ingest_transcripts)
    mcp.tool()(grade_transcript)
    mcp.tool()(improve_run)
    mcp.tool()(improve_status)
    mcp.tool()(list_model_presets)
    mcp.tool()(dry_run_finetune)

    return mcp


def main() -> None:
    """Entry point for ``stateset-agents mcp`` / ``python -m stateset_agents.mcp_server``."""
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
