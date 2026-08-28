"""StateSet Agents MCP server.

Exposes the framework's grade -> curate -> retrain "improve" loop as MCP
tools so any MCP client (Claude Code/Desktop, other agents) can drive it.

Every tool here is a thin wrapper: it validates inputs, calls the existing,
already-tested module functions (``stateset_agents.cli_improve``,
``stateset_agents.data.trajectory_ingest``, ``scripts/grade_transcript.py``,
``stateset_agents/core/model_presets.py``, ``examples/finetune_gspo.py``), and returns a
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

import json
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


def ingest_transcripts(input_path: str, format: str, output_dir: str) -> dict[str, Any]:
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
            from_openai_jsonl(source)
            if fmt == "openai"
            else from_langchain_json(source)
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


async def _grade_transcript_async(history_path: str, reward: str) -> dict[str, Any]:
    """Grade a transcript directly on the caller's event loop."""
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
        rows = await gt.grade_transcript(turns, [], reward_fn)

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
    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:
        return _err(exc)


async def grade_transcript(history_path: str, reward: str) -> dict[str, Any]:
    """Grade a single transcript JSONL, returning mean score + breakdown.

    The grading implementation is already asynchronous, so keep it on
    FastMCP's event loop and avoid dependence on a process-global executor.
    """
    return await _grade_transcript_async(history_path, reward)


async def improve_run(
    transcripts_dir: str,
    reward: str,
    output_dir: str,
    threshold: float = 0.7,
    format: str = "transcripts",
) -> dict[str, Any]:
    """Run the grade -> curate -> next-steps loop asynchronously."""
    try:
        from stateset_agents.cli_improve import (
            ImproveDataError,
            ImproveUsageError,
            run_improve_async,
        )

        try:
            return await run_improve_async(
                transcripts=transcripts_dir,
                reward=reward,
                output=output_dir,
                threshold=threshold,
                format=format,
            )
        except (ImproveUsageError, ImproveDataError) as exc:
            return {"error": str(exc)}
    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:
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
        from stateset_agents.core.model_presets import PRESETS, list_preset_names

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
    """Resolve a finetuning preset and return its dry-run config summary.

    This uses the same builders as the example driver in-process. It never
    downloads model weights or starts training.
    """
    try:
        from stateset_agents.core.model_presets import list_preset_names

        if model_preset not in list_preset_names():
            available = ", ".join(list_preset_names())
            return {
                "error": (
                    f"Unknown model preset {model_preset!r}. "
                    f"Available presets: {available}"
                )
            }

        from examples.finetune_gspo import build_gspo_config, preview_payload
        from examples.model_presets import get_preset

        preset = get_preset(model_preset)
        output_dir = f"./outputs/{model_preset.replace('.', '_')}_gspo"
        config = build_gspo_config(
            preset,
            task="customer_service",
            output_dir=output_dir,
        )
        payload = preview_payload(model_preset, preset, config)
        return {"model_preset": model_preset, "config": payload}
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
