"""StateSet Agents MCP server.

Exposes the framework's grade -> curate -> retrain "improve" loop as MCP
tools so any MCP client (Claude Code/Desktop, other agents) can drive it.

Every tool here is a thin wrapper: it validates inputs, calls the existing,
already-tested module functions (``stateset_agents.cli_improve``,
``stateset_agents.data.trajectory_ingest``, ``stateset_agents.data.grade_transcript``,
``stateset_agents.core.model_presets``, ``stateset_agents.training.gspo_preview``), and returns a
structured dict. No grading/curation/training logic is reimplemented here.

v1 scope: no tool starts real GPU training. ``dry_run_finetune`` only ever
resolves packaged preset configuration (no model download or training).

Usage::

    pip install 'stateset-agents[mcp]'
    stateset-agents mcp --transport stdio

Or run directly::

    python -m stateset_agents.mcp_server
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


async def _grade_transcript_async(
    history_path: str, reward: str, context_file: str | None = None
) -> dict[str, Any]:
    """Grade a transcript directly on the caller's event loop."""
    try:
        history = Path(history_path)
        if not history.exists():
            return {"error": f"Transcript not found: {history}"}

        from stateset_agents.data import grade_transcript as gt

        try:
            reward_fn = gt.get_reward(reward)
        except ValueError as exc:
            return {"error": str(exc)}

        turns = gt.load_transcript(history)
        contexts = gt.load_contexts(
            Path(context_file) if context_file else None,
            expected_count=sum(t["role"] == "assistant" for t in turns),
        )
        rows = await gt.grade_transcript(turns, contexts, reward_fn, reward_name=reward)

        scores = [row["score"] for row in rows if row["score"] is not None]
        mean_score = (sum(scores) / len(scores)) if scores else None

        breakdown_totals: dict[str, list[float]] = {}
        for row in rows:
            for key, value in (row.get("breakdown") or {}).items():
                if isinstance(value, (int, float)):
                    breakdown_totals.setdefault(key, []).append(float(value))
        breakdown = {
            key: (sum(values) / len(values)) if values else 0.0
            for key, values in breakdown_totals.items()
        }

        return {
            "history_path": str(history),
            "reward": reward,
            "assistant_turn_count": len(rows),
            "graded_turn_count": len(scores),
            "ungradable_turn_count": len(rows) - len(scores),
            "mean_score": mean_score,
            "breakdown": breakdown,
            "rows": rows,
        }
    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:
        return _err(exc)


async def grade_transcript(
    history_path: str, reward: str, context_file: str | None = None
) -> dict[str, Any]:
    """Grade a single transcript JSONL, returning mean score + breakdown.

    The grading implementation is already asynchronous, so keep it on
    FastMCP's event loop and avoid dependence on a process-global executor.
    """
    return await _grade_transcript_async(history_path, reward, context_file)


async def improve_run(
    transcripts_dir: str,
    reward: str,
    output_dir: str,
    threshold: float = 0.7,
    format: str = "transcripts",
    context_file: str | None = None,
) -> dict[str, Any]:
    """Run the grade -> curate -> next-steps loop asynchronously."""
    try:
        from stateset_agents.cli_improve import (
            ImproveDataError,
            ImproveUsageError,
            run_improve_async,
        )

        try:
            res = await run_improve_async(
                transcripts=transcripts_dir,
                reward=reward,
                output=output_dir,
                threshold=threshold,
                format=format,
                context_file=context_file,
            )
            return dict(res)
        except (ImproveUsageError, ImproveDataError) as exc:
            return {"error": str(exc)}
    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:
        return _err(exc)


def improve_status(output_dir: str) -> dict[str, Any]:
    """Return the summary JSON written by a previous ``improve_run``."""
    try:
        from stateset_agents.cli_improve import ImproveDataError, get_improve_status

        try:
            return dict(get_improve_status(output_dir))
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

        from stateset_agents.core.model_presets import get_preset
        from stateset_agents.training.gspo_preview import (
            build_gspo_config,
            preview_payload,
        )

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


# ---------------------------------------------------------------------------
# Weight Studio MCP Tools (3D inspection, surgical editing, CAA steering,
# causal circuits, eval gating, and SHA-256 verified .safetensors export)
# ---------------------------------------------------------------------------

_STUDIO_RUNTIMES: dict[str, Any] = {}


def _get_mcp_studio_runtime(workspace: str = ".") -> Any:
    from stateset_agents.weight_studio import StudioRuntime

    resolved = str(Path(workspace).resolve())
    rt = _STUDIO_RUNTIMES.get(resolved)
    if rt is None:
        rt = StudioRuntime(workspace_root=Path(resolved))
        _STUDIO_RUNTIMES[resolved] = rt
    return rt


def studio_list_models(workspace: str = ".", refresh: bool = False) -> dict[str, Any]:
    """List local .safetensors checkpoints, sharded models, and StateSet architecture twins."""
    try:
        rt = _get_mcp_studio_runtime(workspace)
        models = (
            rt.inspector.refresh_catalog() if refresh else rt.inspector.list_models()
        )
        return {
            "workspace": str(rt.inspector.workspace_root),
            "count": len(models),
            "models": models,
        }
    except Exception as exc:
        return _err(exc)


def studio_inspect_model(
    model_id: str,
    workspace: str = ".",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Inspect a model's layer architecture, parameter counts, top layer norms, and active edits."""
    try:
        rt = _get_mcp_studio_runtime(workspace)
        overrides = rt.editor.get_overrides(model_id, checkpoint_id)
        struct = rt.inspector.get_model_structure(
            model_id, checkpoint_id, overrides=overrides if overrides else None
        )
        history = rt.editor.get_history(model_id, checkpoint_id)
        return {
            "model": struct["model"],
            "checkpoint_id": struct["checkpoint_id"],
            "summary": struct["summary"],
            "layers": [
                {
                    "index": layer["index"],
                    "name": layer["name"],
                    "layer_type": layer["layer_type"],
                    "tensor_count": layer["tensor_count"],
                    "frobenius_norm": layer["frobenius_norm"],
                    "has_edits": layer["has_edits"],
                }
                for layer in struct["layers"]
            ],
            "edit_history": history,
        }
    except Exception as exc:
        return _err(exc)


def studio_apply_edit(
    model_id: str,
    op: str,
    target: str,
    params: dict[str, Any] | None = None,
    label: str = "",
    workspace: str = ".",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Apply a non-destructive surgical weight edit (scale, prune, clamp, svd_truncate,
    head_edit, rank1_edit, steer_bake, region_edit, blend, or noise)."""
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.editor.apply_edit(
                model_id,
                op,
                target,
                dict(params or {}),
                checkpoint_id=checkpoint_id,
                label=label,
            )
        )
    except Exception as exc:
        return _err(exc)


def studio_extract_steering(
    model_id: str,
    positive_prompt: str,
    negative_prompt: str,
    name: str = "Steering Vector",
    layer_index: int | None = None,
    workspace: str = ".",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Extract a Contrastive Activation Steering (CAA / RepE) vector and locate the optimal layer."""
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.inference.extract_steering_vector(
                model_id,
                positive_prompt,
                negative_prompt,
                checkpoint_id=checkpoint_id,
                name=name,
                layer_index=layer_index,
            )
        )
    except Exception as exc:
        return _err(exc)


def studio_discover_circuit(
    model_id: str,
    prompt: str,
    top_k: int = 10,
    system_prompt: str = "",
    workspace: str = ".",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Discover the Top-K causal attention heads & MLP cores driving a prompt."""
    try:
        rt = _get_mcp_studio_runtime(workspace)
        res = rt.inference.discover_causal_circuit(
            model_id,
            prompt,
            checkpoint_id=checkpoint_id,
            system_prompt=system_prompt,
            top_k=top_k,
        )
        return {
            "model_id": res["model_id"],
            "scenario": res["scenario"],
            "tokens": res["tokens"],
            "summary": res["summary"],
            "top_nodes": res["top_nodes"],
            "arc_count": len(res["arcs"]),
        }
    except Exception as exc:
        return _err(exc)


def studio_eval_and_export(
    model_id: str,
    export_name: str = "",
    run_merge_probe: bool = True,
    require_go: bool = True,
    max_cases: int = 6,
    workspace: str = ".",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Run the Pre-Export Batch Eval Gate (GO/WARN/BLOCK) and optionally export a
    SHA-256 reload-verified .safetensors checkpoint + portable recipe."""
    try:
        rt = _get_mcp_studio_runtime(workspace)
        gate = rt.inference.run_batch_eval(
            model_id, checkpoint_id=checkpoint_id, max_cases=max_cases
        )
        out: dict[str, Any] = {
            "eval_gate": gate,
            "exported": False,
            "manifest": None,
            "recipe": rt.editor.export_recipe(model_id, checkpoint_id),
        }
        if export_name.strip():
            if require_go and gate["verdict"] != "GO":
                out["export_blocked_reason"] = (
                    f"Gate verdict is {gate['verdict']}; pass require_go=False to override."
                )
            else:
                manifest = rt.editor.export_verified_checkpoint(
                    model_id,
                    export_name.strip(),
                    checkpoint_id=checkpoint_id,
                    run_merge_probe=run_merge_probe,
                )
                out["exported"] = True
                out["manifest"] = manifest
        return out
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server() -> Any:
    """Build and return a configured ``FastMCP`` server instance.

    Raises ``ImportError`` with an install hint if the ``mcp`` package is
    not installed.
    """
    FastMCP = _require_mcp()  # noqa: N806

    mcp = FastMCP(
        "stateset-agents",
        instructions=(
            "Drives the StateSet Agents framework's improvement loop and "
            "3D Weight Studio: ingest logs, grade transcripts, curate data, "
            "preview fine-tuning configs, inspect weights, apply non-destructive "
            "surgical edits, extract CAA steering vectors, discover causal circuits, "
            "and run eval-gated .safetensors exports. No tool starts real GPU "
            "training in this version."
        ),
    )

    mcp.tool()(list_rewards)
    mcp.tool()(ingest_transcripts)
    mcp.tool()(grade_transcript)
    mcp.tool()(improve_run)
    mcp.tool()(improve_status)
    mcp.tool()(list_model_presets)
    mcp.tool()(dry_run_finetune)
    mcp.tool()(studio_list_models)
    mcp.tool()(studio_inspect_model)
    mcp.tool()(studio_apply_edit)
    mcp.tool()(studio_extract_steering)
    mcp.tool()(studio_discover_circuit)
    mcp.tool()(studio_eval_and_export)

    return mcp


def main() -> None:
    """Entry point for ``stateset-agents mcp`` / ``python -m stateset_agents.mcp_server``."""
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
