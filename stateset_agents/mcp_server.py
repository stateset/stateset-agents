"""StateSet Agents MCP server.

Exposes the framework's grade -> curate -> retrain "improve" loop AND the full
3D Weight Studio (model inspection, Wanda/SVD sensitivity profiling, live
inference & Logit Lens, causal circuits, residual-stream tracing, SAE sparse
features, CAA behavioral steering, non-destructive surgical editing with
undo/redo/recipes, Multi-Edit A/B Arena, Surgical Auto-Repair, 95% CI Batch
Eval Gate, PEFT LoRA / `.safetensors` / GGUF v3 / HF Model Card / interactive
3D HTML PR report export, workbench sessions, and the Autonomous
Transcript-to-Surgery Self-Healing Loop) as MCP tools, live resources, and
guided workflow prompts so any MCP client (Claude Code, Claude Desktop, Cursor,
Antigravity, LangGraph, or custom AI agents) can drive it end-to-end.

Every tool is a thin wrapper over the same, already-tested module functions
the CLI and Studio API use. No grading, curation, or surgery logic is
reimplemented here.

Scope note: no MCP tool launches real remote GPU training; ``dry_run_finetune``
only resolves packaged preset configuration.

Usage::

    pip install 'stateset-agents[mcp]'
    stateset-agents mcp --transport stdio --workspace .

Or run directly::

    python -m stateset_agents.mcp_server
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

MCP_INSTALL_HINT = "pip install stateset-agents[mcp]"

_DEFAULT_WORKSPACE: str = ""
_STUDIO_RUNTIMES: dict[str, Any] = {}


def set_default_workspace(workspace: str | Path | None) -> str:
    """Pin the default workspace path used by ``studio_*`` tools and resources."""
    global _DEFAULT_WORKSPACE
    if workspace is None or not str(workspace).strip():
        _DEFAULT_WORKSPACE = ""
        return get_default_workspace()
    resolved = str(Path(workspace).expanduser().resolve())
    _DEFAULT_WORKSPACE = resolved
    return resolved


def get_default_workspace() -> str:
    """Return the active default workspace directory for Weight Studio MCP calls."""
    if _DEFAULT_WORKSPACE:
        return _DEFAULT_WORKSPACE
    env_ws = os.environ.get("STATESET_STUDIO_WORKSPACE", "").strip()
    if env_ws:
        return str(Path(env_ws).expanduser().resolve())
    return str(Path(".").resolve())


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


def _err(exc: Exception, *, hint: str = "", **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": f"{type(exc).__name__}: {exc}"}
    if hint:
        payload["hint"] = hint
    for k, v in extra.items():
        if v is not None:
            payload[k] = v
    return payload


def _studio_err(exc: Exception, rt: Any | None = None) -> dict[str, Any]:
    """Return an actionable, self-correcting error dict for Weight Studio tools."""
    available: list[str] | None = None
    hint = ""
    if rt is not None and isinstance(exc, KeyError):
        try:
            models = rt.inspector.list_models()
            available = [str(m["model_id"]) for m in models[:12]]
            hint = (
                "Pass one of the discovered model_ids below (or call "
                "studio_list_models(refresh=True))."
            )
        except Exception:
            available = None
    return _err(exc, hint=hint, available_models=available)


def _get_mcp_studio_runtime(workspace: str = "") -> Any:
    from stateset_agents.weight_studio import StudioRuntime

    target = (
        workspace.strip()
        if workspace and workspace.strip()
        else get_default_workspace()
    )
    resolved = str(Path(target).expanduser().resolve())
    rt = _STUDIO_RUNTIMES.get(resolved)
    if rt is None:
        rt = StudioRuntime(workspace_root=Path(resolved))
        _STUDIO_RUNTIMES[resolved] = rt
    return rt


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
            res = await run_improve_async(
                transcripts=transcripts_dir,
                reward=reward,
                output=output_dir,
                threshold=threshold,
                format=format,
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
# Weight Studio MCP Tools (3D inspection, sensitivity, live inference,
# circuits, residual stream, SAE features, CAA steering, surgical editing,
# undo/redo/recipes, A/B Arena, Auto-Repair, 95% CI Eval Gate, PEFT LoRA /
# safetensors / GGUF / HF Card / HTML PR report export, sessions, Auto-Heal)
# ---------------------------------------------------------------------------


def studio_list_models(workspace: str = "", refresh: bool = False) -> dict[str, Any]:
    """Discover local .safetensors checkpoints, sharded models, and StateSet architecture twins.

    Start here to obtain valid ``model_id`` values for all other ``studio_*`` tools.
    """
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
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Inspect a model's layer architecture, parameter counts, Frobenius norms, and active edit stack."""
    rt = None
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
        return _studio_err(exc, rt)


def studio_profile_sensitivity(
    model_id: str,
    target_compression: float = 0.35,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Profile per-layer Wanda/Fisher salience and effective SVD rank, synthesizing an adaptive mixed-rank compression recipe.

    Pass the returned ``recommended_recipe`` directly to ``studio_manage_edits(action="apply_recipe", recipe=...)`` to apply it non-destructively.
    """
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        overrides = rt.editor.get_overrides(model_id, checkpoint_id)
        return dict(
            rt.inspector.profile_layer_sensitivity(
                model_id,
                checkpoint_id=checkpoint_id,
                overrides=overrides if overrides else None,
                target_compression=float(target_compression),
            )
        )
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_run_inference(
    model_id: str,
    prompt: str,
    system_prompt: str = "",
    compare_base: bool = True,
    max_tokens: int = 48,
    temperature: float = 0.7,
    steering: dict[str, Any] | None = None,
    causal_patches: list[dict[str, Any]] | None = None,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Run Base vs. Edited inference with Logit Lens, layer activations, optional live CAA steering, and causal patches."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        res = rt.inference.run_inference(
            model_id,
            prompt,
            checkpoint_id=checkpoint_id,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            compare_base=compare_base,
            causal_patches=causal_patches,
            steering=steering,
        )
        return dict(res)
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_discover_circuit(
    model_id: str,
    prompt: str,
    top_k: int = 10,
    system_prompt: str = "",
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Discover the Top-K causal attention heads & MLP cores driving a prompt."""
    rt = None
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
        return _studio_err(exc, rt)


def studio_residual_stream(
    model_id: str,
    prompt: str,
    system_prompt: str = "",
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Trace Base vs. Active Edited residual hidden-state drift, Attention/MLP split, and Logit Lens convergence across every layer."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.inference.compute_residual_stream_3d(
                model_id,
                prompt,
                checkpoint_id=checkpoint_id,
                system_prompt=system_prompt,
            )
        )
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_sae_analyze(
    model_id: str,
    prompt: str,
    layer_index: int | None = None,
    top_k: int = 10,
    clamps: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Decompose a layer's residual activation into sparse monosemantic SAE features (L0, R²) with optional feature clamping."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.inference.analyze_sae_features(
                model_id,
                prompt,
                checkpoint_id=checkpoint_id,
                system_prompt=system_prompt,
                layer_index=layer_index,
                top_k=top_k,
                clamps=clamps,
            )
        )
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_extract_steering(
    model_id: str,
    positive_prompt: str,
    negative_prompt: str,
    name: str = "Steering Vector",
    layer_index: int | None = None,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Extract a Contrastive Activation Steering (CAA / RepE) vector and locate the optimal injection layer."""
    rt = None
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
        return _studio_err(exc, rt)


def studio_apply_edit(
    model_id: str,
    op: str,
    target: str,
    params: dict[str, Any] | None = None,
    label: str = "",
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Apply a non-destructive surgical weight edit.

    Supported ops: ``scale``, ``prune``, ``clamp``, ``svd_truncate``, ``head_edit``,
    ``rank1_edit``, ``steer_bake``, ``region_edit``, ``blend``, ``noise``.
    Supported target selectors: exact tensor name, ``layer:<n>``, ``layer:<lo>-<hi>``,
    ``role:self_attn|linear_attn|mlp|norm|embedding``, ``suffix:<substr>``, or ``all``.
    Original checkpoints are never modified.
    """
    rt = None
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
        return _studio_err(exc, rt)


def studio_manage_edits(
    model_id: str,
    action: str = "history",
    recipe: dict[str, Any] | None = None,
    reset_first: bool = True,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Manage non-destructive surgical edit history and portable recipes.

    ``action`` must be one of:
    - ``"history"``: Inspect active edit stack and undo/redo counts.
    - ``"undo"``: Undo the most recent surgical edit.
    - ``"redo"``: Redo the last undone edit.
    - ``"reset"``: Clear all in-memory edits back to the untouched base checkpoint.
    - ``"export_recipe"``: Export active enabled edits as a portable ``.studio-recipe.json`` dict.
    - ``"apply_recipe"``: Deterministically replay ``recipe`` onto ``model_id``.
    """
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        act = action.strip().lower()
        if act == "history":
            return {
                "action": "history",
                **rt.editor.get_history(model_id, checkpoint_id),
            }
        if act == "undo":
            return {"action": "undo", **rt.editor.undo(model_id, checkpoint_id)}
        if act == "redo":
            return {"action": "redo", **rt.editor.redo(model_id, checkpoint_id)}
        if act == "reset":
            return {"action": "reset", **rt.editor.reset(model_id, checkpoint_id)}
        if act == "export_recipe":
            return {
                "action": "export_recipe",
                "recipe": rt.editor.export_recipe(model_id, checkpoint_id),
            }
        if act == "apply_recipe":
            if not isinstance(recipe, dict):
                return {
                    "error": "apply_recipe requires a 'recipe' dict containing 'operations'."
                }
            return {
                "action": "apply_recipe",
                **rt.editor.apply_recipe(
                    model_id,
                    recipe,
                    checkpoint_id=checkpoint_id,
                    reset_first=reset_first,
                ),
            }
        return {
            "error": (
                f"Unknown action {action!r}. Choose one of: "
                "history, undo, redo, reset, export_recipe, apply_recipe."
            )
        }
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_arena_compare(
    model_id: str,
    prompt: str,
    candidates: list[dict[str, Any]] | None = None,
    reward_fn: str = "composite",
    system_prompt: str = "",
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Compare Base, Active Edited, and candidate surgical edits side-by-side without mutating session history."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.inference.compare_arena_candidates(
                model_id,
                prompt,
                candidates,
                system_prompt=system_prompt,
                checkpoint_id=checkpoint_id,
                reward_fn=reward_fn,
            )
        )
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_auto_repair(
    model_id: str,
    reward_fn: str = "composite",
    max_steps: int = 4,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Automatically bisect and dial back offending surgical edits when the Batch Eval Gate returns WARN/BLOCK until GO is restored."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.editor.auto_repair_edits(
                model_id,
                rt.inference,
                checkpoint_id=checkpoint_id,
                reward_fn=reward_fn,
                max_steps=max_steps,
            )
        )
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_eval_and_export(
    model_id: str,
    export_name: str = "",
    run_merge_probe: bool = True,
    require_go: bool = True,
    max_cases: int = 6,
    reward_fn: str = "composite",
    custom_jsonl: str = "",
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Run the 95% CI Pre-Export Batch Eval Gate (GO/WARN/BLOCK) and optionally write a SHA-256 reload-verified .safetensors checkpoint + portable recipe."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        gate = rt.inference.run_batch_eval(
            model_id,
            checkpoint_id=checkpoint_id,
            custom_jsonl=custom_jsonl or None,
            reward_fn=reward_fn,
            max_cases=max_cases,
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
                    f"Gate verdict is {gate['verdict']}; run studio_auto_repair(...) "
                    "or pass require_go=False to override."
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
        return _studio_err(exc, rt)


def studio_export_lora(
    model_id: str,
    export_name: str = "surgical-lora-v1",
    rank: int = 16,
    alpha: int = 32,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Factorize active Weight Studio surgical diffs ΔW via SVD into a reload-verified PEFT LoRA adapter."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.editor.export_diff_as_lora(
                model_id,
                export_name,
                rank=rank,
                alpha=alpha,
                checkpoint_id=checkpoint_id,
            )
        )
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_export_artifact(
    model_id: str,
    format: str = "gguf",
    export_name: str = "surgical-export-v1",
    quant_type: str = "Q8_0",
    repo_id: str = "",
    include_eval: bool = True,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Export active Weight Studio edits as a deployment or review artifact.

    Supported ``format`` values:
    - ``"gguf"``: GGUF v3 tensor bundle + ready-to-run Ollama ``Modelfile`` (``quant_type``: ``F32``, ``F16``, ``Q8_0``).
    - ``"hf_card"``: Verified Hugging Face Model Card ``README.md`` with Eval Gate scorecard and portable recipe.
    - ``"html_report"``: Offline single-file interactive 3D HTML PR report written to ``outputs/weight_studio_exports/<export_name>/report.html``.
    """
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        fmt = format.strip().lower()
        if fmt == "gguf":
            return dict(
                rt.editor.export_gguf_package(
                    model_id,
                    export_name,
                    quant_type=quant_type,
                    checkpoint_id=checkpoint_id,
                )
            )
        if fmt == "hf_card":
            eval_rep = (
                rt.inference.run_batch_eval(
                    model_id, checkpoint_id=checkpoint_id, max_cases=6
                )
                if include_eval
                else None
            )
            return dict(
                rt.editor.export_hf_model_card(
                    model_id,
                    export_name,
                    repo_id=repo_id,
                    eval_report=eval_rep,
                    checkpoint_id=checkpoint_id,
                    push_to_hub=False,
                )
            )
        if fmt == "html_report":
            from stateset_agents.weight_studio.report_exporter import (
                build_standalone_html_report,
            )

            overrides = rt.editor.get_overrides(model_id, checkpoint_id)
            struct = rt.inspector.get_model_structure(
                model_id,
                checkpoint_id,
                overrides=overrides if overrides else None,
            )
            history = rt.editor.get_history(model_id, checkpoint_id)
            recipe = rt.editor.export_recipe(model_id, checkpoint_id)
            eval_rep = (
                rt.inference.run_batch_eval(
                    model_id, checkpoint_id=checkpoint_id, max_cases=6
                )
                if include_eval
                else None
            )
            sens_rep = rt.inspector.profile_layer_sensitivity(
                model_id,
                checkpoint_id=checkpoint_id,
                overrides=overrides if overrides else None,
            )
            html_doc = build_standalone_html_report(
                model_summary=struct["model"],
                structure_summary=struct["summary"],
                layers=struct["layers"],
                edit_history=history,
                recipe=recipe,
                eval_report=eval_rep,
                sensitivity_report=sens_rep,
            )
            safe_name = (
                "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "-"
                    for ch in export_name.strip()
                ).strip("-")
                or "studio-report"
            )
            out_dir = (
                rt.inspector.workspace_root
                / "outputs"
                / "weight_studio_exports"
                / safe_name
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "report.html"
            report_path.write_text(html_doc, encoding="utf-8")
            return {
                "format": "html_report",
                "export_dir": str(out_dir),
                "report_path": str(report_path),
                "bytes": len(html_doc.encode("utf-8")),
                "eval_verdict": eval_rep["verdict"] if eval_rep else None,
            }
        return {
            "error": (
                f"Unsupported format {format!r}. Choose one of: "
                "gguf, hf_card, html_report."
            )
        }
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_save_session(
    session_name: str,
    model_id: str,
    metadata: dict[str, Any] | None = None,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Persist a named 3D Weight Studio workbench session (recipe + steering/eval metadata) to disk."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.editor.save_session(
                session_name,
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                metadata=metadata,
            )
        )
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_sessions(
    action: str = "list",
    session_name: str = "",
    workspace: str = "",
) -> dict[str, Any]:
    """List saved 3D Weight Studio workbench sessions (``action="list"``) or restore one (``action="load"``)."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        act = action.strip().lower()
        if act == "list":
            sessions = rt.editor.list_sessions()
            return {"action": "list", "count": len(sessions), "sessions": sessions}
        if act == "load":
            if not session_name.strip():
                return {"error": "session_name is required when action='load'."}
            return {
                "action": "load",
                **rt.editor.load_session(session_name.strip()),
            }
        return {"error": f"Unknown action {action!r}. Choose 'list' or 'load'."}
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_auto_heal(
    model_id: str,
    transcripts_path: str = "",
    transcripts_jsonl: str = "",
    reward_fn: str = "customer_support",
    export_format: str = "lora",
    output_name: str = "",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    require_go: bool = True,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Autonomously diagnose failed production transcripts, mine CAA steering, benchmark in Multi-Edit A/B Arena, bake winning edit, verify on 95% CI Eval Gate + Auto-Repair, and export a verified LoRA/safetensors/GGUF/HF Card + HTML PR report."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.healer.run_healing_loop(
                model_id,
                checkpoint_id=checkpoint_id,
                transcripts_path=transcripts_path or None,
                transcripts_jsonl=transcripts_jsonl or None,
                reward_fn=reward_fn,
                export_format=export_format,
                output_name=output_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                require_go=require_go,
            )
        )
    except Exception as exc:
        return _studio_err(exc, rt)


def studio_redteam_audit(
    model_id: str,
    categories: list[str] | None = None,
    probes_path: str = "",
    probes_jsonl: str = "",
    max_probes: int = 10,
    immunize: bool = False,
    export_format: str = "none",
    output_name: str = "",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    workspace: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Audit a model against adversarial attacks (prompt injection, PII exfiltration, unauthorized tool calls, policy bypass, hallucinated commitments) and optionally apply one-click Surgical Immunization (immunize=True) without hurting benign helpfulness."""
    rt = None
    try:
        rt = _get_mcp_studio_runtime(workspace)
        return dict(
            rt.redteam.run_redteam_audit(
                model_id,
                checkpoint_id=checkpoint_id,
                categories=categories,
                probes_path=probes_path or None,
                probes_jsonl=probes_jsonl or None,
                max_probes=max_probes,
                immunize=immunize,
                export_format=export_format,
                output_name=output_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        )
    except Exception as exc:
        return _studio_err(exc, rt)


# ---------------------------------------------------------------------------
# Live MCP Resources (stateset://studio/*)
# ---------------------------------------------------------------------------


def studio_capabilities_manifest() -> dict[str, Any]:
    """Machine-readable capability manifest for AI agents driving Weight Studio."""
    return {
        "server": "stateset-agents",
        "workspace": get_default_workspace(),
        "non_destructive_guarantee": (
            "Original checkpoints are never overwritten. All surgical edits live in "
            "an undoable overlay session; verified exports write new bundles under "
            "outputs/weight_studio_exports/ with SHA-256 reload verification."
        ),
        "edit_ops": [
            "scale",
            "prune",
            "clamp",
            "svd_truncate",
            "head_edit",
            "rank1_edit",
            "steer_bake",
            "region_edit",
            "blend",
            "noise",
        ],
        "target_selectors": [
            "<exact_tensor_name>",
            "layer:<n>",
            "layer:<lo>-<hi>",
            "role:self_attn",
            "role:linear_attn",
            "role:mlp",
            "role:norm",
            "role:embedding",
            "suffix:<substring>",
            "all",
        ],
        "reward_scorers": ["composite", "customer_support", "tool_calling", "gsm8k"],
        "export_formats": ["lora", "safetensors", "gguf", "hf_card", "html_report"],
        "recommended_workflows": {
            "auto_heal_from_logs": [
                "studio_list_models",
                "studio_auto_heal",
            ],
            "targeted_behavior_surgery": [
                "studio_list_models",
                "studio_extract_steering",
                "studio_arena_compare",
                "studio_apply_edit",
                "studio_residual_stream",
                "studio_eval_and_export",
                "studio_export_lora",
            ],
            "adaptive_compression": [
                "studio_list_models",
                "studio_profile_sensitivity",
                "studio_manage_edits(action='apply_recipe')",
                "studio_eval_and_export",
                "studio_auto_repair",
                "studio_export_artifact(format='gguf')",
            ],
        },
    }


def _resource_studio_capabilities() -> str:
    return json.dumps(studio_capabilities_manifest(), indent=2)


def _resource_studio_models() -> str:
    return json.dumps(studio_list_models(), indent=2)


def _resource_studio_steering_presets() -> str:
    rt = _get_mcp_studio_runtime()
    return json.dumps({"presets": rt.inference.list_steering_presets()}, indent=2)


def _resource_studio_scenarios() -> str:
    rt = _get_mcp_studio_runtime()
    return json.dumps({"scenarios": rt.inference.list_scenarios()}, indent=2)


def _resource_studio_sessions() -> str:
    return json.dumps(studio_sessions(action="list"), indent=2)


def _resource_studio_redteam_probes() -> str:
    rt = _get_mcp_studio_runtime()
    return json.dumps({"probes": rt.redteam.list_probes()}, indent=2)


# ---------------------------------------------------------------------------
# Guided MCP Workflow Prompts (agent playbooks)
# ---------------------------------------------------------------------------


def prompt_redteam_and_immunize(
    model_id: str,
    export_format: str = "lora",
) -> str:
    """Guided playbook for Adversarial Red-Team Safety Audit & Surgical Immunization."""
    return (
        f"Run an Adversarial Red-Team Safety Audit and Surgical Immunization on `{model_id}`:\n\n"
        f"1. Call `studio_redteam_audit(model_id={model_id!r}, immunize=False)` to measure baseline "
        "Attack Success Rate (ASR) across prompt injection, PII exfiltration, unauthorized tool calls, "
        "policy bypass, and hallucinated commitments.\n"
        f"2. Call `studio_redteam_audit(model_id={model_id!r}, immunize=True, export_format={export_format!r})` "
        "to extract a Safety Immunization CAA vector, sweep strength in the A/B Arena, bake the winning "
        "surgical edit, verify benign support helpfulness on the 95% CI Batch Eval Gate, and export the "
        "verified bundle + `redteam_report.html`.\n"
        "3. Report baseline vs. immunized ASR, safety score gain, benign eval verdict, and exported paths."
    )


def prompt_auto_heal_model(
    model_id: str,
    transcripts_path: str = "",
    reward: str = "customer_support",
    export_format: str = "lora",
) -> str:
    """Guided playbook for running the Autonomous Transcript-to-Surgery Self-Healing Loop."""
    source = (
        f"the transcripts at `{transcripts_path}`"
        if transcripts_path.strip()
        else "the built-in StateSet production failure scenarios"
    )
    return (
        f"You are operating StateSet 3D Weight Studio via MCP to autonomously heal `{model_id}`.\n\n"
        "Follow these steps in order:\n"
        f"1. Call `studio_inspect_model(model_id={model_id!r})` to confirm baseline layers and existing edits.\n"
        f"2. Call `studio_auto_heal(model_id={model_id!r}, transcripts_path={transcripts_path!r}, "
        f"reward_fn={reward!r}, export_format={export_format!r})` using {source}.\n"
        "3. Inspect the returned diagnosis failure modes, winning Arena CAA strength, 95% CI Batch Eval Gate "
        "verdict (`GO`/`WARN`/`BLOCK`), Auto-Repair status, and exported artifact paths.\n"
        "4. If the verdict is not `GO`, call `studio_auto_repair` and re-check with `studio_eval_and_export`.\n"
        "5. Report a concise engineering summary including before/after reward delta, SHA-256 verification, "
        "and the paths to the exported adapter/bundle and `healing_report.html`."
    )


def prompt_surgical_behavior_edit(
    model_id: str,
    behavior_goal: str,
    reward: str = "composite",
) -> str:
    """Guided playbook for targeted CAA behavioral steering, Arena selection, and verified LoRA export."""
    return (
        f"You are performing non-destructive surgical behavior editing on `{model_id}`.\n"
        f"Goal: {behavior_goal}\n\n"
        "Execute this disciplined loop:\n"
        f"1. Call `studio_extract_steering(model_id={model_id!r}, positive_prompt=..., negative_prompt=..., name=...)` "
        "with clean contrastive prompts that isolate the target behavior.\n"
        "2. Call `studio_arena_compare` with 2–3 candidate `steer_bake` strengths (e.g. 0.5, 0.85, 1.2) on a "
        "realistic test prompt to choose the highest-reward candidate without mutating session history.\n"
        "3. Apply the winner via `studio_apply_edit(op='steer_bake', target=f'layer:{recommended_layer}', ...)`.\n"
        "4. Verify layer-by-layer convergence with `studio_residual_stream` and check the 95% CI Batch Eval Gate "
        f"with `studio_eval_and_export(model_id={model_id!r}, reward_fn={reward!r})`.\n"
        "5. If the gate returns `WARN` or `BLOCK`, call `studio_auto_repair` or `studio_manage_edits(action='undo')`.\n"
        "6. Once `GO`, export a portable PEFT LoRA adapter via `studio_export_lora` and an interactive 3D HTML "
        "PR report via `studio_export_artifact(format='html_report')`."
    )


def prompt_interpretability_audit(model_id: str, prompt: str) -> str:
    """Guided playbook for a full mechanistic interpretability audit."""
    return (
        f"Run a comprehensive mechanistic interpretability audit of `{model_id}` on the prompt:\n"
        f"> {prompt}\n\n"
        "Gather evidence across all four views:\n"
        f"1. `studio_discover_circuit(model_id={model_id!r}, prompt={prompt!r}, top_k=10)` — identify dominant "
        "attention heads, MLP cores, and attention-flow arc density.\n"
        f"2. `studio_residual_stream(model_id={model_id!r}, prompt={prompt!r})` — trace residual norm trajectory, "
        "Attention vs. MLP contribution split, and Logit Lens top-token convergence.\n"
        f"3. `studio_sae_analyze(model_id={model_id!r}, prompt={prompt!r})` — decompose the highest-drift layer "
        "into sparse monosemantic SAE features (`L0`, `R²`) and test clamping the top feature at `0.0` vs. `2.5`.\n"
        f"4. `studio_profile_sensitivity(model_id={model_id!r})` — flag high-salience protected layers vs. "
        "low-salience compressible layers.\n"
        "Synthesize your findings into a concise report highlighting causal bottlenecks and safe edit targets."
    )


def prompt_compress_and_export_model(
    model_id: str,
    target_compression: float = 0.35,
    export_format: str = "gguf",
) -> str:
    """Guided playbook for adaptive Wanda/SVD compression with Eval Gate and Auto-Repair."""
    return (
        f"Compress `{model_id}` to ~{int(target_compression * 100)}% target compression without breaking quality:\n\n"
        f"1. Call `studio_profile_sensitivity(model_id={model_id!r}, target_compression={target_compression})` "
        "to obtain the Wanda/Fisher salience map and `recommended_recipe`.\n"
        "2. Replay the adaptive recipe via `studio_manage_edits(action='apply_recipe', recipe=...)`.\n"
        "3. Run `studio_eval_and_export` to evaluate the 95% CI Batch Eval Gate.\n"
        "4. If the verdict is `WARN` or `BLOCK`, call `studio_auto_repair` to automatically dial back the "
        "most damaging layer edit until the gate passes `GO`.\n"
        f"5. Export the verified bundle using `studio_export_artifact(format={export_format!r})` (or "
        "`studio_export_lora` for PEFT LoRA) and save the session via `studio_save_session`."
    )


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server(*, workspace: str | Path | None = None) -> Any:
    """Build and return a configured ``FastMCP`` server instance.

    Raises ``ImportError`` with an install hint if the ``mcp`` package is
    not installed.
    """
    FastMCP = _require_mcp()  # noqa: N806
    _ensure_scripts_on_path()
    from mcp.types import ToolAnnotations

    if workspace is not None:
        set_default_workspace(workspace)

    mcp = FastMCP(
        "stateset-agents",
        instructions=(
            "Drives StateSet Agents' transcript improve loop and 3D Weight Studio: "
            "discover checkpoints, inspect weights, profile Wanda/SVD sensitivity, "
            "run Base-vs-Edited inference with Logit Lens, discover causal circuits, "
            "trace residual streams, analyze/clamp SAE features, extract & bake CAA "
            "behavioral steering, apply non-destructive surgical edits with undo/redo/"
            "recipes, benchmark in the Multi-Edit A/B Arena, run Surgical Auto-Repair, "
            "gate exports with 95% CI Batch Eval, export PEFT LoRA / .safetensors / "
            "GGUF v3 + Ollama Modelfile / HF Model Card / interactive 3D HTML PR "
            "reports, persist workbench sessions, and run the end-to-end Autonomous "
            "Transcript-to-Surgery Self-Healing Loop (`studio_auto_heal`). "
            "Start by reading `stateset://studio/capabilities` or calling "
            "`studio_list_models()`. Original checkpoints are never overwritten, and "
            "no tool starts real remote GPU training."
        ),
    )

    ro = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
    mut = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )

    # Core improve & finetune preview tools (7)
    mcp.tool(annotations=ro)(list_rewards)
    mcp.tool(annotations=mut)(ingest_transcripts)
    mcp.tool(annotations=ro)(grade_transcript)
    mcp.tool(annotations=mut)(improve_run)
    mcp.tool(annotations=ro)(improve_status)
    mcp.tool(annotations=ro)(list_model_presets)
    mcp.tool(annotations=ro)(dry_run_finetune)

    # 3D Weight Studio tools (18)
    mcp.tool(annotations=ro)(studio_list_models)
    mcp.tool(annotations=ro)(studio_inspect_model)
    mcp.tool(annotations=ro)(studio_profile_sensitivity)
    mcp.tool(annotations=ro)(studio_run_inference)
    mcp.tool(annotations=ro)(studio_discover_circuit)
    mcp.tool(annotations=ro)(studio_residual_stream)
    mcp.tool(annotations=ro)(studio_sae_analyze)
    mcp.tool(annotations=ro)(studio_extract_steering)
    mcp.tool(annotations=mut)(studio_apply_edit)
    mcp.tool(annotations=mut)(studio_manage_edits)
    mcp.tool(annotations=ro)(studio_arena_compare)
    mcp.tool(annotations=mut)(studio_auto_repair)
    mcp.tool(annotations=mut)(studio_eval_and_export)
    mcp.tool(annotations=mut)(studio_export_lora)
    mcp.tool(annotations=mut)(studio_export_artifact)
    mcp.tool(annotations=mut)(studio_save_session)
    mcp.tool(annotations=mut)(studio_sessions)
    mcp.tool(annotations=mut)(studio_auto_heal)
    mcp.tool(annotations=mut)(studio_redteam_audit)

    # Live MCP Resources (6)
    mcp.resource(
        "stateset://studio/capabilities",
        name="studio_capabilities",
        description="Supported Weight Studio edit ops, target selectors, reward scorers, export formats, and recommended agent workflows.",
        mime_type="application/json",
    )(_resource_studio_capabilities)
    mcp.resource(
        "stateset://studio/models",
        name="studio_models",
        description="Live catalog of discovered local .safetensors checkpoints, sharded models, and StateSet architecture twins.",
        mime_type="application/json",
    )(_resource_studio_models)
    mcp.resource(
        "stateset://studio/steering-presets",
        name="studio_steering_presets",
        description="Built-in StateSet contrastive behavioral steering pairs ready for CAA extraction.",
        mime_type="application/json",
    )(_resource_studio_steering_presets)
    mcp.resource(
        "stateset://studio/scenarios",
        name="studio_scenarios",
        description="Built-in StateSet evaluation and interpretability test scenarios.",
        mime_type="application/json",
    )(_resource_studio_scenarios)
    mcp.resource(
        "stateset://studio/sessions",
        name="studio_sessions",
        description="Persisted 3D Weight Studio workbench sessions on disk.",
        mime_type="application/json",
    )(_resource_studio_sessions)
    mcp.resource(
        "stateset://studio/redteam-probes",
        name="studio_redteam_probes",
        description="Built-in adversarial Red-Team safety probes (prompt injection, PII exfiltration, unauthorized tool calls, policy bypass, hallucinated commitments).",
        mime_type="application/json",
    )(_resource_studio_redteam_probes)

    # Guided MCP Workflow Prompts (5)
    mcp.prompt(
        name="auto_heal_model",
        description="Run the Autonomous Transcript-to-Surgery Self-Healing Loop from failing logs to a verified adapter/bundle.",
    )(prompt_auto_heal_model)
    mcp.prompt(
        name="redteam_and_immunize",
        description="Run Adversarial Red-Team Safety Audit and one-click CAA Surgical Immunization without degrading benign helpfulness.",
    )(prompt_redteam_and_immunize)
    mcp.prompt(
        name="surgical_behavior_edit",
        description="Extract CAA steering, benchmark in the Multi-Edit A/B Arena, verify residual convergence, and export a PEFT LoRA.",
    )(prompt_surgical_behavior_edit)
    mcp.prompt(
        name="interpretability_audit",
        description="Perform a full mechanistic interpretability audit (Circuit 3D, Residual Stream, SAE features, and Wanda/SVD sensitivity).",
    )(prompt_interpretability_audit)
    mcp.prompt(
        name="compress_and_export_model",
        description="Adaptively compress a model via Wanda/SVD sensitivity profiling, repair regressions, and export to GGUF/LoRA.",
    )(prompt_compress_and_export_model)

    return mcp


def main() -> None:
    """Entry point for ``stateset-agents mcp`` / ``python -m stateset_agents.mcp_server``."""
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
