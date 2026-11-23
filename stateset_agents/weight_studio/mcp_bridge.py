"""Live Streamable HTTP MCP Bridge & Real-Time Studio Event Bus.

Mounts a spec-compliant MCP JSON-RPC 2.0 Streamable HTTP endpoint directly onto
the running 3D Weight Studio FastAPI server (``POST /api/weight-studio/mcp``)
sharing the exact same live ``StudioRuntime`` instance as the browser UI.

Whenever an external AI agent (Claude Code, Cursor, Antigravity, LangGraph, or
a custom HTTP MCP client) invokes any Weight Studio tool, the bridge:

1. Executes the tool directly against the live ``StudioRuntime``.
2. Publishes a structured event onto ``StudioEventBus`` (``GET /api/weight-studio/events``
   SSE stream), causing any connected 3D Weight Studio browser tab to animate
   liquid-crystal weight morphing, trigger a glowing forward/backprop wave,
   update the Edit History & Eval Gate HUD, and log the live agent action!

Zero extra runtime dependencies beyond FastAPI/stdlib — works even when the
optional ``mcp`` Python SDK extra is not installed.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from typing import Any

MCP_PROTOCOL_VERSION = "2025-03-26"


class StudioEventBus:
    """In-process pub/sub event bus broadcasting live AI-agent & Studio events over SSE."""

    def __init__(self, max_history: int = 40) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._seq = 0

    def publish(
        self,
        event_type: str,
        *,
        source: str = "mcp_agent",
        tool: str = "",
        model_id: str = "",
        summary: str = "",
        mutated: bool = False,
        pulse: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._seq += 1
        event: dict[str, Any] = {
            "seq": self._seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "source": source,
            "tool": tool,
            "model_id": model_id,
            "summary": summary,
            "mutated": mutated,
            "pulse": pulse,
            "payload": payload or {},
        }
        self._history.append(event)
        stale: list[asyncio.Queue[dict[str, Any]]] = []
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    stale.append(q)
            except Exception:
                stale.append(q)
        for q in stale:
            self._subscribers.discard(q)
        return event

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        items = list(self._history)
        return items[-max(1, limit) :]


_RO_ANN = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
_MUT_ANN = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}


def _tool_spec(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
    *,
    read_only: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": list(required or []),
        },
        "annotations": dict(_RO_ANN if read_only else _MUT_ANN),
    }


STUDIO_MCP_TOOLS: list[dict[str, Any]] = [
    _tool_spec(
        "studio_list_models",
        "Discover local .safetensors checkpoints, sharded models, and StateSet architecture twins in the live Weight Studio workspace.",
        {"refresh": {"type": "boolean", "default": False}},
        read_only=True,
    ),
    _tool_spec(
        "studio_inspect_model",
        "Inspect layer architecture, parameter counts, Frobenius norms, and active edit stack.",
        {
            "model_id": {"type": "string"},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=True,
    ),
    _tool_spec(
        "studio_profile_sensitivity",
        "Profile Wanda/Fisher salience & effective SVD rank per layer and synthesize an adaptive mixed-rank compression recipe.",
        {
            "model_id": {"type": "string"},
            "target_compression": {"type": "number", "default": 0.35},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=True,
    ),
    _tool_spec(
        "studio_run_inference",
        "Run Base vs. Edited forward pass with Logit Lens, layer activations, live steering, and causal patches.",
        {
            "model_id": {"type": "string"},
            "prompt": {"type": "string"},
            "system_prompt": {"type": "string", "default": ""},
            "compare_base": {"type": "boolean", "default": True},
            "max_tokens": {"type": "integer", "default": 48},
            "temperature": {"type": "number", "default": 0.7},
            "steering": {"type": "object"},
            "causal_patches": {"type": "array", "items": {"type": "object"}},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id", "prompt"],
        read_only=True,
    ),
    _tool_spec(
        "studio_discover_circuit",
        "Discover Top-K causal attention heads & MLP cores driving a prompt (Circuit 3D).",
        {
            "model_id": {"type": "string"},
            "prompt": {"type": "string"},
            "top_k": {"type": "integer", "default": 10},
            "system_prompt": {"type": "string", "default": ""},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id", "prompt"],
        read_only=True,
    ),
    _tool_spec(
        "studio_residual_stream",
        "Trace Base vs. Edited residual hidden-state drift, Attention/MLP split, and Logit Lens convergence across every layer.",
        {
            "model_id": {"type": "string"},
            "prompt": {"type": "string"},
            "system_prompt": {"type": "string", "default": ""},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id", "prompt"],
        read_only=True,
    ),
    _tool_spec(
        "studio_sae_analyze",
        "Decompose a layer's residual activation into sparse monosemantic SAE features (L0, R²) with optional feature clamping.",
        {
            "model_id": {"type": "string"},
            "prompt": {"type": "string"},
            "layer_index": {"type": "integer"},
            "top_k": {"type": "integer", "default": 10},
            "clamps": {"type": "array", "items": {"type": "object"}},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id", "prompt"],
        read_only=True,
    ),
    _tool_spec(
        "studio_extract_steering",
        "Extract a Contrastive Activation Steering (CAA/RepE) vector and find the optimal injection layer.",
        {
            "model_id": {"type": "string"},
            "positive_prompt": {"type": "string"},
            "negative_prompt": {"type": "string"},
            "name": {"type": "string", "default": "Steering Vector"},
            "layer_index": {"type": "integer"},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id", "positive_prompt", "negative_prompt"],
        read_only=True,
    ),
    _tool_spec(
        "studio_apply_edit",
        "Apply a non-destructive surgical weight edit (scale, prune, clamp, svd_truncate, head_edit, rank1_edit, steer_bake, region_edit, blend, noise) and morph the live 3D Studio.",
        {
            "model_id": {"type": "string"},
            "op": {"type": "string"},
            "target": {"type": "string"},
            "params": {"type": "object"},
            "label": {"type": "string", "default": ""},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id", "op", "target"],
        read_only=False,
    ),
    _tool_spec(
        "studio_manage_edits",
        "Manage edit history and recipes: action = history | undo | redo | reset | export_recipe | apply_recipe.",
        {
            "model_id": {"type": "string"},
            "action": {"type": "string", "default": "history"},
            "recipe": {"type": "object"},
            "reset_first": {"type": "boolean", "default": True},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=False,
    ),
    _tool_spec(
        "studio_arena_compare",
        "Benchmark Base, Active Edited, and candidate surgical edits side-by-side without mutating session history.",
        {
            "model_id": {"type": "string"},
            "prompt": {"type": "string"},
            "candidates": {"type": "array", "items": {"type": "object"}},
            "reward_fn": {"type": "string", "default": "composite"},
            "system_prompt": {"type": "string", "default": ""},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id", "prompt"],
        read_only=True,
    ),
    _tool_spec(
        "studio_auto_repair",
        "Automatically bisect and dial back offending surgical edits until the 95% CI Batch Eval Gate passes GO.",
        {
            "model_id": {"type": "string"},
            "reward_fn": {"type": "string", "default": "composite"},
            "max_steps": {"type": "integer", "default": 4},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=False,
    ),
    _tool_spec(
        "studio_eval_and_export",
        "Run the 95% CI Pre-Export Batch Eval Gate (GO/WARN/BLOCK) and optionally write a SHA-256 reload-verified .safetensors checkpoint.",
        {
            "model_id": {"type": "string"},
            "export_name": {"type": "string", "default": ""},
            "run_merge_probe": {"type": "boolean", "default": True},
            "require_go": {"type": "boolean", "default": True},
            "max_cases": {"type": "integer", "default": 6},
            "reward_fn": {"type": "string", "default": "composite"},
            "custom_jsonl": {"type": "string", "default": ""},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=False,
    ),
    _tool_spec(
        "studio_export_lora",
        "Factorize active surgical weight diffs ΔW via SVD into a reload-verified PEFT LoRA adapter.",
        {
            "model_id": {"type": "string"},
            "export_name": {"type": "string", "default": "surgical-lora-v1"},
            "rank": {"type": "integer", "default": 16},
            "alpha": {"type": "integer", "default": 32},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=False,
    ),
    _tool_spec(
        "studio_export_artifact",
        "Export active edits as gguf (GGUF v3 + Ollama Modelfile), hf_card (Hugging Face README.md), or html_report (offline 3D HTML PR report).",
        {
            "model_id": {"type": "string"},
            "format": {"type": "string", "default": "gguf"},
            "export_name": {"type": "string", "default": "surgical-export-v1"},
            "quant_type": {"type": "string", "default": "Q8_0"},
            "repo_id": {"type": "string", "default": ""},
            "include_eval": {"type": "boolean", "default": True},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=False,
    ),
    _tool_spec(
        "studio_save_session",
        "Persist a named 3D Weight Studio workbench session (recipe + steering/eval metadata) to disk.",
        {
            "session_name": {"type": "string"},
            "model_id": {"type": "string"},
            "metadata": {"type": "object"},
            "checkpoint_id": {"type": "string"},
        },
        ["session_name", "model_id"],
        read_only=False,
    ),
    _tool_spec(
        "studio_sessions",
        "List saved workbench sessions (action='list') or restore a named session (action='load').",
        {
            "action": {"type": "string", "default": "list"},
            "session_name": {"type": "string", "default": ""},
        },
        [],
        read_only=False,
    ),
    _tool_spec(
        "studio_auto_heal",
        "Run the Autonomous Transcript-to-Surgery Self-Healing Loop: grade failing transcripts -> CAA steering -> Arena -> bake -> 95% CI Eval Gate + Auto-Repair -> verified export + HTML PR report.",
        {
            "model_id": {"type": "string"},
            "transcripts_path": {"type": "string", "default": ""},
            "transcripts_jsonl": {"type": "string", "default": ""},
            "reward_fn": {"type": "string", "default": "customer_support"},
            "export_format": {"type": "string", "default": "lora"},
            "output_name": {"type": "string", "default": ""},
            "lora_rank": {"type": "integer", "default": 16},
            "lora_alpha": {"type": "integer", "default": 32},
            "require_go": {"type": "boolean", "default": True},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=False,
    ),
    _tool_spec(
        "studio_redteam_audit",
        "Run the Adversarial Red-Team Safety Audit (prompt injection, PII exfiltration, unauthorized tool calls, policy bypass, hallucinated commitments) and optionally apply one-click Surgical Immunization (immunize=true).",
        {
            "model_id": {"type": "string"},
            "categories": {"type": "array", "items": {"type": "string"}},
            "probes_path": {"type": "string", "default": ""},
            "probes_jsonl": {"type": "string", "default": ""},
            "max_probes": {"type": "integer", "default": 10},
            "immunize": {"type": "boolean", "default": False},
            "export_format": {"type": "string", "default": "none"},
            "output_name": {"type": "string", "default": ""},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=False,
    ),
    _tool_spec(
        "studio_model_tournament",
        "Benchmark 2+ models/checkpoints across StateSet rewards + Red-Team ASR, rank a Pareto leaderboard, synthesize TIES/DARE/SLERP surgical merges of the Top-2 contenders, promote a winning champion, and optionally export a verified bundle.",
        {
            "model_ids": {"type": "array", "items": {"type": "string"}},
            "reward_fn": {"type": "string", "default": "customer_support"},
            "max_cases": {"type": "integer", "default": 4},
            "include_redteam": {"type": "boolean", "default": True},
            "synthesize_merge": {"type": "boolean", "default": True},
            "merge_methods": {"type": "array", "items": {"type": "string"}},
            "export_format": {"type": "string", "default": "none"},
            "output_name": {"type": "string", "default": ""},
        },
        [],
        read_only=False,
    ),
    _tool_spec(
        "studio_canary_gate",
        "Replay shadow production traffic comparing Base vs. Active Edited weights with paired win/tie/loss, 95% CI, severe regression check, and Red-Team ASR guard, issuing PROMOTE/HOLD_CANARY/ROLLBACK and writing to the durable rollout ledger.",
        {
            "model_id": {"type": "string"},
            "stage": {"type": "string", "default": "canary_10"},
            "traffic_path": {"type": "string", "default": ""},
            "traffic_jsonl": {"type": "string", "default": ""},
            "reward_fn": {"type": "string", "default": "composite"},
            "max_requests": {"type": "integer", "default": 6},
            "max_asr": {"type": "number", "default": 0.25},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=False,
    ),
    _tool_spec(
        "studio_autopilot_release",
        "Run the complete end-to-end Production Autopilot: Auto-Heal Logs -> Red-Team Safety Audit & Immunization -> Canary Shadow-Rollout Promotion Gate -> Verified Artifact Export (LoRA/safetensors/GGUF/HF Card) + Rollout Ledger + Offline 3D HTML Release Dossier.",
        {
            "model_id": {"type": "string"},
            "transcripts_path": {"type": "string", "default": ""},
            "transcripts_jsonl": {"type": "string", "default": ""},
            "reward_fn": {"type": "string", "default": "customer_support"},
            "immunize_redteam": {"type": "boolean", "default": True},
            "stage": {"type": "string", "default": "canary_10"},
            "export_format": {"type": "string", "default": "lora"},
            "output_name": {"type": "string", "default": ""},
            "lora_rank": {"type": "integer", "default": 16},
            "lora_alpha": {"type": "integer", "default": 32},
            "checkpoint_id": {"type": "string"},
        },
        ["model_id"],
        read_only=False,
    ),
]

STUDIO_MCP_RESOURCES: list[dict[str, str]] = [
    {
        "uri": "stateset://studio/capabilities",
        "name": "studio_capabilities",
        "description": "Supported edit ops, target selectors, reward scorers, export formats, and recommended workflows.",
        "mimeType": "application/json",
    },
    {
        "uri": "stateset://studio/models",
        "name": "studio_models",
        "description": "Live catalog of discovered checkpoints and StateSet architecture twins.",
        "mimeType": "application/json",
    },
    {
        "uri": "stateset://studio/steering-presets",
        "name": "studio_steering_presets",
        "description": "Built-in contrastive behavioral steering pairs.",
        "mimeType": "application/json",
    },
    {
        "uri": "stateset://studio/scenarios",
        "name": "studio_scenarios",
        "description": "Built-in evaluation and interpretability scenarios.",
        "mimeType": "application/json",
    },
    {
        "uri": "stateset://studio/sessions",
        "name": "studio_sessions",
        "description": "Saved 3D Weight Studio workbench sessions.",
        "mimeType": "application/json",
    },
    {
        "uri": "stateset://studio/redteam-probes",
        "name": "studio_redteam_probes",
        "description": "Built-in adversarial Red-Team safety probes (prompt injection, PII exfiltration, unauthorized tool calls, policy bypass, hallucinated commitments).",
        "mimeType": "application/json",
    },
    {
        "uri": "stateset://studio/rollouts",
        "name": "studio_rollouts",
        "description": "Durable JSONL rollout ledger of Canary Shadow Gate PROMOTE/HOLD_CANARY/ROLLBACK decisions.",
        "mimeType": "application/json",
    },
]

STUDIO_MCP_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "production_autopilot_release",
        "description": "End-to-end Production Autopilot: Auto-Heal failing logs -> Red-Team Safety Audit & Surgical Immunization -> Canary Shadow Promotion Gate -> Verified Export + Release Dossier.",
        "arguments": [
            {"name": "model_id", "required": True},
            {"name": "transcripts_path", "required": False},
            {"name": "export_format", "required": False},
        ],
    },
    {
        "name": "auto_heal_model",
        "description": "Run the Autonomous Transcript-to-Surgery Self-Healing Loop from failing logs to a verified adapter.",
        "arguments": [
            {"name": "model_id", "required": True},
            {"name": "transcripts_path", "required": False},
            {"name": "reward", "required": False},
            {"name": "export_format", "required": False},
        ],
    },
    {
        "name": "redteam_and_immunize",
        "description": "Audit a model against adversarial prompt injection, PII exfiltration, and unauthorized tool attacks, then surgically immunize vulnerable layers without hurting benign helpfulness.",
        "arguments": [
            {"name": "model_id", "required": True},
            {"name": "export_format", "required": False},
        ],
    },
    {
        "name": "model_tournament_and_merge",
        "description": "Benchmark multiple checkpoints on task reward + Red-Team safety, sweep TIES/DARE/SLERP surgical merges of the Top-2, and export the Pareto-optimal champion.",
        "arguments": [
            {"name": "reward", "required": False},
            {"name": "export_format", "required": False},
        ],
    },
    {
        "name": "surgical_behavior_edit",
        "description": "Extract CAA steering, benchmark in the Multi-Edit A/B Arena, verify residual convergence, and export a PEFT LoRA.",
        "arguments": [
            {"name": "model_id", "required": True},
            {"name": "behavior_goal", "required": True},
        ],
    },
    {
        "name": "interpretability_audit",
        "description": "Run Circuit 3D, Residual Stream Logit Lens, SAE feature clamping, and Wanda/SVD sensitivity.",
        "arguments": [
            {"name": "model_id", "required": True},
            {"name": "prompt", "required": True},
        ],
    },
    {
        "name": "compress_and_export_model",
        "description": "Adaptively compress a model via Wanda/SVD sensitivity profiling, auto-repair regressions, and export GGUF/LoRA.",
        "arguments": [
            {"name": "model_id", "required": True},
            {"name": "target_compression", "required": False},
            {"name": "export_format", "required": False},
        ],
    },
]


def _dispatch_studio_tool(
    rt: Any, name: str, args: dict[str, Any]
) -> tuple[dict[str, Any], bool, str, str]:
    """Execute a live Weight Studio MCP tool.

    Returns ``(result_dict, mutated_weights, pulse_kind, summary_text)``.
    """
    model_id = str(args.get("model_id") or "")
    checkpoint_id = args.get("checkpoint_id")

    if name == "studio_list_models":
        models = (
            rt.inspector.refresh_catalog()
            if args.get("refresh")
            else rt.inspector.list_models()
        )
        return (
            {
                "workspace": str(rt.inspector.workspace_root),
                "count": len(models),
                "models": models,
            },
            False,
            "",
            f"Listed {len(models)} models",
        )

    if name == "studio_inspect_model":
        overrides = rt.editor.get_overrides(model_id, checkpoint_id)
        struct = rt.inspector.get_model_structure(
            model_id, checkpoint_id, overrides=overrides if overrides else None
        )
        history = rt.editor.get_history(model_id, checkpoint_id)
        return (
            {
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
            },
            False,
            "forward",
            f"Inspected {model_id} ({len(struct['layers'])} layers)",
        )

    if name == "studio_profile_sensitivity":
        overrides = rt.editor.get_overrides(model_id, checkpoint_id)
        res = rt.inspector.profile_layer_sensitivity(
            model_id,
            checkpoint_id=checkpoint_id,
            overrides=overrides if overrides else None,
            target_compression=float(args.get("target_compression", 0.35)),
        )
        return (
            dict(res),
            False,
            "forward",
            f"Profiled Wanda/SVD sensitivity for {model_id}",
        )

    if name == "studio_run_inference":
        res = rt.inference.run_inference(
            model_id,
            str(args["prompt"]),
            checkpoint_id=checkpoint_id,
            system_prompt=str(args.get("system_prompt") or ""),
            temperature=float(args.get("temperature", 0.7)),
            max_tokens=int(args.get("max_tokens", 48)),
            compare_base=bool(args.get("compare_base", True)),
            causal_patches=args.get("causal_patches"),
            steering=args.get("steering"),
        )
        return dict(res), False, "forward", f"Ran inference probe on {model_id}"

    if name == "studio_discover_circuit":
        res = rt.inference.discover_causal_circuit(
            model_id,
            str(args["prompt"]),
            checkpoint_id=checkpoint_id,
            system_prompt=str(args.get("system_prompt") or ""),
            top_k=int(args.get("top_k", 10)),
        )
        return (
            {
                "model_id": res["model_id"],
                "scenario": res["scenario"],
                "tokens": res["tokens"],
                "summary": res["summary"],
                "top_nodes": res["top_nodes"],
                "arc_count": len(res["arcs"]),
            },
            False,
            "forward",
            f"Discovered Circuit 3D ({len(res['arcs'])} arcs)",
        )

    if name == "studio_residual_stream":
        res = rt.inference.compute_residual_stream_3d(
            model_id,
            str(args["prompt"]),
            checkpoint_id=checkpoint_id,
            system_prompt=str(args.get("system_prompt") or ""),
        )
        return dict(res), False, "forward", f"Traced Residual 3D stream on {model_id}"

    if name == "studio_sae_analyze":
        res = rt.inference.analyze_sae_features(
            model_id,
            str(args["prompt"]),
            checkpoint_id=checkpoint_id,
            system_prompt=str(args.get("system_prompt") or ""),
            layer_index=args.get("layer_index"),
            top_k=int(args.get("top_k", 10)),
            clamps=args.get("clamps"),
        )
        return dict(res), False, "forward", f"Analyzed SAE features on {model_id}"

    if name == "studio_extract_steering":
        res = rt.inference.extract_steering_vector(
            model_id,
            str(args["positive_prompt"]),
            str(args["negative_prompt"]),
            checkpoint_id=checkpoint_id,
            name=str(args.get("name") or "Steering Vector"),
            layer_index=args.get("layer_index"),
        )
        return (
            dict(res),
            False,
            "forward",
            f"Extracted CAA vector '{res.get('name')}' (L{res.get('recommended_layer')})",
        )

    if name == "studio_apply_edit":
        op = str(args["op"])
        target = str(args["target"])
        res = rt.editor.apply_edit(
            model_id,
            op,
            target,
            dict(args.get("params") or {}),
            checkpoint_id=checkpoint_id,
            label=str(args.get("label") or ""),
        )
        return dict(res), True, "backprop", f"Applied {op} → {target}"

    if name == "studio_manage_edits":
        act = str(args.get("action") or "history").strip().lower()
        if act == "history":
            return (
                {"action": "history", **rt.editor.get_history(model_id, checkpoint_id)},
                False,
                "",
                "Checked edit history",
            )
        if act == "undo":
            return (
                {"action": "undo", **rt.editor.undo(model_id, checkpoint_id)},
                True,
                "backprop",
                "Undid last surgical edit",
            )
        if act == "redo":
            return (
                {"action": "redo", **rt.editor.redo(model_id, checkpoint_id)},
                True,
                "backprop",
                "Redid surgical edit",
            )
        if act == "reset":
            return (
                {"action": "reset", **rt.editor.reset(model_id, checkpoint_id)},
                True,
                "backprop",
                "Reset all edits to base weights",
            )
        if act == "export_recipe":
            return (
                {
                    "action": "export_recipe",
                    "recipe": rt.editor.export_recipe(model_id, checkpoint_id),
                },
                False,
                "",
                "Exported portable edit recipe",
            )
        if act == "apply_recipe":
            recipe = args.get("recipe")
            if not isinstance(recipe, dict):
                raise ValueError("apply_recipe requires a 'recipe' dict.")
            return (
                {
                    "action": "apply_recipe",
                    **rt.editor.apply_recipe(
                        model_id,
                        recipe,
                        checkpoint_id=checkpoint_id,
                        reset_first=bool(args.get("reset_first", True)),
                    ),
                },
                True,
                "backprop",
                f"Replayed recipe ({len(recipe.get('operations', []))} ops)",
            )
        raise ValueError(
            f"Unknown action {act!r}. Choose history, undo, redo, reset, export_recipe, or apply_recipe."
        )

    if name == "studio_arena_compare":
        res = rt.inference.compare_arena_candidates(
            model_id,
            str(args["prompt"]),
            args.get("candidates"),
            system_prompt=str(args.get("system_prompt") or ""),
            checkpoint_id=checkpoint_id,
            reward_fn=str(args.get("reward_fn") or "composite"),
        )
        return (
            dict(res),
            False,
            "forward",
            f"Benchmarked {len(res.get('candidates', []))} Arena candidates",
        )

    if name == "studio_auto_repair":
        res = rt.editor.auto_repair_edits(
            model_id,
            rt.inference,
            checkpoint_id=checkpoint_id,
            reward_fn=str(args.get("reward_fn") or "composite"),
            max_steps=int(args.get("max_steps", 4)),
        )
        return dict(res), True, "backprop", f"Auto-Repair → {res.get('final_verdict')}"

    if name == "studio_eval_and_export":
        gate = rt.inference.run_batch_eval(
            model_id,
            checkpoint_id=checkpoint_id,
            custom_jsonl=str(args.get("custom_jsonl") or "") or None,
            reward_fn=str(args.get("reward_fn") or "composite"),
            max_cases=int(args.get("max_cases", 6)),
        )
        out: dict[str, Any] = {
            "eval_gate": gate,
            "exported": False,
            "manifest": None,
            "recipe": rt.editor.export_recipe(model_id, checkpoint_id),
        }
        export_name = str(args.get("export_name") or "").strip()
        if export_name:
            if bool(args.get("require_go", True)) and gate["verdict"] != "GO":
                out["export_blocked_reason"] = (
                    f"Gate verdict is {gate['verdict']}; call studio_auto_repair(...) or pass require_go=False."
                )
            else:
                out["manifest"] = rt.editor.export_verified_checkpoint(
                    model_id,
                    export_name,
                    checkpoint_id=checkpoint_id,
                    run_merge_probe=bool(args.get("run_merge_probe", True)),
                )
                out["exported"] = True
        return out, False, "forward", f"Eval Gate: {gate['verdict']}"

    if name == "studio_export_lora":
        res = rt.editor.export_diff_as_lora(
            model_id,
            str(args.get("export_name") or "surgical-lora-v1"),
            rank=int(args.get("rank", 16)),
            alpha=int(args.get("alpha", 32)),
            checkpoint_id=checkpoint_id,
        )
        return (
            dict(res),
            False,
            "forward",
            f"Exported PEFT LoRA ({res.get('export_dir')})",
        )

    if name == "studio_export_artifact":
        fmt = str(args.get("format") or "gguf").strip().lower()
        export_name = str(args.get("export_name") or "surgical-export-v1")
        if fmt == "gguf":
            res = rt.editor.export_gguf_package(
                model_id,
                export_name,
                quant_type=str(args.get("quant_type") or "Q8_0"),
                checkpoint_id=checkpoint_id,
            )
            return (
                dict(res),
                False,
                "forward",
                f"Exported GGUF v3 ({res.get('quant_type')})",
            )
        if fmt == "hf_card":
            eval_rep = (
                rt.inference.run_batch_eval(
                    model_id, checkpoint_id=checkpoint_id, max_cases=6
                )
                if bool(args.get("include_eval", True))
                else None
            )
            res = rt.editor.export_hf_model_card(
                model_id,
                export_name,
                repo_id=str(args.get("repo_id") or ""),
                eval_report=eval_rep,
                checkpoint_id=checkpoint_id,
            )
            return (
                dict(res),
                False,
                "forward",
                f"Exported HF Model Card ({export_name})",
            )
        if fmt == "html_report":
            from .report_exporter import build_standalone_html_report

            overrides = rt.editor.get_overrides(model_id, checkpoint_id)
            struct = rt.inspector.get_model_structure(
                model_id, checkpoint_id, overrides=overrides if overrides else None
            )
            eval_rep = (
                rt.inference.run_batch_eval(
                    model_id, checkpoint_id=checkpoint_id, max_cases=6
                )
                if bool(args.get("include_eval", True))
                else None
            )
            html_doc = build_standalone_html_report(
                model_summary=struct["model"],
                structure_summary=struct["summary"],
                layers=struct["layers"],
                edit_history=rt.editor.get_history(model_id, checkpoint_id),
                recipe=rt.editor.export_recipe(model_id, checkpoint_id),
                eval_report=eval_rep,
            )
            safe = (
                "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "-"
                    for ch in export_name
                ).strip("-")
                or "studio-report"
            )
            out_dir = (
                rt.inspector.workspace_root / "outputs" / "weight_studio_exports" / safe
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            rp = out_dir / "report.html"
            rp.write_text(html_doc, encoding="utf-8")
            return (
                {
                    "format": "html_report",
                    "report_path": str(rp),
                    "bytes": len(html_doc.encode("utf-8")),
                },
                False,
                "forward",
                f"Exported 3D HTML PR report ({rp.name})",
            )
        raise ValueError(f"Unsupported format {fmt!r}.")

    if name == "studio_save_session":
        res = rt.editor.save_session(
            str(args["session_name"]),
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            metadata=args.get("metadata"),
        )
        return dict(res), False, "", f"Saved session '{args['session_name']}'"

    if name == "studio_sessions":
        act = str(args.get("action") or "list").strip().lower()
        if act == "list":
            sessions = rt.editor.list_sessions()
            return (
                {"action": "list", "count": len(sessions), "sessions": sessions},
                False,
                "",
                f"Listed {len(sessions)} sessions",
            )
        if act == "load":
            res = rt.editor.load_session(str(args.get("session_name") or ""))
            return (
                {"action": "load", **res},
                True,
                "backprop",
                f"Loaded session '{args.get('session_name')}'",
            )
        raise ValueError(f"Unknown session action {act!r}.")

    if name == "studio_auto_heal":
        res = rt.healer.run_healing_loop(
            model_id,
            checkpoint_id=checkpoint_id,
            transcripts_path=str(args.get("transcripts_path") or "") or None,
            transcripts_jsonl=str(args.get("transcripts_jsonl") or "") or None,
            reward_fn=str(args.get("reward_fn") or "customer_support"),
            export_format=str(args.get("export_format") or "lora"),
            output_name=str(args.get("output_name") or ""),
            lora_rank=int(args.get("lora_rank", 16)),
            lora_alpha=int(args.get("lora_alpha", 32)),
            require_go=bool(args.get("require_go", True)),
        )
        verdict = (res.get("eval_gate") or {}).get("verdict", "GO")
        return dict(res), True, "backprop", f"Autonomous Self-Heal → {verdict}"

    if name == "studio_redteam_audit":
        immunize = bool(args.get("immunize", False))
        res = rt.redteam.run_redteam_audit(
            model_id,
            checkpoint_id=checkpoint_id,
            categories=args.get("categories"),
            probes_path=str(args.get("probes_path") or "") or None,
            probes_jsonl=str(args.get("probes_jsonl") or "") or None,
            max_probes=int(args.get("max_probes", 10)),
            immunize=immunize,
            export_format=str(args.get("export_format") or "none"),
            output_name=str(args.get("output_name") or ""),
        )
        action_label = "Red-Team Immunize" if immunize else "Red-Team Audit"
        return (
            dict(res),
            immunize,
            "backprop" if immunize else "forward",
            f"{action_label} → {res['verdict']} (ASR {res['attack_success_rate']*100:.0f}%)",
        )

    if name == "studio_model_tournament":
        res = rt.tournament.run_tournament(
            model_ids=args.get("model_ids"),
            reward_fn=str(args.get("reward_fn") or "customer_support"),
            max_cases=int(args.get("max_cases", 4)),
            include_redteam=bool(args.get("include_redteam", True)),
            synthesize_merge=bool(args.get("synthesize_merge", True)),
            merge_methods=args.get("merge_methods"),
            export_format=str(args.get("export_format") or "none"),
            output_name=str(args.get("output_name") or ""),
        )
        champ = res.get("champion") or {}
        mutated = bool(res.get("promoted_merge"))
        return (
            dict(res),
            mutated,
            "backprop" if mutated else "forward",
            f"Tournament Champion: {champ.get('label', '?')} (fit {champ.get('fitness', 0):.3f})",
        )

    if name == "studio_canary_gate":
        res = rt.canary.run_canary_gate(
            model_id,
            checkpoint_id=checkpoint_id,
            stage=str(args.get("stage") or "canary_10"),
            traffic_path=str(args.get("traffic_path") or "") or None,
            traffic_jsonl=str(args.get("traffic_jsonl") or "") or None,
            reward_fn=str(args.get("reward_fn") or "composite"),
            max_requests=int(args.get("max_requests", 6)),
            max_asr=float(args.get("max_asr", 0.25)),
        )
        return (
            dict(res),
            False,
            "forward",
            f"Canary Gate [{res['stage']}] → {res['decision']} ({res['recommended_traffic_pct']}%)",
        )

    if name == "studio_autopilot_release":
        res = rt.canary.run_autopilot_release(
            model_id,
            checkpoint_id=checkpoint_id,
            transcripts_path=str(args.get("transcripts_path") or "") or None,
            transcripts_jsonl=str(args.get("transcripts_jsonl") or "") or None,
            reward_fn=str(args.get("reward_fn") or "customer_support"),
            immunize_redteam=bool(args.get("immunize_redteam", True)),
            stage=str(args.get("stage") or "canary_10"),
            export_format=str(args.get("export_format") or "lora"),
            output_name=str(args.get("output_name") or ""),
            lora_rank=int(args.get("lora_rank", 16)),
            lora_alpha=int(args.get("lora_alpha", 32)),
        )
        return (
            dict(res),
            True,
            "backprop",
            f"Autopilot → {res['decision']} (traffic {res['recommended_traffic_pct']}%)",
        )

    raise KeyError(f"Unknown MCP tool: {name}")


def _read_studio_resource(rt: Any, uri: str) -> str:
    if uri == "stateset://studio/capabilities":
        return json.dumps(
            {
                "server": "stateset-weight-studio",
                "transport": "streamable-http",
                "workspace": str(rt.inspector.workspace_root),
                "tool_count": len(STUDIO_MCP_TOOLS),
                "live_browser_sync": True,
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
                "export_formats": [
                    "lora",
                    "safetensors",
                    "gguf",
                    "hf_card",
                    "html_report",
                ],
            },
            indent=2,
        )
    if uri == "stateset://studio/models":
        models = rt.inspector.list_models()
        return json.dumps({"count": len(models), "models": models}, indent=2)
    if uri == "stateset://studio/steering-presets":
        return json.dumps({"presets": rt.inference.list_steering_presets()}, indent=2)
    if uri == "stateset://studio/scenarios":
        return json.dumps({"scenarios": rt.inference.list_scenarios()}, indent=2)
    if uri == "stateset://studio/sessions":
        sessions = rt.editor.list_sessions()
        return json.dumps({"count": len(sessions), "sessions": sessions}, indent=2)
    if uri == "stateset://studio/redteam-probes":
        return json.dumps({"probes": rt.redteam.list_probes()}, indent=2)
    if uri == "stateset://studio/rollouts":
        rollouts = rt.canary.list_rollouts(limit=25)
        return json.dumps({"count": len(rollouts), "rollouts": rollouts}, indent=2)
    raise KeyError(f"Unknown MCP resource URI: {uri}")


def _get_studio_prompt(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    model_id = str(arguments.get("model_id") or "preset:deepseek-r1-1.5b")
    if name == "production_autopilot_release":
        fmt = str(arguments.get("export_format") or "lora")
        tp = str(arguments.get("transcripts_path") or "")
        text = (
            f"Run `studio_autopilot_release(model_id={model_id!r}, transcripts_path={tp!r}, "
            f"immunize_redteam=True, export_format={fmt!r})` to heal failing logs, immunize against "
            "adversarial attacks, run the Paired Shadow Canary Gate, record the rollout ledger, and "
            "export a verified bundle + `autopilot_dossier.html`."
        )
        return {
            "description": "Full Production Autopilot release playbook.",
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }
    if name == "redteam_and_immunize":
        fmt = str(arguments.get("export_format") or "lora")
        text = (
            f"Audit `{model_id}` with `studio_redteam_audit(model_id={model_id!r}, immunize=False)` "
            f"to measure baseline Attack Success Rate (ASR) across prompt injection, PII exfiltration, "
            f"unauthorized tool calls, policy bypass, and hallucinated commitments. Then run "
            f"`studio_redteam_audit(model_id={model_id!r}, immunize=True, export_format={fmt!r})` "
            "to surgically immunize the vulnerable layer while protecting benign support helpfulness."
        )
        return {
            "description": "Red-team safety audit and surgical immunization playbook.",
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }
    if name == "model_tournament_and_merge":
        reward = str(arguments.get("reward") or "customer_support")
        fmt = str(arguments.get("export_format") or "lora")
        text = (
            f"Call `studio_model_tournament(reward_fn={reward!r}, include_redteam=True, "
            f"synthesize_merge=True, export_format={fmt!r})` to benchmark top workspace checkpoints "
            "and synthesize a Pareto-optimal TIES/DARE/SLERP champion."
        )
        return {
            "description": "Multi-model Pareto tournament and surgical merge playbook.",
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }
    if name == "auto_heal_model":
        text = (
            f"Call `studio_inspect_model(model_id={model_id!r})`, then run "
            f"`studio_auto_heal(model_id={model_id!r}, transcripts_path={arguments.get('transcripts_path', '')!r}, "
            f"reward_fn={arguments.get('reward', 'customer_support')!r}, "
            f"export_format={arguments.get('export_format', 'lora')!r})`."
        )
        return {
            "description": "Autonomous transcript-to-surgery healing playbook.",
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }
    if name == "surgical_behavior_edit":
        goal = str(arguments.get("behavior_goal") or "improve policy grounding")
        text = (
            f"Extract a CAA steering vector on `{model_id}` for `{goal}` using `studio_extract_steering`, "
            "sweep strengths in `studio_arena_compare`, bake with `studio_apply_edit`, check `studio_residual_stream` "
            "& `studio_eval_and_export`, and export via `studio_export_lora`."
        )
        return {
            "description": "Targeted behavioral surgery playbook.",
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }
    if name == "interpretability_audit":
        prompt = str(arguments.get("prompt") or "Process refund for order #77701")
        text = (
            f"Inspect `{model_id}` on {prompt!r} via `studio_discover_circuit`, "
            "`studio_residual_stream`, `studio_sae_analyze`, and `studio_profile_sensitivity`."
        )
        return {
            "description": "Full mechanistic interpretability audit playbook.",
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }
    if name == "compress_and_export_model":
        text = (
            f"Run `studio_profile_sensitivity(model_id={model_id!r}, "
            f"target_compression={arguments.get('target_compression', 0.35)})`, replay `recommended_recipe` "
            "via `studio_manage_edits(action='apply_recipe')`, verify with `studio_eval_and_export` / "
            f"`studio_auto_repair`, and export via `studio_export_artifact(format={arguments.get('export_format', 'gguf')!r})`."
        )
        return {
            "description": "Adaptive Wanda/SVD compression and verified export playbook.",
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }
    raise KeyError(f"Unknown MCP prompt: {name}")


def handle_mcp_jsonrpc(rt: Any, req: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one MCP JSON-RPC 2.0 request against a live ``StudioRuntime``."""
    req_id = req.get("id")
    method = str(req.get("method") or "")
    params = req.get("params") or {}

    if method == "notifications/initialized":
        return None

    def _ok(result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _err(code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    try:
        if method == "initialize":
            rt.events.publish(
                "agent_connected",
                source="mcp_http",
                tool="initialize",
                summary=f"AI Agent connected ({(params.get('clientInfo') or {}).get('name', 'MCP Client')})",
            )
            return _ok(
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "stateset-weight-studio",
                        "version": "0.84.0",
                    },
                    "instructions": (
                        "Live 3D Weight Studio MCP server. Every tool call runs directly "
                        "against the running Studio runtime and animates connected 3D "
                        "browser viewports in real time."
                    ),
                }
            )

        if method == "ping":
            return _ok({})

        if method == "tools/list":
            return _ok({"tools": STUDIO_MCP_TOOLS})

        if method == "tools/call":
            tool_name = str(params.get("name") or "")
            tool_args = dict(params.get("arguments") or {})
            try:
                result, mutated, pulse, summary = _dispatch_studio_tool(
                    rt, tool_name, tool_args
                )
            except Exception as exc:
                available: list[str] | None = None
                if isinstance(exc, KeyError):
                    try:
                        available = [
                            str(m["model_id"]) for m in rt.inspector.list_models()[:10]
                        ]
                    except Exception:
                        available = None
                err_payload: dict[str, Any] = {"error": f"{type(exc).__name__}: {exc}"}
                if available:
                    err_payload["available_models"] = available
                    err_payload["hint"] = (
                        "Call studio_list_models() to see valid model_ids."
                    )
                return _ok(
                    {
                        "isError": True,
                        "content": [{"type": "text", "text": json.dumps(err_payload)}],
                        "structuredContent": err_payload,
                    }
                )

            rt.events.publish(
                "agent_tool_call",
                source="mcp_http",
                tool=tool_name,
                model_id=str(tool_args.get("model_id") or ""),
                summary=summary,
                mutated=mutated,
                pulse=pulse,
            )
            return _ok(
                {
                    "isError": False,
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, default=str),
                        }
                    ],
                    "structuredContent": result,
                }
            )

        if method == "resources/list":
            return _ok({"resources": STUDIO_MCP_RESOURCES})

        if method == "resources/read":
            uri = str(params.get("uri") or "")
            text = _read_studio_resource(rt, uri)
            return _ok(
                {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": text,
                        }
                    ]
                }
            )

        if method == "prompts/list":
            return _ok({"prompts": STUDIO_MCP_PROMPTS})

        if method == "prompts/get":
            prompt_name = str(params.get("name") or "")
            prompt_args = dict(params.get("arguments") or {})
            return _ok(_get_studio_prompt(prompt_name, prompt_args))

        return _err(-32601, f"Method not found: {method}")
    except KeyError as exc:
        return _err(-32602, str(exc))
    except Exception as exc:
        return _err(-32603, f"{type(exc).__name__}: {exc}")
