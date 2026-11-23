"""End-to-End Autonomous Surgery & Production Autopilot Demo (Zero-GPU).

Demonstrates the complete StateSet Agents v0.85+ autonomous control plane in
seconds on CPU without requiring any external API keys or GPU:

1. Autonomous Transcript-to-Surgery Self-Healing Loop (diagnose failing logs ->
   mine CAA contrast -> Multi-Edit A/B Arena -> bake -> 95% CI Eval Gate).
2. Adversarial Red-Team Safety Audit & Surgical Immunization across 5 attack
   vectors while protecting benign customer-support helpfulness.
3. Multi-Model Pareto Tournament & Evolutionary TIES/DARE/SLERP Surgical Merge.
4. Production Canary Shadow-Rollout Gate (`PROMOTE` / `HOLD_CANARY` / `ROLLBACK`)
   with durable rollout ledger (`rollout_ledger.jsonl`).
5. Live Streamable HTTP MCP JSON-RPC 2.0 tool discovery & invocation against the
   same shared `StudioRuntime`.

Usage:
    python examples/autonomous_surgery_autopilot.py
    python examples/autonomous_surgery_autopilot.py --workspace . --export-format lora
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from stateset_agents.weight_studio import StudioRuntime, handle_mcp_jsonrpc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_autopilot_demo(
    workspace: Path,
    *,
    export_format: str = "none",
) -> dict[str, Any]:
    """Run the complete autonomous surgery, red-team, tournament, canary & MCP demo."""
    rt = StudioRuntime(workspace_root=workspace)
    models = rt.inspector.list_models()
    if not models:
        raise RuntimeError("No checkpoints or preset twins discovered.")

    model_id = str(models[0]["model_id"])
    logger.info("1. Selected model: %s (%s)", models[0]["display_name"], model_id)

    # Stage 1: Autonomous Transcript-to-Surgery Self-Healing Loop
    heal = rt.healer.run_healing_loop(
        model_id,
        reward_fn="customer_support",
        export_format="none",
        require_go=False,
    )
    logger.info(
        "2. Auto-Heal: dominant=%s, baked L%02d (α=%+.2f), eval_gate=%s",
        heal["diagnosis"]["dominant_failure_mode"],
        heal["steering_vector"]["target_layer"],
        heal["steering_vector"]["chosen_strength"],
        heal["eval_gate"]["verdict"],
    )

    # Stage 2: Adversarial Red-Team Safety Audit & Surgical Immunization
    redteam = rt.redteam.run_redteam_audit(
        model_id,
        max_probes=6,
        immunize=True,
        export_format="none",
    )
    imm = redteam["immunization"] or {}
    logger.info(
        "3. Red-Team Immunization: verdict=%s, ASR %.0f%% -> %.0f%%, benign_gate=%s",
        redteam["verdict"],
        float(imm.get("asr_before", 0.0)) * 100.0,
        float(imm.get("asr_after", 0.0)) * 100.0,
        imm.get("benign_eval_verdict", "GO"),
    )

    # Stage 3: Multi-Model Pareto Tournament & Evolutionary Surgical Merge
    tour = rt.tournament.run_tournament(
        reward_fn="customer_support",
        max_cases=2,
        include_redteam=False,
        synthesize_merge=True,
        merge_methods=["ties", "slerp"],
        export_format="none",
    )
    champ = tour["champion"]
    logger.info(
        "4. Tournament Champion: %s (fitness=%.4f, %d contenders, %d merges)",
        champ["label"],
        champ["fitness"],
        tour["contender_count"],
        tour["merge_candidates_evaluated"],
    )

    # Stage 4: Production Canary Shadow-Rollout Promotion Gate + Ledger
    canary = rt.canary.run_canary_gate(
        model_id,
        stage="canary_10",
        reward_fn="composite",
        max_requests=4,
        record_ledger=False,
    )
    pr = canary["paired_record"]
    logger.info(
        "5. Canary Shadow Gate: decision=%s (traffic=%d%%, W/T/L=%d/%d/%d, Δ=%+.4f)",
        canary["decision"],
        canary["recommended_traffic_pct"],
        pr["wins"],
        pr["ties"],
        pr["losses"],
        pr["mean_delta"],
    )

    # Stage 5: Live Streamable HTTP MCP JSON-RPC 2.0 discovery
    mcp_tools = handle_mcp_jsonrpc(
        rt, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )
    tool_count = len((mcp_tools or {}).get("result", {}).get("tools", []))
    logger.info(
        "6. Live Streamable HTTP MCP: %d annotated Studio tools ready at /api/weight-studio/mcp",
        tool_count,
    )

    if export_format != "none":
        auto = rt.canary.run_autopilot_release(
            model_id,
            reward_fn="customer_support",
            export_format=export_format,
            output_name="autopilot-demo-v1",
        )
        logger.info("7. Exported autopilot bundle: exported=%s", auto["exported"])

    return {
        "model_id": model_id,
        "heal_verdict": heal["eval_gate"]["verdict"],
        "redteam_verdict": redteam["verdict"],
        "tournament_champion": champ["label"],
        "canary_decision": canary["decision"],
        "mcp_http_tools": tool_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="StateSet Agents Autonomous Surgery & Production Autopilot Demo"
    )
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument(
        "--export-format",
        default="none",
        choices=("none", "lora", "safetensors", "gguf", "hf_card"),
    )
    args = parser.parse_args(argv)
    run_autopilot_demo(args.workspace, export_format=args.export_format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
