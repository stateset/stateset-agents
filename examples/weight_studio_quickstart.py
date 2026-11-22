"""Quick Start for StateSet 3D Weight Studio.

Demonstrates the full Weight Studio workflow programmatically without requiring a GPU:
1. Discover local ``.safetensors`` checkpoints (or fall back to a built-in architecture twin).
2. Inspect model layer structure and SVD spectrum.
3. Extract a Contrastive Activation Steering (CAA / RepE) vector.
4. Apply non-destructive Wanda pruning and bake the steering vector into weights.
5. Discover the Top-K causal attention circuit for a customer-support prompt.
6. Run the 6-case Pre-Export Batch Eval Gate (GO / WARN / BLOCK).
7. Export a portable ``.studio-recipe.json`` (and optionally a verified ``.safetensors`` artifact
   or launch the live interactive 3D browser UI with ``--serve``).

Usage:
    python examples/weight_studio_quickstart.py
    python examples/weight_studio_quickstart.py --export my-studio-demo
    python examples/weight_studio_quickstart.py --serve --port 8765 --open
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from stateset_agents.weight_studio import StudioRuntime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_quickstart(
    workspace: Path,
    *,
    export_name: str | None = None,
    recipe_out: Path | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end headless Weight Studio workflow and return a summary dict."""
    rt = StudioRuntime(workspace_root=workspace)
    models = rt.inspector.list_models()
    if not models:
        raise RuntimeError("No local checkpoints or preset twins discovered.")

    model = models[0]
    model_id = str(model["model_id"])
    logger.info("1. Loaded model: %s (%s)", model["display_name"], model_id)

    struct = rt.inspector.get_model_structure(model_id)
    summary = struct["summary"]
    logger.info(
        "   Architecture: %d layers, %d tensors, %s params",
        summary["num_layers"],
        summary["num_tensors"],
        f"{summary['total_params']:,}",
    )

    # 2. Extract a Contrastive Activation Steering (CAA) vector
    presets = rt.inference.list_steering_presets()
    preset = presets[0]
    vec = rt.inference.extract_steering_vector(
        model_id,
        preset["positive"],
        preset["negative"],
        name=preset["name"],
    )
    rec_layer = int(vec["recommended_layer"])
    logger.info(
        "2. Extracted CAA vector '%s' (vector_id=%s, layer=%d, peak_sep=%.4f)",
        vec["name"],
        vec["vector_id"],
        rec_layer,
        vec["peak_separation"],
    )

    # 3. Apply non-destructive Wanda pruning + CAA weight bake
    first_layer = struct["layers"][0]
    first_tensor = first_layer["tensors"][0]["name"]
    r_prune = rt.editor.apply_edit(
        model_id,
        "prune",
        first_tensor,
        {"mode": "wanda", "sparsity": 0.25},
        label="Wanda Prune 25%",
    )
    r_bake = rt.editor.apply_edit(
        model_id,
        "steer_bake",
        f"layer:{rec_layer}",
        {
            "concept": vec["name"],
            "strength": 0.65,
            "direction": vec["direction"],
        },
        label=f"Bake CAA '{vec['name']}' (L{rec_layer:02d})",
    )
    logger.info(
        "3. Applied 2 non-destructive surgical edits (%d + %d changed weights)",
        r_prune["edit"]["changed_weights"],
        r_bake["edit"]["changed_weights"],
    )

    # 4. Discover causal circuit for a StateSet support prompt
    prompt = (
        "Customer says: My order #77701 arrived damaged after 9 days "
        "and I was billed twice. Can you refund me?"
    )
    circuit = rt.inference.discover_causal_circuit(model_id, prompt, top_k=8)
    logger.info(
        "4. Discovered %d-node causal circuit (%d 3D attention-flow arcs, top=%s)",
        circuit["summary"]["circuit_nodes"],
        circuit["summary"]["arc_count"],
        circuit["summary"]["top_node"],
    )

    # 5. Run the Pre-Export Batch Eval Gate (GO / WARN / BLOCK)
    gate = rt.inference.run_batch_eval(model_id, max_cases=6)
    logger.info(
        "5. Pre-Export Batch Eval Gate: %s — %s",
        gate["verdict"],
        gate["verdict_reason"],
    )

    # 6. Export portable .studio-recipe.json (and optional verified .safetensors)
    recipe = rt.editor.export_recipe(model_id)
    if recipe_out is not None:
        recipe_out.parent.mkdir(parents=True, exist_ok=True)
        recipe_out.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "6. Wrote portable edit recipe (%d ops) to %s",
            len(recipe["operations"]),
            recipe_out,
        )

    manifest: dict[str, Any] | None = None
    if export_name:
        manifest = rt.editor.export_verified_checkpoint(
            model_id, export_name, run_merge_probe=True
        )
        logger.info(
            "7. Exported SHA-256 verified checkpoint: %s (reload_verified=%s)",
            manifest["export_file"],
            manifest["reload_verified"],
        )

    return {
        "model_id": model_id,
        "steering_vector_id": vec["vector_id"],
        "circuit_nodes": circuit["summary"]["circuit_nodes"],
        "eval_verdict": gate["verdict"],
        "recipe_ops": len(recipe["operations"]),
        "manifest": manifest,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Weight Studio quick start."""
    parser = argparse.ArgumentParser(
        description="StateSet 3D Weight Studio Quick Start"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Workspace directory containing outputs/ and checkpoints/.",
    )
    parser.add_argument(
        "--export",
        dest="export_name",
        default=None,
        help="Optional directory name under outputs/weight_studio_exports/ to write verified .safetensors.",
    )
    parser.add_argument(
        "--recipe-out",
        type=Path,
        default=None,
        help="Optional output path for the portable .studio-recipe.json file.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the interactive 3D WebGL2 browser Studio after running the demo.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for --serve.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port for --serve.")
    parser.add_argument(
        "--open", action="store_true", help="Open browser when using --serve."
    )
    args = parser.parse_args(argv)

    run_quickstart(
        args.workspace,
        export_name=args.export_name,
        recipe_out=args.recipe_out,
    )

    if args.serve:
        from stateset_agents.weight_studio.__main__ import main as studio_main

        serve_argv = [
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--workspace",
            str(args.workspace),
        ]
        if args.open:
            serve_argv.append("--open")
        studio_main(serve_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
