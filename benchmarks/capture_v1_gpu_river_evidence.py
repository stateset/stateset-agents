"""Unified v1.0.0 GA GPU + River AI 0.11 Evidence Capture CLI Runner.

Thin CLI wrapper over ``stateset_agents.utils.colab`` so both standalone CLI
invocations (from any working directory) and bare-wheel Google Colab notebooks
share one canonical, SHA-256 attested evidence pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stateset_agents.utils.colab import capture_v1_evidence as capture_evidence
from stateset_agents.utils.colab import (
    render_v1_evidence_markdown as render_markdown_report,
)
from stateset_agents.utils.colab import (
    run_cuda_chunked_vram_benchmark,
    run_live_hf_open_weight_smoke,
    run_river_verification,
)

__all__ = [
    "capture_evidence",
    "main",
    "render_markdown_report",
    "run_cuda_chunked_vram_benchmark",
    "run_live_hf_open_weight_smoke",
    "run_river_verification",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/v1_ga_evidence"),
        help="Directory where v1_ga_evidence.json and v1_ga_evidence.md will be written.",
    )
    parser.add_argument(
        "--hf-model",
        default="",
        help="Optional HuggingFace causal LM ID for live Step-0 CAA + Chunked GSPO LoRA run.",
    )
    parser.add_argument(
        "--hf-revision",
        default="main",
        help="HuggingFace revision or 40-char commit SHA for --hf-model.",
    )
    parser.add_argument(
        "--hf-steps",
        type=int,
        default=4,
        help="Chunked GSPO optimizer steps for the live HF model smoke.",
    )
    parser.add_argument(
        "--skip-hf-model",
        action="store_true",
        help="Skip downloading/training a HuggingFace model (fast offline verification).",
    )
    parser.add_argument(
        "--river-mode",
        choices=("offline", "preflight", "live"),
        default="offline",
        help="River AI 0.11 verification level: offline, preflight, or live.",
    )
    parser.add_argument(
        "--river-base-model",
        default="Qwen/Qwen3.6-35B-A3B-FP8",
        help="River base model ID to check during --river-mode preflight/live.",
    )
    parser.add_argument(
        "--river-checkpoint-uri",
        default="",
        help="Optional river:// checkpoint URI to query in --river-mode live.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete JSON evidence payload to stdout.",
    )
    args = parser.parse_args(argv)

    result = capture_evidence(
        output_dir=args.output_dir,
        hf_model=args.hf_model or None,
        hf_revision=args.hf_revision,
        hf_steps=args.hf_steps,
        skip_hf_model=args.skip_hf_model,
        river_mode=args.river_mode,
        river_base_model=args.river_base_model,
        river_checkpoint_uri=args.river_checkpoint_uri,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_markdown_report(result))
        print(f"Wrote {args.output_dir / 'v1_ga_evidence.json'}")
        print(f"Wrote {args.output_dir / 'v1_ga_evidence.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
