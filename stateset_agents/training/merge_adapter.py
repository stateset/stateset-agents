"""Merge a LoRA adapter into full base weights — the hybrid-serving fix.

Runs ON the serving pod (``python -m stateset_agents.training.merge_adapter``)
before vLLM starts. Exists because of a disproven claim: vLLM loads LoRA
adapters for hybrid Qwen3.5 models without error and then silently serves
the base weights — the hybrid ``linear_attn`` target modules (``in_proj_qkv``
and friends) never match its LoRA mapping, proven by byte-identical greedy
completions from "adapter" and base (see ``docs/PROOFS.md``, 2026-08-18).

Merging sidesteps the mapping entirely: peft applies every adapter delta to
the modules it was trained on (peft DOES know them — it trained them), and
vLLM then serves an ordinary full checkpoint with no ``--enable-lora`` at
all.

Dry-run/no-GPU behaviour mirrors the sft/harvest modules: print the plan,
exit 0 — so the path is exercisable on CPU-only CI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stateset_agents.training.sft import gpu_available, load_base_model_for_sft

logger = logging.getLogger(__name__)

__all__ = ["main", "merge_adapter"]


def merge_adapter(base_model: str, adapter_dir: Path, output_dir: Path) -> Path:
    """Load base + adapter, merge, save the full checkpoint to ``output_dir``.

    The tokenizer is saved alongside the weights so the output directory is
    a complete, self-sufficient model that ``vllm serve <dir>`` accepts.
    """
    from peft import PeftModel
    from transformers import AutoTokenizer

    model = load_base_model_for_sft(base_model)
    model = PeftModel.from_pretrained(model, str(adapter_dir))  # nosec: B615
    logger.info("merging adapter %s into %s…", adapter_dir, base_model)
    model = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)  # nosec: B615
    tokenizer.save_pretrained(str(output_dir))
    logger.info("merged model saved to %s", output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into full base weights for serving."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    if args.dry_run or not gpu_available():
        print("Merge Plan (dry run — no GPU detected)")
        print(f"  base_model: {args.base_model}")
        print(f"  adapter:    {args.adapter}")
        print(f"  output_dir: {args.output_dir}")
        return 0
    merge_adapter(args.base_model, args.adapter, args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess tests
    raise SystemExit(main())
