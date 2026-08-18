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
from typing import Any

from stateset_agents.training.sft import (
    generate_completions,
    gpu_available,
    load_base_model_for_sft,
)


def remap_adapter_keys(weights: dict, model_param_names: set[str]) -> tuple[dict, int]:
    """Rewrite adapter keys trained on the TEXT extraction for a COMPOSITE.

    ``AutoModelForCausalLM`` extracts the text model from composite
    multimodal checkpoints and renames ``model.language_model.*`` to
    ``model.*`` — so an adapter trained through it does not match the
    composite's module paths, and peft applies NOTHING beyond a warning
    (measured: probe delta exactly 0.0). Inserting ``language_model.``
    back restores the match (measured: 372/372 keys, real deltas).

    Only keys whose composite spelling actually exists on the model are
    rewritten, so text-only models pass through untouched.
    """
    remapped = {}
    changed = 0
    for key, value in weights.items():
        candidate = key.replace("model.layers.", "model.language_model.layers.", 1)
        if candidate != key:
            # Validate against the real module tree: the base parameter the
            # adapter attaches to must exist under the composite spelling.
            base_param = (
                candidate.split(".lora_")[0].replace("base_model.model.", "", 1)
                + ".weight"
            )
            if base_param in model_param_names:
                remapped[candidate] = value
                changed += 1
                continue
        remapped[key] = value
    return remapped, changed


logger = logging.getLogger(__name__)

__all__ = ["main", "merge_adapter"]


#: The effect-probe prompt; any prompt works — the probe compares greedy
#: completions before and after merging, not their content.
PROBE_PROMPT = "Reply with one sentence: what can you help me with?"


def merge_adapter(base_model: str, adapter_dir: Path, output_dir: Path) -> Path:
    """Load base + adapter, merge, save the full checkpoint to ``output_dir``.

    The tokenizer is saved alongside the weights so the output directory is
    a complete, self-sufficient model that ``vllm serve <dir>`` accepts.

    The merge verifies its own effect: one greedy completion is generated
    before the adapter is applied and again from the merged weights, and
    the pair lands in ``merge_probe.json``. Identical completions mean the
    adapter changed nothing observable — the exact silent no-op this module
    exists to prevent — and raise rather than serve a lie.
    """
    import json

    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)  # nosec: B615
    model = _load_full_checkpoint(base_model)
    if gpu_available():
        model = model.to("cuda")
    (base_completion,) = generate_completions(
        model, tokenizer, [PROBE_PROMPT], max_new_tokens=48
    )
    adapter_dir = _adapter_for_model(model, Path(adapter_dir))
    model = PeftModel.from_pretrained(model, str(adapter_dir))  # nosec: B615
    logger.info("merging adapter %s into %s…", adapter_dir, base_model)
    model = model.merge_and_unload()
    (merged_completion,) = generate_completions(
        model, tokenizer, [PROBE_PROMPT], max_new_tokens=48
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    # Composite multimodal checkpoints (the Qwen3.5 family included) need
    # their PROCESSOR artifacts too — without them vLLM's engine dies
    # loading the merged directory (observed live: Qwen3-VL video-processor
    # errors at boot). Text-only models simply have no processor; skip.
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(base_model).save_pretrained(  # nosec: B615
            str(output_dir)
        )
        logger.info("processor artifacts saved alongside the merged weights")
    except Exception as exc:  # noqa: BLE001 - text-only models land here
        logger.info("no processor saved (%s) — fine for text-only models", exc)
    (output_dir / "merge_probe.json").write_text(
        json.dumps(
            {
                "prompt": PROBE_PROMPT,
                "base": base_completion,
                "merged": merged_completion,
                "identical": base_completion == merged_completion,
            },
            indent=2,
        )
    )
    if base_completion == merged_completion:
        raise RuntimeError(
            "merged weights produce a greedy completion byte-identical to "
            "the base model — the adapter had no observable effect; "
            "refusing to serve it as a fine-tune (see merge_probe.json)"
        )
    logger.info("merged model saved to %s (effect probe: differs)", output_dir)
    return output_dir


def _load_full_checkpoint(base_model: str) -> Any:
    """Load the checkpoint in its OWN architecture, composite included.

    ``load_base_model_for_sft`` (via ``AutoModelForCausalLM``) extracts the
    text model from composites, which saves a directory whose config says
    ``model_type=qwen3_5_text`` with ``architectures: None`` — vLLM then
    guesses the architecture and dies loading it (observed live, merge
    attempt 4). Serving needs the hub-identical layout, so composites are
    loaded as themselves.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model)  # nosec: B615
    architectures = config.architectures or []
    if any("ConditionalGeneration" in a for a in architectures):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(base_model)  # nosec: B615
    return load_base_model_for_sft(base_model)


def _adapter_for_model(model: Any, adapter_dir: Path) -> Path:
    """Return an adapter directory whose keys match ``model``'s modules.

    Adapters trained through the text extraction need their keys remapped
    for the composite (see :func:`remap_adapter_keys`); a remapped COPY is
    written next to the merge output, leaving the original untouched.
    """
    import shutil

    from safetensors.torch import load_file, save_file

    tensor_file = adapter_dir / "adapter_model.safetensors"
    if not tensor_file.exists():
        return adapter_dir
    weights = load_file(str(tensor_file))
    param_names = {n for n, _ in model.named_parameters()}
    remapped, changed = remap_adapter_keys(weights, param_names)
    if not changed:
        return adapter_dir
    staging = adapter_dir.parent / f".{adapter_dir.name}-remapped"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(adapter_dir, staging, ignore=shutil.ignore_patterns("checkpoint-*"))
    save_file(remapped, str(staging / "adapter_model.safetensors"))
    logger.info("remapped %d adapter key(s) for the composite architecture", changed)
    return staging


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
