"""
Utilities for exporting fine-tuned checkpoints into serving-ready artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_serving_manifest(
    output_dir: str,
    model_name: str,
    *,
    use_lora: bool,
    use_vllm: bool,
    merged_model_dir: str | None = None,
    recommended: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a lightweight manifest that downstream deployment tools can consume."""
    return {
        "model": model_name,
        "output_dir": output_dir,
        "use_lora": use_lora,
        "use_vllm": use_vllm,
        "merged_model_dir": merged_model_dir,
        "recommended": recommended or {},
    }


def write_serving_manifest(
    output_dir: str,
    model_name: str,
    *,
    use_lora: bool,
    use_vllm: bool,
    merged_model_dir: str | None = None,
    recommended: dict[str, Any] | None = None,
) -> Path:
    """Write the serving manifest beside the training output."""
    path = Path(output_dir) / "serving_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_serving_manifest(
        output_dir,
        model_name,
        use_lora=use_lora,
        use_vllm=use_vllm,
        merged_model_dir=merged_model_dir,
        recommended=recommended,
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Serving manifest written to %s", path)
    return path


def export_merged_model_for_serving(
    *,
    base_model_name: str,
    adapter_dir: str,
    output_dir: str,
    trust_remote_code: bool = True,
) -> str:
    """
    Merge a PEFT adapter into its base model and persist tokenizer/processor assets.

    This supports both text-only CausalLM models and newer multimodal models whose
    config advertises a `*ForConditionalGeneration` architecture.
    """
    try:
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        logger.warning("Missing dependencies for merge export: %s", exc)
        return ""

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # pragma: no cover - older transformers builds
        AutoModelForImageTextToText = None

    try:
        from transformers import AutoProcessor
    except ImportError:  # pragma: no cover - older transformers builds
        AutoProcessor = None

    logger.info("Exporting merged model for serving...")

    config = AutoConfig.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
    )
    architectures = [str(name) for name in getattr(config, "architectures", []) or []]
    uses_conditional_generation = any(
        "ConditionalGeneration" in architecture for architecture in architectures
    )

    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }

    if uses_conditional_generation:
        if AutoModelForImageTextToText is None:
            raise ImportError(
                "This model requires `AutoModelForImageTextToText` for merged export. "
                "Install a recent transformers build with Qwen3.5 support."
            )
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_path)

    processor_saved = False
    if AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(
                base_model_name,
                trust_remote_code=trust_remote_code,
            )
            processor.save_pretrained(output_path)
            processor_saved = True
        except Exception as exc:  # pragma: no cover
            logger.info("Processor export skipped for %s: %s", base_model_name, exc)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.save_pretrained(output_path)
    except Exception as exc:  # pragma: no cover
        if not processor_saved:
            logger.warning(
                "Tokenizer export skipped for %s and no processor was saved: %s",
                base_model_name,
                exc,
            )

    logger.info("Merged model saved to %s", output_path)
    return str(output_path)


__all__ = [
    "build_serving_manifest",
    "export_merged_model_for_serving",
    "write_serving_manifest",
]
