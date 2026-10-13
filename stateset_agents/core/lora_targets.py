"""Keep text-training adapters out of multimodal encoders."""

from __future__ import annotations

from typing import Any

NON_TEXT_STACK_MARKERS = frozenset(
    {
        "vision_tower",
        "vision_model",
        "visual",
        "vision_encoder",
        "image_processor",
        "vision_adapter",
        "vision_projection",
        "vision_projector",
        "perception_encoder",
        "multi_modal_projector",
        "mm_projector",
        "audio_tower",
        "audio_model",
        "audio_encoder",
    }
)


def text_lora_targets(model: Any, targets: list[str]) -> list[str]:
    """Resolve full text paths when a requested suffix also matches an encoder.

    Otherwise preserve suffixes, including unknown names, so PEFT retains its
    normal missing-target validation and adapter checkpoint conventions.
    """
    text_paths: list[str] = []
    encoder_match = False
    for name, _ in model.named_modules():
        if not any(name == target or name.endswith("." + target) for target in targets):
            continue
        if NON_TEXT_STACK_MARKERS.intersection(name.split(".")):
            encoder_match = True
        else:
            text_paths.append(name)
    if not encoder_match:
        return list(targets)
    if not text_paths:
        raise ValueError("LoRA targets match only non-text modules.")
    return sorted(text_paths)
