"""Checkpoint-pinned River text rendering for newly released Nemotron VLMs."""

from __future__ import annotations

from typing import Any


class _NemotronRiverTokenizer:
    """Expose Nemotron's real image ID through River 0.11's metadata lookup.

    The SDK unconditionally resolves an image placeholder when lowering even
    text-only ``model_input`` chunks, but its candidate list omits ``<image>``.
    This lookup alias does not add vocabulary entries or change encoding,
    decoding, prompt tokens, or sampled tokens.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        image_id = tokenizer.convert_tokens_to_ids("<image>")
        if (
            not isinstance(image_id, int)
            or image_id < 0
            or image_id == tokenizer.unk_token_id
        ):
            raise ValueError("Nemotron tokenizer does not expose a valid <image> token")
        self._image_id = image_id

    def convert_tokens_to_ids(self, tokens: Any) -> Any:
        if isinstance(tokens, str) and tokens == "<|image_pad|>":
            return self._image_id
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._tokenizer)


def get_river_renderer(base_model: str, *, thinking: bool = False) -> Any:
    """Build a text/tool renderer without pretending River grants model access.

    River SDK 0.11 recognizes Lightning only. Super VL and Bolt share NVIDIA's
    chat/tool dialect, but need their own checkpoint tokenizer and template.
    Account capabilities must still be checked before calling this factory.
    Image/video RL is intentionally outside this text renderer's contract.
    """
    from stateset_agents.training.nemotron_vl import (
        NEMOTRON_VL_MODELS,
        NEMOTRON_VL_REVISIONS,
    )

    names = {
        NEMOTRON_VL_MODELS["bolt"]: "bolt",
        "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB": "bolt",
        NEMOTRON_VL_MODELS["super-vl"]: "super-vl",
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4": "super-vl",
    }
    family = names.get(base_model)
    if family is None:
        from river_client.renderers import get_renderer

        return get_renderer(base_model, thinking=thinking)

    from river_client.renderers.nemotron import Nemotron35Renderer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        NEMOTRON_VL_MODELS[family],
        revision=NEMOTRON_VL_REVISIONS[family],
        trust_remote_code=True,
    )
    return Nemotron35Renderer(
        _NemotronRiverTokenizer(tokenizer),
        thinking=thinking,
        strip_thinking_from_history=True,
    )


__all__ = ["get_river_renderer"]
