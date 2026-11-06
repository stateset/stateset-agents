"""Inkling 2 text SFT batches using Thinking Machines' native renderer."""

from __future__ import annotations

import sys
from typing import Any


def build_inkling2_sft_batch(
    rows: list[dict[str, Any]], max_length: int = 2048
) -> list[dict[str, Any]]:
    """Render assistant loss weights and shift targets once for Tinker.

    Use the model's native renderer rather than a Hugging Face chat template.
    This path accepts text conversations and tool calls. Media inputs fail
    explicitly instead of being silently dropped.
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    python_version = tuple(sys.version_info[:2])
    if python_version < (3, 11):
        raise RuntimeError("Inkling 2 rendering requires Python 3.11 or newer")
    try:
        # tml-renderers dynamically links libtorch, so import torch first.
        import torch  # noqa: F401
        from tml_renderers import chat, renderers
        from tml_renderers.tinker import (
            training_example_to_tinker_model_input_and_weights,
        )
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Inkling 2 SFT requires tml-renderers with Inkling2 support and "
            "PyTorch >=2.10; install 'stateset-agents[tinker]' on Python >=3.11"
        ) from exc

    renderer_type = getattr(renderers, "Inkling2", None)
    if renderer_type is None:
        raise RuntimeError("Upgrade tml-renderers for Inkling2 support")
    renderer = renderer_type()
    batch: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"SFT row {index} must contain non-empty messages")
        if any(
            not isinstance(message, dict)
            or not isinstance(message.get("content"), (str, type(None)))
            for message in messages
        ):
            raise ValueError(
                f"SFT row {index}: Inkling 2 currently accepts text messages only; "
                "media inputs are unsupported"
            )
        converted = chat.OpenAIMessage.from_openai_chat_dicts(
            messages, tools=row.get("tools")
        )
        for example in renderer.render_for_sft(converted):
            model_input, weights = training_example_to_tinker_model_input_and_weights(
                example
            )
            ids = model_input.to_ints()
            if len(ids) != len(weights):
                raise ValueError("Inkling 2 renderer returned misaligned loss weights")
            ids = ids[:max_length]
            shifted_weights = list(weights[1 : len(ids)])
            if len(ids) < 2 or not any(weight > 0 for weight in shifted_weights):
                continue
            batch.append(
                {
                    "input_ids": ids[:-1],
                    "target_tokens": ids[1:],
                    "weights": shifted_weights,
                }
            )
    return batch
