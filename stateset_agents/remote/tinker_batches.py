"""Inkling 2 text SFT batches using Thinking Machines' native renderer."""

from __future__ import annotations

import logging
import sys
from typing import Any

from stateset_agents.core.chat_masking import truncates_weighted_target

logger = logging.getLogger(__name__)


def _native_messages(chat: Any, row: dict[str, Any]) -> list[Any]:
    """Carry assistant context weights across the OpenAI-to-Inkling conversion."""
    messages = row["messages"]
    converted = chat.OpenAIMessage.from_openai_chat_dicts(
        messages, tools=row.get("tools")
    )
    assistants = [msg for msg in messages if msg.get("role") == "assistant"]
    if all(msg.get("weight", 1) != 0 for msg in assistants):
        return list(converted)

    # Conversion can insert a tool declaration and split one assistant action
    # into text, individual calls and an end marker. Mask all parts of that action.
    parts = [msg.to_messages() for msg in converted]
    model_parts = [
        group
        for msg, group in zip(converted, parts, strict=True)
        if msg.author.kind == chat.AuthorKind.Model
    ]
    if len(model_parts) != len(assistants):
        raise ValueError(
            "Inkling 2 could not align assistant weights with rendered messages"
        )
    for source, group in zip(assistants, model_parts, strict=True):
        if source.get("weight", 1) == 0:
            for part in group:
                metadata = part.message_metadata or chat.MessageMetadata()
                previous = metadata.training_metadata
                metadata.training_metadata = (
                    previous.copy(weight=0.0)
                    if previous is not None
                    else chat.TrainingMetadata(weight=0.0, is_synthetic=False)
                )
                part.message_metadata = metadata
    return [part for group in parts for part in group]


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
    truncated_rows = 0
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
        row_datums: list[dict[str, Any]] = []
        for example in renderer.render_for_sft(_native_messages(chat, row)):
            model_input, weights = training_example_to_tinker_model_input_and_weights(
                example
            )
            ids = model_input.to_ints()
            if len(ids) != len(weights):
                raise ValueError("Inkling 2 renderer returned misaligned loss weights")
            if truncates_weighted_target(messages, weights, max_length):
                # All renderer examples from this graded row belong together.
                row_datums.clear()
                truncated_rows += 1
                break
            ids = ids[:max_length]
            shifted_weights = list(weights[1 : len(ids)])
            if len(ids) < 2 or not any(weight > 0 for weight in shifted_weights):
                continue
            row_datums.append(
                {
                    "input_ids": ids[:-1],
                    "target_tokens": ids[1:],
                    "weights": shifted_weights,
                }
            )
        batch.extend(row_datums)
    if truncated_rows:
        logger.warning(
            "Skipped %d weighted Inkling 2 SFT rows with incomplete targets at max_length=%d",
            truncated_rows,
            max_length,
        )
    return batch
