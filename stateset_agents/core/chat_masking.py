"""Locate assistant turns inside a chat-templated conversation.

Shared by every loss that supervises assistant tokens only (sequence-path
GRPO, local SFT, River/Tinker SFT). Templates differ in ways that break naive
"render a prefix, count its tokens" approaches: Qwen3/Qwen3.5 add an empty
``<think>`` block to the final assistant turn (or to the generation prompt)
that earlier turns lack, so a prefix render is not a prefix of the full
render. Spans are therefore anchored in the *full* render's text and mapped
to tokens through offsets.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

Render = Callable[[list[dict[str, Any]], bool], "str | None"]


def assistant_char_spans(
    render: Render, messages: Sequence[dict[str, Any]]
) -> tuple[str, list[tuple[int, int]]] | None:
    """Character spans of every assistant turn in ``render(messages)``.

    For assistant message ``i`` the span starts where
    ``render(messages[:i], add_generation_prompt=True)`` stops agreeing with
    the full render (which must still cover every earlier message), covers
    the reply plus anything the template inserts before it, and extends over
    the template's end-of-turn suffix taken from ``render(messages[:i+1])``.
    ``render(msgs, add_generation_prompt)`` returns text or ``None`` on
    failure. Returns ``(full_text, spans)``, or ``None`` when the template
    reorders or rewrites history so no faithful span exists.
    """
    messages = list(messages)
    full = render(messages, False)
    if not full:
        return None
    spans: list[tuple[int, int]] = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        before = render(messages[:i], True)
        after = render(messages[: i + 1], False)
        context = render(messages[:i], False) if i else ""
        if before is None or after is None or context is None:
            return None
        span_start = len(os.path.commonprefix([before, full]))
        if span_start < len(context) or not full.startswith(context):
            return None
        span_end = _reply_end(full, after, span_start, message.get("content"))
        if span_end is None:
            return None
        if span_end > span_start:
            spans.append((span_start, span_end))
    return full, spans


def _reply_end(full: str, after: str, start: int, content: Any) -> int | None:
    text = content if isinstance(content, str) else ""
    for candidate in (text, text.strip()):
        if not candidate:
            continue
        found = full.find(candidate, start)
        if found < 0:
            continue
        end = found + len(candidate)
        suffix_at = after.rfind(candidate)
        suffix = after[suffix_at + len(candidate) :] if suffix_at >= 0 else ""
        return end + len(os.path.commonprefix([suffix, full[end:]]))
    # No reply text to anchor on (e.g. a pure tool call): the turn ends where
    # rendering it stops agreeing with the full conversation.
    shared = len(os.path.commonprefix([after, full]))
    return shared if shared > start else None


def chat_template_render(tokenizer: Any, tools: Any | None = None) -> Render:
    """A :data:`Render` over ``tokenizer.apply_chat_template``."""
    extra = {"tools": tools} if tools else {}

    def render(messages: list[dict[str, Any]], add_generation_prompt: bool) -> Any:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **extra,
            )
        except (TypeError, ValueError, RuntimeError, KeyError, IndexError):
            return None
        return text if isinstance(text, str) else None

    return render


def assistant_token_mask(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Any | None = None,
) -> tuple[list[int], list[bool]] | None:
    """Token ids of the full conversation and which of them are assistant output.

    Uses fast-tokenizer offsets when available; otherwise maps span edges by
    encoding text prefixes and verifying they are token prefixes of the whole.
    Returns ``None`` when no faithful mapping exists.
    """
    located = assistant_char_spans(chat_template_render(tokenizer, tools), messages)
    if located is None:
        return None
    full, spans = located
    return _token_mask_for_spans(tokenizer, full, spans)


def last_assistant_token_mask(
    tokenizer: Any, messages: Sequence[dict[str, Any]]
) -> tuple[list[int], list[bool]] | None:
    """Locate only the last assistant action for transition-weighted offline RL."""
    located = assistant_char_spans(chat_template_render(tokenizer), messages)
    if located is None or not located[1] or messages[-1].get("role") != "assistant":
        return None
    full, spans = located
    return _token_mask_for_spans(tokenizer, full, spans[-1:])


def _token_mask_for_spans(
    tokenizer: Any, full: str, spans: list[tuple[int, int]]
) -> tuple[list[int], list[bool]] | None:
    try:
        encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = [int(t) for t in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    except (TypeError, ValueError, KeyError, NotImplementedError):
        return _mask_by_prefix_encoding(tokenizer, full, spans)
    if len(offsets) != len(ids):
        return None
    mask = [
        end > start and any(start < hi and end > lo for lo, hi in spans)
        for start, end in offsets
    ]
    return ids, mask


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return [int(t) for t in ids]


def _mask_by_prefix_encoding(
    tokenizer: Any, full: str, spans: list[tuple[int, int]]
) -> tuple[list[int], list[bool]] | None:
    ids = _encode(tokenizer, full)
    mask = [False] * len(ids)
    for lo, hi in spans:
        head, through = _encode(tokenizer, full[:lo]), _encode(tokenizer, full[:hi])
        if ids[: len(head)] != head or ids[: len(through)] != through:
            return None
        for k in range(len(head), len(through)):
            mask[k] = True
    return ids, mask


__all__ = [
    "assistant_char_spans",
    "assistant_token_mask",
    "chat_template_render",
    "last_assistant_token_mask",
]
