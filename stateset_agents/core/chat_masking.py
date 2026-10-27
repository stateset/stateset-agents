"""Locate assistant turns inside a chat-templated conversation.

Shared by every loss that supervises assistant tokens only (sequence-path
GRPO, local SFT, River/Tinker SFT). Templates differ in ways that break naive
"render a prefix, count its tokens" approaches: Qwen3/Qwen3.5 add an empty
``<think>`` block to the final assistant turn (or to the generation prompt)
that earlier turns lack, so a prefix render is not a prefix of the full
render. Spans are therefore anchored in the *full* render's text and mapped
to tokens through offsets.

A turn is *stable* when the full conversation renders it exactly as the
template renders it when it is the newest message (i.e. as it is generated):
then its span ends exactly where that render ends. Unstable turns (Qwen3
drops the final turn's empty ``<think>`` block once the turn is history, and
strips earlier reasoning) are located by which characters change when only
their content changes, and only if their text survives verbatim.

Assistant messages carrying ``"weight": 0`` (the OpenAI fine-tuning
convention) are context, not targets.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

Render = Callable[[list[dict[str, Any]], bool], "str | None"]

#: Replaces a turn's content to find which characters of the render are the
#: content (private-use code points never occur in real replies).
_SENTINEL = "\ue000\ue001"


#: Roles whose turns the environment (a tool, a code interpreter) wrote.
#: They are context for the policy and never a training target.
OBSERVATION_ROLES = frozenset({"tool", "ipython", "function", "environment"})


def is_supervised(message: dict[str, Any]) -> bool:
    """An assistant message is a target unless it carries ``weight: 0``.

    Only assistant turns are ever targets: system, user and observation
    turns (:data:`OBSERVATION_ROLES`, e.g. tool results) never are.
    """
    return message.get("role") == "assistant" and message.get("weight", 1) != 0


def has_observation_turns(messages: Sequence[Any]) -> bool:
    """Whether ``messages`` (dicts or turns) include an observation turn.

    A loss that cannot locate the assistant spans of such a conversation
    must supervise nothing rather than fall back to the whole token stream,
    which would train the policy to predict tool output.
    """
    for message in messages:
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role in OBSERVATION_ROLES:
            return True
    return False


def assistant_char_spans(
    render: Render,
    messages: Sequence[dict[str, Any]],
    *,
    require_stable: bool = False,
    honor_weights: bool = True,
) -> tuple[str, list[tuple[int, int]]] | None:
    """Character spans of every assistant turn in ``render(messages)``.

    For assistant message ``i`` the span starts where
    ``render(messages[:i], add_generation_prompt=True)`` stops agreeing with
    the full render (which must still cover every earlier message). When the
    full render starts with ``render(messages[:i+1])`` the turn is stable and
    its span ends exactly there — reasoning, reply and end-of-turn suffix,
    however short or repetitive the reply. Otherwise the reply is located by
    a content swap (see :func:`_history_turn_end`); ``require_stable=True``
    rejects such turns instead, because their history render differs from
    what the model generates. With ``honor_weights`` turns with ``weight: 0``
    are context: they yield no span and are not located at all. ``render(msgs, add_generation_prompt)``
    returns text or ``None`` on failure. Returns ``(full_text, spans)``, or
    ``None`` when the template reorders or rewrites history so no faithful
    span exists.
    """
    messages = list(messages)
    full = render(messages, False)
    if not full:
        return None
    turns: list[tuple[int, int, str]] = []  # (message index, start, after)
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        before = render(messages[:i], True)
        after = render(messages[: i + 1], False)
        context = render(messages[:i], False) if i else ""
        if before is None or after is None or context is None:
            return None
        span_start = _turn_start(full, before, context)
        if span_start is None:
            return None
        turns.append((i, span_start, after))
    spans: list[tuple[int, int]] = []
    for position, (i, span_start, after) in enumerate(turns):
        if honor_weights and not is_supervised(messages[i]):
            continue  # context only: its end is never needed
        if full.startswith(after):
            span_end: int | None = len(after)
        elif require_stable:
            return None
        else:
            # A turn ends before the next assistant turn starts.
            limit = turns[position + 1][1] if position + 1 < len(turns) else len(full)
            span_end = _history_turn_end(
                render, messages, i, full, after, span_start, limit
            )
        if span_end is None:
            return None
        if span_end > span_start:
            spans.append((span_start, span_end))
    return full, spans


def _turn_start(full: str, before: str, context: str) -> int | None:
    """Where a turn's target begins in ``full``, or None if history differs.

    ``before`` is the context rendered with the generation prompt; the target
    starts where it stops agreeing with ``full`` (which must still contain
    the whole ``context``). When the turn omits part of the generation
    prompt (a ``<think>`` stub added only to new turns, or an empty one where
    the reply has real reasoning), characters that merely happen to match
    ("<t" of "<think>" vs "<tool_call>") belong to the reply: back off to the
    last line break of the matched prompt part.
    """
    start = len(os.path.commonprefix([before, full]))
    if start < len(context) or not full.startswith(context):
        return None
    if start < len(before):
        line_end = full.rfind("\n", len(context), start)
        if line_end >= 0:
            start = line_end + 1
    return start


def _history_turn_end(
    render: Render,
    messages: list[dict[str, Any]],
    index: int,
    full: str,
    after: str,
    start: int,
    limit: int,
) -> int | None:
    """End of an unstable turn, or None when its text was rewritten.

    The turn's content is swapped for a sentinel: the characters of ``full``
    that change are exactly where the content is rendered, wherever else the
    same words occur (inside reasoning, the template, other turns). That
    region must be the content verbatim (or stripped) — a rewritten reply,
    such as one whose ``<think>`` block the template drops from history, has
    no faithful span. The turn then extends over its end-of-turn suffix, the
    part of ``after`` that follows the content. A template that does not
    render the content at all (some render only the tool call) falls back to
    :func:`_tail_end` within ``[start, limit]``.
    """
    content = messages[index].get("content")
    text = content if isinstance(content, str) else ""
    swapped = list(messages)
    swapped[index] = {**messages[index], "content": _SENTINEL}
    full_swapped = render(swapped, False)
    after_swapped = render(swapped[: index + 1], False)
    if full_swapped is None or after_swapped is None:
        return None
    if full_swapped == full:
        return None if text.strip() else _tail_end(full, after, start, limit)
    head = _common_prefix(full, full_swapped)
    content_end = max(len(full) - _common_suffix(full, full_swapped), head)
    if full[head:content_end] not in {text, text.strip()}:
        return None
    after_head = _common_prefix(after, after_swapped)
    tail_start = max(len(after) - _common_suffix(after, after_swapped), after_head)
    tail = after[tail_start:]
    return content_end + _common_prefix(tail, full[content_end:])


def _tail_end(full: str, after: str, start: int, limit: int) -> int | None:
    """End of a turn whose render shares only its tail with ``after``.

    ``after`` renders the turn as the newest message, which templates may
    decorate at its start (Qwen3 adds an empty ``<think>`` block); the call
    and end-of-turn suffix render identically. The turn ends where the
    longest suffix of ``after`` occurring inside ``full[start:limit]`` ends
    (every shorter suffix of an occurring one also occurs, so the boundary is
    binary-searched); bounding by the next turn keeps an identical later
    turn from being matched.
    """
    low, high = 0, len(after)
    while low < high:
        middle = (low + high) // 2
        if full.find(after[middle:], start, limit) >= 0:
            high = middle
        else:
            low = middle + 1
    tail = after[low:]
    found = full.find(tail, start, limit) if tail else -1
    if found < 0 or found + len(tail) <= start:
        return None
    return found + len(tail)


def _common_prefix(a: str, b: str) -> int:
    return len(os.path.commonprefix([a, b]))


def _common_suffix(a: str, b: str) -> int:
    return len(os.path.commonprefix([a[::-1], b[::-1]]))


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
    require_stable: bool = False,
) -> tuple[list[int], list[bool]] | None:
    """Token ids of the full conversation and which of them are assistant output.

    Uses fast-tokenizer offsets when available; otherwise maps span edges by
    encoding text prefixes and verifying they are token prefixes of the whole.
    Turns with ``weight: 0`` are not marked. Returns ``None`` when no faithful
    mapping exists (with ``require_stable``, also when a target turn renders
    differently in history than when generated).
    """
    located = assistant_char_spans(
        chat_template_render(tokenizer, tools),
        messages,
        require_stable=require_stable,
    )
    if located is None:
        return None
    full, spans = located
    return _token_mask_for_spans(tokenizer, full, spans)


def final_turn_token_mask(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Any | None = None,
) -> tuple[list[int], list[bool]] | None:
    """Mask only the final assistant turn, tolerating rewritten history.

    Unlike :func:`last_assistant_token_mask` this does not need every
    earlier turn to have a faithful span: templates such as Qwen3's strip
    ``<think>`` blocks from earlier replies, yet the final turn's context is
    still exactly the inference-time prompt (the preceding messages rendered
    with the generation prompt, history rewritten the same way). Everything
    the template renders for the final turn after that prompt is the target.
    Messages after the final assistant turn are dropped. Returns ``None``
    when there is no assistant turn or the history does not agree.
    """
    last = max(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        default=None,
    )
    if last is None:
        return None
    render = chat_template_render(tokenizer, tools)
    conversation = list(messages[: last + 1])
    full = render(conversation, False)
    before = render(conversation[:last], True)
    context = render(conversation[:last], False) if last else ""
    if not full or before is None or context is None:
        return None
    start = _turn_start(full, before, context)
    if start is None or start >= len(full):
        return None
    return _token_mask_for_spans(tokenizer, full, [(start, len(full))])


def last_assistant_token_mask(
    tokenizer: Any, messages: Sequence[dict[str, Any]]
) -> tuple[list[int], list[bool]] | None:
    """Locate only the last assistant action for transition-weighted offline RL.

    When the template rewrites earlier turns (Qwen3/Qwen3.5 strip earlier
    ``<think>`` blocks) the whole-conversation spans do not exist, but the
    action does: it is the final turn, generated after the history exactly
    as the template renders it, so :func:`final_turn_token_mask` locates it.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    located = assistant_char_spans(
        chat_template_render(tokenizer), messages, honor_weights=False
    )
    if located is None or not located[1]:
        return final_turn_token_mask(tokenizer, messages)
    full, spans = located
    return _token_mask_for_spans(tokenizer, full, spans[-1:])


def supervised_turn_sequences(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Any | None = None,
) -> tuple[str, list[tuple[list[int], list[bool]]]] | None:
    """Token sequences that train every target turn in its generation context.

    Returns ``("all_assistant_turns", [sequence])`` when every target turn
    (assistant turns without ``weight: 0``) is stable, so one render of the
    conversation supervises them all. When the template renders a target
    turn differently once it is history — Qwen3 drops the empty ``<think>``
    block it adds to the newest turn, Qwen3.5 omits the stub its generation
    prompt carries, and both strip earlier reasoning — a single sequence
    would teach one target in the generated form and another in the history
    form, so the conversation is split: ``("split_per_turn", sequences)``,
    one per target turn, each rendered as the final turn of its prefix
    (:func:`final_turn_token_mask`). ``("no_target", [])`` when no assistant
    turn is a target; ``None`` when a target turn cannot be located.
    """
    messages = list(messages)
    targets = [i for i, m in enumerate(messages) if is_supervised(m)]
    if not targets:
        return "no_target", []
    conversation = messages[: targets[-1] + 1]
    stable = assistant_token_mask(
        tokenizer, conversation, tools=tools, require_stable=True
    )
    if stable is not None:
        return "all_assistant_turns", [stable]
    sequences = []
    for index in targets:
        sequence = final_turn_token_mask(tokenizer, messages[: index + 1], tools=tools)
        if sequence is None:
            return None
        sequences.append(sequence)
    return "split_per_turn", sequences


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
    "OBSERVATION_ROLES",
    "assistant_char_spans",
    "assistant_token_mask",
    "chat_template_render",
    "final_turn_token_mask",
    "has_observation_turns",
    "is_supervised",
    "last_assistant_token_mask",
    "supervised_turn_sequences",
]
