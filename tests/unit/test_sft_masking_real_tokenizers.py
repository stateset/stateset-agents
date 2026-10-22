"""SFT label-masking invariants against the real Qwen3 / Qwen3.5 tokenizers.

Generated multi-turn rows — system prompts, ``<think>`` blocks and
``reasoning_content``, tool calls and results, ``weight: 0`` context replies,
and short replies that collide with template text ("think", "assistant",
"18" inside the reasoning) — are tokenized with ``tokenize_sft_rows`` and
checked against an oracle that only reads the rendered text:

* a row is one sequence exactly when every target turn renders the same in
  history as when it is generated; otherwise it is one sequence per target
  turn, each rendered as the newest turn;
* labels form the target turn(s) only: never system, user or tool text, never
  a ``weight: 0`` reply, and every target reply — reasoning, answer and
  ``<|im_end|>`` — is labeled in full;
* each split sequence is trained after its own generation prompt;
* truncation only cuts, and the counts that feed the run manifest match.

Skips cleanly when transformers or the tokenizers are not in the local cache.
Assertion messages carry the library versions and the failing stage, so a
hosted-CI failure says what differed.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

import pytest

from stateset_agents.core import chat_masking
from stateset_agents.training.sft import tokenize_sft_rows

TOKENIZERS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Refund an order",
            "parameters": {
                "type": "object",
                "properties": {"order": {"type": "string"}},
                "required": ["order"],
            },
        },
    }
]
#: Replies that also occur in the template or the reasoning around them.
COLLIDING = ("think", "assistant", "user", "hi", "in", "18", "ok")
_BLOCK = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>\n?)", re.DOTALL)


@pytest.fixture(scope="module", params=TOKENIZERS)
def tokenizer(request: pytest.FixtureRequest) -> Any:
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            request.param, local_files_only=True
        )
    except (OSError, ValueError) as exc:  # not cached locally
        pytest.skip(f"{request.param} is not in the local Hugging Face cache: {exc}")
    if not getattr(tok, "chat_template", None):
        pytest.skip(f"the cached {request.param} tokenizer has no chat template")
    return tok


def _diagnose(tokenizer: Any, row: dict[str, Any]) -> str:
    """Versions, template identity and the first masking stage that fails."""
    import tokenizers
    import transformers

    template = str(getattr(tokenizer, "chat_template", ""))
    lines = [
        f"transformers={transformers.__version__} tokenizers={tokenizers.__version__}"
        f" class={type(tokenizer).__name__}"
        f" template_sha256={hashlib.sha256(template.encode()).hexdigest()[:12]}",
    ]
    tools = row.get("tools")
    try:
        tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
            **({"tools": tools} if tools else {}),
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        lines.append(f"apply_chat_template raised {exc!r}")
    render = chat_masking.chat_template_render(tokenizer, tools)
    located = chat_masking.assistant_char_spans(render, row["messages"])
    lines.append(f"assistant_char_spans located={located is not None}")
    try:
        tokenizer("x", add_special_tokens=False, return_offsets_mapping=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        lines.append(f"offsets unavailable: {exc!r}")
    lines.append(f"row={row!r}")
    return "\n".join(lines)


def _rows(count: int, seed: int) -> list[dict[str, Any]]:
    """Rows mixing reasoning, tools, weights, colliding replies and lengths."""
    rng = random.Random(seed)
    rows = []
    for r in range(count):
        messages: list[dict[str, Any]] = []
        tooled = False
        if rng.random() < 0.6:
            messages.append({"role": "system", "content": f"System rule s{r} here."})
        exchanges = rng.randint(1, 3)
        for e in range(exchanges):
            messages.append(
                {"role": "user", "content": f"Customer u{r}x{e} asks" + " why" * e}
            )
            if rng.random() < 0.3:
                tooled = True
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "issue_refund",
                                    "arguments": {"order": f"o{r}x{e}"},
                                },
                            }
                        ],
                    }
                )
                messages.append({"role": "tool", "content": f"Tool result t{r}x{e}."})
            kind = rng.random()
            if kind < 0.25:
                reply: dict[str, Any] = {
                    "role": "assistant",
                    "content": rng.choice(COLLIDING),
                }
            else:
                reply = {
                    "role": "assistant",
                    "content": f"Agent reply a{r}x{e}" + " ok" * rng.randint(0, 40),
                }
            style = rng.random()
            if style < 0.25:
                reply["content"] = (
                    f"<think>Reasoning k{r}x{e}.</think>{reply['content']}"
                )
            elif style < 0.5:
                # The answer also occurs inside the reasoning that precedes it.
                reply["reasoning_content"] = (
                    f"16-3-4=9 eggs; 9*2=18 k{r}x{e} {reply['content']}"
                )
            if e < exchanges - 1 and rng.random() < 0.3:
                reply["weight"] = 0
            messages.append(reply)
        row: dict[str, Any] = {"messages": messages}
        if tooled:
            row["tools"] = TOOLS
        rows.append(row)
    return rows


def _render(
    tokenizer: Any,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    generation: bool = False,
) -> str:
    kwargs = {"tools": row["tools"]} if "tools" in row else {}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation, **kwargs
    )


def _targets(row: dict[str, Any]) -> list[int]:
    return [
        i
        for i, m in enumerate(row["messages"])
        if m["role"] == "assistant" and m.get("weight", 1) != 0
    ]


def _stable(tokenizer: Any, row: dict[str, Any]) -> bool:
    """Every target turn renders the same in history as when generated."""
    messages = row["messages"][: _targets(row)[-1] + 1]
    full = _render(tokenizer, row, messages)
    return all(
        full.startswith(_render(tokenizer, row, messages[: i + 1]))
        for i in _targets(row)
    )


def _overlaps(a: int, b: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a < hi and b > lo for lo, hi in ranges)


def _labeled(out: dict[str, list[int]]) -> list[int]:
    return [i for i, label in enumerate(out["labels"]) if label != -100]


def _locate(full: str, messages: list[dict[str, Any]]) -> tuple[list, list]:
    """(non-assistant content ranges, assistant ChatML blocks) of ``full``."""
    others, cursor = [], 0
    for message in messages:
        if message["role"] == "assistant":
            continue
        start = full.find(message["content"], cursor)
        assert start >= 0, message["content"]
        others.append((start, start + len(message["content"])))
        cursor = start + len(message["content"])
    blocks = [(m.start(1), m.end(1)) for m in _BLOCK.finditer(full)]
    return others, blocks


def _reply_payload(full: str, block: tuple[int, int], message: dict[str, Any]):
    """The reply's visible text through ``<|im_end|>`` inside its block."""
    start, end = block
    content = message.get("content") or ""
    anchor = content.split("</think>")[-1] if content else "<tool_call>"
    im_end = full.rfind("<|im_end|>", start, end)
    at = full.rfind(anchor, start, im_end) if anchor else im_end
    assert at >= 0, (anchor, full[start:end])
    return at, im_end + len("<|im_end|>")


def _check_sequence(tokenizer, row, messages, out, supervised: list[int]) -> None:
    """Labels of one sequence are exactly the ``supervised`` turns' targets."""
    full = _render(tokenizer, row, messages)
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    assert out["input_ids"] == list(encoded["input_ids"]), _diagnose(tokenizer, row)
    offsets = [tuple(span) for span in encoded["offset_mapping"]]
    others, blocks = _locate(full, messages)
    replies = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    assert len(blocks) == len(replies)
    target_blocks = [b for i, b in zip(replies, blocks, strict=True) if i in supervised]
    labeled = set(_labeled(out))
    for i in labeled:
        a, b = offsets[i]
        token = tokenizer.decode([out["input_ids"][i]])
        assert out["labels"][i] == out["input_ids"][i]
        assert _overlaps(a, b, target_blocks), (token, _diagnose(tokenizer, row))
        assert not _overlaps(a, b, others), (token, _diagnose(tokenizer, row))
    for index, block in zip(replies, blocks, strict=True):
        if index not in supervised:
            continue
        lo, hi = _reply_payload(full, block, messages[index])
        for i, (a, b) in enumerate(offsets):
            if b > a and lo <= a and b <= hi:
                assert i in labeled, (full[a:b], _diagnose(tokenizer, row))
        reasoning = messages[index].get("reasoning_content") or ""
        if "<think>" in (messages[index].get("content") or ""):
            reasoning = (
                messages[index]["content"].split("<think>")[1].split("</think>")[0]
            )
        if reasoning:
            text = tokenizer.decode([out["input_ids"][i] for i in sorted(labeled)])
            assert reasoning.strip() in text, _diagnose(tokenizer, row)


def test_labels_are_exactly_the_target_turns(tokenizer: Any) -> None:
    modes = set()
    for row in _rows(32, seed=7):
        records, counts = tokenize_sft_rows(tokenizer, [row], 100_000)
        assert records, _diagnose(tokenizer, row)
        targets = _targets(row)
        if _stable(tokenizer, row):
            modes.add("all")
            assert counts["all_assistant_turns"] == 1, _diagnose(tokenizer, row)
            messages = row["messages"][: targets[-1] + 1]
            _check_sequence(tokenizer, row, messages, records[0], targets)
            continue
        modes.add("split")
        assert counts["split_per_turn"] == 1, _diagnose(tokenizer, row)
        assert len(records) == len(targets)
        for index, out in zip(targets, records, strict=True):
            messages = row["messages"][: index + 1]
            _check_sequence(tokenizer, row, messages, out, [index])
            # Trained after its own generation prompt: the unlabeled prefix
            # is the whole history plus a line-aligned part of the prompt.
            first = _labeled(out)[0]
            prompt = tokenizer.decode(out["input_ids"][:first])
            generation = _render(tokenizer, row, messages[:index], generation=True)
            assert generation.startswith(prompt), _diagnose(tokenizer, row)
            assert prompt.startswith(_render(tokenizer, row, messages[:index]))
            assert prompt.endswith("\n")
            assert _labeled(out)[-1] == len(out["labels"]) - 1  # through the end
    assert modes == {"all", "split"}  # both paths were exercised


@pytest.mark.parametrize("reply", COLLIDING)
def test_short_and_colliding_replies_are_fully_labeled(tokenizer, reply) -> None:
    """A reply that also occurs in the template or reasoning used to anchor on
    the first occurrence: "think" labeled only ``<think>``."""
    for extra in ({}, {"reasoning_content": f"16-3-4=9 eggs; 9*2=18 so {reply}"}):
        row = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": reply, **extra},
            ]
        }
        records, counts = tokenize_sft_rows(tokenizer, [row], 100_000)
        assert counts["all_assistant_turns"] == 1, _diagnose(tokenizer, row)
        text = tokenizer.decode(
            [records[0]["input_ids"][i] for i in _labeled(records[0])]
        )
        assert text.endswith(f"{reply}<|im_end|>\n"), (text, _diagnose(tokenizer, row))
        if extra:
            assert extra["reasoning_content"] in text, _diagnose(tokenizer, row)


@pytest.mark.parametrize("max_length", [24, 80, 160, 320])
def test_truncation_only_cuts_and_counts_feed_the_manifest(
    tokenizer: Any, max_length: int
) -> None:
    rows = _rows(16, seed=11)
    expected = dict.fromkeys(
        (
            "all_assistant_turns",
            "split_per_turn",
            "dropped_no_target",
            "dropped_unmaskable",
            "dropped_truncated",
            "partially_truncated",
        ),
        0,
    )
    kept = []
    for row in rows:
        full, _ = tokenize_sft_rows(tokenizer, [row], 100_000)
        assert full, _diagnose(tokenizer, row)
        cut = [
            {key: value[:max_length] for key, value in record.items()}
            for record in full
            if any(i < max_length for i in _labeled(record))
        ]
        lost = any(i >= max_length for record in full for i in _labeled(record))
        mine, _ = tokenize_sft_rows(tokenizer, [row], max_length)
        assert mine == cut
        if not cut:
            expected["dropped_truncated"] += 1
            continue
        expected[
            "all_assistant_turns" if _stable(tokenizer, row) else "split_per_turn"
        ] += 1
        expected["partially_truncated"] += lost
        kept.extend(cut)
    records, counts = tokenize_sft_rows(tokenizer, rows, max_length)
    assert counts == expected
    assert records == kept
