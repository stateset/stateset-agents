"""SFT label-masking invariants against the real Qwen3 / Qwen3.5 tokenizers.

Generated multi-turn rows (system prompts, ``<think>`` blocks, tool calls,
tool results, truncation) are tokenized with ``tokenize_sft_row(s)`` and
checked against an oracle that knows nothing about how spans are located: it
reads the ChatML blocks of the rendered conversation directly.

* every labeled token lies in an assistant block and none touches system,
  user or tool text;
* every supervised assistant reply (and its ``<|im_end|>``) is fully labeled;
  when the template rewrites history only the final block is;
* truncation only cuts: labels at ``max_length`` are a prefix of the
  untruncated labels, and the counts that feed the run manifest
  (``tokenize_sft_rows``) match the oracle.

Skips cleanly when transformers or the tokenizers are not in the local cache.
"""

from __future__ import annotations

import random
import re
from typing import Any

import pytest

from stateset_agents.training.sft import tokenize_sft_row, tokenize_sft_rows

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
_BLOCK = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>\n?)", re.DOTALL)


@pytest.fixture(scope="module", params=TOKENIZERS)
def tokenizer(request: pytest.FixtureRequest) -> Any:
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            request.param, local_files_only=True
        )
    except (OSError, ValueError) as exc:  # not cached locally
        pytest.skip(f"{request.param} is not in the local Hugging Face cache: {exc}")


def _rows(count: int, seed: int) -> list[dict[str, Any]]:
    """Rows mixing system prompts, think blocks, tool calls and lengths."""
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
            if rng.random() < 0.35:
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
            # No edge whitespace: the Qwen templates strip it on render.
            reply = f"Agent reply a{r}x{e}" + " ok" * rng.randint(0, 40)
            if rng.random() < 0.4:
                reply = f"<think>Reasoning k{r}x{e}.</think>{reply}"
            messages.append({"role": "assistant", "content": reply})
        row: dict[str, Any] = {"messages": messages}
        if tooled:
            row["tools"] = TOOLS
        rows.append(row)
    return rows


def _render(tokenizer: Any, row: dict[str, Any]) -> str:
    kwargs = {"tools": row["tools"]} if "tools" in row else {}
    return tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False, **kwargs
    )


def _history_rewritten(tokenizer: Any, row: dict[str, Any]) -> bool:
    """Whether the full render disagrees with how a turn's history renders.

    True when some assistant turn's context, rendered on its own, is not a
    prefix of the full conversation (Qwen3.5 drops the empty ``<think>``
    block of tool-call turns once a later user turn exists), or when an
    earlier reply's text is not in the full render (its ``<think>`` block was
    stripped).
    """
    full = _render(tokenizer, row)
    messages = row["messages"]
    for i, message in enumerate(messages[:-1]):
        if message["role"] != "assistant":
            continue
        context = _render(tokenizer, {**row, "messages": messages[:i]})
        if not full.startswith(context):
            return True
        content = message.get("content") or ""
        if content and content not in full and content.strip() not in full:
            return True
    # The final turn is rendered exactly as it is generated; only its
    # context can disagree with the full render.
    context = _render(tokenizer, {**row, "messages": messages[:-1]})
    return not full.startswith(context)


def _non_assistant_ranges(full: str, row: dict[str, Any]) -> list[tuple[int, int]]:
    ranges, cursor = [], 0
    for message in row["messages"]:
        if message["role"] == "assistant":
            continue
        start = full.find(message["content"], cursor)
        assert start >= 0, message["content"]
        ranges.append((start, start + len(message["content"])))
        cursor = start + len(message["content"])
    return ranges


def _payload_ranges(
    full: str, blocks: list[tuple[int, int]], row: dict[str, Any]
) -> list[tuple[int, int]]:
    """Per assistant block: the reply payload through its ``<|im_end|>``."""
    replies = [m for m in row["messages"] if m["role"] == "assistant"]
    assert len(replies) == len(blocks)
    payloads = []
    for (start, end), message in zip(blocks, replies, strict=True):
        content = message.get("content") or ""
        anchor = content.split("</think>")[-1] if content else "<tool_call>"
        at = full.find(anchor, start, end)
        assert at >= 0, anchor
        im_end = full.find("<|im_end|>", at, end)
        payloads.append((at, im_end + len("<|im_end|>")))
    return payloads


def _overlaps(a: int, b: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a < hi and b > lo for lo, hi in ranges)


def _labeled(out: dict[str, list[int]]) -> list[int]:
    return [i for i, label in enumerate(out["labels"]) if label != -100]


def test_labels_are_exactly_the_assistant_spans(tokenizer: Any) -> None:
    outcomes = set()
    for row in _rows(24, seed=7):
        out = tokenize_sft_row(tokenizer, row, max_length=100_000)
        assert out is not None
        [counts] = [tokenize_sft_rows(tokenizer, [row], 100_000)[1]]
        rewritten = _history_rewritten(tokenizer, row)
        outcome = "final_turn_only" if counts["final_turn_only"] else "all"
        outcomes.add(outcome)
        assert (outcome == "final_turn_only") == rewritten

        full = _render(tokenizer, row)
        encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        assert out["input_ids"] == list(encoded["input_ids"])
        offsets = [tuple(span) for span in encoded["offset_mapping"]]
        blocks = [(m.start(1), m.end(1)) for m in _BLOCK.finditer(full)]
        others = _non_assistant_ranges(full, row)
        payloads = _payload_ranges(full, blocks, row)
        supervised_blocks = blocks[-1:] if rewritten else blocks
        supervised_payloads = payloads[-1:] if rewritten else payloads

        labeled = set(_labeled(out))
        for i in labeled:
            a, b = offsets[i]
            assert out["labels"][i] == out["input_ids"][i]
            assert _overlaps(a, b, supervised_blocks), tokenizer.decode(
                [out["input_ids"][i]]
            )
            assert not _overlaps(a, b, others), tokenizer.decode([out["input_ids"][i]])
        for i, (a, b) in enumerate(offsets):
            if b > a and any(lo <= a and b <= hi for lo, hi in supervised_payloads):
                assert i in labeled, (full[a:b], full)
        # Reasoning the model produced in the supervised final turn is a target;
        # reasoning stripped from history is not even in the input.
        final = row["messages"][-1]["content"]
        if "<think>" in final:
            reasoning = final.split("<think>")[1].split("</think>")[0]
            text = tokenizer.decode([out["input_ids"][i] for i in sorted(labeled)])
            assert reasoning in text
        last_user = max(i for i, m in enumerate(row["messages"]) if m["role"] == "user")
        for message in row["messages"][:last_user]:
            if message["role"] == "assistant" and "<think>" in (
                message.get("content") or ""
            ):
                stripped = message["content"].split("<think>")[1].split("</think>")[0]
                assert stripped not in full
    assert outcomes == {"all", "final_turn_only"}  # both paths were exercised


@pytest.mark.parametrize("max_length", [24, 80, 160, 320])
def test_truncation_only_cuts_and_counts_feed_the_manifest(
    tokenizer: Any, max_length: int
) -> None:
    rows = _rows(16, seed=11)
    expected = {
        "all_assistant_turns": 0,
        "final_turn_only": 0,
        "dropped_unmaskable": 0,
        "dropped_truncated": 0,
        "partially_truncated": 0,
    }
    kept = []
    for row in rows:
        full = tokenize_sft_row(tokenizer, row, max_length=100_000)
        assert full is not None
        cut = tokenize_sft_row(tokenizer, row, max_length=max_length)
        labeled = _labeled(full)
        if not any(i < max_length for i in labeled):
            assert cut is None
            expected["dropped_truncated"] += 1
            continue
        assert cut is not None
        assert cut["input_ids"] == full["input_ids"][:max_length]
        assert cut["labels"] == full["labels"][:max_length]
        assert cut["attention_mask"] == [1] * len(cut["input_ids"])
        expected[
            (
                "final_turn_only"
                if _history_rewritten(tokenizer, row)
                else "all_assistant_turns"
            )
        ] += 1
        expected["partially_truncated"] += any(i >= max_length for i in labeled)
        kept.append(cut)
    records, counts = tokenize_sft_rows(tokenizer, rows, max_length)
    assert counts == expected
    assert records == kept
