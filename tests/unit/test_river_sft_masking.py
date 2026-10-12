"""River/Tinker SFT datums must be the faithful full render, weighted on
assistant tokens only, for templates that rewrite earlier turns (Qwen3 adds a
``<think>`` stub to the final assistant turn only) and for tool calls."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from stateset_agents.core.chat_masking import assistant_token_mask  # noqa: E402
from stateset_agents.remote.river_batches import build_sft_batch  # noqa: E402
from tests.unit.test_loss_computation_real_tokenizer_masks import (  # noqa: E402
    CHATML,
    _byte_tokenizer,
)

QWEN3_LIKE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['role'] == 'assistant' and loop.last %}<think>\n\n</think>\n\n{% endif %}"
    "{% if m.get('tool_calls') %}<tool_call>{{ m['tool_calls'][0]['function']['name'] }}"
    "</tool_call>{% else %}{{ m['content'] }}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "{% endif %}"
)

CONVERSATION = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "Where is order 7?"},
    {"role": "assistant", "content": "Shipped."},
    {"role": "user", "content": "And refund order 8?"},
    {"role": "assistant", "content": "Refunded."},
]


def _weighted_and_full(template, messages, **row):
    tok = _byte_tokenizer(template)
    [datum] = build_sft_batch(
        [{"messages": messages, **row}], tok, max_length=4096, shift_targets=False
    )
    ids, weights = datum["input_ids"], datum["weights"]
    weighted = tok.decode([t for t, w in zip(ids, weights, strict=True) if w > 0])
    return tok, tok.decode(ids), weighted


@pytest.mark.parametrize("template", [CHATML, QWEN3_LIKE])
def test_multi_turn_datum_is_the_exact_render(template):
    tok, full, weighted = _weighted_and_full(template, CONVERSATION)
    assert full == tok.apply_chat_template(CONVERSATION, tokenize=False)
    assert "And refund order 8?" not in weighted
    assert "Shipped." in weighted and "Refunded." in weighted
    assert "Where is order 7?" not in weighted


def test_tool_call_turn_is_kept_and_supervised():
    messages = [
        {"role": "user", "content": "Refund order 8"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "refund"}}],
        },
    ]
    tok, full, weighted = _weighted_and_full(QWEN3_LIKE, messages)
    assert "<tool_call>refund</tool_call>" in weighted
    assert "Refund order 8" not in weighted


def test_assistant_token_mask_rejects_history_rewriting_templates():
    reversing = (
        "{% for m in messages|reverse %}<|im_start|>{{ m['role'] }}\n"
        "{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    assert assistant_token_mask(_byte_tokenizer(reversing), CONVERSATION) is None
