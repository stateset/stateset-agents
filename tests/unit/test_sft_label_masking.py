"""Local/industry SFT supervises assistant tokens only and never trains a row
whose answer was truncated away (it used to set ``labels = input_ids``)."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from stateset_agents.training.sft import tokenize_sft_row  # noqa: E402
from tests.unit.test_loss_computation_real_tokenizer_masks import (  # noqa: E402
    CHATML,
    _byte_tokenizer,
)
from tests.unit.test_river_sft_masking import CONVERSATION, QWEN3_LIKE  # noqa: E402


@pytest.mark.parametrize("template", [CHATML, QWEN3_LIKE])
def test_labels_cover_assistant_tokens_only(template):
    tok = _byte_tokenizer(template)
    out = tokenize_sft_row(tok, {"messages": CONVERSATION}, max_length=4096)
    assert out is not None
    assert len(out["labels"]) == len(out["input_ids"]) == len(out["attention_mask"])
    supervised = tok.decode(
        [
            t
            for t, lab in zip(out["input_ids"], out["labels"], strict=True)
            if lab != -100
        ]
    )
    assert "Shipped." in supervised and "Refunded." in supervised
    assert "Where is order 7?" not in supervised and "Be terse." not in supervised


def test_row_whose_answer_is_truncated_away_is_dropped():
    tok = _byte_tokenizer(CHATML)
    row = {
        "messages": [
            {"role": "system", "content": "x" * 300},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    assert tokenize_sft_row(tok, row, max_length=64) is None
    assert tokenize_sft_row(tok, row, max_length=4096) is not None


# Qwen3-style: earlier assistant turns lose their ``<think>`` block, so their
# replies have no faithful span in the full render.
THINK_STRIPPING = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['role'] == 'assistant' and not loop.last %}"
    "{{ m['content'].split('</think>')[-1] }}"
    "{% else %}{{ m['content'] }}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
REVERSING = (
    "{% for m in messages|reverse %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
THINKING_CONVERSATION = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "Where is order 7?"},
    {"role": "assistant", "content": "<think>look it up</think>Shipped."},
    {"role": "user", "content": "And refund order 8?"},
    {"role": "assistant", "content": "Refunded."},
]


def _supervised(tok, out):
    return tok.decode(
        [
            t
            for t, lab in zip(out["input_ids"], out["labels"], strict=True)
            if lab != -100
        ]
    )


def test_history_rewriting_template_supervises_only_the_final_turn():
    """Unlocatable earlier turns used to label EVERY token (system and user
    included); now only the final assistant turn is a target."""
    tok = _byte_tokenizer(THINK_STRIPPING)
    out = tokenize_sft_row(tok, {"messages": THINKING_CONVERSATION}, max_length=4096)
    assert out is not None
    supervised = _supervised(tok, out)
    assert "Refunded." in supervised
    for prompt_text in ("Be terse.", "Where is order 7?", "And refund", "Shipped."):
        assert prompt_text not in supervised
    # The context is the inference-time render (earlier think block stripped).
    assert "look it up" not in tok.decode(out["input_ids"])


def test_row_with_no_locatable_assistant_span_is_dropped():
    tok = _byte_tokenizer(REVERSING)
    assert (
        tokenize_sft_row(tok, {"messages": THINKING_CONVERSATION}, max_length=4096)
        is None
    )


def test_tokenize_sft_rows_counts_every_outcome():
    from stateset_agents.training.sft import tokenize_sft_rows

    tok = _byte_tokenizer(THINK_STRIPPING)
    long_system = [{"role": "system", "content": "x" * 300}, *CONVERSATION[1:]]
    long_reply = [*CONVERSATION[:-1], {"role": "assistant", "content": "y" * 300}]
    records, counts = tokenize_sft_rows(
        tok,
        [
            {"messages": CONVERSATION},
            {"messages": THINKING_CONVERSATION},
            {"messages": long_system},
            {"messages": long_reply},
        ],
        max_length=200,
    )
    assert len(records) == 3
    assert counts == {
        "all_assistant_turns": 2,
        "final_turn_only": 1,
        "dropped_unmaskable": 0,
        "dropped_truncated": 1,
        "partially_truncated": 1,
    }
    _, unmaskable = tokenize_sft_rows(
        _byte_tokenizer(REVERSING), [{"messages": CONVERSATION}], max_length=4096
    )
    assert unmaskable["dropped_unmaskable"] == 1
