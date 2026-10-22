"""Local/industry SFT supervises assistant tokens only and never trains a row
whose answer was truncated away (it used to set ``labels = input_ids``)."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from stateset_agents.training.sft import (  # noqa: E402
    tokenize_sft_row,
    tokenize_sft_rows,
)
from tests.unit.test_loss_computation_real_tokenizer_masks import (  # noqa: E402
    CHATML,
    _byte_tokenizer,
)
from tests.unit.test_river_sft_masking import CONVERSATION, QWEN3_LIKE  # noqa: E402


@pytest.mark.parametrize("template", [CHATML, QWEN3_LIKE])
def test_labels_cover_assistant_tokens_only(template):
    tok = _byte_tokenizer(template)
    records, _ = tokenize_sft_rows(tok, [{"messages": CONVERSATION}], max_length=4096)
    assert records
    for out in records:
        assert len(out["labels"]) == len(out["input_ids"]) == len(out["attention_mask"])
    supervised = " | ".join(_supervised(tok, out) for out in records)
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


def test_history_rewriting_template_trains_each_turn_as_generated():
    """Unlocatable earlier turns used to label EVERY token (system and user
    included). Each turn is now its own sequence, rendered as the newest
    turn: the earlier reply keeps the reasoning it was generated with, and
    the later one sees the history the template shows at inference."""
    tok = _byte_tokenizer(THINK_STRIPPING)
    records, counts = tokenize_sft_rows(
        tok, [{"messages": THINKING_CONVERSATION}], max_length=4096
    )
    assert counts["split_per_turn"] == 1 and len(records) == 2
    first, second = (_supervised(tok, out) for out in records)
    assert first == "<think>look it up</think>Shipped.<|im_end|>\n"
    assert second == "Refunded.<|im_end|>\n"
    assert "look it up" not in tok.decode(records[1]["input_ids"])
    assert tokenize_sft_row(tok, {"messages": THINKING_CONVERSATION}, 4096) == (
        records[1]
    )


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
    assert len(records) == 4  # the split row contributes two sequences
    assert counts == {
        "all_assistant_turns": 2,
        "split_per_turn": 1,
        "dropped_no_target": 0,
        "dropped_unmaskable": 0,
        "dropped_truncated": 1,
        "partially_truncated": 1,
    }
    _, unmaskable = tokenize_sft_rows(
        _byte_tokenizer(REVERSING), [{"messages": CONVERSATION}], max_length=4096
    )
    assert unmaskable["dropped_unmaskable"] == 1


def test_tool_call_turn_followed_by_more_turns_is_fully_supervised():
    """Qwen3-style templates decorate only the LAST assistant turn, so the
    tool-call turn's own render stopped agreeing with the full conversation
    right after ``<t`` and only that sliver was labeled; the call itself
    went unsupervised."""
    from tests.unit.test_river_sft_masking import QWEN3_LIKE

    messages = [
        {"role": "user", "content": "Refund order 8"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "refund"}}],
        },
        {"role": "tool", "content": "refunded"},
        {"role": "assistant", "content": "Done."},
    ]
    tok = _byte_tokenizer(QWEN3_LIKE)
    records, counts = tokenize_sft_rows(tok, [{"messages": messages}], 4096)
    assert counts["split_per_turn"] == 1
    call, done = (_supervised(tok, out) for out in records)
    assert call == "<tool_call>refund</tool_call><|im_end|>\n"
    assert done == "Done.<|im_end|>\n"


def _prompt_before_target(tok, out):
    first = next(i for i, label in enumerate(out["labels"]) if label != -100)
    return tok.decode(out["input_ids"][:first])


def test_every_target_is_trained_after_its_generation_prompt():
    """Qwen3-style templates render the newest turn with a ``<think>`` stub
    that history drops: one sequence used to train "Shipped." without the
    stub and "Refunded." with it — contradictory targets for one prompt."""
    tok = _byte_tokenizer(QWEN3_LIKE)
    records, _ = tokenize_sft_rows(tok, [{"messages": CONVERSATION}], 4096)
    assert len(records) == 2
    for index, out in zip((2, 4), records, strict=True):
        expected = tok.apply_chat_template(
            CONVERSATION[:index], tokenize=False, add_generation_prompt=True
        )
        assert _prompt_before_target(tok, out) == expected


def test_weight_zero_turns_are_context_not_targets():
    weighted = [
        *CONVERSATION[:2],
        {**CONVERSATION[2], "weight": 0},
        *CONVERSATION[3:],
    ]
    for template in (CHATML, QWEN3_LIKE):
        tok = _byte_tokenizer(template)
        records, counts = tokenize_sft_rows(tok, [{"messages": weighted}], 4096)
        assert len(records) == 1
        supervised = _supervised(tok, records[0])
        assert "Refunded." in supervised and "Shipped." not in supervised
        assert "Shipped." in tok.decode(records[0]["input_ids"])
    unweighted = [
        {**m, "weight": 0} if m["role"] == "assistant" else m for m in weighted
    ]
    records, counts = tokenize_sft_rows(
        _byte_tokenizer(CHATML), [{"messages": unweighted}], 4096
    )
    assert records == [] and counts["dropped_no_target"] == 1


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "x", "weight": 0.5},
        {"role": "user", "content": "x", "weight": 0},
    ],
)
def test_invalid_weights_are_rejected(message):
    tok = _byte_tokenizer(CHATML)
    row = {"messages": [{"role": "user", "content": "q"}, message]}
    with pytest.raises(ValueError, match="weight must be 0 or 1"):
        tokenize_sft_rows(tok, [row], 4096)
