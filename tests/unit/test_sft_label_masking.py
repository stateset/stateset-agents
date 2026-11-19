"""Local/industry SFT supervises assistant tokens only and never trains a row
whose answer was truncated away (it used to set ``labels = input_ids``)."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from stateset_agents.remote.river_batches import build_sft_batch  # noqa: E402
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


def test_weighted_row_requires_a_resolvable_loss_mask():
    reversing = (
        "{% for m in messages|reverse %}{{ m.role }}:{{ m.content }}{% endfor %}"
    )
    messages = [*CONVERSATION[:-1], {**CONVERSATION[-1], "weight": 1}]
    with pytest.raises(ValueError, match="weighted chat row"):
        tokenize_sft_row(_byte_tokenizer(reversing), {"messages": messages}, 4096)


@pytest.mark.parametrize("limit", [-1, 0, 1])
def test_next_token_training_requires_two_positions(limit):
    with pytest.raises(ValueError, match="max_length"):
        tokenize_sft_row(_byte_tokenizer(CHATML), {"messages": CONVERSATION}, limit)


@pytest.mark.parametrize("backend", ["local", "remote_shifted", "remote_unshifted"])
def test_cutoff_preserves_every_selected_token_in_weighted_rows(backend):
    tok = _byte_tokenizer(CHATML)
    messages = [
        {"role": "user", "content": "Check this order."},
        {"role": "assistant", "content": "Checking.", "weight": 0},
        {"role": "user", "content": "Was the refund issued?"},
        {"role": "assistant", "content": "The refund was denied.", "weight": 1},
        {"role": "user", "content": "This trailing text is outside the target."},
    ]

    def targets(turns, limit):
        row = {"messages": turns}
        if backend == "local":
            encoded = tokenize_sft_row(tok, row, limit)
            return [t for t in encoded["labels"] if t != -100] if encoded else []
        data = build_sft_batch(
            [row], tok, limit, shift_targets=backend == "remote_shifted"
        )
        if not data:
            return []
        (item,) = data
        ids = item.get("target_tokens", item["input_ids"])
        return [t for t, w in zip(ids, item["weights"], strict=True) if w > 0]

    full = tokenize_sft_row(tok, {"messages": messages}, 4096)
    labels = full["labels"]
    end = max(i for i, token in enumerate(labels) if token != -100) + 1
    assert end < len(labels)
    assert targets(messages, end) == [t for t in labels if t != -100]
    assert targets(messages, end - 1) == []

    # Keeping an earlier answer still leaves the second selected action incomplete.
    messages[1]["weight"] = 1
    assert targets(messages, end - 5) == []
    legacy = [{k: v for k, v in m.items() if k != "weight"} for m in messages]
    assert targets(legacy, end - 5) and targets(legacy, end - 5) != targets(legacy, end)
