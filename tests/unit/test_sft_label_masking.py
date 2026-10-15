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
