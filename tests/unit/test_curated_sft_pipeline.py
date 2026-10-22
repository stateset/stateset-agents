"""Grade -> curate -> prepare -> tokenize trains only graded, passing replies.

Each curated row carries the whole conversation up to its graded reply; fine-
tuning supervised every assistant turn in it, so a reply that FAILED the
grader was trained whenever a later reply in the same transcript passed, and
a transcript with k passing replies trained its first reply k times.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import grade_transcript as gt  # noqa: E402
import prepare_sft_dataset as prep  # noqa: E402

from stateset_agents.training.sft import (  # noqa: E402
    load_chat_dataset,
    tokenize_sft_rows,
)
from tests.unit.test_loss_computation_real_tokenizer_masks import (  # noqa: E402
    CHATML,
    _byte_tokenizer,
)
from tests.unit.test_river_sft_masking import QWEN3_LIKE  # noqa: E402

TURNS = [
    {"role": "user", "content": "Where is my order?"},
    {"role": "assistant", "content": "BAD REPLY figure it out yourself"},
    {"role": "user", "content": "Please help, order 42."},
    {"role": "assistant", "content": "Order 42 shipped yesterday."},
    {"role": "user", "content": "And the refund?"},
    {"role": "assistant", "content": "Refund 42 is on its way."},
]
SCORES = [{"score": 0.05}, {"score": 0.95}, {"score": 0.9}]


def _supervised(tok, record) -> str:
    return tok.decode(
        [
            t
            for t, lab in zip(record["input_ids"], record["labels"], strict=True)
            if lab != -100
        ]
    )


@pytest.mark.parametrize("template", [CHATML, QWEN3_LIKE])
def test_only_each_graded_passing_reply_is_trained_once(tmp_path, template):
    curated = tmp_path / "curated.jsonl"
    assert gt.write_curated_examples(Path("s.jsonl"), TURNS, SCORES, 0.7, curated) == 2
    entries = prep.filter_entries(prep.load_curated(curated), min_score=0.7)
    sft_file = tmp_path / "sft.jsonl"
    sft_file.write_text(
        "".join(json.dumps(prep.to_chat(entry)) + "\n" for entry in entries),
        encoding="utf-8",
    )
    tok = _byte_tokenizer(template)
    records, counts = tokenize_sft_rows(tok, load_chat_dataset(sft_file), 4096)
    supervised = [_supervised(tok, record) for record in records]
    assert len(supervised) == 2
    joined = " | ".join(supervised)
    assert "BAD REPLY" not in joined
    assert sum("Order 42 shipped" in text for text in supervised) == 1
    assert sum("Refund 42" in text for text in supervised) == 1
    # The failed reply is still the context the passing reply was graded in.
    assert "BAD REPLY" in tok.decode(records[0]["input_ids"])
    assert counts["dropped_unmaskable"] == counts["dropped_no_target"] == 0
