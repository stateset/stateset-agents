"""Offline contracts against cached tokenizers from the actual pinned releases."""

from __future__ import annotations

import os

import pytest

from stateset_agents.core.frontier_models import get_frontier_model
from stateset_agents.training.frontier_sft import (
    encode_frontier_rows,
    prepare_frontier_tokenizer,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "name", ["glm5.3", "glm5.3-flash", "qwen3.8-2.4t-a95b", "qwen3.8-flash-next"]
)
def test_pinned_native_tokenizer_preserves_assistant_labels(name):
    if os.environ.get("STATESET_VERIFY_PINNED_FRONTIER_TOKENIZERS") != "1":
        pytest.skip("requires explicitly prepared pinned tokenizer cache")
    from transformers import AutoTokenizer

    spec = get_frontier_model(name)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=False,
        local_files_only=True,
    )
    prepare_frontier_tokenizer(tokenizer, spec)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Explain the return window."},
                {
                    "role": "assistant",
                    "content": "Returns are accepted within thirty days.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Explain the return window."},
                {
                    "role": "assistant",
                    "content": "Returns are accepted within thirty days.",
                },
                {"role": "user", "content": "Translate 'thank you' into French."},
                {"role": "assistant", "content": "Merci."},
            ]
        },
    ]
    encoded = encode_frontier_rows(tokenizer, rows, 2048)
    for row, batch in zip(rows, encoded, strict=True):
        assert batch["input_ids"] == tokenizer.apply_chat_template(
            row["messages"], tokenize=True, return_dict=False
        )
        assert len(batch["labels"]) == len(batch["input_ids"])
        assert any(token == -100 for token in batch["labels"])
        assert any(token != -100 for token in batch["labels"][1:])
        selected = [token for token in batch["labels"] if token != -100]
        decoded = tokenizer.decode(selected, skip_special_tokens=False)
        assert "Returns are accepted" in decoded
        assert "Explain the return window" not in decoded
        if len(row["messages"]) > 2:
            assert "Merci" in decoded
            assert "Translate" not in decoded
