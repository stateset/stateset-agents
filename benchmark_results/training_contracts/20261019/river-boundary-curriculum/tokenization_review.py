"""Check every prepared chat with the cached target tokenizer, without network."""

import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer

from stateset_agents.core.chat_masking import assistant_token_mask
from stateset_agents.remote.river_batches import build_sft_batch

ROOT = Path("outputs/river-curriculum-review-20261019")
MODEL = "Qwen/Qwen3.5-9B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    lengths = []
    covered = 0
    for seed in range(2026101901, 2026101907):
        path = ROOT / f"seed-{seed}/curriculum/train.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        batch = build_sft_batch(rows, tokenizer, max_length=4096)
        assert len(batch) == len(rows)
        for row, datum in zip(rows, batch, strict=True):
            ids, mask = assistant_token_mask(tokenizer, row["messages"])
            assert len(ids) <= 4096
            assert datum["input_ids"] == ids[:-1]
            assert datum["target_tokens"] == ids[1:]
            assert datum["weights"] == [float(value) for value in mask[1:]]
            supervised = tokenizer.decode(
                [token for token, weight in zip(datum["target_tokens"], datum["weights"], strict=True) if weight]
            )
            actions = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
            assert len(actions) == 3
            assert all(action in supervised for action in actions)
            covered += len(actions)
            lengths.append(len(ids))
    report = {
        "passed": True,
        "model": MODEL,
        "tokenizer_class": type(tokenizer).__name__,
        "serialized_tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "local_files_only": True,
        "max_length": 4096,
        "demonstrations": len(lengths),
        "assistant_actions_supervised": covered,
        "minimum_chat_tokens": min(lengths),
        "maximum_chat_tokens": max(lengths),
        "truncated_chats": 0,
        "dropped_chats": 0,
        "provider_calls": 0,
    }
    (ROOT / "tokenization-review.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
