"""Training-only repetition control with matched row count and identifier format."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path

from stateset_agents.core.environments.refund_policy_environment import FAMILIES
from stateset_agents.data.refund_demonstrations import reference_refund_demonstration


async def repetition_control(canonical, target_count, seed):
    """Keep original cases, then evenly repeat families using fresh identifiers.

    Repeats introduce no new policy facts. Aliases match the curriculum's
    identifier format; lineage maps each alias to its original training case.
    Every teacher transcript is rebuilt and independently replayed.
    """
    rows = copy.deepcopy(sorted(canonical, key=lambda row: row["order_id"]))
    assert len(rows) <= target_count < 2 * len(rows)
    families = sorted(FAMILIES)
    by_family = {
        family: [row for row in rows if row["family"] == family] for family in families
    }
    assert all(by_family.values())
    lineage = [
        {"case_id": row["order_id"], "source_case_id": row["order_id"], "alias": False}
        for row in rows
    ]
    for index in range(target_count - len(rows)):
        family = families[index % len(families)]
        source = by_family[family][(index // len(families)) % len(by_family[family])]
        case = copy.deepcopy(source)
        digest = hashlib.sha256(
            f'original-repetition-v1:{seed}:{index}:{source["order_id"]}'.encode()
        ).hexdigest()
        case["order_id"] = f"curriculum-{digest[:24]}"
        rows.append(case)
        lineage.append(
            {
                "case_id": case["order_id"],
                "source_case_id": source["order_id"],
                "alias": True,
            }
        )
    examples = []
    for case in rows:
        demonstration = await reference_refund_demonstration(case)
        examples.append(
            {
                "messages": demonstration["messages"],
                "metadata": {
                    "split": "train",
                    "case_id": case["order_id"],
                    "case_hash": demonstration["case_hash"],
                    "family": case["family"],
                    "source": "original_training_repetition_control",
                },
            }
        )
    return rows, examples, lineage


async def inspect():
    from transformers import AutoTokenizer
    from stateset_agents.remote.river_batches import build_sft_batch

    root = Path("/home/dom/stateset-agents/outputs")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)
    results = []
    for seed in range(2026101901, 2026101907):
        canonical = json.loads(
            (
                root / f"river-six-seed-recovery-20261019/seed-{seed}/data/train.json"
            ).read_text()
        )
        curriculum = root / f"river-curriculum-review-20261019/seed-{seed}/curriculum"
        contrast = [
            json.loads(line)
            for line in (curriculum / "train.jsonl").read_text().splitlines()
        ]
        _, control, _ = await repetition_control(canonical, len(contrast), seed)
        counts = {}
        for name, rows in (("control", control), ("boundary", contrast)):
            data = build_sft_batch(rows, tokenizer, max_length=4096)
            assert len(data) == len(rows)
            counts[name] = {
                "rows": len(data),
                "input_tokens": sum(len(d["input_ids"]) for d in data),
                "target_tokens": sum(sum(d["weights"]) for d in data),
            }
        results.append(
            {
                "seed": seed,
                **counts,
                "input_ratio": counts["control"]["input_tokens"]
                / counts["boundary"]["input_tokens"],
                "target_ratio": counts["control"]["target_tokens"]
                / counts["boundary"]["target_tokens"],
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(inspect())
