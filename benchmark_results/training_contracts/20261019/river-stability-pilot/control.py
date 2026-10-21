"""Training-only repetition of the audited legacy curriculum, matched by family."""

from collections import Counter
import copy
import hashlib

from stateset_agents.data.refund_demonstrations import reference_refund_demonstration


async def repetition_control(canonical, target_counts, seed):
    """Preserve all source rows, then repeat existing facts to meet family quotas."""
    rows = copy.deepcopy(sorted(canonical, key=lambda row: row["order_id"]))
    counts = Counter(row["family"] for row in rows)
    if (
        not rows
        or set(counts) != set(target_counts)
        or any(type(n) is not int or n < counts[f] for f, n in target_counts.items())
    ):
        raise ValueError("Family quotas must preserve every existing training case")
    if len({r["order_id"] for r in rows}) != len(rows):
        raise ValueError("Source training IDs must be unique")
    by_family = {
        family: [r for r in rows if r["family"] == family] for family in counts
    }
    lineage = [
        {"case_id": r["order_id"], "source_case_id": r["order_id"], "alias": False}
        for r in rows
    ]
    used_ids = {r["order_id"] for r in rows}
    for family in sorted(counts):
        for index in range(target_counts[family] - counts[family]):
            source = by_family[family][index % len(by_family[family])]
            case = copy.deepcopy(source)
            identity = hashlib.sha256(
                f'legacy-family-repetition-v1:{seed}:{family}:{index}:{source["order_id"]}'.encode()
            ).hexdigest()
            case["order_id"] = f"curriculum-{identity[:24]}"
            if case["order_id"] in used_ids:
                raise ValueError("Control alias identity collision")
            used_ids.add(case["order_id"])
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
        teacher = await reference_refund_demonstration(case)
        examples.append(
            {
                "messages": teacher["messages"],
                "metadata": {
                    "split": "train",
                    "case_id": case["order_id"],
                    "case_hash": teacher["case_hash"],
                    "family": case["family"],
                    "source": "legacy_family_repetition_control",
                },
            }
        )
    return rows, examples, lineage
