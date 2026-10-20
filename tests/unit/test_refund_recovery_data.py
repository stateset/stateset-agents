"""Canonical train-only recovery exports, split protection, and tamper checks."""

import hashlib
import json

import pytest

from stateset_agents.data.refund_recovery import (
    audit_refund_recovery_data,
    prepare_refund_recovery_data,
)


async def bundle(path):
    return await prepare_refund_recovery_data(
        path, seed=901, train_count=10, validation_count=10, test_count=10
    )


async def test_export_has_replayed_train_only_targets_and_rejects_overwrite(tmp_path):
    path = tmp_path / "data"
    manifest = await bundle(path)
    audited = await audit_refund_recovery_data(path)
    assert audited["passed"] and audited["replayed_train_demonstrations"] == 10
    assert manifest["reviewed_by_humans"] is False
    train = json.loads((path / "train.json").read_text())
    train_ids = {r["order_id"] for r in train}
    records = [
        json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()
    ]
    assert {r["metadata"]["case_id"] for r in records} == train_ids
    assert all(r["metadata"]["split"] == "train" for r in records)
    assert all(
        "recovery" not in message["content"] or message["role"] == "user"
        for record in records
        for message in record["messages"]
    )
    with pytest.raises(ValueError, match="overwrite"):
        await bundle(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "heldout",
        "split",
        "observation",
        "duplicate",
        "missing",
        "metadata-type",
        "case-type",
        "seed",
        "families",
    ],
)
async def test_rehashed_bad_provenance_or_transcript_is_rejected(tmp_path, mutation):
    path = tmp_path / "data"
    manifest = await bundle(path)
    records = [
        json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()
    ]
    if mutation == "heldout":
        heldout = json.loads((path / "test.json").read_text())[0]
        records[0]["metadata"]["case_id"] = heldout["order_id"]
    elif mutation == "split":
        records[0]["metadata"]["split"] = "test"
    elif mutation == "observation":
        records[0]["messages"][2]["content"] = "{}"
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "missing":
        records.pop()
    elif mutation == "metadata-type":
        records[0]["metadata"] = []
    elif mutation == "case-type":
        records[0]["metadata"]["case_id"] = []
    elif mutation == "seed":
        manifest["seed"] += 1
    else:
        manifest["families"]["train"] = {}
    raw = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    (path / "train.jsonl").write_text(raw)
    manifest["artifacts"]["train.jsonl"] = hashlib.sha256(raw.encode()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        await audit_refund_recovery_data(path)
