"""Reviewed conversation intake with private source-group and temporal isolation."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from stateset_agents.data.curation import curate_conversations
from stateset_agents.evaluation.agent_runs import content_hash

from .storage import (
    decode_json,
    private_directory,
    private_path,
    read_bytes,
    read_json,
    write_json,
)


def prepare_dataset(
    source: Path,
    destination: Path,
    *,
    validation_after: str,
    test_after: str,
    review: Path | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Snapshot reviewed conversations privately; synthetic fixtures never imply review.

    Reviewed inputs require an attestation bound to the exact source bytes, an
    independent reviewer, approved use, and per-record provenance. Attestations
    are local assertions, not independent proof of identity or data rights.
    """
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be boolean")
    private_path(destination)
    if synthetic and review is not None:
        raise ValueError("Synthetic fixtures cannot carry customer review attestations")
    if not synthetic:
        private_path(source)
        if review is None:
            raise ValueError("Customer data requires a separate review attestation")
        private_path(review)
    data = read_bytes(source)
    rows = [decode_json(line) for line in data.decode().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Conversation JSONL must contain objects")
    attestation = read_json(review) if review is not None else None
    if not synthetic:
        if not isinstance(attestation, dict) or attestation.get("schema_version") != 1:
            raise ValueError("Invalid review attestation")
        for field in ("reviewer", "collector", "rubric_id"):
            if (
                not isinstance(attestation.get(field), str)
                or not attestation[field].strip()
            ):
                raise ValueError(f"Review requires {field}")
        if attestation["reviewer"] == attestation["collector"]:
            raise ValueError("Review must be independent of the collector")
        if (
            attestation.get("approved_for_training") is not True
            or attestation.get("approved_for_evaluation") is not True
        ):
            raise ValueError("Review must approve training and evaluation use")
        if attestation.get("source_sha256") != hashlib.sha256(data).hexdigest():
            raise ValueError("Review does not match source bytes")
        reviewed_at = datetime.fromisoformat(
            str(attestation.get("reviewed_at", "")).replace("Z", "+00:00")
        )
        if reviewed_at.tzinfo is None:
            raise ValueError("Review timestamp must include a timezone")
    ids: set[str] = set()
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip() or record_id in ids:
            raise ValueError("Every record needs a distinct record_id")
        ids.add(record_id)
        if synthetic and row.get("source_kind") != "synthetic":
            raise ValueError(
                "Synthetic mode requires every record to declare source_kind=synthetic"
            )
        if not synthetic and (
            row.get("source_kind") != "customer"
            or not isinstance(row.get("provenance_id"), str)
            or not row["provenance_id"].strip()
        ):
            raise ValueError(
                "Customer records require provenance_id and source_kind=customer"
            )
    if not synthetic and (
        not isinstance(attestation, dict)
        or not isinstance(attestation.get("record_ids"), list)
        or len(attestation["record_ids"]) != len(ids)
        or any(not isinstance(x, str) for x in attestation["record_ids"])
        or set(attestation["record_ids"]) != ids
    ):
        raise ValueError("Review must cover every input record exactly once")
    result = curate_conversations(
        rows, validation_after=validation_after, test_after=test_after
    )
    if any(not result["splits"][split] for split in ("train", "validation", "test")):
        raise ValueError("Every temporal/source-isolated split must be nonempty")
    destination = private_path(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_directory(destination)
    inventory = {}
    for split, records in result["splits"].items():
        write_json(destination / f"{split}.json", records)
        inventory[split] = content_hash(records)
    if attestation is not None:
        write_json(destination / "review.json", attestation)
    manifest = {
        "schema_version": 1,
        "kind": "stateset-private-conversation-dataset",
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "source_kind": "synthetic" if synthetic else "customer",
        "review_status": (
            "synthetic_unreviewed" if synthetic else "attested_independent_review"
        ),
        "review_sha256": content_hash(attestation) if attestation is not None else None,
        "curation": result["manifest"],
        "split_sha256": inventory,
        "production_quality_established": False,
    }
    write_json(destination / "dataset.json", manifest)
    verify_dataset(destination)
    return manifest


def verify_dataset(root: Path) -> dict[str, Any]:
    """Rehash every split and review; reject cross-split sources and duplicates."""
    root = private_directory(root)
    manifest = read_json(root / "dataset.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != "stateset-private-conversation-dataset"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("Invalid private dataset manifest")
    expected_review = {
        "synthetic": "synthetic_unreviewed",
        "customer": "attested_independent_review",
    }
    if (
        manifest.get("source_kind") not in expected_review
        or manifest.get("review_status") != expected_review[manifest["source_kind"]]
    ):
        raise ValueError("Invalid dataset source/review classification")
    if manifest.get("production_quality_established") is not False:
        raise ValueError("Dataset intake cannot establish production quality")
    groups: set[str] = set()
    contents: set[str] = set()
    cases: set[str] = set()
    identities: set[str] = set()
    for split in ("train", "validation", "test"):
        rows = read_json(root / f"{split}.json")
        if (
            not isinstance(rows, list)
            or not rows
            or content_hash(rows) != manifest["split_sha256"][split]
        ):
            raise ValueError("Dataset split changed")
        if any(
            not isinstance(r, dict)
            or r.get("source_kind") != manifest["source_kind"]
            or r.get("split") != split
            for r in rows
        ):
            raise ValueError("Dataset rows disagree with source kind or partition")
        split_ids = {r["record_id"] for r in rows}
        split_cases = {
            content_hash(r["evaluation_case"]) for r in rows if "evaluation_case" in r
        }
        if len(split_ids) != len(rows) or identities & split_ids or cases & split_cases:
            raise ValueError(
                "Evaluation case or record identity leakage across partitions"
            )
        identities.update(split_ids)
        cases.update(split_cases)
        split_groups = {r["source_group"] for r in rows}
        split_contents = {content_hash(r["messages"]) for r in rows}
        if groups & split_groups or contents & split_contents:
            raise ValueError("Dataset leakage across partitions")
        groups.update(split_groups)
        contents.update(split_contents)
    if manifest["review_sha256"] is not None:
        if content_hash(read_json(root / "review.json")) != manifest["review_sha256"]:
            raise ValueError("Dataset review changed")
    if manifest["source_kind"] == "customer" and manifest["review_sha256"] is None:
        raise ValueError("Customer review is missing")
    return dict(manifest)
