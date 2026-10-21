"""Training-only adjacent-boundary contrasts for the synthetic refund policy.

Derive examples from canonical training facts only. Existing validation/test
facts are used solely to reject overlap, never to supply examples or labels.
This prospective curriculum does not change the benchmark or establish learning.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)
from stateset_agents.data.refund_demonstrations import (
    ENVIRONMENT,
    _load_data,
    _publish,
    _sft_row,
    reference_refund_demonstration,
)
from stateset_agents.evaluation.agent_runs import content_hash

CURRICULUM_POLICY = "adjacent_boundary_v1"
PARAMETER_POLICY = "policy_parameter_contrasts_v1"
ARITHMETIC_POLICY = "borrow_boundary_contrasts_v1"
ARITHMETIC_QUANTUM = 1000
PARAMETER_WINDOWS = (3, 10, 21, 60, 90)
SOURCE = "reference_policy_boundary_contrast"
_ELIGIBLE_FAMILIES = {"eligible", "partial_refund", "window_boundary"}


def _contrast_cases(
    splits: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep all original training cases and add deduplicated adjacent triples."""
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    held_out = {
        fingerprint(row) for name in ("validation", "test") for row in splits[name]
    }
    held_out_ids = {
        row["order_id"] for name in ("validation", "test") for row in splits[name]
    }
    rows = copy.deepcopy(sorted(splits["train"], key=lambda row: row["order_id"]))
    by_facts: dict[str, dict[str, Any]] = {}
    lineage = []
    for row in rows:
        key = fingerprint(row)
        if key in held_out or row["order_id"] in held_out_ids:
            raise ValueError("Training cases overlap validation or test")
        by_facts.setdefault(key, row)
        lineage.append(
            {
                "case_id": row["order_id"],
                "case_hash": content_hash(row),
                "source_case_id": row["order_id"],
                "source_case_hash": content_hash(row),
                "source_split": "train",
                "derived": False,
            }
        )
    groups = []
    used_ids = {row["order_id"] for row in rows} | held_out_ids
    # Iterate the original split, never the growing list of derived cases.
    for source in sorted(splits["train"], key=lambda row: row["order_id"]):
        if source["family"] not in _ELIGIBLE_FAMILIES:
            continue
        if (
            source["chargeback_open"]
            or source["status"] != "delivered"
            or source["refunded_cents"] >= source["paid_cents"]
            or source["return_window_days"] < 1
        ):
            raise ValueError("Boundary contrast requires eligible training facts")
        group: dict[str, Any] = {
            "source_case_id": source["order_id"],
            "source_case_hash": content_hash(source),
            "cases": [],
        }
        for offset in (-1, 0, 1):
            row = copy.deepcopy(source)
            row["days_since_delivery"] = row["return_window_days"] + offset
            row["family"] = (
                "outside_window"
                if offset > 0
                else "window_boundary" if offset == 0 else source["family"]
            )
            if offset < 0 and row["family"] == "window_boundary":
                row["family"] = "eligible"
            key = fingerprint(row)
            if key in held_out:
                raise ValueError("Derived training case overlaps validation or test")
            if key in by_facts:
                row = by_facts[key]
            else:
                identity = content_hash(
                    {
                        "policy": CURRICULUM_POLICY,
                        "source_case_hash": content_hash(source),
                        "offset": offset,
                    }
                )
                row["order_id"] = f"curriculum-{identity[:24]}"
                if row["order_id"] in used_ids:
                    raise ValueError("Derived training case identity collision")
                used_ids.add(row["order_id"])
                rows.append(row)
                by_facts[key] = row
                lineage.append(
                    {
                        "case_id": row["order_id"],
                        "case_hash": content_hash(row),
                        "source_case_id": source["order_id"],
                        "source_case_hash": content_hash(source),
                        "source_split": "train",
                        "derived": True,
                        "boundary_offset": offset,
                    }
                )
            group["cases"].append(
                {
                    "boundary_offset": offset,
                    "case_id": row["order_id"],
                    "case_hash": content_hash(row),
                }
            )
        groups.append(group)
    if not groups:
        raise ValueError("Training split has no eligible boundary contrast sources")
    return sorted(rows, key=lambda row: row["order_id"]), lineage, groups


def _parameter_cases(
    splits: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Contrast new window lengths while retaining priority and note facts."""
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    held_out = {
        fingerprint(row) for name in ("validation", "test") for row in splits[name]
    }
    used_ids = {
        row["order_id"] for name in ("validation", "test") for row in splits[name]
    }
    rows = copy.deepcopy(sorted(splits["train"], key=lambda row: row["order_id"]))
    by_facts = {fingerprint(row): row for row in rows}
    lineage = []
    for row in rows:
        if fingerprint(row) in held_out or row["order_id"] in used_ids:
            raise ValueError("Training cases overlap validation or test")
        used_ids.add(row["order_id"])
        lineage.append(
            {
                "case_id": row["order_id"],
                "case_hash": content_hash(row),
                "source_case_id": row["order_id"],
                "source_case_hash": content_hash(row),
                "source_split": "train",
                "derived": False,
            }
        )
    counts: Counter[str] = Counter()
    groups = []
    # Assignment is balanced within source family and uses training IDs only.
    # Held-out outcomes never choose sources, window values, or supervision.
    for source in sorted(splits["train"], key=lambda row: row["order_id"]):
        family = source["family"]
        window = PARAMETER_WINDOWS[counts[family] % len(PARAMETER_WINDOWS)]
        counts[family] += 1
        group: dict[str, Any] = {
            "source_case_id": source["order_id"],
            "source_case_hash": content_hash(source),
            "return_window_days": window,
            "cases": [],
        }
        for offset in (-1, 0, 1):
            row = copy.deepcopy(source)
            row["return_window_days"] = window
            row["days_since_delivery"] = window + offset
            # Misleading notes remain explicitly tracked across every decision.
            if family != "misleading_note":
                if row["chargeback_open"]:
                    row["family"] = "chargeback"
                elif row["refunded_cents"] == row["paid_cents"]:
                    row["family"] = "already_refunded"
                elif row["status"] != "delivered":
                    row["family"] = "undelivered"
                elif offset > 0:
                    row["family"] = "outside_window"
                elif offset == 0:
                    row["family"] = "window_boundary"
                else:
                    row["family"] = (
                        "partial_refund" if row["refunded_cents"] else "eligible"
                    )
            key = fingerprint(row)
            if key in held_out:
                raise ValueError("Derived training case overlaps validation or test")
            if key in by_facts:
                row = by_facts[key]
            else:
                identity = content_hash(
                    {
                        "policy": PARAMETER_POLICY,
                        "source_case_hash": content_hash(source),
                        "return_window_days": window,
                        "offset": offset,
                    }
                )
                row["order_id"] = f"curriculum-{identity[:24]}"
                if row["order_id"] in used_ids:
                    raise ValueError("Derived training case identity collision")
                used_ids.add(row["order_id"])
                rows.append(row)
                by_facts[key] = row
                lineage.append(
                    {
                        "case_id": row["order_id"],
                        "case_hash": content_hash(row),
                        "source_case_id": source["order_id"],
                        "source_case_hash": content_hash(source),
                        "source_split": "train",
                        "derived": True,
                        "return_window_days": window,
                        "boundary_offset": offset,
                    }
                )
            group["cases"].append(
                {
                    "boundary_offset": offset,
                    "case_id": row["order_id"],
                    "case_hash": content_hash(row),
                }
            )
        groups.append(group)
    if not groups:
        raise ValueError("Training split has no parameter contrast sources")
    return sorted(rows, key=lambda row: row["order_id"]), lineage, groups


def _arithmetic_cases(
    splits: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain parameter contrasts and vary training-only partial refund amounts."""
    rows, lineage, groups = _parameter_cases(splits)
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    held_out = {
        fingerprint(row) for split in ("validation", "test") for row in splits[split]
    }
    used_ids = {row["order_id"] for split in splits for row in splits[split]}
    used_ids.update(row["order_id"] for row in rows)
    by_facts = {fingerprint(row): row for row in rows}
    for source in sorted(splits["train"], key=lambda row: row["order_id"]):
        if not (
            0 < source["refunded_cents"] < source["paid_cents"]
            and source["status"] == "delivered"
            and not source["chargeback_open"]
            and source["days_since_delivery"] <= source["return_window_days"]
        ):
            continue
        quantum = ARITHMETIC_QUANTUM
        paid_anchor = ((source["paid_cents"] + quantum - 1) // quantum) * quantum
        refunded_anchor = min(
            max(quantum, (source["refunded_cents"] // quantum) * quantum),
            paid_anchor - 3,
        )
        group: dict[str, Any] = {
            "kind": "arithmetic",
            "source_case_id": source["order_id"],
            "source_case_hash": content_hash(source),
            "quantum": quantum,
            "paid_anchor": paid_anchor,
            "refunded_anchor": refunded_anchor,
            "cases": [],
        }
        for paid_offset in (-1, 0, 1):
            for refunded_offset in (-1, 0, 1):
                row = copy.deepcopy(source)
                row["paid_cents"] = paid_anchor + paid_offset
                row["refunded_cents"] = refunded_anchor + refunded_offset
                row["days_since_delivery"] = row["return_window_days"]
                row["family"] = (
                    "misleading_note"
                    if source["family"] == "misleading_note"
                    else "window_boundary"
                )
                key = fingerprint(row)
                if key in held_out:
                    raise ValueError(
                        "Derived training case overlaps validation or test"
                    )
                if key in by_facts:
                    row = by_facts[key]
                else:
                    identity = content_hash(
                        {
                            "policy": ARITHMETIC_POLICY,
                            "source_case_hash": content_hash(source),
                            "paid_offset": paid_offset,
                            "refunded_offset": refunded_offset,
                        }
                    )
                    row["order_id"] = f"curriculum-{identity[:24]}"
                    if row["order_id"] in used_ids:
                        raise ValueError("Derived training case identity collision")
                    used_ids.add(row["order_id"])
                    by_facts[key] = row
                    rows.append(row)
                    lineage.append(
                        {
                            "case_id": row["order_id"],
                            "case_hash": content_hash(row),
                            "source_case_id": source["order_id"],
                            "source_case_hash": content_hash(source),
                            "source_split": "train",
                            "derived": True,
                            "kind": "arithmetic",
                            "quantum": quantum,
                            "paid_offset": paid_offset,
                            "refunded_offset": refunded_offset,
                        }
                    )
                group["cases"].append(
                    {
                        "case_id": row["order_id"],
                        "case_hash": content_hash(row),
                        "paid_offset": paid_offset,
                        "refunded_offset": refunded_offset,
                    }
                )
        groups.append(group)
    if not any(group.get("kind") == "arithmetic" for group in groups):
        raise ValueError("Training split has no partial refund arithmetic sources")
    return sorted(rows, key=lambda row: row["order_id"]), lineage, groups


async def _build_curriculum(
    data_dir: Path,
    policy: str = CURRICULUM_POLICY,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    builders = {
        CURRICULUM_POLICY: (_contrast_cases, SOURCE),
        PARAMETER_POLICY: (_parameter_cases, "reference_policy_parameter_contrast"),
        ARITHMETIC_POLICY: (_arithmetic_cases, "reference_policy_arithmetic_contrast"),
    }
    if policy not in builders:
        raise ValueError(f"Unsupported refund curriculum policy: {policy}")
    data_manifest, splits = _load_data(data_dir)
    builder, source = builders[policy]
    cases, lineage, groups = builder(splits)
    origins = {entry["case_id"]: entry for entry in lineage}
    selected = []
    for row in cases:
        candidate = await reference_refund_demonstration(row)
        item = _sft_row(row, candidate["messages"], source)
        item["metadata"].update(
            {
                "curriculum_policy": policy,
                "source_case_id": origins[row["order_id"]]["source_case_id"],
                "source_case_hash": origins[row["order_id"]]["source_case_hash"],
            }
        )
        selected.append(item)
    manifest = {
        "schema_version": 1,
        "environment": ENVIRONMENT,
        "source": source,
        "scope": "training_only",
        "curriculum_policy": policy,
        "seed": data_manifest["seed"],
        "data_manifest_hash": content_hash(data_manifest),
        "source_training_cases_hash": content_hash(splits["train"]),
        "selected": len(selected),
        "original_cases": len(splits["train"]),
        "derived_cases": len(cases) - len(splits["train"]),
        "contrast_groups": len(groups),
        "families": dict(Counter(row["family"] for row in cases)),
    }
    if policy in (PARAMETER_POLICY, ARITHMETIC_POLICY):
        manifest["parameter_windows"] = list(PARAMETER_WINDOWS)
        manifest["boundary_offsets"] = [-1, 0, 1]
    if policy == ARITHMETIC_POLICY:
        manifest["arithmetic_quantum"] = ARITHMETIC_QUANTUM
        manifest["arithmetic_offsets"] = [-1, 0, 1]
        manifest["arithmetic_contrast_groups"] = sum(
            group.get("kind") == "arithmetic" for group in groups
        )
    documents = {
        "train.json": cases,
        "lineage.json": lineage,
        "contrast_groups.json": groups,
    }
    return manifest, documents, selected


async def prepare_refund_curriculum(
    data_dir: Path,
    output: Path,
    *,
    policy: str = CURRICULUM_POLICY,
) -> dict[str, Any]:
    """Publish verified synthetic SFT data derived only from prepared train cases.

    Preserve all original training cases, add the preceding, inclusive, and
    following day around their existing windows by default. The explicit
    parameter policy contrasts five new window lengths across all source
    families, including higher priority denials, chargebacks and untrusted notes.
    The arithmetic policy retains parameter contrasts and adds partial-refund
    subtraction contrasts around decimal boundaries at the inclusive final day.
    All policies deduplicate facts and use no held-out labels or provider calls.
    """
    manifest, documents, selected = await _build_curriculum(data_dir, policy)
    return _publish(output, manifest, documents, selected)


async def audit_refund_curriculum(
    data_dir: Path,
    output: Path,
    *,
    policy: str = CURRICULUM_POLICY,
) -> dict[str, Any]:
    """Reconstruct the curriculum and replay every expected teacher transcript.

    Artifact hashes alone cannot admit altered facts, lineage, or assistant
    actions. A prepared source bundle is required; it cannot be another derived
    curriculum or a collection of validation/test cases.
    """
    expected, documents, selected = await _build_curriculum(data_dir, policy)
    manifest = json.loads((output / "data_manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Curriculum manifest must be an object")
    artifacts = manifest.get("artifacts")
    names = set(documents) | {"train.jsonl"}
    if not isinstance(artifacts, dict) or set(artifacts) != names:
        raise ValueError("Curriculum artifact manifest is incomplete")
    if content_hash(manifest) != content_hash({**expected, "artifacts": artifacts}):
        raise ValueError("Curriculum provenance differs from the training source")
    for name in sorted(names):
        raw = (output / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != artifacts[name]:
            raise ValueError(f"Curriculum artifact changed: {name}")
        actual = (
            [json.loads(line) for line in raw.decode().splitlines()]
            if name == "train.jsonl"
            else json.loads(raw)
        )
        wanted = selected if name == "train.jsonl" else documents[name]
        if content_hash(actual) != content_hash(wanted):
            raise ValueError(f"Curriculum reconstruction differs: {name}")
    return {
        "passed": True,
        "curriculum_policy": policy,
        "data_manifest_hash": content_hash(manifest),
        "replayed_demonstrations": len(selected),
        "derived_cases": expected["derived_cases"],
        "contrast_groups": expected["contrast_groups"],
    }
