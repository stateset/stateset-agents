"""Deterministic recovery fixtures and public-observation reference trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.core.environments.refund_recovery_environment import (
    RECOVERY_FAMILIES,
    RECOVERY_PROTOCOL,
    RefundRecoveryEnvironment,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.implementation import package_implementation


def refund_recovery_cases(
    split: str,
    *,
    count: int = 80,
    seed: int = 0,
    excluded: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Generate ten balanced fault families; reject aliases and prior facts.

    Train/validation use the original policy's 14/30-day windows; test uses
    7/45-day windows. Recovery labels are private environmental fault settings.
    Source order facts can be adjusted only to make a stale-fact event real.
    All data remains fictional, with no human-reviewed quality claim.
    """
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Use positive integer count and nonnegative integer seed")
    rows = refund_policy_benchmark(split, count=count, seed=seed)
    used_ids = {row["order_id"] for row in excluded}
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    used_facts = {fingerprint(row) for row in excluded}
    result = []
    for index, source in enumerate(rows):
        row = copy.deepcopy(source)
        family = RECOVERY_FAMILIES[index % len(RECOVERY_FAMILIES)]
        row["policy_source_family"] = row["family"]
        row["family"] = row["recovery"] = family
        identity = content_hash(
            {
                "protocol": RECOVERY_PROTOCOL,
                "split": split,
                "count": count,
                "seed": seed,
                "index": index,
            }
        )
        row["order_id"] = f"recovery-{identity[:24]}"
        if family == "stale_chargeback":
            row["chargeback_open"] = False
        if family == "stale_refund" and row["refunded_cents"] == row["paid_cents"]:
            row["refunded_cents"] -= 1
        RefundRecoveryEnvironment.validate_scenario(row)
        facts = fingerprint(row)
        if row["order_id"] in used_ids or facts in used_facts:
            raise ValueError("Recovery fixture overlaps an excluded or generated order")
        used_ids.add(row["order_id"])
        used_facts.add(facts)
        result.append(row)
    return result


def reference_recovery_action(observation: dict[str, Any] | None) -> dict[str, Any]:
    """Choose a recovery intent using public backend knowledge only."""
    resolve = {"tool": "resolve_refund", "args": {"intent": "resolve_refund"}}
    if observation is None:
        return resolve
    status = observation.get("status")
    if status == "applied":
        outcome = "resolved"
    elif status == "unauthorized":
        outcome = "unauthorized"
    elif status in {"pending", "unavailable"}:
        outcome = "handoff"
    elif status == "outcome_unknown":
        return {"tool": "reconcile", "args": {}}
    elif status == "retryable" and observation.get("retry_safe") is True:
        return resolve
    elif (
        status == "not_applied"
        and observation.get("retry_safe") is True
        and observation.get("fenced") is True
    ):
        return resolve
    else:
        raise ValueError("No safe reference action for unrecognized backend knowledge")
    return {"tool": "finish", "args": {"outcome": outcome}}


async def reference_recovery_demonstration(row: dict[str, Any]) -> dict[str, Any]:
    """Create a teacher transcript from executed public observations and replay it."""
    env = RefundRecoveryEnvironment()
    state = await env.reset(copy.deepcopy(row))
    messages = copy.deepcopy(state.context["messages"])
    observation = None
    for _ in range(env.max_turns):
        action = reference_recovery_action(observation)
        message = {"role": "assistant", "content": json.dumps(action, sort_keys=True)}
        messages.append(message)
        state, reward, done, info = await env.step(state, ConversationTurn(**message))
        messages.extend(copy.deepcopy(info["messages"]))
        observation = json.loads(info["messages"][0]["content"])
        if done:
            if reward != 1.0 or info["metrics"]["task_success"] != 1:
                raise ValueError("Reference recovery trajectory did not succeed")
            candidate = {
                "case_id": row["order_id"],
                "case_hash": content_hash(row),
                "messages": messages,
                "truncated": None,
            }
            await replay_recovery_candidate(row, candidate)
            return candidate
    raise ValueError("Reference recovery trajectory exceeded its turn limit")


async def replay_recovery_candidate(
    row: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Require an attributed, complete transcript that actually completes safely."""
    if (
        candidate.get("case_id") != row["order_id"]
        or candidate.get("case_hash") != content_hash(row)
        or candidate.get("truncated", "missing") is not None
    ):
        raise ValueError("Candidate identity or truncation differs from recovery case")
    messages = candidate.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Candidate needs a transcript")
    env = RefundRecoveryEnvironment()
    state = await env.reset(copy.deepcopy(row))
    initial = copy.deepcopy(state.context["messages"])
    if content_hash(messages[: len(initial)]) != content_hash(initial):
        raise ValueError("Recovery candidate prompt differs from environment")
    index, done, reward = len(initial), False, 0.0
    info: dict[str, Any] = {}
    while index < len(messages):
        message = messages[index]
        if (
            done
            or not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] != "assistant"
        ):
            raise ValueError("Invalid or post-terminal recovery action")
        state, reward, done, info = await env.step(state, ConversationTurn(**message))
        observations = info["messages"]
        if content_hash(
            messages[index + 1 : index + 1 + len(observations)]
        ) != content_hash(observations):
            raise ValueError("Recovery observation differs from executed backend")
        index += 1 + len(observations)
    if not done or reward != 1.0 or info["metrics"]["policy_violations"] != 0:
        raise ValueError("Recovery candidate did not finish safely")
    return {
        "passed": True,
        "case_id": row["order_id"],
        "turns": state.turn_count,
        "transcript_hash": hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode()
        ).hexdigest(),
    }


async def prepare_refund_recovery_data(
    output: Path,
    *,
    seed: int = 0,
    train_count: int = 160,
    validation_count: int = 40,
    test_count: int = 80,
    excluded: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Publish immutable canonical splits and train-only replayed demonstrations.

    The manifest is written last as the commit marker. A failed write leaves an
    incomplete directory that is never automatically reused or overwritten.
    Exclusion IDs/fingerprints record local consistency, not external review.
    """
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to overwrite an existing recovery bundle")
    working, splits = list(excluded), {}
    for split, count in (
        ("train", train_count),
        ("validation", validation_count),
        ("test", test_count),
    ):
        splits[split] = refund_recovery_cases(
            split, count=count, seed=seed, excluded=working
        )
        working.extend(splits[split])
    records = []
    for row in splits["train"]:
        candidate = await reference_recovery_demonstration(row)
        records.append(
            {
                "messages": candidate["messages"],
                "metadata": {
                    "environment": RECOVERY_PROTOCOL,
                    "split": "train",
                    "source": "synthetic_recovery_reference",
                    "case_id": candidate["case_id"],
                    "case_hash": candidate["case_hash"],
                },
            }
        )
    serialized = {
        f"{split}.json": json.dumps(rows, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
        for split, rows in splits.items()
    }
    serialized["train.jsonl"] = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records
    )
    exclusions = {
        "order_ids": sorted({r["order_id"] for r in excluded}),
        "fact_fingerprints": sorted(
            {RefundPolicyEnvironment.case_fingerprint(r) for r in excluded}
        ),
    }
    manifest = {
        "schema_version": 1,
        "environment": RECOVERY_PROTOCOL,
        "seed": seed,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "families": {
            split: dict(sorted(Counter(r["recovery"] for r in rows).items()))
            for split, rows in splits.items()
        },
        "scope": "synthetic_train_only_demonstrations",
        "reviewed_by_humans": False,
        "implementation": package_implementation(),
        "exclusions": exclusions,
        "artifacts": {
            name: hashlib.sha256(data.encode()).hexdigest()
            for name, data in serialized.items()
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in serialized.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(data)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    return manifest


async def audit_refund_recovery_data(output: Path) -> dict[str, Any]:
    """Regenerate canonical splits, verify train provenance, and replay teachers."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Recovery manifest must be an object")
    if (
        manifest.get("schema_version") != 1
        or type(manifest.get("schema_version")) is not int
        or manifest.get("environment") != RECOVERY_PROTOCOL
        or manifest.get("scope") != "synthetic_train_only_demonstrations"
        or manifest.get("reviewed_by_humans") is not False
        or manifest.get("implementation") != package_implementation()
    ):
        raise ValueError("Recovery manifest does not match this synthetic runtime")
    expected_files = {"train.json", "validation.json", "test.json", "train.jsonl"}
    if (
        not isinstance(manifest.get("artifacts"), dict)
        or set(manifest["artifacts"]) != expected_files
    ):
        raise ValueError("Recovery bundle has invalid artifact inventory")
    for name, digest in manifest["artifacts"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Recovery bundle artifact changed")
    if set(manifest.get("counts", {})) != {"train", "validation", "test"}:
        raise ValueError("Recovery bundle requires three canonical splits")
    exclusions = manifest.get("exclusions")
    if (
        not isinstance(exclusions, dict)
        or set(exclusions) != {"order_ids", "fact_fingerprints"}
        or any(
            not isinstance(exclusions[name], list)
            or any(not isinstance(v, str) for v in exclusions[name])
            for name in exclusions
        )
    ):
        raise ValueError("Invalid recovery exclusion commitment")
    ids, facts = set(exclusions["order_ids"]), set(exclusions["fact_fingerprints"])
    splits = {}
    for split in ("train", "validation", "test"):
        rows = json.loads((output / f"{split}.json").read_text())
        expected = refund_recovery_cases(
            split, count=manifest["counts"][split], seed=manifest["seed"]
        )
        if content_hash(rows) != content_hash(expected):
            raise ValueError("Recovery split differs from canonical seed/split/count")
        new_ids = {r["order_id"] for r in rows}
        new_facts = {RefundPolicyEnvironment.case_fingerprint(r) for r in rows}
        if ids & new_ids or facts & new_facts:
            raise ValueError("Recovery split overlaps another split or exclusion")
        ids.update(new_ids)
        facts.update(new_facts)
        splits[split] = rows
    expected_families = {
        split: dict(sorted(Counter(r["recovery"] for r in rows).items()))
        for split, rows in splits.items()
    }
    if manifest.get("families") != expected_families:
        raise ValueError("Recovery family counts differ from canonical splits")
    records = [
        json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
    ]
    by_id, seen = {r["order_id"]: r for r in splits["train"]}, set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("metadata"), dict):
            raise ValueError("Recovery demonstration requires a metadata object")
        metadata = record.get("metadata", {})
        key = metadata.get("case_id")
        if (
            not isinstance(key, str)
            or key not in by_id
            or key in seen
            or metadata
            != {
                "environment": RECOVERY_PROTOCOL,
                "split": "train",
                "source": "synthetic_recovery_reference",
                "case_id": key,
                "case_hash": content_hash(by_id[key]),
            }
        ):
            raise ValueError(
                "Recovery demonstration lacks unique train-only provenance"
            )
        await replay_recovery_candidate(
            by_id[key],
            {
                "case_id": key,
                "case_hash": metadata["case_hash"],
                "messages": record.get("messages"),
                "truncated": None,
            },
        )
        seen.add(key)
    if seen != set(by_id):
        raise ValueError("Recovery training coverage is incomplete")
    return {
        "passed": True,
        "replayed_train_demonstrations": len(seen),
        "counts": manifest["counts"],
        "manifest_hash": content_hash(manifest),
        "provider_calls": 0,
    }
