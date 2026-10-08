"""Verified SFT and rejection-sampling data for the synthetic refund policy.

Only canonical training cases can enter exported SFT data. Model-authored
success claims and rewards are never used to accept a trajectory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.remote.river_rl import atomic_json, exclusive_run

ENVIRONMENT = "refund-policy-v2"


class RejectedTrajectory(ValueError):
    """A candidate cannot be admitted to verified SFT data."""


def _message(tool: str, **args: Any) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": json.dumps({"tool": tool, "args": args}, sort_keys=True),
    }


async def replay_refund_candidate(
    row: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, str]]:
    """Replay an exact transcript; return canonical messages only on success.

    Initial prompts and every environment observation must match the sandbox.
    All assistant actions are re-executed in a fresh ledger. A terminal success
    must end the transcript; incomplete, truncated, or extra turns are rejected.
    """
    if candidate.get("case_id") != row["order_id"] or candidate.get(
        "case_hash"
    ) != content_hash(row):
        raise RejectedTrajectory("case_mismatch")
    if "truncated" not in candidate or candidate["truncated"] is not None:
        raise RejectedTrajectory("truncated_or_unknown")
    messages = candidate.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(m, dict)
            or set(m) != {"role", "content"}
            or not isinstance(m["content"], str)
            for m in messages
        )
    ):
        raise RejectedTrajectory("invalid_messages")
    env = RefundPolicyEnvironment()
    state = await env.reset(row)
    prefix = state.context["messages"]
    if messages[: len(prefix)] != prefix:
        raise RejectedTrajectory("prompt_mismatch")
    position = len(prefix)
    total_reward = 0.0
    while position < len(messages):
        message = messages[position]
        if message["role"] != "assistant":
            raise RejectedTrajectory("expected_assistant")
        state, reward, done, info = await env.step(
            state, ConversationTurn(role="assistant", content=message["content"])
        )
        position += 1
        total_reward += reward
        if done:
            if position != len(messages):
                raise RejectedTrajectory("extra_turns_after_completion")
            if (
                info["metrics"]["task_success"] != 1
                or info["metrics"]["policy_violations"] != 0
                or total_reward != 1
            ):
                raise RejectedTrajectory("unsuccessful_replay")
            return copy.deepcopy(messages)
        observations = info["messages"]
        if messages[position : position + len(observations)] != observations:
            raise RejectedTrajectory("observation_mismatch")
        position += len(observations)
    raise RejectedTrajectory("incomplete_episode")


async def reference_refund_demonstration(row: dict[str, Any]) -> dict[str, Any]:
    """Build a rule-based teacher demonstration using only observed order facts.

    This is synthetic supervision, not a model rollout or evidence of learning.
    A second fresh sandbox verifies the completed transcript before it is used.
    """
    env = RefundPolicyEnvironment()
    state = await env.reset(row)
    messages = copy.deepcopy(state.context["messages"])
    lookup = _message("lookup_order", order_id=row["order_id"])
    state, _, _, info = await env.step(state, ConversationTurn(**lookup))
    messages.extend([lookup, *info["messages"]])
    facts = json.loads(info["messages"][0]["content"])
    if facts["chargeback_open"]:
        decision = _message("escalate", order_id=facts["order_id"], reason="chargeback")
    elif facts["refunded_cents"] == facts["paid_cents"]:
        decision = _message(
            "deny", order_id=facts["order_id"], reason="already_refunded"
        )
    elif facts["status"] != "delivered":
        decision = _message("deny", order_id=facts["order_id"], reason="not_delivered")
    elif facts["days_since_delivery"] > facts["return_window_days"]:
        decision = _message("deny", order_id=facts["order_id"], reason="outside_window")
    else:
        decision = _message(
            "refund",
            order_id=facts["order_id"],
            amount_cents=facts["paid_cents"] - facts["refunded_cents"],
        )
    state, _, _, info = await env.step(state, ConversationTurn(**decision))
    messages.extend([decision, *info["messages"], _message("finish")])
    candidate = {
        "case_id": row["order_id"],
        "case_hash": content_hash(row),
        "messages": messages,
        "truncated": None,
    }
    await replay_refund_candidate(row, candidate)
    return candidate


def _sft_row(
    row: dict[str, Any], messages: list[dict[str, str]], source: str
) -> dict[str, Any]:
    return {
        "messages": messages,
        "metadata": {
            "case_id": row["order_id"],
            "case_hash": content_hash(row),
            "family": row["family"],
            "split": "train",
            "source": source,
        },
    }


def _publish(
    output: Path,
    manifest: dict[str, Any],
    documents: dict[str, Any],
    sft_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish a complete bundle by rename; failed writes leave no partial data.

    The sibling control directory holds the writer lock and private staging
    directories. Never remove that lock while writers may be active. An abrupt
    process exit can leave an unreferenced staging directory there, which cannot
    be mistaken for a published dataset and does not prevent a fresh retry.
    """
    output = output.absolute()
    control = output.parent / f".{output.name}.publication"
    with exclusive_run(control):
        if output.is_symlink():
            raise ValueError("Dataset output must not be a symbolic link")
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("Output directory is not empty; choose a new directory")
        mode = stat.S_IMODE(output.stat().st_mode)
        with tempfile.TemporaryDirectory(prefix="stage-", dir=control) as temporary:
            staging = Path(temporary)
            for name, document in documents.items():
                atomic_json(staging / name, document)
            if sft_rows:
                with (staging / "train.jsonl").open("x", encoding="utf-8") as stream:
                    for row in sft_rows:
                        stream.write(
                            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                        )
                    stream.flush()
                    os.fsync(stream.fileno())
            names = list(documents) + (["train.jsonl"] if sft_rows else [])
            published = {
                **manifest,
                "artifacts": {
                    name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                    for name in sorted(names)
                },
            }
            atomic_json(staging / "data_manifest.json", published)
            staging.chmod(mode)
            # rmdir refuses a destination filled by another writer. It also
            # makes rename portable to platforms that cannot replace a directory.
            output.rmdir()
            os.rename(staging, output)
            if os.name != "nt":
                for directory in (control, output.parent):
                    fd = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
    return published


def _splits(seed: int, counts: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    if set(counts) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test counts")
    return {
        name: refund_policy_benchmark(name, count, seed)
        for name, count in counts.items()
    }


async def prepare_refund_data(
    output: Path,
    *,
    seed: int = 42,
    train_count: int = 256,
    validation_count: int = 64,
    test_count: int = 128,
) -> dict[str, Any]:
    """Export train-only verified teacher chats plus separate held-out cases."""
    counts = {"train": train_count, "validation": validation_count, "test": test_count}
    splits = _splits(seed, counts)
    demos = []
    for row in sorted(splits["train"], key=lambda r: r["order_id"]):
        candidate = await reference_refund_demonstration(row)
        demos.append(_sft_row(row, candidate["messages"], "reference_policy"))
    return _publish(
        output,
        {
            "schema_version": 1,
            "environment": ENVIRONMENT,
            "source": "reference_policy",
            "seed": seed,
            "counts": counts,
            "selected": len(demos),
            "split_hashes": {name: content_hash(rows) for name, rows in splits.items()},
            "families": dict(
                sorted(Counter(row["family"] for row in splits["train"]).items())
            ),
        },
        {f"{name}.json": rows for name, rows in splits.items()},
        demos,
    )


def _load_data(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads((directory / "data_manifest.json").read_text())
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("environment") != ENVIRONMENT
        or manifest.get("source") != "reference_policy"
    ):
        raise ValueError("Expected a prepared refund-policy-v2 data bundle")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Dataset counts are missing")
    seed = manifest.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("Dataset seed must be a nonnegative integer")
    expected = _splits(seed, counts)
    artifacts = manifest.get("artifacts")
    names = {"train.json", "validation.json", "test.json", "train.jsonl"}
    if not isinstance(artifacts, dict) or set(artifacts) != names:
        raise ValueError("Dataset artifact manifest is incomplete")
    for name in names:
        if (
            hashlib.sha256((directory / name).read_bytes()).hexdigest()
            != artifacts[name]
        ):
            raise ValueError(f"Dataset artifact changed: {name}")
    for name, rows in expected.items():
        if json.loads((directory / f"{name}.json").read_text()) != rows:
            raise ValueError(f"Saved {name} cases are not the canonical split")
    if manifest.get("split_hashes") != {
        name: content_hash(rows) for name, rows in expected.items()
    }:
        raise ValueError("Split hashes do not match canonical cases")
    return manifest, expected


async def _select_refund_candidates(
    candidates: list[Any],
    train: dict[str, dict[str, Any]],
    held_out: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay all candidates and deterministically select one success per case.

    Both publication and study auditing use this rule. Callers must first
    validate collection provenance against the canonical training split.
    """
    audit: list[dict[str, Any]] = []
    valid: dict[str, list[tuple[tuple[int, int, str], int, list[dict[str, str]]]]] = {}
    for index, candidate in enumerate(candidates):
        case_id = candidate.get("case_id") if isinstance(candidate, dict) else None
        entry = {
            "index": index,
            "case_id": case_id,
            "candidate_hash": content_hash(candidate),
            "status": "rejected",
        }
        audit.append(entry)
        if not isinstance(case_id, str) or case_id not in train:
            entry["reason"] = (
                "held_out_case"
                if isinstance(case_id, str) and case_id in held_out
                else "unknown_case"
            )
            continue
        try:
            messages = await replay_refund_candidate(train[case_id], candidate)
        except RejectedTrajectory as exc:
            entry["reason"] = str(exc)
            continue
        entry["status"] = "not_selected"
        rank = (
            sum(m["role"] == "assistant" for m in messages),
            sum(len(m["content"]) for m in messages),
            content_hash(messages),
        )
        valid.setdefault(case_id, []).append((rank, index, messages))
    selected = []
    for case_id in sorted(valid):
        _, index, messages = min(valid[case_id], key=lambda item: (item[0], item[1]))
        audit[index]["status"] = "selected"
        selected.append(_sft_row(train[case_id], messages, "verified_model_rollout"))
    return selected, audit


def _selection_summary(
    train: dict[str, dict[str, Any]],
    selected: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize exact selection coverage, including unsuccessful families."""
    families = {
        family: {
            "available_cases": sum(r["family"] == family for r in train.values()),
            "selected_cases": sum(r["metadata"]["family"] == family for r in selected),
        }
        for family in sorted({r["family"] for r in train.values()})
    }
    return {
        "selected": len(selected),
        "candidates": len(audit),
        "rejected": sum(e["status"] == "rejected" for e in audit),
        "not_selected": sum(e["status"] == "not_selected" for e in audit),
        "families": families,
    }


async def filter_refund_candidates(
    data_dir: Path,
    candidates_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Replay a collection artifact and select at most one success per train case.

    Choose the shortest verified transcript (assistant turns, then characters,
    then canonical hash). Rejected and duplicate candidates remain in the audit.
    An empty harvest publishes its audit and manifest without a training file.
    """
    data_manifest, splits = _load_data(data_dir)
    collection = json.loads(candidates_path.read_text())
    train = {row["order_id"]: row for row in splits["train"]}
    if (
        not isinstance(collection, dict)
        or type(collection.get("schema_version")) is not int
        or collection["schema_version"] != 1
        or collection.get("environment") != ENVIRONMENT
        or type(collection.get("seed")) is not int
        or collection["seed"] != data_manifest["seed"]
    ):
        raise ValueError("Collection must match the prepared benchmark and seed")
    checkpoint = collection.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint["path"]
    ):
        raise ValueError("Collection checkpoint identity is required")
    if collection.get("case_hashes") != {
        key: content_hash(row) for key, row in train.items()
    }:
        raise ValueError("Collection training cases differ from the prepared split")
    if type(collection.get("complete")) is not bool or not isinstance(
        collection.get("candidates"), list
    ):
        raise ValueError("Collection requires candidates and a completion flag")
    collection_hash = content_hash(collection)
    held_out = {
        row["order_id"] for name in ("validation", "test") for row in splits[name]
    }
    selected, audit = await _select_refund_candidates(
        collection["candidates"], train, held_out
    )
    manifest = {
        "schema_version": 1,
        "environment": ENVIRONMENT,
        "source": "verified_model_rollout",
        "seed": data_manifest["seed"],
        **_selection_summary(train, selected, audit),
        "collection_complete": collection["complete"],
        "collection_hash": collection_hash,
        "data_manifest_hash": content_hash(data_manifest),
        "source_checkpoint": collection.get("checkpoint"),
        "selection": "one per case; fewest assistant turns, then characters, then transcript hash",
    }
    return _publish(output, manifest, {"replay_audit.json": audit}, selected)
