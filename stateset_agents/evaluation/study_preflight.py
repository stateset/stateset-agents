"""Prepare study inputs and inspect local readiness without provider requests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from stateset_agents.data.refund_demonstrations import (
    RejectedTrajectory,
    _load_data,
    prepare_refund_data,
    replay_refund_candidate,
)
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.agent_study import StudyConfig, build_plan
from stateset_agents.remote.river_rl import exclusive_run
from stateset_agents.remote.river_runtime import (
    inspect_native_runtime as _runtime_checks,
)
from stateset_agents.remote.river_runtime import (
    native_runtime_issues,
)


def _load_plan(directory: Path) -> tuple[dict[str, Any], StudyConfig]:
    plan = json.loads((directory / "study_plan.json").read_text())
    if not isinstance(plan, dict) or not isinstance(plan.get("config"), dict):
        raise ValueError("Invalid study plan")
    config = StudyConfig(**plan["config"])
    if plan != build_plan(config):
        raise ValueError(
            "Study plan or installed implementation changed; use a new study"
        )
    return plan, config


async def _check_data(
    directory: Path, config: StudyConfig, seed: int
) -> dict[str, Any]:
    manifest, splits = _load_data(directory)
    if manifest["seed"] != seed or splits != config.splits(seed):
        raise ValueError("Prepared dataset differs from the planned seed or splits")
    rows = [
        json.loads(line)
        for line in (directory / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if (
        type(manifest.get("selected")) is not int
        or manifest["selected"] != len(rows)
        or len(rows) != config.train_count
    ):
        raise ValueError("Prepared training example count differs from the plan")
    expected = {case["order_id"]: case for case in splits["train"]}
    seen = set()
    for row in rows:
        meta = row.get("metadata") if isinstance(row, dict) else None
        if not isinstance(meta, dict) or not isinstance(meta.get("case_id"), str):
            raise ValueError("Prepared training example has no case identity")
        case_id = meta["case_id"]
        if (
            case_id not in expected
            or case_id in seen
            or meta.get("split") != "train"
            or meta.get("source") != "reference_policy"
            or meta.get("case_hash") != content_hash(expected[case_id])
            or meta.get("family") != expected[case_id]["family"]
        ):
            raise ValueError(
                "Prepared training example does not match its planned case"
            )
        seen.add(case_id)
        try:
            await replay_refund_candidate(
                expected[case_id],
                {
                    "case_id": case_id,
                    "case_hash": meta["case_hash"],
                    "messages": row.get("messages"),
                    "truncated": None,
                },
            )
        except RejectedTrajectory as exc:
            raise ValueError(
                f"Prepared training example failed sandbox replay: {exc}"
            ) from exc
    return {
        "seed": seed,
        "train_examples": manifest["selected"],
        "replayed_examples": len(rows),
    }


async def prepare_study_data(directory: Path) -> dict[str, Any]:
    """Prepare every planned dataset, reusing only matching, intact bundles.

    Validate existing bundles before creating any missing ones. The study lock
    serializes this operation; interruption leaves complete bundles reusable.
    No provider credentials, SDK imports, or shell commands are required.
    """
    with exclusive_run(directory):
        plan, config = _load_plan(directory)
        existing = {}
        for seed in config.seeds:
            destination = directory / f"seed-{seed}" / "data"
            if destination.is_symlink():
                raise ValueError("Study dataset must not be a symbolic link")
            if destination.exists() and (
                not destination.is_dir() or any(destination.iterdir())
            ):
                existing[seed] = await _check_data(destination, config, seed)
        datasets = []
        for seed in config.seeds:
            if seed in existing:
                datasets.append({**existing[seed], "status": "reused"})
                continue
            destination = directory / f"seed-{seed}" / "data"
            await prepare_refund_data(
                destination,
                seed=seed,
                train_count=config.train_count,
                validation_count=config.validation_count,
                test_count=config.test_count,
            )
            datasets.append(
                {**await _check_data(destination, config, seed), "status": "created"}
            )
        return {
            "schema_version": 1,
            "plan_hash": plan["plan_hash"],
            "datasets": datasets,
            "provider_requests": 0,
        }


def preflight_study(directory: Path) -> dict[str, Any]:
    """Run offline preflight outside an event loop; never execute provider jobs.

    Async applications should await ``preflight_study_async`` instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(preflight_study_async(directory))
    raise RuntimeError("Await preflight_study_async() inside an active event loop")


async def preflight_study_async(directory: Path) -> dict[str, Any]:
    """Inspect local runtime and replay training examples without provider calls."""
    plan, config = _load_plan(directory)
    runtime = _runtime_checks()
    issues = native_runtime_issues(runtime)
    datasets = []
    for seed in config.seeds:
        try:
            data = await _check_data(directory / f"seed-{seed}" / "data", config, seed)
            datasets.append({**data, "passed": True})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            datasets.append({"seed": seed, "passed": False, "reason": str(exc)})
            issues.append(f"Seed {seed}: prepare or repair its planned dataset")
    return {
        "schema_version": 1,
        "plan_hash": plan["plan_hash"],
        "local_checks_passed": not issues,
        "runtime": runtime,
        "datasets": datasets,
        "issues": issues,
        "stages": {
            "total": len(plan["stages"]),
            "paid": sum(stage["paid"] for stage in plan["stages"]),
        },
        "limits": plan["limits"],
        "unverified": [
            "Provider authentication, model availability, funding and pricing",
            "Tokenizer availability and provider connectivity",
            "Provider billing and optimizer/input-token costs",
        ],
        "provider_requests": 0,
        "scope": "Local preparation only; passing does not authorize spending or prove learning gains.",
    }
