"""Prepare, run once, and audit an exploratory River agent-system comparison.

Preparation and audit are offline. ``launch`` uses paid inference, a hidden
credential prompt, a frozen package snapshot, and a finite subprocess deadline.
No training or model promotion is performed. See docs/REFUND_RESOLUTION_TOOL.md.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.data.refund_arithmetic_stress import refund_arithmetic_stress_cases
from stateset_agents.data.refund_stress import refund_policy_stress_cases
from stateset_agents.evaluation.agent_runs import (
    content_hash,
    evaluation_report,
    validate_report,
)
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.refund_delegation_trace import replay_delegation_trace
from stateset_agents.evaluation.refund_trace import replay_refund_trace
from stateset_agents.remote.river_environment import (
    river_environment_factory,
    trajectory_case_identity,
    trajectory_environment_trace,
    trajectory_truncation,
)
from stateset_agents.remote.river_refund_delegation import (
    river_refund_delegation_factory,
    trajectory_delegation_trace,
)
from stateset_agents.remote.river_rl import atomic_json, exclusive_run
from stateset_agents.remote.rollout_budget import RolloutAdmissionBudget
from stateset_agents.training.river_refund import EVALUATION_SETTINGS
from stateset_agents.utils.async_calls import closing_async_generator

ARMS = ("raw", "delegated")
MODEL = "Qwen/Qwen3.5-9B"
SEED = 2026109001
TOKEN_LIMIT = 2 * (74 + 170) * 1024


def read(path: Path) -> Any:
    """Read a retained JSON artifact."""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(path: Path, value: Any) -> None:
    """Persist an exclusive attempt marker before provider activity."""
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(root: Path, repo: Path) -> dict[str, Any]:
    """Freeze inputs, fresh case selection, limits, and source before inference."""
    root.mkdir(parents=True, exist_ok=False)
    prior = repo / "outputs/river-arithmetic-pilot-20261020"
    history = read(prior / "historical-sources.json")
    for name in ("control-cases", "arithmetic-cases", "validation", "test"):
        path = prior / "data" / f"{name}.json"
        history[str(path.relative_to(repo))] = sha(path)
    excluded = refund_policy_benchmark("train", count=128, seed=20261020)
    for name, digest in history.items():
        path = repo / name
        if sha(path) != digest:
            raise ValueError("Historical dataset changed before preparation")
        rows = read(path)
        if isinstance(rows, list):
            excluded.extend(
                row
                for row in rows
                if isinstance(row, dict) and "return_window_days" in row
            )
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    rejected = []
    for attempt in range(100):
        case_seed = SEED + attempt * 100
        working, proposed = list(excluded), {}
        try:
            for split, count, window, quantum in (
                ("validation", 32, 18, 10),
                ("test", 128, 25, 100),
            ):
                ordinary = refund_policy_benchmark(split, count=count, seed=case_seed)
                ids, facts = {r["order_id"] for r in working}, {
                    fingerprint(r) for r in working
                }
                if (
                    ids & {r["order_id"] for r in ordinary}
                    or facts & {fingerprint(r) for r in ordinary}
                    or len({fingerprint(r) for r in ordinary}) != len(ordinary)
                ):
                    raise ValueError("Ordinary case overlap")
                working.extend(ordinary)
                windows = refund_policy_stress_cases(
                    split, seed=case_seed, windows=(window,), excluded=working
                )
                working.extend(windows)
                arithmetic = refund_arithmetic_stress_cases(
                    split,
                    seed=case_seed,
                    windows=(window,),
                    quantum=quantum,
                    excluded=working,
                )
                working.extend(arithmetic)
                proposed[split] = sorted(
                    ordinary + windows + arithmetic, key=lambda r: r["order_id"]
                )
        except ValueError as exc:
            if "overlap" not in str(exc).lower():
                raise
            rejected.append({"seed": case_seed, "reason": str(exc)})
            continue
        break
    else:
        raise ValueError("No fresh input seed found in 100 prospective candidates")
    assert {split: len(rows) for split, rows in proposed.items()} == {
        "validation": 74,
        "test": 170,
    }
    pointer = prior / "arithmetic/river_checkpoint.json"
    checkpoint = read(pointer)
    assert checkpoint["base_model"] == MODEL
    atomic_json(root / "checkpoint.json", checkpoint)
    atomic_json(root / "historical-sources.json", history)
    for split, rows in proposed.items():
        atomic_json(root / f"{split}.json", rows)
    implementation = package_implementation()
    runtime = root / "runtime"
    for name, digest in implementation["files"].items():
        source = repo / "stateset_agents" / name
        assert sha(source) == digest
        target = runtime / "stateset_agents" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(Path(__file__), runtime / "experiment.py")
    plan = {
        "schema_version": 1,
        "kind": "exploratory-agent-system-comparison",
        "base_model": MODEL,
        "seed": SEED,
        "case_seed": case_seed,
        "checkpoint": checkpoint["checkpoint"],
        "arms": list(ARMS),
        "implementation": implementation,
        "input_sha256": {
            name: sha(root / name)
            for name in (
                "checkpoint.json",
                "historical-sources.json",
                "validation.json",
                "test.json",
            )
        },
        "runner_sha256": sha(runtime / "experiment.py"),
        "evaluation": EVALUATION_SETTINGS,
        "counts": {split: len(rows) for split, rows in proposed.items()},
        "case_generation": {
            "rejected_input_seeds": rejected,
            "maximum_prospective_seeds": 100,
        },
        "limits": {
            "reserved_output_tokens": TOKEN_LIMIT,
            "wall_seconds": 1200,
            "request_timeout_seconds": 180,
            "provider_retries": False,
            "attempts_per_split_arm": 1,
            "concurrency_per_arm": 6,
        },
        "acceptance": "Delegated success >= raw success and zero delegated violations/truncations in every overall/family/suite/suite-family cell. Validation must pass before either test arm is sampled.",
        "attribution": "Same fixed weights, different agent/tool protocols and prompts. Tool-selected amounts/priority are system correctness, not model learning. Intent and executed call counts reported separately.",
        "training_updates": 0,
        "quality_certification": False,
        "cost_usd": None,
        "dollar_cap_enforced": False,
        "previous_arithmetic_test": "308 cases excluded and never sampled by this study",
        "freshness": {
            "historical_source_files": len(history),
            "overlapping_case_ids": 0,
            "overlapping_fact_fingerprints": 0,
        },
    }
    plan["plan_hash"] = content_hash(plan)
    atomic_json(root / "plan.json", plan)
    return {
        "prepared": True,
        "root": str(root),
        "counts": plan["counts"],
        "plan_hash": plan["plan_hash"],
    }


def validate(root: Path) -> dict[str, Any]:
    """Require the frozen package, runner, checkpoint, and cases to match plan."""
    plan = read(root / "plan.json")
    assert (
        content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
        == plan["plan_hash"]
    )
    assert package_implementation() == plan["implementation"]
    assert sha(Path(__file__)) == plan["runner_sha256"]
    for name, digest in plan["input_sha256"].items():
        assert sha(root / name) == digest
    for name, digest in plan["implementation"]["files"].items():
        assert sha(root / "runtime/stateset_agents" / name) == digest
    return plan


def suite(row: dict[str, Any]) -> str:
    """Label a case for prespecified descriptive safety cells."""
    if "arithmetic_stress" in row:
        return "arithmetic"
    if "stress" in row:
        return "window"
    return "ordinary"


def gate(reports: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact paired nonregression with zero delegated safety failures."""
    for arm, expected in (
        ("raw", "refund-policy-v2"),
        ("delegated", "refund-delegation-v1"),
    ):
        if reports[arm].get("environment") != expected:
            raise ValueError("Comparison arm has the wrong environment protocol")
    by_id = {r["order_id"]: r for r in cases}
    summaries = {
        arm: validate_report(reports[arm], expected_split=reports[arm]["split"])
        for arm in ARMS
    }
    outcomes = {
        arm: {r["case_id"]: r for r in reports[arm]["outcomes"]} for arm in ARMS
    }
    assert all(set(rows) == set(by_id) for rows in outcomes.values())
    cells: dict[str, list[str]] = {"overall": list(by_id)}
    for key, row in by_id.items():
        for cell in (
            f"family:{row['family']}",
            f"suite:{suite(row)}",
            f"suite-family:{suite(row)}:{row['family']}",
        ):
            cells.setdefault(cell, []).append(key)
    result = {}
    for name, ids in cells.items():
        raw = sum(outcomes["raw"][key]["success"] for key in ids)
        delegated = sum(outcomes["delegated"][key]["success"] for key in ids)
        violations = sum(
            outcomes["delegated"][key]["policy_violations"] > 0 for key in ids
        )
        truncated = sum(
            outcomes["delegated"][key]["truncated"] is not None for key in ids
        )
        result[name] = {
            "total": len(ids),
            "raw_passed": raw,
            "delegated_passed": delegated,
            "delegated_violations": violations,
            "delegated_truncated": truncated,
            "passed": delegated >= raw and violations == truncated == 0,
        }
    return {
        "passed": all(cell["passed"] for cell in result.values()),
        "cells": result,
        "summaries": summaries,
    }


async def evaluate(
    root: Path,
    plan: dict[str, Any],
    session: Any,
    renderer: Any,
    checkpoint: Any,
    cases: list[dict[str, Any]],
    split: str,
    arm: str,
    budget: RolloutAdmissionBudget,
) -> dict[str, Any]:
    """Evaluate one sealed arm once, then require complete independent replay."""
    from river_client import rl

    async def reserve() -> None:
        await asyncio.to_thread(budget.reserve)

    factory = (
        river_refund_delegation_factory(before_reset=reserve)
        if arm == "delegated"
        else river_environment_factory(
            RefundPolicyEnvironment,
            before_reset=reserve,
            truncation_reward=-1,
            record_trace=True,
        )
    )
    engine = rl.RolloutEngine(
        rl.CheckpointSampler(
            session,
            base_model=MODEL,
            checkpoint=checkpoint,
            tokenizer=renderer.tokenizer,
        ),
        env=factory,
        renderer=renderer,
        budget=rl.Budget(
            **{k: v for k, v in plan["evaluation"].items() if k.startswith("max_")}
        ),
        schedule=rl.Schedule(concurrency=6),
        temperature=0,
        top_p=1.0,
        top_k=-1,
        seed=SEED,
    )
    by_id, outcomes = {row["order_id"]: row for row in cases}, []
    reader = (
        trajectory_delegation_trace
        if arm == "delegated"
        else trajectory_environment_trace
    )
    async with closing_async_generator(
        engine.rollout(
            cases,
            group_size=1,
            completion=rl.GroupCompletion(mode="wait", min_members=1),
        )
    ) as groups:
        async for group in groups:
            for traj in group:
                outcomes.append(
                    {
                        **trajectory_case_identity(traj, cases=by_id),
                        "environment_trace": reader(traj),
                        "reward": traj.reward,
                        "success": traj.metrics.get("task_success", 0) == 1,
                        "policy_violations": traj.metrics.get("policy_violations", 0),
                        "tool_calls": traj.metrics.get("tool_calls", 0),
                        "executed_tool_calls": traj.metrics.get(
                            "executed_tool_calls", traj.metrics.get("tool_calls", 0)
                        ),
                        "duplicate_intents": traj.metrics.get("duplicate_intents", 0),
                        "generated_tokens": traj.generated_tokens,
                        "elapsed_seconds": traj.elapsed,
                        "truncated": trajectory_truncation(traj),
                    }
                )
                atomic_json(
                    root / f"{arm}-{split}-progress.json",
                    {"completed": len(outcomes), "expected": len(cases)},
                )
    report = evaluation_report(
        environment=(
            "refund-delegation-v1" if arm == "delegated" else "refund-policy-v2"
        ),
        base_model=MODEL,
        seed=SEED,
        cases=by_id,
        settings=plan["evaluation"],
        checkpoint=asdict(checkpoint),
        outcomes=outcomes,
        case_families={row["order_id"]: row["family"] for row in cases},
        implementation=plan["implementation"],
        split=split,
    )
    await audit_report(report, cases)
    atomic_json(root / f"{arm}-{split}.json", report)
    return report


async def audit_report(report: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    """Bind every report field to the supplied cases and replayed transitions."""
    validate_report(report, expected_split=report["split"])
    by_id = {row["order_id"]: row for row in cases}
    assert report["case_hashes"] == {
        key: content_hash(row) for key, row in by_id.items()
    }
    for outcome in report["outcomes"]:
        case, trace = by_id[outcome["case_id"]], outcome["environment_trace"]
        if report["environment"] == "refund-delegation-v1":
            replay = await replay_delegation_trace(case, trace)
            assert replay["executed_tool_calls"] == outcome["executed_tool_calls"]
            assert replay["duplicate_intents"] == outcome["duplicate_intents"]
        else:
            assert report["environment"] == "refund-policy-v2"
            replay = await replay_refund_trace("refund-policy-v2", case, trace)
            assert replay["tool_calls"] == outcome["executed_tool_calls"]
        for field in (
            "case_hash",
            "reward",
            "success",
            "policy_violations",
            "tool_calls",
            "truncated",
        ):
            assert replay[field] == outcome[field]


async def run_evaluations(
    root: Path,
    plan: dict[str, Any],
    session: Any,
    renderer: Any,
    checkpoint: Any,
    budget: RolloutAdmissionBudget,
) -> None:
    """Seal both arms before sampling each split and stop on validation failure."""
    for split in ("validation", "test"):
        cases = read(root / f"{split}.json")
        for arm in ARMS:
            seal(
                root / f"{arm}-{split}-attempt.json",
                {
                    "plan_hash": plan["plan_hash"],
                    "checkpoint": asdict(checkpoint),
                    "case_sha256": plan["input_sha256"][f"{split}.json"],
                },
            )
        values = await asyncio.gather(
            *(
                evaluate(
                    root, plan, session, renderer, checkpoint, cases, split, arm, budget
                )
                for arm in ARMS
            )
        )
        decision = gate(dict(zip(ARMS, values, strict=True)), cases)
        atomic_json(root / f"{split}-decision.json", decision)
        print(
            json.dumps(
                {
                    "split": split,
                    "passed": decision["passed"],
                    "counts": {
                        arm: report["passed"]
                        for arm, report in zip(ARMS, values, strict=True)
                    },
                }
            ),
            flush=True,
        )
        if split == "validation" and not decision["passed"]:
            atomic_json(
                root / "result.json",
                {
                    "status": "validation_blocked_test_not_attempted",
                    "test_attempted": False,
                    "accepted": False,
                    "training_updates": 0,
                },
            )
            return
    atomic_json(
        root / "result.json",
        {
            "status": "completed",
            "test_attempted": True,
            "accepted": decision["passed"],
            "training_updates": 0,
            "quality_certification": False,
        },
    )


def worker(root: Path) -> None:
    """Own SDK resources inside a finite, externally supervised process."""
    import river_client as river

    from stateset_agents.remote.river_renderers import get_river_renderer
    from stateset_agents.remote.river_runtime import (
        require_model_access,
        require_native_runtime,
    )

    plan = validate(root)
    require_native_runtime()
    with exclusive_run(root / "worker"):
        seal(root / "worker-attempt.json", {"plan_hash": plan["plan_hash"]})
        budget = RolloutAdmissionBudget(
            root / "rollout-budget.json",
            limit=TOKEN_LIMIT,
            trajectory_tokens=1024,
            fingerprint=plan["plan_hash"],
            create=True,
        )
        with closing(
            river.Client(
                api_key=os.environ["RIVER_API_KEY"], timeout=180, enable_retries=False
            )
        ) as client:
            require_model_access(client, MODEL)
            renderer = get_river_renderer(MODEL, thinking=False)
            checkpoint = river.Checkpoint(
                path=plan["checkpoint"], step=44, checkpoint_type="inference"
            )
            with client.session(
                experiment="stateset-delegation-pilot",
                role="fixed-checkpoint-evaluation",
            ) as session:
                asyncio.run(
                    run_evaluations(root, plan, session, renderer, checkpoint, budget)
                )


def launch(root: Path) -> dict[str, Any]:
    """Launch one bounded paid attempt; never persist the credential or retry."""
    from stateset_agents.remote.river_runtime import river_preflight

    plan = validate(root)
    assert not (
        root / "execution-attempt.json"
    ).exists(), "Never retry a sealed comparison"
    credential = (
        os.environ.get("RIVER_API_KEY")
        or getpass.getpass("River credential (hidden): ").strip()
    )
    assert credential
    os.environ["RIVER_API_KEY"] = credential
    seal(
        root / "execution-attempt.json",
        {"plan_hash": plan["plan_hash"], "started_at_unix": time.time()},
    )
    preflight = river_preflight([MODEL], live=True, timeout_seconds=20)
    atomic_json(root / "preflight-live.json", preflight)
    assert preflight["passed"]
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "runtime"),
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": "1",
    }
    started, timed_out = time.monotonic(), False
    with (root / "live-run.txt").open("x") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(root / "runtime/experiment.py"),
                "worker",
                "--output",
                str(root),
            ],
            cwd=root / "runtime",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=plan["limits"]["wall_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=10)
    summary = {
        "exit_code": code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "plan_hash": plan["plan_hash"],
        "result": (
            read(root / "result.json") if (root / "result.json").exists() else None
        ),
    }
    atomic_json(root / "execution-summary.json", summary)
    if code:
        raise SystemExit(code)
    return summary


async def audit(root: Path) -> dict[str, Any]:
    """Replay saved outcomes and recompute every gate without provider calls."""
    plan = validate(root)
    total = 0
    for split in ("validation", "test"):
        if not (root / f"{split}-decision.json").exists():
            continue
        cases = read(root / f"{split}.json")
        reports = {arm: read(root / f"{arm}-{split}.json") for arm in ARMS}
        for arm, report in reports.items():
            expected = (
                "refund-delegation-v1" if arm == "delegated" else "refund-policy-v2"
            )
            assert report["environment"] == expected
            assert report["implementation"] == plan["implementation"]
            assert report["selected_checkpoint"]["path"] == plan["checkpoint"]
            assert report["settings"] == plan["evaluation"]
            await audit_report(report, cases)
            total += len(report["outcomes"])
        assert content_hash(gate(reports, cases)) == content_hash(
            read(root / f"{split}-decision.json")
        )
    assert total >= 148
    return {
        "passed": True,
        "replayed_outcomes": total,
        "training_updates": 0,
        "provider_calls": 0,
    }


def main() -> None:
    """Dispatch preparation, paid execution, or offline audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "validate", "launch", "worker", "audit")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "prepare":
        result = prepare(root, args.repo.resolve())
    elif args.command == "validate":
        result = {"valid": True, "plan_hash": validate(root)["plan_hash"]}
    elif args.command == "audit":
        result = asyncio.run(audit(root))
    elif args.command == "launch":
        result = launch(root)
    else:
        worker(root)
        return
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
