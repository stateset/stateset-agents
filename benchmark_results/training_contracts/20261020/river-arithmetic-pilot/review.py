"""Reconstruct the arithmetic experiment and replay all completed trajectories."""

import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from stateset_agents.data.refund_curriculum import (
    ARITHMETIC_POLICY,
    audit_refund_curriculum,
)
from stateset_agents.data.refund_demonstrations import replay_refund_candidate
from stateset_agents.data.refund_arithmetic_stress import refund_arithmetic_stress_cases
from stateset_agents.data.refund_stress import refund_policy_stress_cases
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.evaluation.agent_runs import content_hash, validate_report
from stateset_agents.evaluation.refund_trace import replay_refund_trace
from stateset_agents.training.river_refund import EVALUATION_SETTINGS

REPO = Path("/home/dom/stateset-agents")
ROOT = Path(__file__).parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    spec = importlib.util.spec_from_file_location(
        "review_" + name, ROOT / "runtime" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def main():
    runner, control, gates = load("experiment"), load("control"), load("gates")
    plan = runner.validate()
    execution = read(ROOT / "execution-summary.json")
    prior = read(ROOT / "prior-evidence.json")
    for name, digest in prior.items():
        assert sha(REPO / name) == digest
    canonical = REPO / "outputs/river-six-seed-recovery-20261019/seed-2026101901/data"
    assert (
        await audit_refund_curriculum(
            canonical, ROOT / "curriculum", policy=ARITHMETIC_POLICY
        )
        == plan["curriculum_audit"]
    )
    assert read(ROOT / "curriculum/data_manifest.json") == plan["curriculum_manifest"]
    assert (ROOT / "curriculum/train.json").read_bytes() == (
        ROOT / "data/arithmetic-cases.json"
    ).read_bytes()
    assert (ROOT / "curriculum/train.jsonl").read_bytes() == (
        ROOT / "data/arithmetic.jsonl"
    ).read_bytes()
    candidate = read(ROOT / "data/arithmetic-cases.json")
    previous = (
        REPO
        / "outputs/river-boundary-stability-20261019/attempt-2/seed-2026102901/curriculum"
    )
    cases, examples, lineage = await control.repetition_control(
        read(previous / "train.json"),
        dict(Counter(r["family"] for r in candidate)),
        runner.SEED,
    )
    assert cases == read(ROOT / "data/control-cases.json")
    assert lineage == read(ROOT / "data/control-lineage.json")
    assert examples == [
        json.loads(line)
        for line in (ROOT / "data/control.jsonl").read_text().splitlines()
    ]
    assert sha(ROOT / "historical-sources.json") == plan["historical_sources_sha256"]
    history = read(ROOT / "historical-sources.json")
    excluded, historical_holdouts = [], set()
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    for name, digest in history.items():
        path = REPO / name
        assert sha(path) == digest
        rows = read(path)
        if isinstance(rows, list):
            orders = [
                r for r in rows if isinstance(r, dict) and "return_window_days" in r
            ]
            excluded.extend(orders)
            if path.name in ("validation.json", "test.json", "primary-test.json"):
                historical_holdouts.update(fingerprint(r) for r in orders)
    assert not historical_holdouts & {fingerprint(r) for r in candidate}
    excluded.extend(candidate + cases)
    for split, count, windows, quantum, repetitions in (
        ("validation", 32, (5, 18, 40), 10, 1),
        ("test", 128, (8, 25, 75), 100, 2),
    ):
        ordinary = refund_policy_benchmark(
            split, count=count, seed=plan["evaluation_case_seed"]
        )
        ids, facts = {r["order_id"] for r in excluded}, {
            fingerprint(r) for r in excluded
        }
        assert not ids & {r["order_id"] for r in ordinary}
        assert not facts & {fingerprint(r) for r in ordinary}
        excluded.extend(ordinary)
        windows_cases = refund_policy_stress_cases(
            split, seed=plan["evaluation_case_seed"], windows=windows, excluded=excluded
        )
        excluded.extend(windows_cases)
        arithmetic = refund_arithmetic_stress_cases(
            split,
            seed=plan["evaluation_case_seed"],
            windows=windows,
            quantum=quantum,
            repetitions=repetitions,
            excluded=excluded,
        )
        excluded.extend(arithmetic)
        assert sorted(
            ordinary + windows_cases + arithmetic, key=lambda r: r["order_id"]
        ) == read(ROOT / "data" / f"{split}.json")
    replayed_teachers = 0
    for arm in runner.ARMS:
        by_id = {c["order_id"]: c for c in read(ROOT / "data" / f"{arm}-cases.json")}
        for row in [
            json.loads(l)
            for l in (ROOT / "data" / f"{arm}.jsonl").read_text().splitlines()
        ]:
            meta = row["metadata"]
            assert meta["split"] == "train"
            await replay_refund_candidate(
                by_id[meta["case_id"]],
                {
                    "case_id": meta["case_id"],
                    "case_hash": meta["case_hash"],
                    "messages": row["messages"],
                    "truncated": None,
                },
            )
            replayed_teachers += 1
    parent = read(ROOT / "original/river_checkpoint.json")
    checkpoints, training, updates = {"initial": parent["checkpoint"]}, {}, 0
    for arm in runner.ARMS:
        path = ROOT / arm / "river_checkpoint.json"
        if not path.exists():
            training[arm] = {"status": "no_completed_checkpoint"}
            continue
        pointer = read(path)
        manifest = read(ROOT / arm / "stateset_manifest.json")
        expected_updates = plan["exposure"][arm]["updates"]
        assert pointer["steps"] == expected_updates
        assert pointer["sft_options"] == {
            "checkpoint": parent["checkpoint"],
            "seed": runner.SEED,
            "shuffle": True,
        }
        assert manifest["dataset_sha256"] == plan["input_sha256"][f"data/{arm}.jsonl"]
        assert manifest["dataset_rows"] == plan["exposure"][arm]["rows"]
        assert manifest["hyperparameters"]["checkpoint"] == parent["checkpoint"]
        assert math.isfinite(pointer["final_loss"])
        result = read(ROOT / f"{arm}-training-result.json")
        assert result["status"] == "succeeded"
        checkpoints[arm] = pointer["checkpoint"]
        updates += pointer["steps"]
        training[arm] = {
            "status": "succeeded",
            "updates": pointer["steps"],
            "final_loss": pointer["final_loss"],
            "provider_reported_training_tokens": pointer["tokens"],
            "cost_usd": result["cost_usd"],
            "duration_s": result["duration_s"],
        }
    assert len(set(checkpoints.values())) == len(checkpoints)
    summaries, replayed = {}, Counter()
    decisions = {}
    for split in ("validation", "test"):
        cases = read(ROOT / "data" / f"{split}.json")
        by_id = {r["order_id"]: r for r in cases}
        reports, summaries[split] = {}, {}
        for arm in runner.ARMS:
            path = ROOT / f"{arm}-{split}.json"
            if not path.exists():
                summaries[split][arm] = {
                    "status": "missing_or_incomplete",
                    "attempt_sealed": (ROOT / f"{arm}-{split}-attempt.json").exists(),
                }
                continue
            report = read(path)
            computed = validate_report(report, expected_split=split)
            assert all(report[k] == v for k, v in computed.items())
            assert report["implementation"] == plan["implementation"]
            assert (
                report["settings"] == EVALUATION_SETTINGS
                and report["seed"] == runner.SEED
            )
            assert report["case_hashes"] == {
                key: content_hash(c) for key, c in by_id.items()
            }
            assert report["selected_checkpoint"]["path"] == checkpoints[arm]
            assert read(ROOT / f"{arm}-{split}-audit.json")["passed"]
            assert read(ROOT / f"{arm}-{split}-attempt.json") == {
                "plan_hash": plan["plan_hash"],
                "checkpoint": report["selected_checkpoint"],
                "split_sha256": plan["input_sha256"][f"data/{split}.json"],
            }
            for o in report["outcomes"]:
                checked = await replay_refund_trace(
                    "refund-policy-v2",
                    by_id[o["case_id"]],
                    o["environment_trace"],
                    truncation_reward=report["settings"]["truncation_reward"],
                )
                assert all(
                    checked[k] == o[k]
                    for k in (
                        "case_hash",
                        "reward",
                        "success",
                        "policy_violations",
                        "tool_calls",
                        "truncated",
                    )
                )
                replayed[split] += 1
            suites = {"ordinary": [], "window": [], "arithmetic": []}
            for o in report["outcomes"]:
                c = by_id[o["case_id"]]
                suite = (
                    "arithmetic"
                    if "arithmetic_stress" in c
                    else "window" if "stress" in c else "ordinary"
                )
                suites[suite].append(o)
            summaries[split][arm] = {
                "status": "complete",
                "passed": report["passed"],
                "total": report["total"],
                "violation_cases": sum(
                    o["policy_violations"] > 0 for o in report["outcomes"]
                ),
                "truncated_cases": sum(
                    o["truncated"] is not None for o in report["outcomes"]
                ),
                "suites": {
                    k: {"passed": sum(o["success"] for o in rows), "total": len(rows)}
                    for k, rows in suites.items()
                },
            }
            reports[arm] = report
        if len(reports) == 2:
            expected = gates.assess(
                reports["control"],
                reports["arithmetic"],
                cases,
                split=split,
                require_arithmetic_gain=split == "test",
            )
            actual = read(ROOT / f"{split}-decision.json")
            assert actual == {"plan_hash": plan["plan_hash"], **expected}
            decisions[split] = actual
    budget = read(ROOT / "rollout-budget.json")
    assert budget["fingerprint"] == plan["plan_hash"]
    assert (
        budget["reserved_generated_tokens"]
        == budget["admitted_trajectories"] * 1024
        <= runner.TOKEN_LIMIT
    )
    if "validation" in decisions and not decisions["validation"]["passed"]:
        assert not any(
            (ROOT / f"{arm}-test-attempt.json").exists() for arm in runner.ARMS
        )
        assert not replayed["test"]
    result = (
        read(ROOT / "pilot-result.json")
        if (ROOT / "pilot-result.json").exists()
        else None
    )
    if execution["exit_code"] == 0:
        assert (
            result is not None and updates == plan["limits"]["additional_sft_updates"]
        )
        assert replayed["validation"] == 316
        assert replayed["test"] == (616 if result["test_attempted"] else 0)
        assert budget["admitted_trajectories"] == sum(replayed.values())
    summary = {
        "schema_version": 1,
        "scope": "single_parent_arithmetic_pilot_replay",
        "execution": execution,
        "pilot_result": result,
        "training": training,
        "decisions": decisions,
        "additional_sft_updates_acknowledged": updates,
        "teacher_transcripts_replayed": replayed_teachers,
        "evaluations_replayed": dict(replayed),
        "arms": summaries,
        "budget": budget,
        "prior_evidence_artifacts_verified_unchanged": len(prior),
        "provider_requests_during_review": 0,
        "quality_certification": False,
        "limitation": plan["scope"],
    }
    (ROOT / "review-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "pilot_result",
                    "additional_sft_updates_acknowledged",
                    "teacher_transcripts_replayed",
                    "evaluations_replayed",
                    "arms",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
