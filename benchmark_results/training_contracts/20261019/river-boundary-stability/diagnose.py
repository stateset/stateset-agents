"""Replay previous validation evidence using its exact frozen implementation."""

import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO = Path("/home/dom/stateset-agents")
SOURCE = REPO / "outputs/river-boundary-six-seed-20261019"
ROOT = Path(__file__).parent
sys.path.insert(0, str(SOURCE / "runtime"))

from stateset_agents.evaluation.agent_runs import content_hash, validate_report
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.refund_trace import replay_refund_trace


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failure_kind(case, steps):
    applied = False
    for step in steps:
        action = json.loads(step["action"]["content"])
        if step["metrics"]["policy_violations"]:
            if action["tool"] == "finish" and action["args"]:
                return "finish_arguments"
            if action["args"].get("order_id") != case["order_id"]:
                return "order_identity"
            if applied:
                return "repeated_resolution"
            if (
                action["tool"] == "deny"
                and action["args"].get("reason") == "outside_window"
                and case["days_since_delivery"] <= case["return_window_days"]
            ):
                return "window_comparison"
            if action["tool"] == "refund" and action["args"].get("amount_cents") != (
                case["paid_cents"] - case["refunded_cents"]
            ):
                return "refund_amount"
            return "other_policy_action"
        applied |= any('"applied"' in m["content"] for m in step["observations"])
    return "incomplete"


async def main():
    implementation = package_implementation()
    assert implementation == read(SOURCE / "plan.json")["implementation"]
    for name, digest in implementation["files"].items():
        assert sha(SOURCE / "runtime/stateset_agents" / name) == digest
    totals = {arm: Counter() for arm in ("original", "repeat", "boundary")}
    seeds, inputs = [], {}
    for seed in range(2026101931, 2026101937):
        directory = SOURCE / f"seed-{seed}"
        casepath = directory / "data/validation.json"
        cases = {c["order_id"]: c for c in read(casepath)}
        inputs[str(casepath.relative_to(REPO))] = sha(casepath)
        entry = {"seed": seed, "arms": {}}
        for arm in totals:
            path = directory / f"{arm}-validation.json"
            report = read(path)
            validate_report(report, expected_split="validation")
            assert report["implementation"] == implementation
            assert report["case_hashes"] == {
                key: content_hash(c) for key, c in cases.items()
            }
            inputs[str(path.relative_to(REPO))] = sha(path)
            failures = []
            for outcome in report["outcomes"]:
                case = cases[outcome["case_id"]]
                replay = await replay_refund_trace(
                    report["environment"],
                    case,
                    outcome["environment_trace"],
                    truncation_reward=report["settings"]["truncation_reward"],
                )
                assert all(
                    replay[k] == outcome[k]
                    for k in (
                        "case_hash",
                        "success",
                        "reward",
                        "policy_violations",
                        "tool_calls",
                        "truncated",
                    )
                )
                totals[arm]["total"] += 1
                totals[arm]["passed"] += replay["success"]
                if not replay["success"]:
                    kind = failure_kind(case, outcome["environment_trace"]["steps"])
                    totals[arm]["failure_" + kind] += 1
                    failures.append(
                        {
                            "case_id": case["order_id"],
                            "case_hash": content_hash(case),
                            "family": case["family"],
                            "kind": kind,
                            "trace_hash": replay["trace_hash"],
                            "return_window_days": case["return_window_days"],
                            "days_since_delivery": case["days_since_delivery"],
                        }
                    )
            entry["arms"][arm] = {
                "passed": report["passed"],
                "total": report["total"],
                "failures": failures,
            }
        seeds.append(entry)
    result = {
        "schema_version": 1,
        "scope": "validation_only_diagnosis",
        "provider_requests": 0,
        "test_outcomes_read": 0,
        "implementation": implementation,
        "source_sha256": inputs,
        "totals": totals,
        "seeds": seeds,
        "replayed_trajectories": 576,
        "limitations": [
            "Failure categories are descriptive, not causal evidence.",
            "New data choices are exploratory after the earlier study.",
            "No prior test labels or outcomes supply new supervision.",
        ],
    }
    (ROOT / "validation-diagnosis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"replayed": 576, "totals": totals}, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
