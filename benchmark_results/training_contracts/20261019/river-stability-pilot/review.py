"""Replay this pilot and reconstruct its decisions without provider requests."""

import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from stateset_agents.data.refund_demonstrations import replay_refund_candidate
from stateset_agents.data.refund_curriculum import (
    PARAMETER_POLICY,
    audit_refund_curriculum,
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
    assert (
        ROOT / "execution-summary.json"
    ).exists(), "Do not finalize a live execution"
    execution = read(ROOT / "execution-summary.json")
    prior = read(ROOT / "prior-evidence.json")
    for name, digest in prior.items():
        assert sha(REPO / name) == digest
    data = ROOT / "data"
    parameter = read(data / "parameter-cases.json")
    ready_path = (
        REPO / "outputs/river-boundary-stability-20261019/attempt-3/readiness.json"
    )
    assert sha(ready_path) == plan["source_readiness_sha256"]
    entry = read(ready_path)["parents"][0]
    source_curriculum = REPO / entry["curriculum_path"]
    assert (
        await audit_refund_curriculum(
            REPO / entry["canonical_data"], source_curriculum, policy=PARAMETER_POLICY
        )
    ) == entry["curriculum_audit"]
    assert (data / "parameter-cases.json").read_bytes() == (
        source_curriculum / "train.json"
    ).read_bytes()
    assert (data / "parameter.jsonl").read_bytes() == (
        source_curriculum / "train.jsonl"
    ).read_bytes()
    legacy = read(
        REPO
        / "outputs/river-curriculum-review-20261019/seed-2026101901/curriculum/train.json"
    )
    cases, examples, lineage = await control.repetition_control(
        legacy, dict(Counter(r["family"] for r in parameter)), runner.SEED
    )
    assert cases == read(data / "control-cases.json")
    assert lineage == read(data / "control-lineage.json")
    assert examples == [
        json.loads(line) for line in (data / "control.jsonl").read_text().splitlines()
    ]
    training_replays = 0
    for arm in ("control", "parameter"):
        by_id = {r["order_id"]: r for r in read(data / f"{arm}-cases.json")}
        for row in [
            json.loads(line)
            for line in (data / f"{arm}.jsonl").read_text().splitlines()
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
            training_replays += 1
    parent = read(ROOT / "original/river_checkpoint.json")
    checkpoints, updates, training = {"original": parent["checkpoint"]}, 0, {}
    for arm in ("control", "parameter"):
        pointer_path = ROOT / arm / "river_checkpoint.json"
        if not pointer_path.exists():
            training[arm] = {"status": "no_completed_checkpoint"}
            continue
        pointer = read(pointer_path)
        manifest = read(ROOT / arm / "stateset_manifest.json")
        assert pointer["steps"] == 32
        assert pointer["sft_options"] == {
            "checkpoint": parent["checkpoint"],
            "seed": runner.SEED,
            "shuffle": True,
        }
        assert manifest["dataset_sha256"] == sha(data / f"{arm}.jsonl")
        assert manifest["dataset_rows"] == 512
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
            "duration_s": result["duration_s"],
            "cost_usd": result["cost_usd"],
        }
    assert len(set(checkpoints.values())) == len(checkpoints)
    reports, summaries, replayed = {}, {}, Counter()
    for split in ("validation", "test"):
        cases = read(data / f"{split}.json")
        by_id = {r["order_id"]: r for r in cases}
        reports[split], summaries[split] = {}, {}
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
            assert all(report[key] == value for key, value in computed.items())
            assert report["case_hashes"] == {
                key: content_hash(row) for key, row in by_id.items()
            }
            assert report["implementation"] == plan["implementation"]
            assert report["settings"] == EVALUATION_SETTINGS
            assert report["selected_checkpoint"]["path"] == checkpoints[arm]
            assert report["seed"] == runner.SEED
            assert read(ROOT / f"{arm}-{split}-audit.json")["passed"]
            assert read(ROOT / f"{arm}-{split}-attempt.json") == {
                "plan_hash": plan["plan_hash"],
                "checkpoint": report["selected_checkpoint"],
                "split_sha256": plan["input_sha256"][f"data/{split}.json"],
            }
            for outcome in report["outcomes"]:
                checked = await replay_refund_trace(
                    "refund-policy-v2",
                    by_id[outcome["case_id"]],
                    outcome["environment_trace"],
                    truncation_reward=report["settings"]["truncation_reward"],
                )
                assert all(
                    checked[field] == outcome[field]
                    for field in (
                        "case_hash",
                        "reward",
                        "success",
                        "policy_violations",
                        "tool_calls",
                        "truncated",
                    )
                )
                replayed[split] += 1
            reports[split][arm] = report
            strata = {
                "ordinary": [
                    o for o in report["outcomes"] if "stress" not in by_id[o["case_id"]]
                ],
                "stress": [
                    o for o in report["outcomes"] if "stress" in by_id[o["case_id"]]
                ],
            }
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
                "distributions": {
                    k: {"passed": sum(o["success"] for o in rows), "total": len(rows)}
                    for k, rows in strata.items()
                },
            }
        decision_path = ROOT / f"{split}-decision.json"
        if len(reports[split]) == 3:
            expected = {
                "parameter_vs_control": gates.assess(
                    reports[split]["control"],
                    reports[split]["parameter"],
                    cases,
                    split=split,
                    min_gain=0.03 if split == "test" else 0,
                ),
                "parameter_vs_original": gates.assess(
                    reports[split]["original"],
                    reports[split]["parameter"],
                    cases,
                    split=split,
                    min_gain=0,
                ),
            }
            actual = read(decision_path)
            assert actual["comparisons"] == expected
            assert actual["passed"] == all(v["passed"] for v in expected.values())
    decision = (
        read(ROOT / "validation-decision.json")
        if (ROOT / "validation-decision.json").exists()
        else None
    )
    if decision is not None and not decision["passed"]:
        assert not any(
            (ROOT / f"{arm}-test-attempt.json").exists() for arm in runner.ARMS
        )
        assert replayed["test"] == 0
    budget = read(ROOT / "rollout-budget.json")
    assert budget["fingerprint"] == plan["plan_hash"]
    assert (
        budget["reserved_generated_tokens"]
        == budget["admitted_trajectories"] * 1024
        <= runner.TOKEN_LIMIT
    )
    result = (
        read(ROOT / "pilot-result.json")
        if (ROOT / "pilot-result.json").exists()
        else None
    )
    if execution["exit_code"] == 0:
        assert result is not None and updates == 64 and replayed["validation"] == 312
        assert (result["test_attempted"] and replayed["test"] == 600) or (
            not result["test_attempted"]
            and replayed["test"] == 0
            and decision is not None
            and not decision["passed"]
        )
        assert budget["admitted_trajectories"] == sum(replayed.values())
    summary = {
        "schema_version": 1,
        "scope": "replayed_single_parent_stability_pilot",
        "execution": execution,
        "pilot_result": result,
        "training": training,
        "additional_sft_updates_acknowledged": updates,
        "teacher_transcripts_replayed": training_replays,
        "evaluations_replayed": dict(replayed),
        "arms": summaries,
        "budget": budget,
        "provider_requests_during_review": 0,
        "prior_evidence_artifacts_verified_unchanged": len(prior),
        "quality_certification": False,
        "limitation": plan["inference_unit"],
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
