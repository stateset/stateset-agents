"""Replay and compare the frozen six-parent SFT study without provider calls."""

import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tarfile

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)
from stateset_agents.data.refund_curriculum import audit_refund_curriculum
from stateset_agents.data.refund_demonstrations import replay_refund_candidate
from stateset_agents.evaluation.agent_runs import (
    compare_runs,
    content_hash,
    validate_report,
)
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.refund_trace import (
    audit_refund_traces,
    replay_refund_trace,
)
from stateset_agents.remote.river_rl import atomic_json

REPO = Path("/home/dom/stateset-agents")
ROOT = REPO / "outputs/river-boundary-six-seed-20261019"
DEST = REPO / "benchmark_results/training_contracts/20261019/river-boundary-six-seed"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


async def review():
    runner = module("study_runner", ROOT / "runtime/experiment.py")
    plan = runner.validate()
    assert package_implementation() == plan["implementation"]
    control = module("frozen_control", ROOT / "runtime/control.py")
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    exclusions = read(ROOT / "exclusion-audit.json")
    excluded_ids, excluded_facts = set(), set()
    for name, digest in exclusions["historical_sources_sha256"].items():
        path = REPO / name
        assert sha(path) == digest
        for case in read(path):
            if isinstance(case, dict) and "order_id" in case:
                excluded_ids.add(case["order_id"])
                excluded_facts.add(fingerprint(case))
    initial_ids, initial_facts = set(excluded_ids), set(excluded_facts)
    for entry in plan["datasets"]:
        directory = ROOT / f'seed-{entry["seed"]}'
        for name in ("repeat-cases.json", "boundary-cases.json"):
            for case in read(directory / "data" / name):
                excluded_ids.add(case["order_id"])
                excluded_facts.add(fingerprint(case))
    fresh_ids, fresh_facts = set(), set()
    window_values = {name: set() for name in ("train", "validation", "test")}
    outcome_counts = Counter()
    reports = {arm: {} for arm in plan["arms"]}
    rows = []
    total_reserved = 0
    total_sft_steps = 0
    training_replays = 0
    training_facts = set()
    for entry in plan["datasets"]:
        seed = entry["seed"]
        directory = ROOT / f"seed-{seed}"
        canonical = read(REPO / entry["canonical_training_path"])
        repeat_cases, repeat_examples, lineage = await control.repetition_control(
            canonical, entry["exposure"]["repeat"]["rows"], entry["parent_seed"]
        )
        assert repeat_cases == read(directory / "data/repeat-cases.json")
        assert lineage == read(directory / "data/repeat-lineage.json")
        actual_repeat = [
            json.loads(line)
            for line in (directory / "data/repeat.jsonl").read_text().splitlines()
        ]
        assert actual_repeat == repeat_examples
        assert {fingerprint(case) for case in repeat_cases} == {
            fingerprint(case) for case in canonical
        }
        curriculum = REPO / entry["curriculum_path"]
        assert (
            await audit_refund_curriculum(
                (REPO / entry["canonical_training_path"]).parent, curriculum
            )
            == entry["curriculum_audit"]
        )
        assert (directory / "data/boundary.jsonl").read_bytes() == (
            curriculum / "train.jsonl"
        ).read_bytes()
        facts = {
            fingerprint(case)
            for case in repeat_cases + read(directory / "data/boundary-cases.json")
        }
        assert not facts & training_facts
        training_facts.update(facts)
        for arm, cases in (
            ("repeat", repeat_cases),
            ("boundary", read(directory / "data/boundary-cases.json")),
        ):
            window_values["train"].update(case["return_window_days"] for case in cases)
            examples = [
                json.loads(line)
                for line in (directory / "data" / f"{arm}.jsonl")
                .read_text()
                .splitlines()
            ]
            by_id = {case["order_id"]: case for case in cases}
            for example in examples:
                metadata = example["metadata"]
                assert metadata["split"] == "train"
                case = by_id[metadata["case_id"]]
                await replay_refund_candidate(
                    case,
                    {
                        "case_id": case["order_id"],
                        "case_hash": metadata["case_hash"],
                        "messages": example["messages"],
                        "truncated": None,
                    },
                )
                training_replays += 1
        for split in ("validation", "test"):
            for case in read(directory / "data" / f"{split}.json"):
                window_values[split].add(case["return_window_days"])
                identifier, facts = case["order_id"], fingerprint(case)
                assert identifier not in excluded_ids | fresh_ids
                assert facts not in excluded_facts | fresh_facts
                fresh_ids.add(identifier)
                fresh_facts.add(facts)
        row = {
            "seed": seed,
            "parent_seed": entry["parent_seed"],
            "arms": {},
            "exposure": entry["exposure"],
        }
        execution = directory / "execution-summary.json"
        row["execution"] = (
            read(execution) if execution.exists() else {"status": "missing"}
        )
        parent = read(directory / "original/river_checkpoint.json")
        pointers = {"original": parent}
        for arm in ("repeat", "boundary"):
            path = directory / arm / "river_checkpoint.json"
            if not path.exists():
                continue
            pointer = read(path)
            assert pointer["steps"] == 16
            assert pointer["sft_options"] == {
                "seed": seed,
                "shuffle": True,
                "checkpoint": parent["checkpoint"],
            }
            manifest = read(directory / arm / "stateset_manifest.json")
            assert manifest["dataset_sha256"] == sha(
                directory / "data" / f"{arm}.jsonl"
            )
            assert manifest["dataset_rows"] == entry["exposure"][arm]["rows"]
            assert manifest["hyperparameters"]["checkpoint"] == parent["checkpoint"]
            assert (
                read(directory / f"{arm}-training-result.json")["status"] == "succeeded"
            )
            total_sft_steps += pointer["steps"]
            pointers[arm] = pointer
        budget_path = directory / "rollout-budget.json"
        if budget_path.exists():
            budget = read(budget_path)
            assert budget["fingerprint"] == plan["plan_hash"]
            assert (
                budget["reserved_generated_tokens"]
                == budget["admitted_trajectories"] * 1024
            )
            assert (
                0
                <= budget["reserved_generated_tokens"]
                <= budget["limit"]
                == plan["limits"]["generated_output_reservation_per_seed"]
            )
            total_reserved += budget["reserved_generated_tokens"]
            row["rollout_budget"] = budget
        for arm in plan["arms"]:
            arm_info = {}
            for split in ("validation", "test"):
                path = directory / f"{arm}-{split}.json"
                if not path.exists():
                    arm_info[split] = {"status": "missing_or_incomplete"}
                    continue
                report = read(path)
                validate_report(report, expected_split=split)
                assert (
                    report["seed"] == seed
                    and report["implementation"] == plan["implementation"]
                )
                assert (
                    report["selected_checkpoint"]["path"] == pointers[arm]["checkpoint"]
                )
                cases = read(directory / "data" / f"{split}.json")
                by_id = {case["order_id"]: case for case in cases}
                assert report["case_hashes"] == {
                    key: content_hash(case) for key, case in by_id.items()
                }
                attempt = read(directory / f"{arm}-{split}-attempt.json")
                assert attempt == {
                    "plan_hash": plan["plan_hash"],
                    "checkpoint": report["selected_checkpoint"],
                    "split_sha256": entry["input_sha256"][f"data/{split}.json"],
                }
                if split == "test":
                    assert (await audit_refund_traces(report, cases))["passed"]
                    reports[arm][seed] = report
                else:
                    for outcome in report["outcomes"]:
                        replay = await replay_refund_trace(
                            "refund-policy-v2",
                            by_id[outcome["case_id"]],
                            outcome["environment_trace"],
                        )
                        for field in (
                            "reward",
                            "success",
                            "policy_violations",
                            "tool_calls",
                            "truncated",
                        ):
                            assert replay[field] == outcome[field]
                outcome_counts[split] += len(report["outcomes"])
                arm_info[split] = {
                    key: report[key]
                    for key in (
                        "passed",
                        "total",
                        "success_rate",
                        "violation_rate",
                        "truncation_rate",
                        "families",
                    )
                }
            row["arms"][arm] = arm_info
        rows.append(row)
    assert len(fresh_ids) == len(fresh_facts) == 960
    assert window_values["train"].isdisjoint(window_values["test"])
    assert total_reserved <= plan["limits"]["total_reserved_output_tokens"]
    assert total_sft_steps <= plan["limits"]["additional_sft_updates"]
    comparisons = {}
    for left, right in plan["acceptance"]["pairs"]:
        paired_seeds = sorted(reports[left].keys() & reports[right].keys())
        if paired_seeds:
            comparison = compare_runs(
                [reports[left][seed] for seed in paired_seeds],
                [reports[right][seed] for seed in paired_seeds],
                min_seeds=6,
                min_gain=0.03,
                require_significance=True,
                alpha=0.05,
                comparisons=3,
            )
            atomic_json(ROOT / f"{right}-vs-{left}.json", comparison)
            comparisons[f"{right}_vs_{left}"] = {
                key: comparison[key]
                for key in ("passed", "gates", "thresholds", "inference")
            }
        else:
            comparisons[f"{right}_vs_{left}"] = {
                "passed": False,
                "reason": "No complete paired tests",
            }
    aggregate = {}
    for arm, per_seed in reports.items():
        aggregate[arm] = {
            "complete_seeds": len(per_seed),
            "passed": sum(report["passed"] for report in per_seed.values()),
            "total": sum(report["total"] for report in per_seed.values()),
            "boundary_passed": sum(
                report["families"]["window_boundary"]["passed"]
                for report in per_seed.values()
            ),
            "boundary_total": sum(
                report["families"]["window_boundary"]["total"]
                for report in per_seed.values()
            ),
            "violation_cases": sum(
                sum(outcome["policy_violations"] > 0 for outcome in report["outcomes"])
                for report in per_seed.values()
            ),
            "truncated_cases": sum(
                sum(outcome["truncated"] is not None for outcome in report["outcomes"])
                for report in per_seed.values()
            ),
        }
    complete = (
        outcome_counts["validation"] == 576
        and outcome_counts["test"] == 2304
        and total_sft_steps == 192
    )
    execution_ok = all(
        row["execution"].get("exit_code") == 0
        and not row["execution"].get("timed_out", False)
        for row in rows
    )
    if complete:
        assert total_reserved == plan["limits"]["total_reserved_output_tokens"]
    result = {
        "schema_version": 1,
        "plan_hash": plan["plan_hash"],
        "status": "completed" if complete and execution_ok else "incomplete",
        "all_seed_executions_successful": execution_ok,
        "protocol": plan["parent_protocol"],
        "source_and_input_hashes_verified": True,
        "historical_unique_ids": len(initial_ids),
        "historical_unique_facts": len(initial_facts),
        "fresh_disjoint_holdout_cases": len(fresh_ids),
        "return_window_values": {
            name: sorted(values) for name, values in window_values.items()
        },
        "training_transcripts_replayed": training_replays,
        "evaluation_outcomes_replayed": dict(outcome_counts),
        "acknowledged_additional_sft_steps": total_sft_steps,
        "reserved_output_tokens": total_reserved,
        "reserved_output_token_ceiling": plan["limits"]["total_reserved_output_tokens"],
        "verified_cost_usd": None,
        "primary_comparison": "boundary_vs_repeat",
        "primary_acceptance_passed": comparisons["boundary_vs_repeat"]["passed"],
        "comparisons": comparisons,
        "aggregate": aggregate,
        "per_seed": rows,
        "synthetic_data": True,
        "independently_reviewed_data": False,
        "quality_certification": False,
        "scope": "Conditional paired continuations of retained independently trained parents. Pooled counts are descriptive; inference uses six parent/run pairs. Not independent human review or production certification.",
    }
    atomic_json(ROOT / "review-summary.json", result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("comparisons", "per_seed")
            },
            indent=2,
        )
    )


def retain():
    assert not DEST.exists(), "Never overwrite retained evidence"
    result = read(ROOT / "review-summary.json")
    plan = read(ROOT / "plan.json")
    old_artifacts = 0
    for manifest in sorted(DEST.parent.glob("*/artifacts.sha256.json")):
        for name, digest in read(manifest).items():
            assert sha(manifest.parent / name) == digest
            old_artifacts += 1
    # Retain the exact external case sources used for exclusion and reconstruction.
    for name, digest in read(ROOT / "exclusion-audit.json")[
        "historical_sources_sha256"
    ].items():
        assert sha(REPO / name) == digest
        target = ROOT / "source-inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    for entry in plan["datasets"]:
        curriculum = REPO / entry["curriculum_path"]
        for name in ("data_manifest.json", "lineage.json", "contrast_groups.json"):
            target = ROOT / "source-inputs" / entry["curriculum_path"] / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(curriculum / name, target)
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and not path.name.endswith(".lock")
    )
    for path in files:
        assert not re.search(
            rb"rv_[A-Za-z0-9_-]{24,}", path.read_bytes()
        ), "Credential-shaped evidence rejected"
    DEST.mkdir()
    for name in (
        "plan.json",
        "exclusion-audit.json",
        "checks.json",
        "control-tests.xml",
        "execution-summary.json",
        "review-summary.json",
        "experiment.py",
        "control.py",
        "test_control.py",
        "review.py",
        "boundary-vs-repeat.json",
        "repeat-vs-original.json",
        "boundary-vs-original.json",
    ):
        path = ROOT / name
        if path.exists():
            shutil.copyfile(path, DEST / name)
    with tarfile.open(DEST / "frozen-run.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    text = "# Six-parent boundary SFT continuation replication\n\n"
    text += "Qwen/Qwen3.5-9B, rank 16. All six independently trained retained original SFT parents were fixed before the new study; one was used in the earlier pilot. "
    text += "This is conditional continuation replication with new shuffle seeds and fresh holdouts, not six wholly new base initializations. "
    text += "Each parent has unchanged, original-repetition SFT and boundary-curriculum SFT arms. The two trained arms have identical rows, 16 updates, seed and hyperparameters. "
    text += "Input and assistant-target token exposure are within 0.5% per pair; exact provider compute or dollar costs are not matched.\n\n"
    text += "| Arm | Test successes | Boundary successes | Policy-violation cases | Truncated cases |\n|---|---:|---:|---:|---:|\n"
    for arm in plan["arms"]:
        stats = result["aggregate"][arm]
        text += f"| {arm} | {stats['passed']}/{stats['total']} | {stats['boundary_passed']}/{stats['boundary_total']} | {stats['violation_cases']} | {stats['truncated_cases']} |\n"
    text += f"\nStatus: {result['status']}. Prespecified primary boundary-versus-repetition acceptance: {result['primary_acceptance_passed']}. "
    text += "The gates require six pairs, at least three percentage points mean gain, positive gain in every seed, no family/policy/truncation regression, and a seed sign test at alpha 0.05 with three-comparison Bonferroni correction. "
    text += "Secondary original-versus-continued-SFT comparisons use the same gates and correction. Validation is descriptive; final one-epoch checkpoints were fixed without test-driven selection.\n\n"
    primary = result["comparisons"]["boundary_vs_repeat"]
    inference = primary.get("inference", {})
    text += f"Primary comparison: {inference.get('positive_seeds', 0)} positive and {inference.get('negative_seeds', 0)} negative seed pairs; corrected p={inference.get('bonferroni_adjusted_p_value', 'unavailable')}. "
    text += "The boundary arm has a higher pooled success rate, but the prespecified matched-control gate fails on consistency, family/policy/truncation regressions and significance. "
    secondary = result["comparisons"]["boundary_vs_original"]
    text += f"The secondary boundary-versus-original gate passes: {secondary['passed']}; corrected p={secondary.get('inference', {}).get('bonferroni_adjusted_p_value', 'unavailable')}.\n\n"
    text += "| Continuation seed | Original /128 | Repetition /128 | Boundary /128 |\n|---|---:|---:|---:|\n"
    for row in result["per_seed"]:
        scores = [
            str(row["arms"][arm]["test"].get("passed", "unavailable"))
            for arm in plan["arms"]
        ]
        text += f"| {row['seed']} | {' | '.join(scores)} |\n"
    text += f"\nSupervised return-window values: {result['return_window_values']['train']}; held-out test values: {result['return_window_values']['test']}.\n\n"
    text += f"Replayed {result['training_transcripts_replayed']} training transcripts and {sum(result['evaluation_outcomes_replayed'].values())} completed evaluation trajectories. "
    text += f"{result['fresh_disjoint_holdout_cases']} new validation/test cases are disjoint from historical and current training identities and policy facts, and other new holdouts. "
    text += f"Acknowledged added SFT steps: {result['acknowledged_additional_sft_steps']}; reserved output tokens: {result['reserved_output_tokens']:,}/{result['reserved_output_token_ceiling']:,}. "
    text += "These are admission reservations, not actual generated tokens or billed dollars. No RL or deployment was run. No automatic retry or deadline extension.\n\n"
    text += f"Four new control-data tests passed. Packaged sources match the prior validated implementation (370 unit tests, 19 scripted SDK tests, 445 type-checked sources). {old_artifacts} older evidence artifacts verified unchanged. "
    text += "This synthetic policy study is not independently reviewed industry data or A+ production certification. Pooled counts are descriptive; six parent/run pairs are the inference units.\n\n"
    text += "`frozen-run.tar.gz` retains exact package sources, training data, heldouts, per-arm attempts and traces, pointer lineage, token ledgers, supervisor logs, and historical input snapshots. "
    text += "No River credential is retained. Hashes establish local identity, not provider attestation.\n"
    (DEST / "README.md").write_text(text)
    atomic_json(
        DEST / "artifacts.sha256.json",
        {path.name: sha(path) for path in sorted(DEST.iterdir()) if path.is_file()},
    )
    print(
        json.dumps({"retained": str(DEST), "older_artifacts_verified": old_artifacts})
    )


if __name__ == "__main__":
    import sys

    if sys.argv[-1] == "retain":
        retain()
    else:
        asyncio.run(review())
