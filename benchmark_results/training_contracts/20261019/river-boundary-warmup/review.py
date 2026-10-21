"""Replay and retain the frozen boundary warm-start experiment without API calls."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tarfile

from stateset_agents.evaluation.agent_runs import content_hash, validate_report
from stateset_agents.evaluation.checkpoint_selection import validate_validation_cases
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.refund_trace import (
    audit_refund_traces,
    replay_refund_trace,
    select_replayed_validation_checkpoint,
)
from stateset_agents.remote.river_rl import atomic_json

REPO = Path("/home/dom/stateset-agents")
ROOT = REPO / "outputs/river-boundary-warmup-20261019"
DEST = REPO / "benchmark_results/training_contracts/20261019/river-boundary-warmup"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def review():
    plan = read(ROOT / "plan.json")
    assert (
        content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
        == plan["plan_hash"]
    )
    assert package_implementation() == plan["implementation"]
    assert sha(ROOT / "runtime/experiment.py") == plan["runner_sha256"]
    assert sha(ROOT / "runtime/evaluation.py") == plan["evaluation_helper_sha256"]
    assert (
        sha(ROOT / "original/river_checkpoint.json") == plan["original_pointer_sha256"]
    )
    for name, digest in plan["input_artifacts_sha256"].items():
        assert sha(ROOT / name) == digest
    assert sha(ROOT / "exclusion-audit.json") == plan["exclusion_audit_sha256"]
    assert sha(ROOT / "data-check.json") == plan["data_check_sha256"]
    for name, digest in read(ROOT / "exclusion-audit.json")["sources_sha256"].items():
        assert sha(REPO / name) == digest
    result = {
        "schema_version": 1,
        "plan_hash": plan["plan_hash"],
        "source_and_inputs_verified": True,
        "execution": read(ROOT / "execution-summary.json"),
        "synthetic_data": True,
        "independently_reviewed_data": False,
        "quality_gain_verified": False,
        "generalized_rl_gain_verified": False,
        "verified_cost_usd": None,
        "reserved_output_tokens": 0,
        "budgets": {},
        "arms": {},
    }
    original = read(ROOT / "original/river_checkpoint.json")
    warmup = None
    if (ROOT / "warmup/river_checkpoint.json").exists():
        warmup = read(ROOT / "warmup/river_checkpoint.json")
        assert warmup["steps"] == plan["expected_sft_updates"]
        assert warmup["sft_options"] == {
            "seed": plan["seed"],
            "shuffle": True,
            "checkpoint": original["checkpoint"],
        }
        manifest = read(ROOT / "warmup/stateset_manifest.json")
        assert manifest["dataset_sha256"] == sha(ROOT / "data/train.jsonl")
        assert manifest["dataset_rows"] == 255
        assert manifest["hyperparameters"]["checkpoint"] == original["checkpoint"]
        assert read(ROOT / "sft-result.json")["status"] == "succeeded"
        result["sft_warmup"] = {
            "acknowledged_training_steps": warmup["steps"],
            "starting_checkpoint": original["checkpoint"],
            "checkpoint": warmup["checkpoint"],
            "final_loss": warmup["final_loss"],
            "training_source_verified": True,
        }
    for name, path, limit in (
        ("external_tests", ROOT / "external-budget.json", 262144),
        ("native_rl", ROOT / "rl/rollout_budget.json", 622592),
    ):
        if not path.exists():
            continue
        budget = read(path)
        expected_fingerprint = (
            plan["plan_hash"]
            if name == "external_tests"
            else content_hash(read(ROOT / "rl/run_manifest.json"))
        )
        assert budget["fingerprint"] == expected_fingerprint
        assert budget["limit"] == limit
        assert (
            budget["reserved_generated_tokens"]
            == budget["admitted_trajectories"] * 1024
            <= limit
        )
        result["budgets"][name] = budget
        result["reserved_output_tokens"] += budget["reserved_generated_tokens"]
    assert (
        result["reserved_output_tokens"]
        <= plan["limits"]["total_reserved_output_tokens"]
    )
    test_cases = read(ROOT / "data/test.json")
    reports = {}
    for arm, path in (
        ("original", ROOT / "original-test.json"),
        ("warmup", ROOT / "warmup-test.json"),
        ("rl", ROOT / "rl/test_results.json"),
    ):
        if not path.exists():
            result["arms"][arm] = {"status": "incomplete"}
            continue
        report = read(path)
        validate_report(report)
        assert report["implementation"] == plan["implementation"]
        assert report["seed"] == plan["seed"]
        assert report["case_hashes"] == {
            case["order_id"]: content_hash(case) for case in test_cases
        }
        replay = await audit_refund_traces(report, test_cases)
        assert replay["passed"]
        if arm == "rl":
            attempt = read(ROOT / "rl/test_attempt.json")
            assert attempt["test_split_hash"] == plan["split_hashes"]["test"]
            assert attempt["selected_checkpoint"] == report["selected_checkpoint"]
            assert report["test_attempt_hash"] == content_hash(attempt)
        else:
            attempt = read(ROOT / f"{arm}-test-attempt.json")
            assert attempt["test_hash"] == plan["split_hashes"]["test"]
            assert attempt["plan_hash"] == plan["plan_hash"]
            assert attempt["checkpoint"] == report["selected_checkpoint"]
            pointer = original if arm == "original" else warmup
            assert report["selected_checkpoint"]["path"] == pointer["checkpoint"]
        reports[arm] = report
        result["arms"][arm] = {
            "status": "completed",
            "passed": report["passed"],
            "total": report["total"],
            "success_rate": report["success_rate"],
            "violation_rate": report["violation_rate"],
            "truncation_rate": report["truncation_rate"],
            "families": report["families"],
            "test_replay": replay,
            "checkpoint": report["selected_checkpoint"],
        }
    if (ROOT / "rl/run_manifest.json").exists():
        assert warmup is not None
        spec = importlib.util.spec_from_file_location(
            "frozen_experiment", ROOT / "runtime/experiment.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from stateset_agents.training.river_refund import run_manifest

        splits = {
            name: read(ROOT / "data" / f"{name}.json")
            for name in ("train", "validation", "test")
        }
        manifest = read(ROOT / "rl/run_manifest.json")
        assert manifest == run_manifest(module.rl_args(), splits)
        assert manifest["checkpoint"] == warmup["checkpoint"]
        binding = read(ROOT / "stage-binding.json")
        assert binding == {
            "plan_hash": plan["plan_hash"],
            "run_manifest_hash": content_hash(manifest),
            "warmup_pointer_sha256": sha(ROOT / "warmup/river_checkpoint.json"),
        }
        for split, rows in splits.items():
            assert read(ROOT / "rl" / f"{split}.json") == rows
        validation = (
            read(ROOT / "rl/validation_results.json")
            if (ROOT / "rl/validation_results.json").exists()
            else []
        )
        cases = {row["order_id"]: row for row in splits["validation"]}
        for entry in validation:
            validate_validation_cases(entry, cases=cases)
            for outcome in entry["outcomes"]:
                replay = await replay_refund_trace(
                    "refund-policy-v2",
                    cases[outcome["case_id"]],
                    outcome["environment_trace"],
                )
                assert replay["reward"] == outcome["reward"]
        result["validation_outcomes_replayed"] = sum(
            len(entry["outcomes"]) for entry in validation
        )
        if (ROOT / "rl/training_activity.json").exists():
            result["rl_training_activity"] = read(ROOT / "rl/training_activity.json")
        if (ROOT / "rl/validation_selection.json").exists():
            selected = await select_replayed_validation_checkpoint(
                validation,
                steps=4,
                cases=cases,
                environment="refund-policy-v2",
                training_progress=read(ROOT / "rl/training_metrics.json"),
                recovery_receipts=(
                    read(ROOT / "rl/recovery_receipts.json")
                    if (ROOT / "rl/recovery_receipts.json").exists()
                    else {}
                ),
                run_manifest_hash=content_hash(manifest),
            )
            selection = read(ROOT / "rl/validation_selection.json")
            assert selected["selection_activity"] == selection
            assert reports["rl"]["selected_checkpoint"] == selected["checkpoint"]
            assert reports["rl"]["validation_selection_hash"] == content_hash(selection)
            result["validation_selection"] = selection
            atomic_json(
                ROOT / "selected/river_checkpoint.json",
                {
                    "provider": "river",
                    "base_model": plan["base_model"],
                    "lora": warmup["lora"],
                    "checkpoint": selected["checkpoint"]["path"],
                    "plan_hash": plan["plan_hash"],
                    "source_checkpoint": warmup["checkpoint"],
                    "selected_validation_step": selected["step"],
                    "selected_training_activity": selection[
                        "selected_training_activity"
                    ],
                    "quality_gain_verified": False,
                    "note": "Hosted River pointer; validation selects within the boundary-SFT/RL lineage. No test-driven deployment promotion.",
                },
            )
    if len(reports) >= 2:
        comparisons = {}
        for left, right in (
            ("original", "warmup"),
            ("warmup", "rl"),
            ("original", "rl"),
        ):
            if left not in reports or right not in reports:
                continue
            old = {row["case_id"]: row for row in reports[left]["outcomes"]}
            new = {row["case_id"]: row for row in reports[right]["outcomes"]}
            comparisons[f"{right}_vs_{left}"] = {
                "baseline_passed": reports[left]["passed"],
                "candidate_passed": reports[right]["passed"],
                "success_gain": reports[right]["success_rate"]
                - reports[left]["success_rate"],
                "wins": sum(
                    new[key]["success"] and not old[key]["success"] for key in old
                ),
                "regressions": sum(
                    old[key]["success"] and not new[key]["success"] for key in old
                ),
            }
        result["paired_comparisons"] = comparisons
    result["status"] = "completed" if len(reports) == 3 else "incomplete"
    if (ROOT / "evaluation-completion-plan.json").exists():
        completion = read(ROOT / "evaluation-completion-plan.json")
        assert (
            content_hash({k: v for k, v in completion.items() if k != "plan_hash"})
            == completion["plan_hash"]
        )
        assert completion["original_plan_hash"] == plan["plan_hash"]
        assert (
            completion["training_calls"]
            == completion["token_allocation_increase"]
            == completion["deadline_extension_seconds"]
            == 0
        )
        assert (
            sha(ROOT / "runtime/complete_evaluation.py") == completion["runner_sha256"]
        )
        for name, digest in completion["source_hashes"].items():
            assert sha(ROOT / name) == digest
        assert not (ROOT / "rl/test_attempt.json").exists()
        assert not (ROOT / "rl/test_results.json").exists()
        activity = read(ROOT / "rl/training_activity.json")
        assert (
            activity["observed_optimizer_updates"]
            == activity["unknown_update_batches"]
            == 0
        )
        assert (
            activity["observed_skipped_batches"] == activity["completed_batches"] == 3
        )
        result["arms"]["rl"] = {
            "status": "stopped_no_optimizer_updates_test_unattempted",
            "training_activity": activity,
        }
        result["evaluation_completion"] = {
            "plan_hash": completion["plan_hash"],
            "protocol_deviation": completion["protocol_deviation"],
            "execution": read(ROOT / "evaluation-completion-summary.json"),
            "additional_training_calls": 0,
            "token_allocation_increase": 0,
            "deadline_extension_seconds": 0,
        }
        if set(reports) == {"original", "warmup"}:
            assert result["evaluation_completion"]["execution"]["exit_code"] == 0
            assert not result["evaluation_completion"]["execution"]["timed_out"]
            for arm in ("original", "warmup"):
                assert (
                    read(ROOT / f"{arm}-test-attempt.json")["completion_plan_hash"]
                    == completion["plan_hash"]
                )
            result["status"] = "warmup_evaluated_rl_stopped_without_updates"
    atomic_json(ROOT / "review-summary.json", result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("arms", "validation_selection", "budgets")
            },
            indent=2,
        )
    )


def retain():
    assert not DEST.exists(), "Never overwrite retained evidence"
    summary = read(ROOT / "review-summary.json")
    old_files = 0
    for manifest in sorted(DEST.parent.glob("*/artifacts.sha256.json")):
        for name, digest in read(manifest).items():
            assert sha(manifest.parent / name) == digest
            old_files += 1
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
        "data-check.json",
        "execution-summary.json",
        "review-summary.json",
        "checks.json",
        "preflight-tests.xml",
        "experiment.py",
        "review.py",
    ):
        shutil.copyfile(ROOT / name, DEST / name)
    with tarfile.open(DEST / "frozen-run.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    with tarfile.open(DEST / "source-and-tests.tar.gz", "w:gz") as archive:
        for name in (
            "stateset_agents/remote/river.py",
            "stateset_agents/remote/job.py",
            "tests/unit/test_remote_river_executor.py",
            "tests/unit/test_river_client_lifecycle.py",
            "docs/RIVER_PROVIDER.md",
        ):
            archive.add(REPO / name, arcname=name)
    rows = ["| Checkpoint | Successes | Boundary successes |", "|---|---:|---:|"]
    for arm in ("original", "warmup", "rl"):
        item = summary["arms"][arm]
        rows.append(
            f"| {arm} | {item['passed']}/128 | {item['families']['window_boundary']['passed']}/16 |"
            if "passed" in item
            else f"| {arm} | Not tested | Not tested |"
        )
    text = "# River boundary SFT warm-start and RL pilot\n\n"
    text += "Separate user-authorized synthetic pilot on Qwen/Qwen3.5-9B, rank 16. Original checkpoint fixed to the first prior study seed. "
    text += "Warm-start SFT uses 255 training-only verified curriculum demonstrations for one epoch (16 batches), then four planned native CISPO RL batches. "
    text += "The 64 fresh validation and 128 fresh test cases are disjoint by identity and policy facts from retained historical cases. "
    if "evaluation_completion" in summary:
        text += "Native RL stopped at three consecutive skipped batches, with zero optimizer updates. Its test remains unattempted. "
        text += "The original failed execution and protocol are preserved. An explicit evaluation-completion plan was frozen before test sampling to run the two already planned fixed SFT arms once. "
        text += "This changes stage ordering after the RL stop; it adds no training, output-token allocation, or deadline extension.\n\n"
    else:
        text += "Validation selects the RL-lineage checkpoint before any test sampling. Fixed original and warmup SFT checkpoints are then tested once on paired cases.\n\n"
    text += "\n".join(rows) + "\n\n"
    text += "Single-seed observations do not establish reliable gains or A+ certification. Original versus warmup also changes training exposure; it cannot isolate curriculum effects. "
    text += "Data is synthetic and has no independent human review. Check the family, policy-violation and truncation results in `review-summary.json`.\n\n"
    text += f"Execution: {summary['status']}; supervisor exit {summary['execution']['exit_code']}; timeout {summary['execution']['timed_out']}. "
    text += f"Generated-output admission reservations: {summary['reserved_output_tokens']:,} against 884,736. These are not actual usage or billed dollars. "
    text += "SFT has a separate one-epoch, 16-batch cap with input/assistant token counts in `data-check.json`. No operational retries.\n\n"
    text += "Validation and completed test traces were replayed offline; checkpoint lineage, split hashes, run binding and admission ledgers verified. "
    text += f"370 unit tests and 19 scripted SDK tests passed; 445 sources passed full type checking. {old_files} older evidence artifacts verified unchanged.\n\n"
    text += "`frozen-run.tar.gz` retains package sources, input data, exact actions, token ledgers, supervisor logs and checkpoint pointers. "
    text += "`source-and-tests.tar.gz` retains the explicit River SFT checkpoint option, regression tests and documentation. "
    text += "The warmup checkpoint pointer is reusable through StateSet; no deployment or test-driven promotion was created. No River credential is retained. Hashes are local provenance, not provider attestation.\n"
    (DEST / "README.md").write_text(text)
    atomic_json(
        DEST / "artifacts.sha256.json",
        {path.name: sha(path) for path in sorted(DEST.iterdir()) if path.is_file()},
    )
    print(json.dumps({"retained": str(DEST), "previous_artifacts_verified": old_files}))


if __name__ == "__main__":
    import sys

    if sys.argv[-1] == "retain":
        retain()
    else:
        asyncio.run(review())
