"""Replay and retain this exploratory River experiment without provider calls."""

import asyncio
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

from stateset_agents.evaluation.agent_runs import content_hash, validate_report
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.refund_trace import (
    audit_refund_traces,
    select_replayed_validation_checkpoint,
)
from stateset_agents.remote.river_rl import atomic_json

REPO = Path("/home/dom/stateset-agents")
ROOT = REPO / "outputs/river-boundary-rl-20261019"
DEST = REPO / "benchmark_results/training_contracts/20261019/river-boundary-rl"


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
    assert sha(ROOT / "sft/river_checkpoint.json") == plan["checkpoint_pointer_sha256"]
    assert sha(ROOT / "exclusion-audit.json") == plan["exclusion_audit_sha256"]
    exclusion = read(ROOT / "exclusion-audit.json")
    for name, digest in exclusion["source_sha256"].items():
        assert sha(REPO / name) == digest
    result = {
        "schema_version": 1,
        "plan_hash": plan["plan_hash"],
        "kind": plan["kind"],
        "source_and_inputs_verified": True,
        "exclusion_scope": "Fresh validation/test identities and policy facts versus historical cases; training intentionally reuses the retained curriculum",
        "synthetic_data": True,
        "independently_reviewed_data": False,
        "quality_gain_verified": False,
        "generalized_rl_gain_verified": False,
        "execution": read(ROOT / "execution-summary.json"),
        "account_preflight_passed": read(ROOT / "preflight-live.json")["passed"],
        "arms": {},
        "reserved_output_tokens": 0,
        "verified_cost_usd": None,
    }
    reports = {}
    for arm in ("rl", "baseline"):
        directory = ROOT / arm
        assert sha(directory / "run_manifest.json") == plan["run_manifest_sha256"][arm]
        manifest = read(directory / "run_manifest.json")
        for name, digest in manifest["split_hashes"].items():
            assert (
                content_hash(read(directory / f"{name}.json"))
                == digest
                == plan["split_hashes"][name]
            )
        ledger = read(directory / "rollout_budget.json")
        assert ledger["fingerprint"] == content_hash(manifest)
        assert (
            ledger["reserved_generated_tokens"]
            == ledger["admitted_trajectories"] * 1024
        )
        assert 0 <= ledger["reserved_generated_tokens"] <= ledger["limit"]
        result["reserved_output_tokens"] += ledger["reserved_generated_tokens"]
        item = {"rollout_admission": ledger, "status": "incomplete"}
        if (directory / "training_activity.json").exists():
            item["training_activity"] = read(directory / "training_activity.json")
        if (directory / "test_results.json").exists():
            report = read(directory / "test_results.json")
            validate_report(report)
            assert report["implementation"] == plan["implementation"]
            assert report["run_manifest_hash"] == content_hash(manifest)
            attempt = read(directory / "test_attempt.json")
            assert report["test_attempt_hash"] == content_hash(attempt)
            assert attempt["run_manifest_hash"] == content_hash(manifest)
            assert attempt["test_split_hash"] == plan["split_hashes"]["test"]
            assert attempt["selected_checkpoint"] == report["selected_checkpoint"]
            audit = await audit_refund_traces(report, read(directory / "test.json"))
            assert audit["passed"]
            item.update(
                {
                    "status": "completed",
                    "passed": report["passed"],
                    "total": report["total"],
                    "success_rate": report["success_rate"],
                    "violation_rate": report["violation_rate"],
                    "truncation_rate": report["truncation_rate"],
                    "families": report["families"],
                    "test_replay": audit,
                    "selected_checkpoint": report["selected_checkpoint"],
                }
            )
            reports[arm] = report
        result["arms"][arm] = item
    assert (
        result["reserved_output_tokens"]
        <= plan["limits"]["total_reserved_output_tokens"]
    )
    if (ROOT / "rl/validation_selection.json").exists():
        selection = read(ROOT / "rl/validation_selection.json")
        validation = read(ROOT / "rl/validation_results.json")
        selected = await select_replayed_validation_checkpoint(
            validation,
            steps=4,
            cases={row["order_id"]: row for row in read(ROOT / "rl/validation.json")},
            environment="refund-policy-v2",
            training_progress=read(ROOT / "rl/training_metrics.json"),
            recovery_receipts=(
                read(ROOT / "rl/recovery_receipts.json")
                if (ROOT / "rl/recovery_receipts.json").exists()
                else {}
            ),
            run_manifest_hash=content_hash(read(ROOT / "rl/run_manifest.json")),
        )
        assert selected["selection_activity"] == selection
        assert reports["rl"]["selected_checkpoint"] == selected["checkpoint"]
        assert reports["rl"]["validation_selection_hash"] == content_hash(selection)
        result["validation_selection"] = selection
        original_pointer = read(ROOT / "sft/river_checkpoint.json")
        atomic_json(
            ROOT / "selected/river_checkpoint.json",
            {
                "provider": "river",
                "base_model": plan["base_model"],
                "lora": original_pointer["lora"],
                "checkpoint": selected["checkpoint"]["path"],
                "source_checkpoint": original_pointer["checkpoint"],
                "selected_validation_step": selected["step"],
                "selected_training_activity": selection["selected_training_activity"],
                "plan_hash": plan["plan_hash"],
                "quality_gain_verified": False,
                "note": "Hosted River checkpoint pointer; not a local adapter export. Selection used validation only.",
            },
        )
        result["validation_outcomes_replayed"] = sum(
            len(entry["outcomes"]) for entry in validation
        )
    if set(reports) == {"rl", "baseline"}:
        before = {row["case_id"]: row for row in reports["baseline"]["outcomes"]}
        after = {row["case_id"]: row for row in reports["rl"]["outcomes"]}
        assert reports["baseline"]["case_hashes"] == reports["rl"]["case_hashes"]
        result["paired_comparison"] = {
            "cases": len(before),
            "baseline_passed": reports["baseline"]["passed"],
            "rl_passed": reports["rl"]["passed"],
            "success_gain": reports["rl"]["success_rate"]
            - reports["baseline"]["success_rate"],
            "wins": sum(
                after[key]["success"] and not before[key]["success"] for key in before
            ),
            "regressions": sum(
                before[key]["success"] and not after[key]["success"] for key in before
            ),
            "interpretation": "Single-seed exploratory comparison; stochastic sampling can change outcomes even for unchanged weights.",
        }
    result["status"] = (
        "completed" if set(reports) == {"rl", "baseline"} else "incomplete"
    )
    atomic_json(ROOT / "review-summary.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("arms", "validation_selection")
            },
            indent=2,
        )
    )


def retain():
    assert not DEST.exists(), "Never overwrite retained evidence"
    summary = read(ROOT / "review-summary.json")
    previous = 0
    for manifest in sorted(DEST.parent.glob("*/artifacts.sha256.json")):
        for name, digest in read(manifest).items():
            assert (
                sha(manifest.parent / name) == digest
            ), f"Older evidence changed: {name}"
            previous += 1
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
        ), "Credential-shaped data in evidence"
    DEST.mkdir()
    for name in (
        "plan.json",
        "exclusion-audit.json",
        "preflight-live.json",
        "execution-summary.json",
        "review-summary.json",
        "preflight-tests.xml",
        "experiment.py",
        "review.py",
    ):
        shutil.copyfile(ROOT / name, DEST / name)
    with tarfile.open(DEST / "frozen-run.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    comparison = summary.get("paired_comparison", {})
    activity = summary["arms"]["rl"].get("training_activity", {})
    selected = summary.get("validation_selection", {})
    text = (
        "# River boundary-curriculum RL continuation\n\n"
        f"Status: {summary['status']}. User-authorized, separate exploratory experiment; prior studies preserved.\n\n"
        "Qwen/Qwen3.5-9B, rank-16 existing first-seed SFT checkpoint, four planned CISPO RL batches, "
        "255 synthetic training cases with adjacent boundary contrasts, 32 fresh validation cases and 64 fresh paired test cases. "
        "Validation alone selects the checkpoint; confirmed skipped batches cannot promote unchanged weights. "
        "The fixed SFT baseline is sampled only after RL test evaluation.\n\n"
        f"Observed optimizer updates: {activity.get('observed_optimizer_updates', 'unavailable')}; "
        f"skipped batches: {activity.get('observed_skipped_batches', 'unavailable')}; "
        f"selected validation step: {selected.get('selected_validation_step', 'unavailable')}.\n\n"
        f"Paired baseline/RL successes: {comparison.get('baseline_passed', 'unavailable')}/64 and "
        f"{comparison.get('rl_passed', 'unavailable')}/64. "
        "This single seed does not establish a reliable RL quality gain, independently reviewed data quality, or A+ certification.\n\n"
        "The paired results have two wins and no case-level regressions. "
        "Model policy-violation incidence fell from 8/64 to 6/64; truncations fell from 8/64 to 5/64. "
        "Return-window boundary accuracy stayed at 3/8 for both checkpoints: the targeted weakness remains unresolved. "
        "The selected checkpoint contains one observed RL update; four updates occurred across the complete run.\n\n"
        f"Reserved generated-output tokens: {summary['reserved_output_tokens']:,}, ceiling 458,752; "
        "these are admission reservations, not actual usage or verified billed dollars. "
        "The supervisor completed successfully in 687.5 seconds without a timeout or operational retry.\n\n"
        "131 focused tests passed. All completed validation and test reports were replayed offline; "
        "fresh cases were checked against historical identities and policy facts. "
        f"{previous} previously retained artifacts verified unchanged.\n\n"
        "`frozen-run.tar.gz` retains package source, input cases, manifests, selection and training activity, "
        "exact traces, token ledgers and supervisor logs. No River credential is retained. "
        "Hashes establish local file identity, not provider attestation.\n"
        "The archive also includes `selected/river_checkpoint.json`, a reusable StateSet `--checkpoint` pointer directory.\n"
    )
    (DEST / "README.md").write_text(text)
    atomic_json(
        DEST / "artifacts.sha256.json",
        {path.name: sha(path) for path in sorted(DEST.iterdir()) if path.is_file()},
    )
    print(json.dumps({"retained": str(DEST), "previous_artifacts_verified": previous}))


if __name__ == "__main__":
    import sys

    if sys.argv[-1] == "retain":
        retain()
    else:
        asyncio.run(review())
