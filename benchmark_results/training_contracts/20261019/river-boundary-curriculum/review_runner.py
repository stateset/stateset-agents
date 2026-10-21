"""Offline validation diagnosis and prospective training-data preparation."""

import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)
from stateset_agents.data.refund_curriculum import (
    audit_refund_curriculum,
    prepare_refund_curriculum,
)
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.refund_trace import replay_refund_trace

ROOT = Path("outputs/river-curriculum-review-20261019")
STUDY = Path("outputs/river-six-seed-recovery-20261019")
SEEDS = range(2026101901, 2026101907)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


async def main():
    sources = {}
    diagnostics = []
    total_replayed = 0
    for seed in SEEDS:
        directory = STUDY / f"seed-{seed}"
        cases_path = directory / "data/validation.json"
        history_path = directory / "sft_rl/validation_results.json"
        cases = {row["order_id"]: row for row in read(cases_path)}
        sources[str(cases_path)] = sha(cases_path)
        sources[str(history_path)] = sha(history_path)
        for entry in read(history_path):
            failures = Counter()
            errors = Counter()
            ids = []
            boundary_count = 0
            for outcome in entry["outcomes"]:
                case = cases[outcome["case_id"]]
                assert outcome["case_hash"] == content_hash(case)
                result = await replay_refund_trace(
                    "refund-policy-v2", case, outcome["environment_trace"]
                )
                assert result["reward"] == outcome["reward"]
                total_replayed += 1
                ids.append(case["order_id"])
                boundary_count += case["family"] == "window_boundary"
                if not result["success"]:
                    failures[case["family"]] += 1
                    for step in outcome["environment_trace"]["steps"]:
                        for message in step["observations"]:
                            observation = json.loads(message["content"])
                            if "error" in observation:
                                errors[observation["error"]] += 1
            assert len(ids) == len(set(ids)) == len(cases)
            assert set(ids) == set(cases)
            diagnostics.append(
                {
                    "seed": seed,
                    "validation_step": entry["step"],
                    "cases": len(cases),
                    "boundary_cases": boundary_count,
                    "failed_cases_by_family": dict(failures),
                    "rejected_actions_by_error": dict(errors),
                }
            )
    initial = [row for row in diagnostics if row["validation_step"] == 0]
    save(
        "validation-diagnostics.json",
        {
            "scope": "validation_only; no test outcomes inspected",
            "source_artifacts": sources,
            "replayed_validation_outcomes": total_replayed,
            "initial_sft_boundary_cases": sum(row["boundary_cases"] for row in initial),
            "initial_sft_boundary_failures": sum(
                row["failed_cases_by_family"].get("window_boundary", 0)
                for row in initial
            ),
            "evaluations": diagnostics,
        },
    )

    # Exclusion only: these cases never supply a transcript, label or diagnostic.
    exclusions = []
    for seed in SEEDS:
        directory = STUDY / f"seed-{seed}"
        exclusions.extend(directory / f"data/{split}.json" for split in ("validation", "test"))
        exclusions.append(directory / "primary-test.json")
    for directory in (
        Path("outputs/river-sft-warmup-20261018/data"),
        Path("outputs/river-prompt-fix-20261018-live/run"),
        Path("outputs/river-integration-review-20261018-live/run"),
    ):
        exclusions.extend(directory / f"{split}.json" for split in ("train", "validation", "test"))
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    excluded_facts = set()
    excluded_ids = set()
    excluded_sources = {}
    for path in exclusions:
        excluded_sources[str(path)] = sha(path)
        for case in read(path):
            excluded_facts.add(fingerprint(case))
            excluded_ids.add(case["order_id"])

    bundles = []
    all_training_ids = set()
    all_training_facts = set()
    for seed in SEEDS:
        source = STUDY / f"seed-{seed}/data"
        output = ROOT / f"seed-{seed}/curriculum"
        manifest = await prepare_refund_curriculum(source, output)
        audit = await audit_refund_curriculum(source, output)
        cases = read(output / "train.json")
        ids = {row["order_id"] for row in cases}
        facts = {fingerprint(row) for row in cases}
        assert not ids & excluded_ids
        assert not facts & excluded_facts
        assert not ids & all_training_ids
        assert not facts & all_training_facts
        all_training_ids.update(ids)
        all_training_facts.update(facts)
        bundles.append({"seed": seed, "manifest": manifest, "audit": audit})
    summary = {
        "complete": True,
        "quality_certification": False,
        "provider_calls": 0,
        "model_updates": 0,
        "validation_replayed": total_replayed,
        "training_demonstrations": sum(b["manifest"]["selected"] for b in bundles),
        "original_training_cases": sum(b["manifest"]["original_cases"] for b in bundles),
        "derived_training_cases": sum(b["manifest"]["derived_cases"] for b in bundles),
        "contrast_groups": sum(b["manifest"]["contrast_groups"] for b in bundles),
        "excluded_cases_unique_by_facts": len(excluded_facts),
        "exclusion_sources": excluded_sources,
        "cross_seed_training_overlap": 0,
        "training_overlap_with_reserved_or_historical_cases": 0,
        "bundles": bundles,
        "limitations": [
            "Synthetic policy teacher verification is not human review or model quality.",
            "No live fine-tuning or post-change held-out evaluation was run.",
            "Existing exposed test outcomes remain diagnostic evidence only.",
            "A future study needs a new frozen protocol, fresh excluded test cases, and bounded spend.",
        ],
    }
    save("review-summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("bundles", "exclusion_sources")}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
