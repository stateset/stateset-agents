"""Prepare training-only stability data and untouched stress holdouts offline."""

import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.data.refund_curriculum import (
    PARAMETER_POLICY,
    PARAMETER_WINDOWS,
    audit_refund_curriculum,
)
from stateset_agents.data.refund_stress import STRESS_POLICY, refund_policy_stress_cases
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.implementation import package_implementation

REPO = Path("/home/dom/stateset-agents")
ROOT = Path(__file__).parent
PREVIOUS = REPO / "outputs/river-boundary-six-seed-20261019"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


async def main():
    if (ROOT / "readiness.json").exists():
        raise ValueError(
            "Dataset preparation is already completed; use a new output directory"
        )
    prior = {}
    for manifest_path in sorted(
        (REPO / "benchmark_results/training_contracts").rglob("artifacts.sha256.json")
    ):
        for name, digest in read(manifest_path).items():
            path = manifest_path.parent / name
            assert sha(path) == digest
            prior[str(path.relative_to(REPO))] = digest
    write(ROOT / "prior-evidence.json", prior)
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    history = dict(read(PREVIOUS / "exclusion-audit.json")["historical_sources_sha256"])
    for directory in sorted(PREVIOUS.glob("seed-*")):
        for path in sorted((directory / "data").glob("*cases.json")):
            history[str(path.relative_to(REPO))] = sha(path)
    # Even unused cases published by rejected offline preparations are excluded.
    for split in ("validation", "test"):
        for path in sorted(ROOT.parent.rglob(f"{split}.json")):
            history[str(path.relative_to(REPO))] = sha(path)
        for split in ("validation", "test"):
            path = directory / "data" / f"{split}.json"
            history[str(path.relative_to(REPO))] = sha(path)
    excluded = []
    for name, digest in history.items():
        path = REPO / name
        assert sha(path) == digest
        rows = read(path)
        if isinstance(rows, list):
            excluded.extend(
                row
                for row in rows
                if isinstance(row, dict) and "return_window_days" in row
            )
    write(ROOT / "historical-sources.json", history)
    entries, training = [], []
    # Prepare every parent's data first so each holdout excludes all new training facts.
    for parent_seed in range(2026101901, 2026101907):
        seed = parent_seed + 1000
        canonical = (
            REPO / f"outputs/river-six-seed-recovery-20261019/seed-{parent_seed}/data"
        )
        old = (
            REPO
            / f"outputs/river-curriculum-review-20261019/seed-{parent_seed}/curriculum"
        )
        # Legacy output must still reconstruct byte-for-byte after adding the opt-in policy.
        legacy_audit = await audit_refund_curriculum(canonical, old)
        curriculum = ROOT.parent / "attempt-2" / f"seed-{seed}" / "curriculum"
        manifest = read(curriculum / "data_manifest.json")
        audit = await audit_refund_curriculum(
            canonical, curriculum, policy=PARAMETER_POLICY
        )
        cases = read(curriculum / "train.json")
        training.extend(cases)
        entries.append(
            {
                "seed": seed,
                "parent_seed": parent_seed,
                "canonical_data": str(canonical.relative_to(REPO)),
                "canonical_manifest_sha256": sha(canonical / "data_manifest.json"),
                "curriculum_path": str(curriculum.relative_to(REPO)),
                "curriculum_manifest": manifest,
                "curriculum_audit": audit,
                "legacy_audit": legacy_audit,
            }
        )
    assert len({fingerprint(c) for c in training}) == len(training)
    excluded.extend(training)
    fresh, counts = [], Counter()
    for entry in entries:
        seed = entry["seed"]
        directory = ROOT / f"parent-{entry['parent_seed']}"
        entry["holdouts"] = {}
        specification = (
            ("validation", 32, (5, 18, 40)),
            ("test", 128, (8, 25, 75)),
        )
        rejected = []
        # Reject collisions prospectively using only generated inputs, before
        # publishing any cases or obtaining model outcomes. Never retry a model.
        for trial in range(100):
            candidate = seed + trial * 100
            proposed, working = {}, list(excluded)
            try:
                for split, count, windows in specification:
                    ordinary = refund_policy_benchmark(
                        split, count=count, seed=candidate
                    )
                    used_ids = {c["order_id"] for c in working}
                    used_facts = {fingerprint(c) for c in working}
                    if (
                        used_ids & {c["order_id"] for c in ordinary}
                        or used_facts & {fingerprint(c) for c in ordinary}
                        or len({fingerprint(c) for c in ordinary}) != len(ordinary)
                    ):
                        raise ValueError(
                            "Ordinary cases overlap previous or generated cases"
                        )
                    working.extend(ordinary)
                    grid = refund_policy_stress_cases(
                        split, seed=candidate, windows=windows, excluded=working
                    )
                    working.extend(grid)
                    proposed[split] = (ordinary, grid)
            except ValueError as exc:
                if "overlap" not in str(exc):
                    raise
                rejected.append({"seed": candidate, "reason": str(exc)})
                continue
            break
        else:
            raise ValueError(
                "No collision-free input seed found within the offline limit"
            )
        entry["seed"] = candidate
        entry["rejected_input_seeds"] = rejected
        excluded = working
        for split, count, windows in specification:
            ordinary, grid = proposed[split]
            cases = sorted(ordinary + grid, key=lambda c: c["order_id"])
            path = directory / f"{split}.json"
            write(path, cases)
            counts[split] += len(cases)
            fresh.extend(cases)
            entry["holdouts"][split] = {
                "sha256": sha(path),
                "case_hash": content_hash(cases),
                "ordinary": len(ordinary),
                "stress": len(grid),
                "total": len(cases),
                "stress_windows": list(windows),
                "stress_source_families": dict(
                    Counter(c["stress"]["source_family"] for c in grid)
                ),
                "families": dict(Counter(c["family"] for c in cases)),
            }
    assert len({c["order_id"] for c in fresh}) == len(fresh)
    assert len({fingerprint(c) for c in fresh}) == len(fresh)
    result = {
        "schema_version": 1,
        "scope": "offline_stability_dataset_preparation",
        "status": "data_prepared_no_provider_execution",
        "curriculum_policy": PARAMETER_POLICY,
        "training_parameter_windows": list(PARAMETER_WINDOWS),
        "stress_policy": STRESS_POLICY,
        "implementation": package_implementation(),
        "parents": entries,
        "training_rows": len(training),
        "fresh_holdouts": len(fresh),
        "holdout_counts": counts,
        "historical_source_files": len(history),
        "prior_evidence_artifacts_verified": len(prior),
        "identity_and_fact_overlap": 0,
        "heldouts_evaluated": 0,
        "provider_requests": 0,
        "unverified": [
            "Model quality on the new distribution",
            "Matched-control training exposure",
            "Provider compute and billed cost",
            "Independently reviewed production data",
        ],
        "required_before_live_run": [
            "Freeze a bounded matched-control pilot plan and exact runtime.",
            "Tokenize both arms with the actual selected model tokenizer and match exposure.",
            "Prespecify per-family and stress-stratum regression gates and a single test attempt.",
            "Run validation first; inspect test only after freezing selection.",
        ],
    }
    write(ROOT / "readiness.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "training_rows",
                    "fresh_holdouts",
                    "holdout_counts",
                    "prior_evidence_artifacts_verified",
                )
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
