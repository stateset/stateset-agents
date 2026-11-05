"""Independently reconstruct prepared stability inputs, without provider calls."""

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
    audit_refund_curriculum,
)
from stateset_agents.data.refund_stress import refund_policy_stress_cases
from stateset_agents.evaluation.implementation import package_implementation

REPO = Path("/home/dom/stateset-agents")
ROOT = Path(__file__).parent
PREPARED = ROOT / "attempt-3"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def main():
    readiness = read(PREPARED / "readiness.json")
    assert readiness["implementation"] == package_implementation()
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    prior = read(PREPARED / "prior-evidence.json")
    for name, digest in prior.items():
        assert sha(REPO / name) == digest
    historical = read(PREPARED / "historical-sources.json")
    excluded = []
    for name, digest in historical.items():
        path = REPO / name
        assert sha(path) == digest
        rows = read(path)
        if isinstance(rows, list):
            excluded.extend(
                r for r in rows if isinstance(r, dict) and "return_window_days" in r
            )
    training, replays = [], 0
    for entry in readiness["parents"]:
        source = REPO / entry["canonical_data"]
        assert sha(source / "data_manifest.json") == entry["canonical_manifest_sha256"]
        curriculum = REPO / entry["curriculum_path"]
        audit = await audit_refund_curriculum(
            source, curriculum, policy=PARAMETER_POLICY
        )
        assert audit == entry["curriculum_audit"]
        training.extend(read(curriculum / "train.json"))
        replays += audit["replayed_demonstrations"]
        old = (
            REPO
            / f"outputs/river-curriculum-review-20261019/seed-{entry['parent_seed']}/curriculum"
        )
        assert await audit_refund_curriculum(source, old) == entry["legacy_audit"]
    assert (
        len({fingerprint(r) for r in training})
        == len(training)
        == readiness["training_rows"]
    )
    excluded.extend(training)
    used_ids = {r["order_id"] for r in excluded}
    used_facts = {fingerprint(r) for r in excluded}
    counts = Counter()
    for entry in readiness["parents"]:
        for split, count, windows in (
            ("validation", 32, (5, 18, 40)),
            ("test", 128, (8, 25, 75)),
        ):
            path = PREPARED / f"parent-{entry['parent_seed']}" / f"{split}.json"
            assert sha(path) == entry["holdouts"][split]["sha256"]
            ordinary = refund_policy_benchmark(split, count=count, seed=entry["seed"])
            grid = refund_policy_stress_cases(
                split, seed=entry["seed"], windows=windows
            )
            expected = sorted(ordinary + grid, key=lambda r: r["order_id"])
            assert read(path) == expected
            for row in expected:
                assert row["order_id"] not in used_ids
                assert fingerprint(row) not in used_facts
                used_ids.add(row["order_id"])
                used_facts.add(fingerprint(row))
                counts[split] += 1
    assert dict(counts) == readiness["holdout_counts"]
    assert sum(counts.values()) == readiness["fresh_holdouts"]
    result = {
        "passed": True,
        "scope": "offline_data_and_trace_integrity",
        "provider_requests": 0,
        "model_holdout_outcomes": 0,
        "teacher_transcripts_reconstructed_and_replayed": replays,
        "holdout_cases_reconstructed": dict(counts),
        "id_and_fact_overlap": 0,
        "legacy_curriculum_bundles_reconstructed": 6,
        "prior_evidence_artifacts_verified_unchanged": len(prior),
        "validation_trajectories_replayed_in_frozen_prior_runtime": read(
            ROOT / "validation-diagnosis.json"
        )["replayed_trajectories"],
        "implementation": readiness["implementation"],
    }
    (ROOT / "data-checks.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "implementation"}, sort_keys=True
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
