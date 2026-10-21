"""Retain a completed pilot, its frozen implementation, and exact replay evidence."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET

REPO = Path("/home/dom/stateset-agents")
ROOT = Path(__file__).parent
DEST = REPO / "benchmark_results/training_contracts/20261019/river-stability-pilot"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = read(ROOT / "review-summary.json")
    plan = read(ROOT / "plan.json")
    assert (ROOT / "execution-summary.json").exists()
    suites = list(ET.parse(ROOT / "pilot-tests.xml").getroot().iter("testsuite"))
    assert sum(int(s.get("tests", 0)) for s in suites) == 8
    assert sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites) == 0
    prior = read(ROOT / "prior-evidence.json")
    for name, digest in prior.items():
        assert sha(REPO / name) == digest
    snapshots = ROOT / "source-inputs"
    snapshots.mkdir(exist_ok=False)
    source = (
        REPO / "outputs/river-curriculum-review-20261019/seed-2026101901/curriculum"
    )
    shutil.copytree(source, snapshots / "legacy-curriculum")
    prepared = REPO / "outputs/river-boundary-stability-20261019"
    shutil.copyfile(prepared / "attempt-3/readiness.json", snapshots / "readiness.json")
    shutil.copyfile(
        prepared / "attempt-3/historical-sources.json",
        snapshots / "historical-sources.json",
    )
    DEST.mkdir(parents=True, exist_ok=False)
    files = [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]
    for path in files:
        assert not re.search(
            rb"\brv_[A-Za-z0-9_-]{20,}\b", path.read_bytes()
        ), f"Credential-like content in {path.name}"
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in files}
    (ROOT / "archive-inputs.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    archive = DEST / "frozen-run.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in files + [ROOT / "archive-inputs.json"]:
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    with tarfile.open(archive) as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        assert set(members) == set(manifest) | {"archive-inputs.json"}
        for name, member in members.items():
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(
                ROOT / name
            )
    for name in (
        "plan.json",
        "review-summary.json",
        "execution-summary.json",
        "experiment.py",
        "control.py",
        "gates.py",
        "test_pilot.py",
        "pilot-tests.xml",
        "review.py",
        "retain.py",
        "archive-inputs.json",
        "validation-decision.json",
        "test-decision.json",
    ):
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, DEST / name)
    rows = []
    for split in ("validation", "test"):
        for arm in ("original", "control", "parameter"):
            result = summary["arms"][split][arm]
            if result["status"] != "complete":
                state = (
                    "incomplete (attempt consumed)"
                    if result["attempt_sealed"]
                    else "not attempted"
                )
                rows.append(f"| {split} | {arm} | {state} | — | — | — |")
                continue
            ordinary, stress = (
                result["distributions"]["ordinary"],
                result["distributions"]["stress"],
            )
            rows.append(
                f"| {split} | {arm} | {result['passed']}/{result['total']} | "
                f"{ordinary['passed']}/{ordinary['total']} | {stress['passed']}/{stress['total']} | "
                f"{result['violation_cases']} / {result['truncated_cases']} |"
            )
    decisions = {}
    for split in ("validation", "test"):
        path = ROOT / f"{split}-decision.json"
        if path.exists():
            d = read(path)
            decisions[split] = {
                "passed": d["passed"],
                "parameter_vs_control_gain": d["comparisons"]["parameter_vs_control"][
                    "gain"
                ],
            }
    test_path = ROOT / "test-decision.json"
    test_decision = read(test_path) if test_path.exists() else None
    failed_strata = []
    if test_decision is not None:
        for name, item in test_decision["comparisons"]["parameter_vs_control"][
            "strata"
        ].items():
            if any(
                not item[key]
                for key in (
                    "success_nonregression",
                    "policy_nonregression",
                    "truncation_nonregression",
                )
            ):
                failed_strata.append(name)
    acceptance = (
        "PASSED"
        if summary["pilot_result"] and summary["pilot_result"]["pilot_accepted"]
        else "FAILED"
    )
    text = f"""# One-parent River stability pilot

**Prespecified pilot acceptance: {acceptance}.** The candidate remains unpromoted.
Primary regression strata: {", ".join(failed_strata) or "none recorded"}.

The parameter arm gains 14 successes (7 percentage points) over control, but the
partial-refund family declines from 19/19 to 18/19, stress cases sourced from
partial-refund orders decline from 8/9 to 7/9, and eight-day stress truncation
increases from zero cases to one. These overlapping strata fail the frozen
nonregression gates. The secondary comparison against the original passes.

Qwen/Qwen3.5-9B, rank 16, training seed {plan['training_seed']}, fresh evaluation
case seed {plan['evaluation_case_seed']}. The first retained original SFT parent
(2026101901) was fixed before evaluation. It has already participated in earlier
pilots: this is a conditional, exploratory experiment with one parent, not an
independent multi-seed replication or population-significance result.

The primary comparison is broader policy-parameter contrasts versus additional
legacy-curriculum SFT. Both train from the same original SFT checkpoint with a
fresh optimizer, one epoch, 512 rows, 32 updates, batch 16, learning rate 2e-5,
rank 16, shuffle seed and family counts. The control preserves all 255 legacy
curriculum rows and repeats only those existing facts to meet the new arm's
family quotas. Near-matched local input exposure differs by 0.063%; supervised
assistant tokens differ by 0.053%, below the frozen 0.5% tolerance. These limits
do not establish equal provider compute or billed cost.

The parameter arm uses the training-only `policy_parameter_contrasts_v1` policy
from the preceding offline preparation. Teacher transcripts were re-executed in
fresh sandboxes. All 304 evaluation cases are disjoint from historical inputs,
all new training facts, unused earlier preparations and the other five parents'
fresh holdouts. Validation contains 32 ordinary plus 72 stress cases, test 128
ordinary plus 72 stress cases. Stress windows 5/18/40 and 8/25/75 remain distinct
from the new 3/10/21/60/90 training windows. Original 14/30 training cases remain.
Private stress metadata is never included in model observations.

Both trained checkpoints are fixed final-epoch checkpoints. The parameter arm
must show no validation success, policy-violation or truncation regression versus
both control and original across all, family, ordinary/stress, stress source
family and stress-window marginal strata. Failure stops without test. Otherwise
all three test attempts are durably sealed before sampling. Primary test
acceptance requires at least 3 percentage points gain versus control and the
same safety strata, plus nonregression versus original. No tuning or retry.

| Split | Arm | Successes | Ordinary | Stress | Violation / truncated cases |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Execution exit code: {summary['execution']['exit_code']}, timed out:
{summary['execution']['timed_out']}; elapsed
{summary['execution']['elapsed_seconds']:.2f} seconds. Pilot result:
`{summary['pilot_result']}`. Reconstructed decisions: `{decisions}`.

Acknowledged added SFT updates: {summary['additional_sft_updates_acknowledged']}.
Replayed {summary['teacher_transcripts_replayed']} teacher transcripts and
{sum(summary['evaluations_replayed'].values())} completed evaluation trajectories.
Output reservations: {summary['budget']['reserved_generated_tokens']:,} /
{summary['budget']['limit']:,} tokens. Reservations are admission limits, not
actual generated tokens or billed dollars. Actual dollar cost is unavailable.
The original workload caps were 64 SFT updates, 933,888 reserved output tokens,
30 minutes, SDK requests bounded to 180 seconds and no operational or transport
retries. No RL, deployment, or additional provider calls during review.

Eight new control/gate tests passed, including masked stratum regressions,
corrupted aggregate rejection and exact teacher replay. The packaged sources
are unchanged from the prior 182-test, 446-source mypy validation. The worker
socketpair wakeup probe passed outside the sandbox. Independent offline review
recomputed each saved gate and replayed the executed environment actions.
{len(prior)} older retained evidence artifacts verified unchanged.

The archive retains exact package/runner sources, matched datasets, untouched or
consumed heldouts, durable attempts, provider pointer lineage, logs, token ledger,
and replay evidence. Every member was byte-verified. No credential is retained.
Hashes establish local identity, not provider attestation. This synthetic pilot
does not establish A+ performance or independently reviewed production readiness.
"""
    (DEST / "README.md").write_text(text)
    artifacts = {p.name: sha(p) for p in sorted(DEST.iterdir()) if p.is_file()}
    (DEST / "artifacts.sha256.json").write_text(
        json.dumps(artifacts, indent=2, sort_keys=True) + "\n"
    )
    for name, digest in artifacts.items():
        assert sha(DEST / name) == digest
    print(
        json.dumps(
            {
                "retained_artifacts": len(artifacts),
                "archive_members_verified": len(members),
                "archive_bytes": archive.stat().st_size,
                "prior_artifacts_unchanged": len(prior),
            }
        )
    )


if __name__ == "__main__":
    main()
