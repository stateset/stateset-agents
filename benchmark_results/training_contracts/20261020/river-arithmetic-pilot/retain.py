"""Retain the arithmetic curriculum, frozen pilot and independent replay audit."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET

REPO = Path("/home/dom/stateset-agents")
ROOT = Path(__file__).parent
DEST = REPO / "benchmark_results/training_contracts/20261020/river-arithmetic-pilot"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary, plan = read(ROOT / "review-summary.json"), read(ROOT / "plan.json")
    for name, expected in (("unit-tests.xml", 199), ("pilot-tests.xml", 9)):
        suites = list(ET.parse(ROOT / name).getroot().iter("testsuite"))
        assert sum(int(s.get("tests", 0)) for s in suites) == expected
        assert not sum(
            int(s.get("errors", 0)) + int(s.get("failures", 0)) for s in suites
        )
    prior = read(ROOT / "prior-evidence.json")
    for name, digest in prior.items():
        assert sha(REPO / name) == digest
    sources = ROOT / "source-inputs"
    sources.mkdir(exist_ok=False)
    for name, digest in read(ROOT / "historical-sources.json").items():
        path = REPO / name
        assert sha(path) == digest
        target = sources / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    canonical = REPO / "outputs/river-six-seed-recovery-20261019/seed-2026101901/data"
    shutil.copytree(canonical, sources / "canonical")
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
    inventory = {str(p.relative_to(ROOT)): sha(p) for p in files}
    (ROOT / "archive-inputs.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    archive = DEST / "frozen-run.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in files + [ROOT / "archive-inputs.json"]:
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    with tarfile.open(archive) as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        assert set(members) == set(inventory) | {"archive-inputs.json"}
        for name, member in members.items():
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(
                ROOT / name
            )
    for name in (
        "plan.json",
        "review-summary.json",
        "execution-summary.json",
        "validation-decision.json",
        "test-decision.json",
        "compatibility-checks.json",
        "unit-tests.xml",
        "pilot-tests.xml",
        "experiment.py",
        "preparation.py",
        "control.py",
        "gates.py",
        "test_pilot.py",
        "review.py",
        "retain.py",
        "archive-inputs.json",
    ):
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, DEST / name)
    rows = []
    for split in ("validation", "test"):
        for arm in ("control", "arithmetic"):
            item = summary["arms"][split][arm]
            if item["status"] != "complete":
                state = (
                    "incomplete (attempt consumed)"
                    if item["attempt_sealed"]
                    else "not attempted"
                )
                rows.append(f"| {split} | {arm} | {state} | — | — | — | — |")
                continue
            suites = item["suites"]
            values = [
                f"{suites[k]['passed']}/{suites[k]['total']}"
                for k in ("ordinary", "window", "arithmetic")
            ]
            rows.append(
                f"| {split} | {arm} | {item['passed']}/{item['total']} | {' | '.join(values)} | "
                f"{item['violation_cases']} / {item['truncated_cases']} |"
            )
    accepted = summary["pilot_result"] and summary["pilot_result"]["pilot_accepted"]
    failures = {}
    for split, decision in summary["decisions"].items():
        failures[split] = [
            name
            for name, item in decision["strata"].items()
            if any(
                not item[k]
                for k in (
                    "success_nonregression",
                    "policy_nonregression",
                    "truncation_nonregression",
                )
            )
        ]
    (DEST / "README.md").write_text(f"""# Borrow-boundary River SFT continuation pilot

**Pilot accepted: {bool(accepted)}.** Result: `{summary['pilot_result']}`.
Regression strata: `{failures}`. No deployment or production certification.

Validation stopped the pilot before any test attempt. Both arms passed 157/158.
Arithmetic improved from 53/54 to 54/54, while the existing window suite declined
from 72/72 to 71/72. The candidate denied an already-refunded order with reason
outside_window before correcting to already_refunded. That priority violation
fails the strict success rule even though no financial ledger error is applied.
The fully-refunded family declines from 13/13 to 12/13. The 308 test cases remain
unattempted, with no test seal or sampling. The candidate remains unpromoted.

The previous policy-parameter pilot passed 197/200 test cases but failed its
partial-refund safety gates. Its three failures involved subtraction; a separate
validation failure involved order-ID copying. This exploratory new hypothesis
uses only canonical training facts, never those exposed amounts, IDs or answers.
Both new arms load that retained, unpromoted candidate with fresh optimizers.
This is one conditional parent, not a new base-model initialization or multi-seed
population result. Qwen/Qwen3.5-9B, rank 16, seed {plan['training_seed']}.

The packaged, opt-in `borrow_boundary_contrasts_v1` curriculum retains all 512
parameter-curriculum rows and adds 189 training-only amount contrasts: 21 eligible
partial-refund sources, each with a nine-case payment/refund grid at offsets
-1/0/+1 around thousand-cent anchors on the inclusive final day. Small-order
clipping keeps all new cases partially refunded. Other order facts and untrusted
notes remain intact. Teacher transcripts perform the exact refund and finish
explicitly. Audit reconstructs every amount, group, lineage and teacher action.
Both prior curriculum policies remain byte-reproducible: 12 old bundles checked.

Control repeats only earlier parameter-curriculum facts. Both arms use 701 rows,
44 added SFT updates, identical family counts, one epoch, batch 16, lr 2e-5,
rank, initialization and shuffle seed. Local input-token exposure differs by
0.194%; supervised assistant tokens differ by 0.070%, within the frozen 0.5%
tolerance. Amount/window mixtures, provider compute and billed dollars are not
exactly matched; this does not isolate a single mechanistic cause of improvement.

Fresh case seed: {plan['evaluation_case_seed']}. Each arm has 158 validation cases
(32 ordinary, 72 window stress, 54 arithmetic stress) and 308 test cases
(128 ordinary, 72 window stress, 108 arithmetic stress). Arithmetic training uses
thousand-cent anchors; validation uses tens and test uses hundreds. Every
arithmetic cell draws its own source order. Day offsets -1/0 and payment/refund
offsets -1/0/+1 are represented. All heldouts are ID/fact-disjoint from historical
cases, all current training facts, unused earlier preparations and each other.
Private stress metadata is never exposed in model observations.

Fixed final-epoch checkpoints. Validation must show no success, policy-violation
or truncation regression overall and across the frozen families, suites,
suite-families, stress source families/windows, and arithmetic amount/day offsets.
Existing window-stress source strata are preserved separately so new arithmetic
gains cannot mask their regressions. If validation fails, test stays unattempted.
Otherwise both test attempts are sealed before sampling. Test must additionally
gain at least one arithmetic success versus control. No tuning or retries.

| Split | Arm | Successes | Ordinary | Window stress | Arithmetic stress | Violation / truncated cases |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Acknowledged added SFT updates: {summary['additional_sft_updates_acknowledged']}.
Replayed {summary['teacher_transcripts_replayed']} teacher transcripts and
{sum(summary['evaluations_replayed'].values())} completed evaluation trajectories.
Execution exit code {summary['execution']['exit_code']}, timed out
{summary['execution']['timed_out']}, elapsed {summary['execution']['elapsed_seconds']:.2f}s.
Output reservations {summary['budget']['reserved_generated_tokens']:,} /
{summary['budget']['limit']:,}. Reservations are admission allowances, not actual
generated usage or billed dollars. Actual cost remains unavailable. Original
caps: 88 SFT updates, 954,368 reserved output tokens, 40 minutes, request timeout
180s, no transport or operational retries. No RL or deployment ran.

199 unit/CLI tests and nine pilot checks passed. Full mypy passed 447 packaged
sources; Ruff, Black, isort and repository hygiene checks passed. The SDK worker
socketpair probe passed outside the sandbox. Independent offline review
reconstructed the source data, fresh holdouts and control, replayed all published
actions and recomputed every saved gate. {len(prior)} older evidence artifacts
verified unchanged. The archive retains exact package and runner sources,
datasets, provenance, attempts, pointer lineage, logs, token ledger and audits;
all {len(members)} archive members were verified byte-for-byte. No credential is
retained. Local hashes do not establish provider attestation, independently
reviewed industry quality, or A+ production readiness.
""")
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
                "prior_artifacts_unchanged": len(prior),
                "archive_bytes": archive.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
