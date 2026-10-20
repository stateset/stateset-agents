"""Retain the offline stability improvement and its independently checked inputs."""

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

REPO = Path("/home/dom/stateset-agents")
ROOT = Path(__file__).parent
DEST = REPO / "benchmark_results/training_contracts/20261019/river-boundary-stability"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = read(ROOT / "data-checks.json")
    assert checks["passed"]
    readiness = read(ROOT / "attempt-3/readiness.json")
    xml = ET.parse(ROOT / "unit-tests.xml")
    suites = list(xml.getroot().iter("testsuite"))
    assert sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites) == 0
    assert sum(int(s.get("tests", 0)) for s in suites) == 182
    DEST.mkdir(parents=True, exist_ok=False)
    inputs = set(read(ROOT / "attempt-3/historical-sources.json"))
    inputs.update(read(ROOT / "validation-diagnosis.json")["source_sha256"])
    for entry in readiness["parents"]:
        directories = [
            REPO / entry["canonical_data"],
            REPO
            / f"outputs/river-curriculum-review-20261019/seed-{entry['parent_seed']}/curriculum",
        ]
        for directory in directories:
            inputs.update(
                str(p.relative_to(REPO)) for p in directory.iterdir() if p.is_file()
            )
    old_runtime = (
        REPO / "outputs/river-boundary-six-seed-20261019/runtime/stateset_agents"
    )
    old_implementation = read(ROOT / "validation-diagnosis.json")["implementation"]
    for name, digest in old_implementation["files"].items():
        assert sha(old_runtime / name) == digest
        path = ROOT / "snapshot/previous-runtime/stateset_agents" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_runtime / name, path)
    for name, digest in readiness["implementation"]["files"].items():
        source = REPO / "stateset_agents" / name
        assert sha(source) == digest
        path = ROOT / "snapshot/current-runtime/stateset_agents" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
    for name in sorted(inputs):
        path = ROOT / "snapshot/inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, path)
    archived = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    (ROOT / "archive-inputs.json").write_text(
        json.dumps(archived, indent=2, sort_keys=True) + "\n"
    )
    archive = DEST / "frozen-preparation.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    with tarfile.open(archive) as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        assert set(members) == set(archived) | {"archive-inputs.json"}
        for name, member in members.items():
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(
                ROOT / name
            )
    for source, name in [
        ("attempt-3/readiness.json", "readiness.json"),
        ("data-checks.json", "data-checks.json"),
        ("validation-diagnosis.json", "validation-diagnosis.json"),
        ("preparation-attempts.json", "preparation-attempts.json"),
        ("unit-tests.xml", "unit-tests.xml"),
        ("diagnose.py", "diagnose.py"),
        ("audit.py", "audit.py"),
        ("retain.py", "retain.py"),
        ("attempt-3/prepare.py", "prepare.py"),
        ("archive-inputs.json", "archive-inputs.json"),
    ]:
        shutil.copyfile(ROOT / source, DEST / name)
    (DEST / "README.md").write_text(
        """# Refund policy stability data and validation diagnosis

Offline system improvement following the six-parent SFT replication. No River
job, optimizer update, model evaluation of new holdouts, or deployment ran here.

The exact previous frozen package replayed all 576 prior validation trajectories:
original 171/192, original-repetition 189/192, boundary 189/192. The boundary
failures comprise one date comparison, one refund arithmetic error and one
corrupted order ID. Categories describe the first invalid action, not a proven
cause of the model behavior. No previous test outcomes supply new supervision.

The opt-in `policy_parameter_contrasts_v1` curriculum preserves every original
training case and adds three adjacent-day contrasts around one of 3, 10, 21, 60,
90 days, assigned within source family. It retains monetary, delivery, chargeback
and untrusted-note facts. All eight source families participate. A teacher reads
lookup facts, performs exactly one resolution and finishes with empty arguments;
every transcript is independently replayed. All six legacy curriculum bundles
still reconstruct exactly under the default policy.

Prepared 3,072 teacher transcripts (512 per parent, including 768 original rows)
and 1,824 fresh held-out cases: 624 validation and 1,200 test. Each parent has 32
ordinary plus 72 stress validation cases, and 128 ordinary plus 72 stress test
cases. Stress windows are 5/18/40 for validation and 8/25/75 for test, distinct
from the new training windows. Original training windows 14/30 remain present.
These are changed, explicitly prospective synthetic evaluation distributions.
Every grid cell uses its own source order; private stratum metadata stays outside
model observations. ID and observable-fact overlap is zero against historical
inputs, all new training facts, earlier unused preparations and other fresh cases.

Two offline preparations stopped on identity or fact reuse before any model
evaluation. Their complete training data and partial heldouts remain retained.
The completed preparation reaudits training data and logs prospective input-seed
rejections with a finite 100-candidate limit per parent. These are input checks,
not retries of a model, holdout outcomes, paid jobs or a failed acceptance gate.

Validation: 182 unit/CLI tests passed, full mypy passed for 446 packaged sources,
and Ruff, Black, isort and repository hygiene checks passed. Independent audit
reconstructed/replayed all 3,072 teachers and reconstructed all 1,824 holdouts.
593 existing retained evidence artifacts verified unchanged. The archive retains
current and prior package sources, source inputs, all preparation attempts,
fresh datasets, provenance, audits and test results. Archive members were checked
byte-for-byte. Hashes establish local identity, not provider attestation.

Model quality, actual provider compute and billed cost are unmeasured. The next
live experiment must freeze a bounded matched-control pilot, tokenize and match
exposure with the actual model tokenizer, declare family and stress regression
gates, and seal a single test attempt after validation selection. This preparation
does not establish A+ performance or independently reviewed production readiness.
"""
    )
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
                "archive_files_verified": len(members),
                "archive_bytes": archive.stat().st_size,
                "prior_artifacts_unchanged": 593,
            }
        )
    )


if __name__ == "__main__":
    main()
