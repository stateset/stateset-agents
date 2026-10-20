"""Retain this completed offline review without rewriting older evidence."""

import hashlib
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("outputs/river-curriculum-review-20261019")
DESTINATION = Path("benchmark_results/training_contracts/20261019/river-boundary-curriculum")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    tests = {}
    for name in ("focused-tests.xml", "regression-tests.xml"):
        suites = ET.parse(ROOT / name).getroot()
        cases = suites.findall(".//testcase")
        assert cases and not suites.findall(".//failure") and not suites.findall(".//error")
        tests[name] = {
            "cases": len(cases),
            "skipped": len(suites.findall(".//skipped")),
            "passed": len(cases) - len(suites.findall(".//skipped")),
        }
    checks = {
        "tests": tests,
        "test_scope": "new curriculum, demonstration replay, policy/action integrity, atomic publication, trace audit, original study/provenance/preflight",
        "mypy": {"passed": True, "source_files": 440},
        "ruff": "passed on changed Python sources and new tests",
        "black": "passed on changed Python sources and new tests",
        "isort": "passed on changed Python sources and new tests",
        "repo_hygiene": "passed",
        "git_diff_check": "passed on changed files",
        "receipt_note": "JUnit is the test receipt; other entries summarize observed tool results, not raw transcripts.",
    }
    tokenization = json.loads((ROOT / "tokenization-review.json").read_text())
    assert tokenization["passed"] and tokenization["demonstrations"] == 1515
    checks["tokenization"] = tokenization
    save(ROOT / "checks.json", checks)
    DESTINATION.mkdir(parents=True, exist_ok=False)
    for name in (
        "README.md", "review-summary.json", "validation-diagnostics.json",
        "exclusion-fingerprints.json", "source-implementation.json", "checks.json",
        "focused-tests.xml", "regression-tests.xml", "review_runner.py", "retain_evidence.py",
        "tokenization-review.json", "tokenization_review.py",
    ):
        shutil.copy2(ROOT / name, DESTINATION / name)
    sources = [
        Path("stateset_agents/data/refund_curriculum.py"),
        Path("stateset_agents/cli_benchmark.py"),
        Path("tests/unit/test_refund_curriculum.py"),
        Path("docs/RIVER_PROVIDER.md"),
        Path("CHANGELOG.md"),
    ]
    with tarfile.open(DESTINATION / "source-and-checks.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=path.as_posix())
    save(DESTINATION / "changed-file-hashes.json", {str(path): sha(path) for path in sources})
    with tarfile.open(DESTINATION / "curriculum-bundles.tar.gz", "w:gz") as archive:
        for seed in range(2026101901, 2026101907):
            directory = ROOT / f"seed-{seed}/curriculum"
            archive.add(directory, arcname=f"seed-{seed}/curriculum")
    hashes = {
        path.relative_to(DESTINATION).as_posix(): sha(path)
        for path in sorted(DESTINATION.rglob("*")) if path.is_file()
    }
    save(DESTINATION / "artifacts.sha256.json", hashes)
    for name, expected in hashes.items():
        assert sha(DESTINATION / name) == expected
    print(json.dumps({"retained_files": len(hashes) + 1, "tests": tests, "directory": str(DESTINATION)}, indent=2))


if __name__ == "__main__":
    main()
