"""A crashed, incomplete, or high-severity scan must never produce a green gate."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.security_workflow_gate import check_reports, main


@pytest.fixture
def reports(tmp_path):
    documents = {
        "bandit": {"errors": [], "results": []},
        "pip-audit": {
            "dependencies": [{"name": "example", "version": "1", "vulns": []}]
        },
        "semgrep": {
            "version": "1.175.0",
            "errors": [],
            "results": [],
            "paths": {"scanned": ["stateset_agents/example.py"]},
        },
        "trivy": {
            "SchemaVersion": 2,
            "ArtifactName": ".",
            "Results": [{"Target": "requirements.txt"}],
        },
    }
    for name, report in documents.items():
        (tmp_path / f"{name}-report.json").write_text(json.dumps(report))
        (tmp_path / f"{name}-exit-code.txt").write_text("0\n")
    return tmp_path, documents


def write(reports, name, payload):
    (reports[0] / f"{name}-report.json").write_text(json.dumps(payload))


def test_complete_clean_scan_passes_and_cli_writes_decision(reports):
    root, _ = reports
    assert check_reports(root)["passed"]
    output = root / "gate.json"
    assert main(["--reports-dir", str(root), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["passed"] and set(result["scans"]) == {
        "bandit",
        "pip-audit",
        "semgrep",
        "trivy",
    }
    assert result["scans"]["semgrep"]["scanned"] == 1


def test_missing_or_ignored_package_sources_prevent_a_clean_scan(reports):
    root, _ = reports
    source = root / "stateset_agents"
    source.mkdir()
    (source / "example.py").write_text("value = 1\n")
    (source / "__init__.py").write_text("")
    assert check_reports(root, source_root=source)["passed"]
    (source / "unreadable.py").write_text("value = 2\n")
    result = check_reports(root, source_root=source)
    assert not result["passed"]
    assert any("unreadable.py" in issue for issue in result["issues"])
    assert not check_reports(root, source_root=root / "missing")["passed"]


@pytest.mark.parametrize("name", ["bandit", "pip-audit", "semgrep", "trivy"])
@pytest.mark.parametrize(
    "failure",
    [
        "missing_report",
        "empty_report",
        "invalid_json",
        "wrong_shape",
        "empty_object",
        "missing_exit",
        "crashed",
        "bad_exit",
        "finding_exit_without_findings",
    ],
)
def test_incomplete_or_failed_scanners_fail_closed(reports, name, failure):
    root, _ = reports
    report = root / f"{name}-report.json"
    status = root / f"{name}-exit-code.txt"
    if failure == "missing_report":
        report.unlink()
    elif failure == "empty_report":
        report.write_text("")
    elif failure == "invalid_json":
        report.write_text("not json")
    elif failure == "wrong_shape":
        report.write_text("[]")
    elif failure == "empty_object":
        report.write_text("{}")
    elif failure == "missing_exit":
        status.unlink()
    elif failure == "crashed":
        status.write_text("2")
    elif failure == "bad_exit":
        status.write_text("success")
    else:
        status.write_text("1")
    result = check_reports(root)
    assert not result["passed"]
    assert any(issue.startswith(name + ":") for issue in result["issues"])


@pytest.mark.parametrize(
    "severity,blocked",
    [
        ("ERROR", True),
        ("HIGH", True),
        ("CRITICAL", True),
        ("WARNING", False),
        ("MEDIUM", False),
        ("INFO", False),
        ("LOW", False),
    ],
)
def test_semgrep_severity_aliases_cannot_bypass_gate(reports, severity, blocked):
    root, documents = reports
    payload = copy.deepcopy(documents["semgrep"])
    payload["results"] = [{"extra": {"severity": severity}}]
    write(reports, "semgrep", payload)
    (root / "semgrep-exit-code.txt").write_text("1")
    assert check_reports(root)["passed"] is (not blocked)


@pytest.mark.parametrize("name", ["bandit", "semgrep"])
@pytest.mark.parametrize("errors", [None, "", [{"message": "scan timed out"}]])
def test_partial_scan_errors_are_blocking_even_with_zero_findings(
    reports, name, errors
):
    _, documents = reports
    payload = copy.deepcopy(documents[name])
    payload["errors"] = errors
    write(reports, name, payload)
    assert not check_reports(reports[0])["passed"]


@pytest.mark.parametrize(
    "patch",
    [
        {"paths": {}},
        {"paths": {"scanned": []}},
        {"version": ""},
        {"results": [None]},
        {"results": [{"extra": {}}]},
        {"results": [{"extra": {"severity": "not_known"}}]},
    ],
)
def test_semgrep_requires_complete_scan_metadata(reports, patch):
    payload = {**reports[1]["semgrep"], **patch}
    write(reports, "semgrep", payload)
    assert not check_reports(reports[0])["passed"]


@pytest.mark.parametrize("field", ["Vulnerabilities", "Secrets", "Misconfigurations"])
def test_trivy_high_findings_in_each_scanner_are_blocking(reports, field):
    payload = copy.deepcopy(reports[1]["trivy"])
    payload["Results"][0][field] = [{"Severity": "HIGH", "Status": "FAIL"}]
    write(reports, "trivy", payload)
    result = check_reports(reports[0])
    assert not result["passed"] and result["scans"]["trivy"]["blocked"] == 1


@pytest.mark.parametrize("status", ["PASS", "EXCEPTION"])
def test_trivy_nonfailed_checks_do_not_become_vulnerabilities(reports, status):
    payload = copy.deepcopy(reports[1]["trivy"])
    payload["Results"][0]["Misconfigurations"] = [
        {"Severity": "HIGH", "Status": status}
    ]
    write(reports, "trivy", payload)
    assert check_reports(reports[0])["passed"]


@pytest.mark.parametrize(
    "result",
    [
        {"Target": "x", "Vulnerabilities": [None]},
        {"Target": "x", "Vulnerabilities": [{}]},
        {"Target": "x", "Secrets": None},
        {"Target": "x", "Misconfigurations": [{"Severity": "HIGH", "Status": []}]},
        {},
    ],
)
def test_trivy_malformed_entries_are_not_silently_skipped(reports, result):
    payload = {**reports[1]["trivy"], "Results": [result]}
    write(reports, "trivy", payload)
    assert not check_reports(reports[0])["passed"]


def test_workflow_preserves_all_exit_codes_and_always_evaluates_and_uploads():
    path = Path(__file__).resolve().parents[2] / ".github/workflows/security.yml"
    workflow = yaml.safe_load(path.read_text())
    steps = workflow["jobs"]["security-scan"]["steps"]
    gate = next(
        step
        for step in steps
        if "scripts.security_workflow_gate" in step.get("run", "")
    )
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact")
    )
    assert gate["if"] == upload["if"] == "always()"
    assert "--source-root stateset_agents" in gate["run"]
    assert steps.index(gate) < steps.index(upload)
    for name in ("bandit", "pip-audit", "semgrep", "trivy"):
        scan = next(
            step["run"]
            for step in steps
            if f"> {name}-exit-code.txt" in step.get("run", "")
        )
        assert "|| scan_exit_code=$?" in scan
        assert "|| true" not in scan
        assert f"{name}-exit-code.txt" in upload["with"]["path"]
        if name == "bandit":
            # Bandit's progress bar can appear on stdout before JSON output.
            assert "-o bandit-report.json" in scan
    assert "--config=auto" not in path.read_text()


@pytest.mark.parametrize("scanner_exit", [0, 2])
def test_workflow_shell_gate_propagates_failure_and_saves_evidence(
    reports, scanner_exit
):
    """Execute the checked-in gate step with a clean report or a crashed scanner."""
    root, _ = reports
    repo = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((repo / ".github/workflows/security.yml").read_text())
    gate = next(
        step["run"]
        for step in workflow["jobs"]["security-scan"]["steps"]
        if "scripts.security_workflow_gate" in step.get("run", "")
    )
    source = root / "stateset_agents"
    source.mkdir()
    (source / "example.py").write_text("value = 1\n")
    (root / "semgrep-exit-code.txt").write_text(str(scanner_exit))
    summary = root / "summary.txt"
    environment = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(repo),
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", gate],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    passed = scanner_exit == 0
    assert result.returncode == (0 if passed else 1), result.stderr
    decision = json.loads((root / "security-gate.json").read_text())
    assert decision["passed"] is passed
    assert ("Security gate passed" if passed else "Security gate failed") in (
        summary.read_text()
    )
    if not passed:
        assert any("code 2" in issue for issue in decision["issues"])
