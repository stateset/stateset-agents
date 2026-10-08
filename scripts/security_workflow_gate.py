"""Validate complete scanner reports and their process outcomes before passing CI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.security_exceptions import classify_pip_audit_findings


@dataclass(frozen=True)
class ScanSummary:
    """Counts from a structurally valid scan, without disclosing matched source."""

    findings: int
    blocked: int
    scanned: int | None = None


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _objects(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be an array of objects")
    return value


def _scanned_results(payload: Any, name: str) -> list[dict[str, Any]]:
    report = _object(payload, name)
    if not isinstance(report.get("errors"), list):
        raise ValueError(f"{name} report is missing its errors array")
    if report["errors"]:
        raise ValueError(f"{name} scan contains execution errors")
    return _objects(report.get("results"), f"{name} results")


def _severity(value: Any, *, aliases: bool = False, unknown: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("Finding severity must be explicit text")
    severity = value.upper()
    if aliases:
        severity = {"ERROR": "HIGH", "WARNING": "MEDIUM", "INFO": "LOW"}.get(
            severity, severity
        )
    if severity not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"} and not (
        unknown and severity == "UNKNOWN"
    ):
        raise ValueError("Finding severity is not recognized")
    return severity


def validate_bandit_report(payload: Any) -> ScanSummary:
    """Reject scan errors and count medium-or-higher Bandit findings."""
    results = _scanned_results(payload, "Bandit")
    blocked = sum(
        _severity(item.get("issue_severity")) in {"MEDIUM", "HIGH", "CRITICAL"}
        for item in results
    )
    return ScanSummary(len(results), blocked)


def validate_semgrep_report(payload: Any) -> ScanSummary:
    """Reject partial/empty scans and recognize old and new severity names."""
    results = _scanned_results(payload, "Semgrep")
    if not isinstance(payload.get("version"), str) or not payload["version"]:
        raise ValueError("Semgrep report is missing its scanner version")
    paths = _object(payload.get("paths"), "Semgrep paths").get("scanned")
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(p, str) or not p for p in paths)
    ):
        raise ValueError("Semgrep report must identify scanned files")
    blocked = 0
    for item in results:
        extra = _object(item.get("extra"), "Semgrep finding extra")
        blocked += _severity(extra.get("severity"), aliases=True) in {
            "HIGH",
            "CRITICAL",
        }
    return ScanSummary(len(results), blocked, len(paths))


def validate_trivy_report(payload: Any) -> ScanSummary:
    """Validate Trivy results, including secrets and failed misconfigurations."""
    report = _object(payload, "Trivy")
    if type(report.get("SchemaVersion")) is not int or report["SchemaVersion"] != 2:
        raise ValueError("Expected Trivy report schema 2")
    if not isinstance(report.get("ArtifactName"), str) or not report["ArtifactName"]:
        raise ValueError("Trivy report must identify the scanned artifact")
    results = _objects(report.get("Results"), "Trivy Results")
    if not results:
        raise ValueError("Trivy report contains no scanned targets")
    findings = blocked = 0
    for result in results:
        if not isinstance(result.get("Target"), str) or not result["Target"]:
            raise ValueError("Trivy result must identify its target")
        for key in ("Vulnerabilities", "Secrets", "Misconfigurations"):
            for item in _objects(result.get(key, []), f"Trivy {key}"):
                findings += 1
                severity = _severity(item.get("Severity"), unknown=True)
                if key == "Misconfigurations":
                    status = item.get("Status")
                    if not isinstance(status, str) or status not in {
                        "PASS",
                        "FAIL",
                        "EXCEPTION",
                    }:
                        raise ValueError("Trivy misconfiguration status is invalid")
                    if status != "FAIL":
                        continue
                blocked += severity in {"HIGH", "CRITICAL"}
    return ScanSummary(findings, blocked, len(results))


def validate_audit_report(payload: Any) -> ScanSummary:
    """Apply the existing reviewed dependency exceptions."""
    approved, blocked = classify_pip_audit_findings(payload)
    return ScanSummary(
        len(approved) + len(blocked), len(blocked), len(payload["dependencies"])
    )


VALIDATORS = {
    "bandit": validate_bandit_report,
    "pip-audit": validate_audit_report,
    "semgrep": validate_semgrep_report,
    "trivy": validate_trivy_report,
}


def check_reports(
    directory: Path, *, source_root: Path | None = None
) -> dict[str, Any]:
    """Fail closed on missing reports, execution failures, and blocking findings."""
    issues: list[str] = []
    scans: dict[str, Any] = {}
    for name, validator in VALIDATORS.items():
        try:
            exit_code = int(
                (directory / f"{name}-exit-code.txt")
                .read_text(encoding="utf-8")
                .strip()
            )
            allowed = {0} if name == "trivy" else {0, 1}
            if exit_code not in allowed:
                raise ValueError(f"scanner exited with code {exit_code}")
            payload = json.loads(
                (directory / f"{name}-report.json").read_text(encoding="utf-8")
            )
            summary = validator(payload)
            if name == "semgrep" and source_root is not None:
                expected = {
                    path.relative_to(source_root.parent).as_posix()
                    for path in source_root.rglob("*.py")
                    if path.is_file() and path.stat().st_size
                }
                if not expected:
                    raise ValueError(
                        "no packaged Python sources found for coverage check"
                    )
                scanned = {
                    Path(path).as_posix() for path in payload["paths"]["scanned"]
                }
                missing = sorted(expected - scanned)
                if missing:
                    raise ValueError(
                        f"{len(missing)} packaged sources were not scanned: {', '.join(missing[:5])}"
                    )
            if exit_code == 1 and not summary.findings:
                raise ValueError("finding exit code without reported findings")
            scans[name] = {"exit_code": exit_code, **asdict(summary)}
            if summary.blocked:
                issues.append(f"{name}: {summary.blocked} blocking findings")
        except (OSError, ValueError) as exc:
            issues.append(f"{name}: {exc}")
    return {"schema_version": 1, "passed": not issues, "scans": scans, "issues": issues}


def main(argv: list[str] | None = None) -> int:
    """Check CI scanner artifacts and optionally save a machine-readable decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Require every nonempty Python source under this package in the Semgrep report",
    )
    args = parser.parse_args(argv)
    result = check_reports(args.reports_dir, source_root=args.source_root)
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
