#!/usr/bin/env python3
"""Fail the build if Bandit or pip-audit reports contain unapproved findings.

Used by `make security-scan-strict` (see Makefile). Extracted from an
inline Makefile heredoc because multi-line heredocs are not portable
across `make` recipe lines without `.ONESHELL` (each recipe line runs
in its own shell by default, which broke the previous inline version).
"""

import json
import sys
from pathlib import Path
from typing import Any

from scripts.security_exceptions import classify_pip_audit_findings


def _load_json_lenient(text: str) -> Any:
    """Parse a JSON object out of text that may have extra content around it.

    Retain compatibility with Bandit output that may include surrounding text.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in output")
    obj, _end = json.JSONDecoder().raw_decode(text, start)
    return obj


def main() -> int:
    bandit_path = Path("bandit-report.json")
    audit_path = Path("pip-audit-report.json")

    if not bandit_path.exists() or not bandit_path.read_text().strip():
        print("Bandit report not generated")
        return 1

    try:
        bandit_payload = _load_json_lenient(bandit_path.read_text())
    except Exception as exc:
        print(f"Bandit output parse failed: {exc}")
        return 1

    bandit_results = []
    if isinstance(bandit_payload, dict):
        bandit_results = bandit_payload.get("results", [])
    elif isinstance(bandit_payload, list):
        bandit_results = bandit_payload

    high_findings = [
        item
        for item in bandit_results
        if str(item.get("issue_severity", "")).upper() in {"MEDIUM", "HIGH", "CRITICAL"}
    ]

    if high_findings:
        for item in high_findings[:10]:
            print(
                f"Bandit: {item.get('filename')}:{item.get('line_number')} "
                f"{item.get('test_id')} {item.get('issue_severity')}"
            )
        print(
            f"Bandit: failing with {len(high_findings)} medium/high/critical findings"
        )
        return 1

    if not audit_path.exists() or not audit_path.read_text().strip():
        print("pip-audit report not generated; ensure pip-audit is installed")
        return 1

    try:
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        approved, blocked = classify_pip_audit_findings(audit_payload)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"pip-audit output parse failed: {exc}")
        return 1

    for finding in approved:
        print(f"Approved, time-limited dependency exception: {finding}")
    if blocked:
        for finding in blocked[:10]:
            print(f"Dependency vulnerability: {finding}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
