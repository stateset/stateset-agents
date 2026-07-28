#!/usr/bin/env python3
"""Fail the build if bandit/safety reports contain high-severity findings.

Used by `make security-scan-strict` (see Makefile). Extracted from an
inline Makefile heredoc because multi-line heredocs are not portable
across `make` recipe lines without `.ONESHELL` (each recipe line runs
in its own shell by default, which broke the previous inline version).
"""

import json
import sys
from pathlib import Path
from typing import Any


def _load_json_lenient(text: str) -> Any:
    """Parse a JSON object out of text that may have extra content around it.

    Newer `safety` CLI releases (3.x) print a banner/deprecation notice and
    "brought to you by safetycli.com" ASCII art to stdout even with --json,
    surrounding (not just prefixing) the actual JSON payload. Locate the
    first '{' and let json.JSONDecoder consume only the balanced object that
    follows, ignoring any trailing banner text.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in output")
    obj, _end = json.JSONDecoder().raw_decode(text, start)
    return obj


def main() -> int:
    bandit_path = Path("bandit-report.json")
    safety_path = Path("safety-report.json")

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

    if not safety_path.exists() or not safety_path.read_text().strip():
        print("Safety report not generated; ensure safety is installed")
        return 1

    try:
        safety_payload = _load_json_lenient(safety_path.read_text())
    except Exception as exc:
        print(f"Safety output parse failed: {exc}")
        return 1

    if isinstance(safety_payload, list):
        vulns = safety_payload
    elif isinstance(safety_payload, dict):
        vulns = safety_payload.get("vulnerabilities", [])
    else:
        vulns = []

    high = [
        v for v in vulns if str(v.get("severity", "")).upper() in {"HIGH", "CRITICAL"}
    ]

    if high:
        for v in high[:10]:
            print(
                f"High severity vulnerability: {v.get('package_name', 'unknown')} {v.get('id', '')}"
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
