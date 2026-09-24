"""Dependency audit failures remain blocking across the release gates."""

import json
from datetime import date
from pathlib import Path

import pytest

from scripts import security_exceptions
from scripts.check_security_findings import main


def _write_reports(directory: Path, vulnerabilities: list[dict[str, object]]) -> None:
    (directory / "bandit-report.json").write_text(
        json.dumps({"errors": [], "results": []}), encoding="utf-8"
    )
    (directory / "pip-audit-report.json").write_text(
        json.dumps(
            {
                "dependencies": [
                    {
                        "name": "accelerate",
                        "version": "1.14.0",
                        "vulns": vulnerabilities,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_security_gate_accepts_only_reviewed_accelerate_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    class FixedDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 9, 23)

    monkeypatch.setattr(security_exceptions, "date", FixedDate)
    approved = {
        "id": "PYSEC-2026-3804",
        "aliases": ["GHSA-4j2p-28q2-5m79", "CVE-2026-69112"],
    }
    _write_reports(tmp_path, [approved])
    assert main() == 0

    _write_reports(tmp_path, [approved, {"id": "PYSEC-new", "aliases": []}])
    assert main() == 1


def test_security_gate_rejects_missing_or_malformed_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bandit-report.json").write_text(
        '{"errors": [], "results": []}', encoding="utf-8"
    )
    assert main() == 1
    (tmp_path / "pip-audit-report.json").write_text("{}", encoding="utf-8")
    assert main() == 1
