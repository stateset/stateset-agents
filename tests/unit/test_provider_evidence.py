"""Tests for fail-closed retained provider evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[2] / "benchmarks" / "provider_evidence.py"
ROOT = MODULE.parents[1]
SPEC = importlib.util.spec_from_file_location("provider_evidence", MODULE)
assert SPEC is not None and SPEC.loader is not None
provider_evidence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provider_evidence
SPEC.loader.exec_module(provider_evidence)


def _report(provider: str, status: str = "passed") -> dict[str, Any]:
    passed = status == "passed"
    return {
        "schema_version": 1,
        "billable_resources_created": 0,
        "results": [
            {
                "provider": provider,
                "status": status,
                "checked_at": "2026-08-27T12:00:00+00:00",
                "duration_ms": 10,
                "checks": {"billable_resources_created": 0},
                "cleanup_verified": passed,
                "error": None if passed else "missing credentials",
            }
        ],
    }


def test_complete_provider_matrix_passes() -> None:
    report = provider_evidence.validate_matrix(
        [_report(provider) for provider in ("river", "runpod", "fireworks")]
    )
    assert report["passed"] is True
    assert report["skipped"] == []


def test_skipped_provider_fails_closed() -> None:
    reports = [_report("river"), _report("runpod"), _report("fireworks", "skipped")]
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="not 'passed'"):
        provider_evidence.validate_matrix(reports)

    diagnostic = provider_evidence.validate_matrix(reports, allow_skipped=True)
    assert diagnostic == {
        "schema_version": 1,
        "passed": False,
        "providers": ["river", "runpod", "fireworks"],
        "skipped": ["fireworks"],
    }


def test_rejects_duplicate_or_missing_provider() -> None:
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="duplicate"):
        provider_evidence.validate_matrix(
            [_report("river"), _report("river"), _report("fireworks")]
        )
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="mismatch"):
        provider_evidence.validate_matrix([_report("river"), _report("runpod")])


def test_loader_rejects_billable_or_malformed_report(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    billable = _report("runpod")
    billable["billable_resources_created"] = 1
    path.write_text(json.dumps(billable), encoding="utf-8")
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="billable"):
        provider_evidence.load_reports([path])


def test_retained_provider_matrix_is_explicitly_incomplete() -> None:
    reports = provider_evidence.load_reports(
        [ROOT / "benchmark_results" / "provider_canaries"]
    )
    diagnostic = provider_evidence.validate_matrix(reports, allow_skipped=True)
    assert diagnostic["passed"] is False
    assert diagnostic["skipped"] == ["fireworks"]
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="fireworks"):
        provider_evidence.validate_matrix(reports)
