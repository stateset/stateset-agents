"""Tests for fail-closed retained provider evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
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
    checks: dict[str, Any] = {"billable_resources_created": 0}
    if provider == "river":
        checks.update({"health": True, "capabilities": ["model"]})
    elif provider == "runpod":
        checks.update(
            {
                "pods_observed": 0,
                "local_cleanup_leases": 0,
                "canary_leftovers": [],
                "ephemeral_training_leftovers": [],
            }
        )
    elif provider == "fireworks":
        checks.update(
            {
                "models_observed": 1,
                "jobs_observed": 0,
                "deployments_observed": 0,
                "canary_leftovers": [],
            }
        )
    elif provider == "coreweave":
        checks.update(
            {
                "provider": "coreweave",
                "authenticated": True,
                "can_create_jobs": True,
            }
        )
    elif provider == "nebius":
        checks.update(
            {
                "provider": "nebius",
                "authenticated": True,
                "response_type": "dict",
            }
        )
    return {
        "schema_version": 2,
        "kind": "stateset-provider-canary-evidence",
        "framework_version": "0.54.0",
        "harness_commit": "a" * 40,
        "harness_clean": True,
        "billable_resources_created": 0,
        "results": [
            {
                "provider": provider,
                "status": status,
                "checked_at": "2026-08-27T12:00:00+00:00",
                "duration_ms": 10,
                "checks": checks,
                "cleanup_verified": passed,
                "error": None if passed else "missing credentials",
            }
        ],
    }


def test_complete_provider_matrix_passes() -> None:
    report = provider_evidence.validate_matrix(
        [_report(provider) for provider in provider_evidence.REQUIRED_PROVIDERS],
        minimum_schema_version=2,
        expected_commit="a" * 40,
        expected_version="0.54.0",
    )
    assert report["passed"] is True
    assert report["skipped"] == []
    assert report["max_age_days"] == 30
    assert report["harness_commit"] == "a" * 40


def test_a_plus_provider_matrix_rejects_legacy_or_wrong_source() -> None:
    reports = [_report(provider) for provider in provider_evidence.REQUIRED_PROVIDERS]
    reports[0]["schema_version"] = 1
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="schema_version"):
        provider_evidence.validate_matrix(reports, minimum_schema_version=2)


def test_provider_matrix_rejects_missing_adapter_observations() -> None:
    reports = [_report(provider) for provider in provider_evidence.REQUIRED_PROVIDERS]
    reports[1]["results"][0]["checks"].pop("ephemeral_training_leftovers")
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="runpod"):
        provider_evidence.validate_matrix(reports)

    reports = [_report(provider) for provider in provider_evidence.REQUIRED_PROVIDERS]
    reports[3]["results"][0]["checks"]["can_create_jobs"] = False
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="coreweave"):
        provider_evidence.validate_matrix(reports)

    reports[0]["schema_version"] = 2
    reports[0]["harness_commit"] = "b" * 40
    with pytest.raises(
        provider_evidence.ProviderEvidenceError, match="commit mismatch"
    ):
        provider_evidence.validate_matrix(
            reports,
            minimum_schema_version=2,
            expected_commit="a" * 40,
            expected_version="0.54.0",
        )

    reports[0]["harness_commit"] = "a" * 40
    reports[0]["harness_clean"] = False
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="incomplete"):
        provider_evidence.validate_matrix(reports, minimum_schema_version=2)


def test_skipped_provider_fails_closed() -> None:
    reports = [
        _report(provider, "skipped" if provider == "fireworks" else "passed")
        for provider in provider_evidence.REQUIRED_PROVIDERS
    ]
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="not 'passed'"):
        provider_evidence.validate_matrix(reports)

    diagnostic = provider_evidence.validate_matrix(reports, allow_skipped=True)
    assert diagnostic["schema_version"] == 1
    assert diagnostic["passed"] is False
    assert diagnostic["providers"] == list(provider_evidence.REQUIRED_PROVIDERS)
    assert diagnostic["skipped"] == ["fireworks"]


def test_rejects_duplicate_or_missing_provider() -> None:
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="duplicate"):
        provider_evidence.validate_matrix(
            [_report("river"), _report("river"), _report("fireworks")]
        )
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="mismatch"):
        provider_evidence.validate_matrix([_report("river"), _report("runpod")])


def test_rejects_stale_or_future_provider_evidence() -> None:
    reports = [_report(provider) for provider in provider_evidence.REQUIRED_PROVIDERS]
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)
    reports[0]["results"][0]["checked_at"] = "2026-07-01T00:00:00+00:00"
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="older than"):
        provider_evidence.validate_matrix(reports, now=now)

    reports[0]["results"][0]["checked_at"] = "2026-09-11T00:00:00+00:00"
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="future"):
        provider_evidence.validate_matrix(reports, now=now)


def test_loader_rejects_billable_or_malformed_report(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    billable = _report("runpod")
    billable["billable_resources_created"] = 1
    path.write_text(json.dumps(billable), encoding="utf-8")
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="billable"):
        provider_evidence.load_reports([path])

    malformed = _report("runpod")
    malformed["results"][0]["duration_ms"] = -1
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="envelope"):
        provider_evidence.load_reports([path])


def test_loader_rejects_symlinked_provider_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps(_report("runpod")), encoding="utf-8")
    link = tmp_path / "linked.json"
    link.symlink_to(target)
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="symlinked"):
        provider_evidence.load_reports([link])


def test_retained_provider_matrix_is_explicitly_incomplete() -> None:
    reports = provider_evidence.load_reports(
        [ROOT / "benchmark_results" / "provider_canaries"]
    )
    historical = ("river", "runpod", "fireworks")
    diagnostic = provider_evidence.validate_matrix(
        reports, required=historical, allow_skipped=True
    )
    assert diagnostic["passed"] is False
    assert diagnostic["skipped"] == ["fireworks"]
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="fireworks"):
        provider_evidence.validate_matrix(reports, required=historical)
    with pytest.raises(provider_evidence.ProviderEvidenceError, match="coreweave"):
        provider_evidence.validate_matrix(reports, allow_skipped=True)
