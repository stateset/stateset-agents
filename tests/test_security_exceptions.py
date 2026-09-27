"""Tests for the narrowly scoped release security exception."""

from datetime import date

import pytest

from scripts.security_exceptions import (
    classify_pip_audit_findings,
    is_approved_pip_audit_exception,
)


def test_pip_audit_exception_requires_exact_advisory_and_expiry() -> None:
    vulnerability = {
        "id": "PYSEC-2026-3804",
        "aliases": ["GHSA-4j2p-28q2-5m79", "CVE-2026-69112"],
    }
    assert is_approved_pip_audit_exception(
        "accelerate", "1.14.0", vulnerability, today=date(2026, 9, 23)
    )
    assert not is_approved_pip_audit_exception(
        "accelerate", "1.15.0", vulnerability, today=date(2026, 9, 23)
    )
    assert not is_approved_pip_audit_exception(
        "nltk", "1.14.0", vulnerability, today=date(2026, 9, 23)
    )
    assert not is_approved_pip_audit_exception(
        "accelerate",
        "1.14.0",
        {**vulnerability, "aliases": []},
        today=date(2026, 9, 23),
    )
    assert not is_approved_pip_audit_exception(
        "accelerate", "1.14.0", vulnerability, today=date(2027, 1, 1)
    )


def test_pip_audit_classification_fails_closed() -> None:
    payload = {
        "dependencies": [
            {
                "name": "accelerate",
                "version": "1.14.0",
                "vulns": [
                    {
                        "id": "PYSEC-2026-3804",
                        "aliases": ["GHSA-4j2p-28q2-5m79", "CVE-2026-69112"],
                    }
                ],
            },
            {"name": "nltk", "version": "3.10.3", "vulns": [{"id": "PYSEC-2026-3740"}]},
        ]
    }
    approved, blocked = classify_pip_audit_findings(payload, today=date(2026, 9, 23))
    assert approved == ["accelerate 1.14.0: PYSEC-2026-3804"]
    assert blocked == ["nltk 3.10.3: PYSEC-2026-3740"]
    for invalid in ({}, {"dependencies": []}, {"dependencies": [{}]}):
        with pytest.raises(ValueError):
            classify_pip_audit_findings(invalid)
