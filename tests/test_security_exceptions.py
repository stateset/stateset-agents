"""Tests for the narrowly scoped release security exception."""

from datetime import date

from scripts.security_exceptions import is_approved_safety_exception

FINDING = {
    "package_name": "accelerate",
    "analyzed_version": "1.14.0",
    "vulnerability_id": "SFTY-20260810-08074",
    "CVE": "CVE-2026-69112",
}


def test_approved_exception_matches_only_reviewed_finding() -> None:
    assert is_approved_safety_exception(FINDING, today=date(2026, 9, 23))
    for key in FINDING:
        changed = {**FINDING, key: "different"}
        assert not is_approved_safety_exception(changed, today=date(2026, 9, 23))
        missing = {name: value for name, value in FINDING.items() if name != key}
        assert not is_approved_safety_exception(missing, today=date(2026, 9, 23))
    assert not is_approved_safety_exception(None, today=date(2026, 9, 23))
    assert not is_approved_safety_exception(FINDING, today=date(2027, 1, 1))
