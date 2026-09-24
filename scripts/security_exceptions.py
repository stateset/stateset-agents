"""Narrow, time-limited exceptions for the release security gate."""

from datetime import date
from typing import Any

# Approved for v0.56.0: upstream closed the proposed fixes as outside its
# security policy. Reassess before this date or when a patched release appears.
ACCELERATE_EXCEPTION_REVIEW_DATE = date(2026, 12, 31)


def is_approved_safety_exception(issue: Any, *, today: date | None = None) -> bool:
    """Recognize only the reviewed Accelerate lockfile finding."""
    if not isinstance(issue, dict):
        return False
    if today is None:
        today = date.today()
    return today <= ACCELERATE_EXCEPTION_REVIEW_DATE and all(
        issue.get(key) == value
        for key, value in {
            "package_name": "accelerate",
            "analyzed_version": "1.14.0",
            "vulnerability_id": "SFTY-20260810-08074",
            "CVE": "CVE-2026-69112",
        }.items()
    )
