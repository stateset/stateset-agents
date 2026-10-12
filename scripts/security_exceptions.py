"""Narrow, time-limited exceptions for the release security gate."""

from datetime import date
from typing import Any

# Approved for v0.56.0: upstream closed the proposed fixes as outside its
# security policy. Reassess before this date or when a patched release appears.
ACCELERATE_EXCEPTION_REVIEW_DATE = date(2026, 12, 31)


def is_approved_safety_exception(issue: Any, *, today: date | None = None) -> bool:
    """Recognize only narrowly reviewed Safety findings (exact match + expiry)."""
    if not isinstance(issue, dict):
        return False
    if today is None:
        today = date.today()
    # Catalog of approved exceptions: exact package, analyzed version, Safety id, and CVE.
    approved: list[dict[str, Any]] = [
        {
            "package_name": "accelerate",
            "analyzed_version": "1.14.0",
            "vulnerability_id": "SFTY-20260810-08074",
            "CVE": "CVE-2026-69112",
            "review_until": ACCELERATE_EXCEPTION_REVIEW_DATE,
        },
        # Safety pulls nltk transitively; no patched release exists yet.
        {
            "package_name": "nltk",
            "analyzed_version": "3.10.3",
            "vulnerability_id": "SFTY-20260902-58666",
            "CVE": "CVE-2026-81726",
            # Time-bound; reassess or remove when an upstream fix lands.
            "review_until": date(2027, 3, 31),
        },
        # datasets 5.0.0 path traversal; upgrade to 5.0.1+ when compatible everywhere.
        {
            "package_name": "datasets",
            "analyzed_version": "5.0.0",
            "vulnerability_id": "SFTY-20260724-01754",
            "CVE": "CVE-2026-66007",
            "review_until": date(2027, 3, 31),
        },
    ]
    for entry in approved:
        if today <= entry["review_until"] and all(
            issue.get(key) == value
            for key, value in {
                "package_name": entry["package_name"],
                "analyzed_version": entry["analyzed_version"],
                "vulnerability_id": entry["vulnerability_id"],
                "CVE": entry["CVE"],
            }.items()
        ):
            return True
    return False
