"""Timezone-aware UTC time for the framework (one implementation).

``datetime.utcnow()`` returns a naive datetime and is deprecated since Python
3.12. Use :func:`utc_now` for timezone-aware values and :func:`utc_isoformat`
for the ``...Z`` wire strings the gateway has always emitted.
"""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["as_utc", "utc_isoformat", "utc_now"]


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def as_utc(moment: datetime) -> datetime:
    """Return ``moment`` in UTC; naive values (older records) are taken as UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def utc_isoformat(moment: datetime | None = None) -> str:
    """Format ``moment`` (default: now) as ``YYYY-MM-DDTHH:MM:SS[.ffffff]Z``.

    This is byte-for-byte the string ``naive_utc.isoformat() + "Z"`` produced,
    for naive (assumed UTC) and aware inputs alike.
    """
    value = utc_now() if moment is None else as_utc(moment)
    return value.replace(tzinfo=None).isoformat() + "Z"
