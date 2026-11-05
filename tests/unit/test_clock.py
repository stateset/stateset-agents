"""Timezone-aware UTC helpers replace the deprecated ``datetime.utcnow()``."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stateset_agents.utils.clock import as_utc, utc_isoformat, utc_now

PACKAGE = Path(__file__).resolve().parents[2] / "stateset_agents"


def test_utc_now_is_timezone_aware() -> None:
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_utc_isoformat_matches_the_legacy_naive_plus_z_format() -> None:
    naive = datetime(2026, 10, 22, 18, 30, 5, 123456)
    legacy = naive.isoformat() + "Z"
    assert utc_isoformat(naive) == legacy
    assert utc_isoformat(naive.replace(tzinfo=timezone.utc)) == legacy
    shifted = naive.replace(tzinfo=timezone.utc).astimezone(
        timezone(timedelta(hours=-7))
    )
    assert utc_isoformat(shifted) == legacy
    whole = datetime(2026, 10, 22, 18, 30, 5)
    assert utc_isoformat(whole) == "2026-10-22T18:30:05Z"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{6})?Z", utc_isoformat())


def test_as_utc_takes_naive_values_as_utc() -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0)
    assert as_utc(naive) == naive.replace(tzinfo=timezone.utc)
    other = datetime(2026, 1, 1, 5, 0, 0, tzinfo=timezone(timedelta(hours=-7)))
    assert as_utc(other) == naive.replace(tzinfo=timezone.utc)


def test_no_naive_utc_clock_calls_remain_in_the_package() -> None:
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        if path.name == "clock.py":
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if re.search(r"\butcnow\b|\butcfromtimestamp\b", line):
                offenders.append(f"{path.relative_to(PACKAGE)}:{lineno}")
    assert offenders == [], (
        "Use stateset_agents.utils.clock.utc_now()/utc_isoformat() instead of "
        "the deprecated naive datetime.utcnow(): " + ", ".join(offenders)
    )


def test_training_queue_reads_legacy_naive_timestamps_as_utc() -> None:
    from stateset_agents.api.training_queue import TrainingQueue

    decoded = TrainingQueue._decode(
        '{"created_at": "2026-01-01T12:00:00", "started_at": null, '
        '"completed_at": "2026-01-01T12:00:00+00:00"}'
    )
    assert decoded["created_at"] == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert decoded["completed_at"] == decoded["created_at"]
    assert decoded["started_at"] is None
