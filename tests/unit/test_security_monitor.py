"""Regression tests for bounded security event retention."""

from datetime import datetime, timedelta, timezone

import pytest

from stateset_agents.utils.security import SecurityMonitor


def test_security_monitor_retains_only_recent_events() -> None:
    """A stream of auth failures cannot grow the in-memory event list forever."""
    monitor = SecurityMonitor(max_events=11)
    for index in range(12):
        monitor.log_security_event(
            "authentication_failure", {"index": index}, severity="debug"
        )

    assert len(monitor.events) == 11
    assert [event["details"]["index"] for event in monitor.events] == list(range(1, 12))
    assert len(monitor.get_recent_events()) == 11
    assert monitor.detect_anomalies() == [
        {
            "type": "high_frequency",
            "event_type": "authentication_failure",
            "count": 11,
            "threshold": 10,
        }
    ]


def test_security_monitor_reads_legacy_naive_utc_events() -> None:
    """Older naive UTC event timestamps remain usable after the UTC change."""
    monitor = SecurityMonitor(max_events=2)
    monitor.events.append(
        {
            "timestamp": (datetime.now(timezone.utc) - timedelta(hours=1))
            .replace(tzinfo=None)
            .isoformat(),
            "type": "legacy",
        }
    )
    assert len(monitor.get_recent_events(hours=2)) == 1
    assert monitor.get_recent_events(hours=0) == []


@pytest.mark.parametrize("max_events", [0, -1, True, 1.5, "10"])
def test_security_monitor_rejects_invalid_capacity(max_events: object) -> None:
    """A non-positive or non-integer retention limit is a configuration error."""
    with pytest.raises(ValueError, match="max_events"):
        SecurityMonitor(max_events=max_events)  # type: ignore[arg-type]
