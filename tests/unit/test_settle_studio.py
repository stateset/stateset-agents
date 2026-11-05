"""The studio settle helper waits out slow runners without hiding stuck work."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


class FakeTime:
    """Monotonic time that only moves when the helper sleeps."""

    def __init__(self, step: float) -> None:
        self.now, self.step = 0.0, step

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += self.step


def studio_finishing_at(clock: FakeTime, done_at: float | None) -> SimpleNamespace:
    """A studio whose submission result is applied by the first refresh after it."""
    studio = SimpleNamespace(
        planning=SimpleNamespace(busy=False),
        submission=SimpleNamespace(busy=True),
        refreshes=0,
    )

    def refresh() -> None:
        studio.refreshes += 1
        if done_at is not None and clock.now >= done_at:
            studio.submission.busy = False

    studio.refresh = refresh
    return studio


@pytest.fixture
def fake_time(monkeypatch, settle_preview):
    clock = FakeTime(step=0.005)
    monkeypatch.setattr(sys.modules[settle_preview.__module__], "time", clock)
    return clock


def test_slow_runner_submission_settles_within_the_default_budget(
    fake_time, settle_preview
):
    """Windows CI took over 10 s to verify a launch; that is slow, not stuck."""
    studio = studio_finishing_at(fake_time, done_at=15.0)
    settle_preview(studio)
    assert not studio.submission.busy
    assert 15.0 <= fake_time.now < 15.1


def test_work_finishing_during_the_final_sleep_is_applied(fake_time, settle_preview):
    fake_time.step = 1.0
    studio = studio_finishing_at(fake_time, done_at=3.0)
    settle_preview(studio, 3)
    assert not studio.submission.busy and studio.refreshes == 4


def test_stuck_submission_still_fails_at_the_deadline(fake_time, settle_preview):
    fake_time.step = 1.0
    studio = studio_finishing_at(fake_time, done_at=None)
    with pytest.raises(AssertionError, match="Submission did not finish in 3.0s"):
        settle_preview(studio, 3)
    assert studio.refreshes == 4 and fake_time.now == 3.0


def test_idle_studio_is_not_refreshed(fake_time, settle_preview):
    studio = studio_finishing_at(fake_time, done_at=0.0)
    studio.submission.busy = False
    settle_preview(studio)
    assert studio.refreshes == 0 and fake_time.now == 0.0
