"""Helpers for asynchronous terminal interactions."""

import time

import pytest


def settle_studio(studio, timeout=10):
    """Apply finished preview and launch/stop/review work on the UI thread."""
    deadline = time.monotonic() + timeout
    while (studio.planning.busy or studio.submission.busy) and (
        time.monotonic() < deadline
    ):
        studio.refresh()
        time.sleep(0.005)
    assert not studio.planning.busy, "Preview did not finish"
    assert not studio.submission.busy, "Submission did not finish"


@pytest.fixture
def settle_preview():
    """Wait for preview completion while applying results on the UI thread."""
    return settle_studio
