"""Helpers for asynchronous terminal interactions."""

import time

import pytest


@pytest.fixture
def settle_preview():
    """Wait for preview completion while applying results on the UI thread."""

    def settle(studio):
        deadline = time.monotonic() + 10
        while studio.planning.busy and time.monotonic() < deadline:
            studio.refresh()
            time.sleep(0.005)
        assert not studio.planning.busy, "Preview did not finish"

    return settle
