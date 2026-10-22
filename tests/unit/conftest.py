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


@pytest.fixture(scope="session")
def _isolated_runpod_lease_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("runpod-leases")


@pytest.fixture(autouse=True)
def _no_real_runpod_leases(monkeypatch, _isolated_runpod_lease_dir):
    """Keep tests away from the developer's real cleanup leases.

    Executors built without ``lease_dir`` would otherwise read, sweep or
    write ~/.cache/stateset-agents/runpod-leases.
    """
    import sys

    module = sys.modules.get("stateset_agents.remote.runpod")
    if module is None:
        import stateset_agents.remote.runpod as module
    monkeypatch.setattr(module, "DEFAULT_RUNPOD_LEASE_DIR", _isolated_runpod_lease_dir)


@pytest.fixture
def settle_preview():
    """Wait for preview completion while applying results on the UI thread."""
    return settle_studio
