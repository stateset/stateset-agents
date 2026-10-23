"""Helpers for asynchronous terminal interactions."""

import time

import pytest

#: Upper bound for one preview, launch, stop or review to finish. Planning,
#: review and launch each hash every package source file; on loaded Windows
#: CI runners (four xdist workers, real-time scanning) one pass took several
#: seconds and a 10 s budget ran out. Settling returns as soon as the work is
#: applied, so the bound only matters for work that is really stuck, and it
#: stays below pytest's 60 s per-test timeout to report which step stuck.
SETTLE_TIMEOUT_S = 40.0


def settle_studio(studio, timeout=SETTLE_TIMEOUT_S):
    """Apply finished preview and launch/stop/review work on the UI thread.

    The deadline is checked only after a refresh, so work that finishes
    during the final sleep is still applied rather than reported as stuck.
    """
    start = time.monotonic()
    while studio.planning.busy or studio.submission.busy:
        studio.refresh()
        if time.monotonic() - start >= timeout:
            break
        time.sleep(0.005)
    elapsed = time.monotonic() - start
    assert not studio.planning.busy, f"Preview did not finish in {elapsed:.1f}s"
    assert not studio.submission.busy, f"Submission did not finish in {elapsed:.1f}s"


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
