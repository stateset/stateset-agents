"""Start-time tokens distinguish a process from a later one reusing its PID."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.utils import process_probe


def test_live_process_has_a_stable_token_and_an_exited_one_has_none():
    if process_probe.process_start_token(os.getpid()) is None:
        assert sys.platform == "win32"  # Windows offers tokens only via psutil
        pytest.skip("No start-time token source on this host")
    assert process_probe.process_start_token(
        os.getpid()
    ) == process_probe.process_start_token(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert process_probe.process_start_token(child.pid) is None


@pytest.mark.parametrize("pid", [0, -1, True, "1", 2**40])
def test_invalid_pids_have_no_token(pid):
    assert process_probe.process_start_token(pid) is None


def test_non_linux_uses_psutil_creation_time(monkeypatch):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(process_probe, "sys", SimpleNamespace(platform="darwin"))
    token = process_probe.process_start_token(os.getpid())
    assert token == f"{psutil.Process(os.getpid()).create_time():.6f}"


def test_macos_without_psutil_falls_back_to_ps(monkeypatch):
    monkeypatch.setattr(process_probe, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(
            returncode=0, stdout="Ss   Thu Oct 22 10:00:00 2026\n"
        ),
    )
    assert process_probe.process_start_token(42) == "Thu Oct 22 10:00:00 2026"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(returncode=0, stdout="Z  x\n"),
    )
    assert process_probe.process_start_token(42) is None
