"""Process status must never signal a Windows console or leak process handles."""

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from stateset_agents.utils import process_probe as probe


@pytest.mark.parametrize("pid", [0, -1, True, "42", 1.5])
def test_invalid_process_ids_never_reach_the_os(monkeypatch, pid):
    monkeypatch.setattr(
        probe, "_windows_api", lambda: pytest.fail("invalid PID reached OS")
    )
    monkeypatch.setattr(
        probe.os, "kill", lambda *_: pytest.fail("invalid PID was signalled")
    )
    with pytest.raises(ValueError):
        probe.probe_process(pid)


def test_oversize_process_id_never_wraps_into_a_different_pid(monkeypatch):
    monkeypatch.setattr(
        probe, "_windows_api", lambda: pytest.fail("PID overflow reached OS")
    )
    with pytest.raises(OverflowError):
        probe.probe_process(2**32)


@pytest.mark.parametrize(
    "state,exception",
    [(258, None), (0, ProcessLookupError), (0xFFFFFFFF, OSError), (128, OSError)],
)
def test_windows_probe_waits_without_signalling_and_closes_handle(
    monkeypatch, state, exception
):
    api = SimpleNamespace(
        OpenProcess=MagicMock(return_value=2**40),
        WaitForSingleObject=MagicMock(return_value=state),
        CloseHandle=MagicMock(return_value=1),
    )
    monkeypatch.setattr(probe, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(probe, "_windows_api", lambda: api)
    monkeypatch.setattr(probe, "_last_windows_error", lambda: 6)
    monkeypatch.setattr(
        probe.os, "kill", lambda *_: pytest.fail("Windows probe sent a signal")
    )
    if exception:
        with pytest.raises(exception):
            probe.probe_process(42)
    else:
        assert probe.probe_process(42) is None
    api.OpenProcess.assert_called_once_with(0x00100000, False, 42)
    api.WaitForSingleObject.assert_called_once_with(2**40, 0)
    api.CloseHandle.assert_called_once_with(2**40)


@pytest.mark.parametrize(
    "code,exception",
    [
        (87, ProcessLookupError),
        (1168, ProcessLookupError),
        (5, PermissionError),
        (6, OSError),
    ],
)
def test_windows_open_failure_keeps_absence_and_uncertainty_distinct(
    monkeypatch, code, exception
):
    api = SimpleNamespace(
        OpenProcess=MagicMock(return_value=0), CloseHandle=MagicMock()
    )
    monkeypatch.setattr(probe, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(probe, "_windows_api", lambda: api)
    monkeypatch.setattr(probe, "_last_windows_error", lambda: code)
    with pytest.raises(exception):
        probe.probe_process(42)
    api.CloseHandle.assert_not_called()


def test_windows_handle_cleanup_failure_is_not_a_successful_probe(monkeypatch):
    api = SimpleNamespace(
        OpenProcess=lambda *_: 42,
        WaitForSingleObject=lambda *_: 258,
        CloseHandle=lambda *_: 0,
    )
    monkeypatch.setattr(probe, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(probe, "_windows_api", lambda: api)
    monkeypatch.setattr(probe, "_last_windows_error", lambda: 6)
    with pytest.raises(OSError, match="close process"):
        probe.probe_process(42)


def test_windows_api_declares_pointer_sized_handles(monkeypatch):
    api = MagicMock()
    loader = MagicMock(return_value=api)
    monkeypatch.setattr(ctypes, "WinDLL", loader, raising=False)
    monkeypatch.setattr(probe, "sys", SimpleNamespace(platform="win32"))
    assert probe._windows_api.__wrapped__() is api
    loader.assert_called_once_with("kernel32", use_last_error=True)
    assert api.OpenProcess.restype is wintypes.HANDLE
    assert api.WaitForSingleObject.argtypes == [wintypes.HANDLE, wintypes.DWORD]
    assert api.CloseHandle.argtypes == [wintypes.HANDLE]


def test_current_process_probe_does_not_interrupt_the_caller(monkeypatch):
    if sys.platform == "win32":
        monkeypatch.setattr(
            probe.os, "kill", lambda *_: pytest.fail("Windows used os.kill")
        )
    assert probe.probe_process(os.getpid()) is None


def test_real_child_process_presence_and_exit_are_observed_without_signals():
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    )
    try:
        assert probe.probe_process(child.pid) is None
        child.communicate(b"x", timeout=10)
        assert child.returncode == 0
        with pytest.raises(ProcessLookupError):
            probe.probe_process(child.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
