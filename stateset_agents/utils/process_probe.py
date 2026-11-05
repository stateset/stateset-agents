"""Probe process presence without sending Windows console control events."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from functools import lru_cache
from typing import Any

_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 0x00000102


@lru_cache(maxsize=1)
def _windows_api() -> Any:
    from ctypes import wintypes

    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise RuntimeError("Windows process probing requires Windows")
    api = loader("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    return api


def _last_windows_error() -> int:
    reader = getattr(ctypes, "get_last_error", None)
    if reader is None:
        raise RuntimeError("Windows process probing requires Windows")
    return int(reader())


def _is_windows() -> bool:
    return sys.platform == "win32"


def probe_process(pid: int) -> None:
    """Return if a process is present; raise if absent or presence is unknown.

    ProcessLookupError means the process has exited or is not present.
    PermissionError and other OS errors preserve uncertainty for callers.
    This establishes presence only, not process ownership or PID identity.

    POSIX uses the OS's signal-zero probe. Windows instead opens a handle
    with only SYNCHRONIZE access and performs a zero-time wait: signal zero
    there is CTRL_C_EVENT, which must never be used as a presence probe.
    """
    if type(pid) is not int or pid < 1:
        raise ValueError("Process ID must be a positive integer")
    if pid > 0xFFFFFFFF:
        raise OverflowError("Process ID exceeds the supported range")
    if not _is_windows():
        os.kill(pid, 0)
        return
    api = _windows_api()
    handle = api.OpenProcess(_SYNCHRONIZE, False, pid)
    if not handle:
        error = _last_windows_error()
        if error in {87, 1168}:  # Invalid parameter / not found.
            raise ProcessLookupError(errno.ESRCH, "Process is not present")
        if error == 5:
            raise PermissionError(errno.EACCES, "Cannot query process presence")
        raise OSError(error, "Cannot open process presence handle")
    try:
        state = api.WaitForSingleObject(handle, 0)
        if state == _WAIT_OBJECT_0:
            raise ProcessLookupError(errno.ESRCH, "Process has exited")
        if state != _WAIT_TIMEOUT:
            raise OSError(_last_windows_error(), "Cannot query process presence")
    finally:
        if not api.CloseHandle(handle):
            raise OSError(_last_windows_error(), "Cannot close process presence handle")


def process_start_token(pid: int) -> str | None:
    """A token that differs between a process and a later one reusing its PID.

    Linux reads the start time in clock ticks from ``/proc/<pid>/stat``.
    Elsewhere the process creation time comes from ``psutil`` when it is
    installed, or ``ps -o lstart=`` on macOS. ``None`` means no token is
    available: the process is gone or a zombie, or the platform offers none.
    Reading a token never signals the process.
    """
    if type(pid) is not int or pid < 1 or pid > 0xFFFFFFFF:
        return None
    if sys.platform.startswith("linux"):
        return _proc_start_token(pid)
    else:
        return _portable_start_token(pid)


def _proc_start_token(pid: int) -> str | None:
    """Linux: start time in clock ticks, ``None`` for zombies or gone PIDs."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
            fields = stream.read().rsplit(")", 1)[1].split()
        return None if fields[0] in {"Z", "X"} else fields[19]
    except (OSError, IndexError, UnicodeDecodeError):
        return None


def _portable_start_token(pid: int) -> str | None:
    """Other platforms: psutil's creation time, or ``ps`` on macOS."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            return f"{process.create_time():.6f}"
        except (psutil.Error, OSError, ValueError):
            return None
    if sys.platform == "darwin":
        return _ps_start_token(pid)
    return None


def _ps_start_token(pid: int) -> str | None:
    """macOS fallback without psutil: the process start time reported by ps."""
    import subprocess

    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "stat=", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = result.stdout.strip()
    if result.returncode != 0 or not line or line.startswith("Z"):
        return None
    parts = line.split(None, 1)
    return parts[1].strip() if len(parts) == 2 and parts[1].strip() else None
