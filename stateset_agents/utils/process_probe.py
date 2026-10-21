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
