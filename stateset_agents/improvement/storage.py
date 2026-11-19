"""Private, bounded records for improvement workspaces outside Git checkouts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MAX_RECORD_BYTES = 64 * 1024 * 1024


def plain_path(path: Path) -> Path:
    """Reject parent traversal and symbolic links, including intermediate links."""
    absolute = path.expanduser().absolute()
    if ".." in absolute.parts:
        raise ValueError("Paths cannot contain parent traversal")
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise ValueError("Artifact paths cannot traverse symbolic links")
    return absolute


def private_path(path: Path) -> Path:
    """Reject paths inside a Git checkout, including linked worktrees."""
    absolute = plain_path(path)
    for parent in (absolute, *absolute.parents):
        marker = parent / ".git"
        # Some sandboxes reserve an empty .git directory outside any checkout.
        # Actual repositories have HEAD; linked worktrees use a .git file.
        if marker.is_file() or marker.is_symlink() or (marker / "HEAD").exists():
            raise ValueError(
                "Keep customer data and run evidence outside Git repositories"
            )
    return absolute


def private_directory(path: Path) -> Path:
    """Create an owner-only workspace outside every detected Git checkout."""
    path = private_path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_mode & 0o077:
        raise ValueError("Private workspace directory must have mode 0700")
    return path


def read_bytes(path: Path) -> bytes:
    """Read one bounded regular file without following symbolic links."""
    path = plain_path(path)
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RECORD_BYTES:
            raise ValueError("Expected a bounded regular artifact")
        data = stream.read(MAX_RECORD_BYTES + 1)
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("Artifact exceeds the size limit")
    return data


def digest(path: Path) -> str:
    """Hash retained bytes, rejecting symlink and special-file artifacts."""
    return hashlib.sha256(read_bytes(path)).hexdigest()


def decode_json(data: bytes | str) -> Any:
    """Reject duplicate fields and nonfinite numbers in JSON inputs."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError("Nonfinite JSON number")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def read_json(path: Path) -> Any:
    """Read a bounded strict JSON artifact."""
    return decode_json(read_bytes(path))


def write_json(path: Path, value: Any) -> None:
    """Atomically publish a new owner-only JSON record without replacing data."""
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    )
    write_bytes(path, payload)


def write_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bounded owner-only bytes, exclusively and durably."""
    path = private_path(path)
    private_directory(path.parent)
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError("Artifact exceeds the size limit")
    descriptor, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


@contextmanager
def workspace_lock(root: Path) -> Iterator[None]:
    """Serialize POSIX workspace operations; process death releases the lock."""
    import fcntl

    root = private_directory(root)
    descriptor = os.open(
        root / ".workflow.lock",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Invalid workspace lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except BlockingIOError as exc:
        raise ValueError("Another workflow operation is running") from exc
    finally:
        os.close(descriptor)
