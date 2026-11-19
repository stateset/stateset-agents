"""Atomic single-file snapshot publication with revision and durability receipts."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def open_regular(path: Path, flags: int) -> int:
    """Open a regular file, rejecting symlinks and nonblocking special files."""
    if path.is_symlink():
        raise ValueError("Snapshot paths must not be symlinks")
    descriptor = os.open(
        path,
        flags
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Snapshot paths must be regular files")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def writer_lock(path: Path, conflict: type[RuntimeError]) -> Iterator[None]:
    """Acquire a stable cross-process lock; retain the original memory lock name."""
    with exclusive_file_lock(
        path.with_name(f".{path.name}.stateset-memory.lock"), conflict
    ):
        yield


@contextmanager
def exclusive_file_lock(path: Path, conflict: type[RuntimeError]) -> Iterator[None]:
    """Hold a nonblocking file lock until the context exits or the process dies."""
    descriptor = open_regular(path, os.O_RDWR | os.O_CREAT)
    try:
        try:
            if sys.platform == "win32":  # pragma: no cover - Windows
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise conflict("Another writer owns this snapshot path") from error
        except OSError as error:
            if os.name == "nt" and error.errno in (13, 36):
                raise conflict("Another writer owns this snapshot path") from error
            raise
        yield
    finally:
        os.close(descriptor)


def sync_directory(path: Path) -> bool:
    """Acknowledge directory fsync on POSIX without claiming it on other platforms."""
    if os.name != "posix":  # pragma: no cover
        return False
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def publish_bytes(
    path: Path,
    raw: bytes,
    *,
    expected_revision: str | None,
    read_revision: Callable[[], str],
    status: Callable[[dict[str, Any]], None],
    conflict: type[RuntimeError],
    sync: Callable[[Path], bool] = sync_directory,
) -> None:
    """Publish under a revision lock; report replacement separately from syncing.

    The parent must exist. Cooperating writers share a trusted directory. Cleanup
    only removes this operation's staging file, never orphaned stages from a crash.
    """
    status({"phase": "checking_revision"})
    with writer_lock(path, conflict):
        try:
            observed = read_revision()
        except FileNotFoundError:
            observed = None
        status({"observed_revision": observed})
        if observed != expected_revision:
            raise conflict("Snapshot file revision changed; inspect before retrying")
        stage: str | None = None
        initiating_error: BaseException | None = None
        try:
            status({"phase": "staging"})
            descriptor, stage = tempfile.mkstemp(
                prefix=f".{path.name}.stage-", dir=path.parent
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            status(
                {
                    "file_synced": True,
                    "phase": "replacing",
                    "publication": "unconfirmed",
                }
            )
            os.replace(stage, path)
            status({"publication": "published", "phase": "syncing_directory"})
            synced = sync(path.parent)
            status({"directory_synced": synced, "phase": "finished"})
        except BaseException as error:
            initiating_error = error
            raise
        finally:
            if stage is not None:
                try:
                    Path(stage).unlink(missing_ok=True)
                except BaseException as error:
                    status({"cleanup_error_type": type(error).__name__})
                    if initiating_error is None:
                        raise


def json_value(value: Any, ancestors: frozenset[int] = frozenset()) -> Any:
    """Copy exact JSON types without coercion or invoking arbitrary serializers."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (dict, list):
        if id(value) in ancestors:
            raise ValueError("Snapshot context cannot contain cycles")
        ancestors = ancestors | {id(value)}
        if type(value) is list:
            return [json_value(item, ancestors) for item in value]
        if not all(type(key) is str for key in value):
            raise ValueError("Snapshot context keys must be strings")
        return {key: json_value(item, ancestors) for key, item in value.items()}
    raise ValueError("Snapshot data must contain only finite JSON values")


def canonical_json(value: Any) -> bytes:
    """Encode portable data as deterministic UTF-8 JSON bytes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
