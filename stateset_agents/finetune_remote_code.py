"""Review-bound RunPod package choices without loading credentials or providers.

A release record pins a version, not a remote artifact. A wheel record identifies
archive bytes, not equivalence to the local checkout or installed dependencies.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

MAX_WHEEL_BYTES = 64 * 1024 * 1024
WHEEL_ENV = "STATESET_AGENTS_WHEEL"
SNAPSHOT_DIRECTORY = "remote-code"
_CHUNK_BYTES = 1024 * 1024
_VERSION = re.compile(r"[0-9][A-Za-z0-9.!+_-]{0,127}\Z")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,195}\.whl\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class RemoteCodeSelection:
    """A public package identity plus an unpersisted preparation source path."""

    package_version: str
    wheel_filename: str | None = None
    wheel_sha256: str | None = None
    wheel_bytes: int | None = None
    source_path: Path | None = field(default=None, repr=False, compare=False)

    def describe(self) -> dict[str, Any]:
        """Return a normalized review record that never exposes the source path."""
        record: dict[str, Any] = {
            "schema_version": 1,
            "provider": "runpod",
            "mode": "wheel" if self.wheel_filename is not None else "release",
            "package_version": self.package_version,
        }
        if self.wheel_filename is not None:
            record.update(
                wheel_filename=self.wheel_filename,
                wheel_sha256=self.wheel_sha256,
                wheel_bytes=self.wheel_bytes,
            )
        elif self.wheel_sha256 is not None or self.wheel_bytes is not None:
            raise ValueError("Remote code selection is incomplete")
        return normalize_remote_code(record)


def normalize_remote_code(
    record: Any, *, package_version: str | None = None
) -> dict[str, Any]:
    """Validate exact package-review fields without filesystem or environment I/O."""
    common = {"schema_version", "provider", "mode", "package_version"}
    if (
        not isinstance(record, dict)
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("provider") != "runpod"
        or record.get("mode") not in ("release", "wheel")
    ):
        raise ValueError(
            "Remote code review is missing or unsupported; prepare a fresh run"
        )
    version = record.get("package_version")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError("Remote package review requires a bounded explicit version")
    if package_version is not None and version != package_version:
        raise ValueError("Remote package version differs from the reviewed framework")
    if record["mode"] == "release":
        if set(record) != common:
            raise ValueError("Release package review contains unsupported fields")
    else:
        if set(record) != common | {"wheel_filename", "wheel_sha256", "wheel_bytes"}:
            raise ValueError("Wheel package review contains unsupported fields")
        name, digest, size = (
            record["wheel_filename"],
            record["wheel_sha256"],
            record["wheel_bytes"],
        )
        if not isinstance(name, str) or not _FILENAME.fullmatch(name):
            raise ValueError("Remote wheel requires a bounded plain .whl filename")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("Remote wheel checksum is invalid")
        if type(size) is not int or not 0 < size <= MAX_WHEEL_BYTES:
            raise ValueError("Remote wheel bytes exceed the 64 MiB limit")
    return dict(record)


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _private_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or (
        os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ValueError("Remote code snapshots require private regular directories")


def _wheel_identity(
    path: Path, *, copy_to: BinaryIO | None = None, private: bool = False
) -> tuple[str, int]:
    if not _FILENAME.fullmatch(path.name) or path.is_symlink():
        raise ValueError("Remote wheel must be a plain regular .whl file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= MAX_WHEEL_BYTES
        ):
            raise ValueError(
                "Remote wheel must be a nonempty regular file within 64 MiB"
            )
        if private and os.name != "nt" and stat.S_IMODE(before.st_mode) & 0o077:
            raise ValueError("Remote wheel snapshot must remain private")
        path_before = path.lstat()
        # On Windows, stat's ctime may be creation time while fstat's ctime
        # is change time (CPython #157671). Compare compatible identity fields
        # across APIs, then retain full before/after stamps within each API.
        if (
            not stat.S_ISREG(path_before.st_mode)
            or _stamp(before)[:4] != _stamp(path_before)[:4]
        ):
            raise ValueError("Remote wheel changed during inspection")
        digest, total = hashlib.sha256(), 0
        while total <= MAX_WHEEL_BYTES:
            chunk = os.read(descriptor, min(_CHUNK_BYTES, MAX_WHEEL_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_WHEEL_BYTES:
                raise ValueError("Remote wheel exceeds the 64 MiB limit")
            digest.update(chunk)
            if copy_to is not None:
                copy_to.write(chunk)
        if (
            total != before.st_size
            or _stamp(before) != _stamp(os.fstat(descriptor))
            or _stamp(path_before) != _stamp(path.lstat())
        ):
            raise ValueError("Remote wheel changed during inspection")
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def capture_remote_code(
    provider: str,
    package_version: str,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> RemoteCodeSelection | None:
    """Capture the RunPod wheel override once, before review, without credentials."""
    if provider != "runpod":
        return None
    selection = RemoteCodeSelection(package_version)
    selection.describe()
    value = (os.environ if environ is None else environ).get(WHEEL_ENV, "")
    if not isinstance(value, str):
        raise ValueError("Remote wheel override must be a path string")
    if not value.strip():
        return selection
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = (cwd if cwd is not None else Path.cwd()) / path
    path = path.absolute()
    digest, size = _wheel_identity(path)
    return RemoteCodeSelection(package_version, path.name, digest, size, path)


def snapshot_remote_code(run: Path, selection: RemoteCodeSelection) -> dict[str, Any]:
    """Privately copy exactly the reviewed wheel, refusing replacement or reuse."""
    record = selection.describe()
    if record["mode"] == "release":
        if selection.source_path is not None:
            raise ValueError("Release package review cannot contain a wheel source")
        return record
    if (
        selection.source_path is None
        or selection.source_path.name != record["wheel_filename"]
    ):
        raise ValueError("Reviewed wheel preparation source is unavailable")
    _private_directory(run)
    directory = run / SNAPSHOT_DIRECTORY
    directory.mkdir(mode=0o700)
    target = directory / record["wheel_filename"]
    created = False
    try:
        _private_directory(directory)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(target, flags, 0o600)
        created = True
        try:
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            identity = _wheel_identity(selection.source_path, copy_to=stream)
            if identity != (record["wheel_sha256"], record["wheel_bytes"]):
                raise ValueError("Remote wheel differs from the reviewed bytes")
            stream.flush()
            os.fsync(stream.fileno())
            if os.name != "nt":
                os.fchmod(stream.fileno(), 0o400)
        verify_remote_code(run, record, package_version=selection.package_version)
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass
        raise
    return record


def verify_remote_code(
    run: Path, record: Any, *, package_version: str | None = None
) -> Path | None:
    """Revalidate retained wheel bytes, ignoring ambient source paths and overrides."""
    normalized = normalize_remote_code(record, package_version=package_version)
    if normalized["mode"] == "release":
        return None
    _private_directory(run)
    directory = run / SNAPSHOT_DIRECTORY
    _private_directory(directory)
    path: Path = directory / normalized["wheel_filename"]
    identity = _wheel_identity(path, private=True)
    if identity != (normalized["wheel_sha256"], normalized["wheel_bytes"]):
        raise ValueError("Retained remote wheel differs from the reviewed bytes")
    return path
