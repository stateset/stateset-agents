"""Stage complete checkpoint directories and recover interrupted replacements.

The local filesystem must support atomic same-directory rename and advisory
locking. All cooperating readers and writers lock the stable parent directory
(on Windows, a persistent sibling lock file). External HF readers do not take
this lock and must not read while a checkpoint is being replaced.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
COMPLETION_FILE = ".stateset-checkpoint.json"


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def _lock(parent: Path) -> Iterator[None]:
    """Fail promptly on concurrent access rather than blocking an event loop."""
    if sys.platform == "win32":
        import msvcrt

        with (parent / ".stateset-checkpoint.lock").open("a+b") as stream:
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("Checkpoint directory is busy") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        descriptor = os.open(parent, os.O_RDONLY)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("Checkpoint directory is busy") from exc
            yield
        finally:
            os.close(descriptor)


def _inventory(path: Path, *, sync: bool = False) -> dict[str, str]:
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Checkpoint artifacts cannot contain symlinks")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError("Checkpoint artifacts must be regular files")
        if item == path / COMPLETION_FILE:
            continue
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if sync:
                os.fsync(stream.fileno())
        files[item.relative_to(path).as_posix()] = digest.hexdigest()
    if sync:
        for directory in sorted(
            (item for item in path.rglob("*") if item.is_dir()), reverse=True
        ):
            _sync_directory(directory)
        _sync_directory(path)
    return files


def validate_checkpoint_publication(path: Path, *, required: bool = False) -> bool:
    """Verify a new checkpoint's complete file inventory before loading weights."""
    marker = path / COMPLETION_FILE
    if not marker.exists() and not marker.is_symlink():
        if required:
            raise ValueError("Checkpoint publication is incomplete")
        return False
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Invalid checkpoint completion marker")
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Invalid checkpoint completion marker") from exc
    if (
        not isinstance(state, dict)
        or set(state) != {"schema_version", "files"}
        or type(state["schema_version"]) is not int
        or state["schema_version"] != 1
        or not isinstance(state["files"], dict)
        or "training_state.pt" not in state["files"]
        or state["files"] != _inventory(path)
    ):
        raise ValueError("Checkpoint artifacts differ from the completed publication")
    return True


def _paths(path: Path) -> tuple[Path, Path]:
    return (
        path.with_name(f".{path.name}.stateset-publication.json"),
        path.with_name(f".{path.name}.stateset-previous"),
    )


def _recover(path: Path) -> None:
    journal, previous = _paths(path)
    if not journal.exists() and not journal.is_symlink():
        return
    if journal.is_symlink() or not journal.is_file():
        raise ValueError("Invalid checkpoint publication journal")
    try:
        record = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Invalid checkpoint publication journal") from exc
    if (
        not isinstance(record, dict)
        or set(record) != {"schema_version", "staging"}
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or not isinstance(record["staging"], str)
        or not record["staging"].startswith(f".{path.name}.stateset-stage-")
        or Path(record["staging"]).name != record["staging"]
        or any(character in record["staging"] for character in ("/", "\\", ":"))
    ):
        raise ValueError("Invalid checkpoint publication journal")
    staging = path.parent / record["staging"]
    for directory in (path, previous, staging):
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError("Invalid checkpoint publication directory")
    # A missing staging directory means its complete contents were published.
    if not staging.exists() and path.exists():
        validate_checkpoint_publication(path, required=True)
        if previous.exists():
            shutil.rmtree(previous)
        journal.unlink()
        _sync_directory(path.parent)
        return
    if not path.exists() and previous.exists():
        os.replace(previous, path)
        _sync_directory(path.parent)
    elif path.exists() and previous.exists():
        raise ValueError("Ambiguous interrupted checkpoint publication")
    # Persist rollback before deleting staging: otherwise another crash could
    # mistake a restored legacy directory for a newly published generation.
    journal.unlink()
    _sync_directory(path.parent)
    if staging.exists():
        shutil.rmtree(staging)


@contextlib.contextmanager
def checkpoint_read(path: Path) -> Iterator[Path]:
    """Recover a replaced directory and hold its lock through the entire load."""
    if not path.parent.exists():
        yield path
        return
    with _lock(path.parent):
        _recover(path)
        yield path


@contextlib.contextmanager
def checkpoint_write(path: Path) -> Iterator[Path]:
    """Write off to the side; publish only after every artifact is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock(path.parent):
        _recover(path)
        journal, previous = _paths(path)
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("Checkpoint destination must be a directory")
        if previous.exists() or previous.is_symlink():
            raise ValueError("Checkpoint backup exists without a publication journal")
        if (
            path.exists()
            and any(path.iterdir())
            and not (path / "training_state.pt").is_file()
        ):
            raise ValueError("Refusing to replace a directory that is not a checkpoint")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{path.name}.stateset-stage-", dir=path.parent)
        )
        journal_temporary = staging / ".publication-journal"
        try:
            yield staging
            files = _inventory(staging, sync=True)
            if "training_state.pt" not in files:
                raise ValueError("Checkpoint save did not produce training_state.pt")
            _write_json(
                staging / COMPLETION_FILE, {"schema_version": 1, "files": files}
            )
            _sync_directory(staging)
            _write_json(
                journal_temporary, {"schema_version": 1, "staging": staging.name}
            )
            os.replace(journal_temporary, journal)
            _sync_directory(staging)
            _sync_directory(path.parent)
            if path.exists():
                os.replace(path, previous)
                _sync_directory(path.parent)
            os.replace(staging, path)
            _sync_directory(path.parent)
        except BaseException:
            # If publishing failed after moving the old directory, restore it.
            # Keep the journal if recovery itself fails so the next access retries.
            try:
                if previous.exists():
                    if path.exists():
                        os.replace(path, staging)
                    os.replace(previous, path)
                    _sync_directory(path.parent)
                elif not staging.exists() and path.exists() and journal.exists():
                    os.replace(path, staging)
                journal.unlink(missing_ok=True)
                _sync_directory(path.parent)
                if staging.exists():
                    shutil.rmtree(staging)
            except OSError:
                logger.exception("Checkpoint rollback requires recovery on next access")
            raise
        else:
            # The new directory is committed. Cleanup failure must not turn a
            # successful publication into an apparent failed save; retry on load.
            try:
                _recover(path)
            except OSError:
                logger.warning(
                    "Checkpoint published; cleanup will retry on access", exc_info=True
                )
