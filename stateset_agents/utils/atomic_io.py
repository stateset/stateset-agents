"""Crash-safe file primitives shared by every layer.

``atomic_json`` replaces a JSON artifact without exposing a partial write, and
``exclusive_run`` holds an OS-released lock on an output directory so two
drivers cannot interleave writes to one run. They lived in
``stateset_agents.remote.river_rl`` (which still re-exports them); framework
layers -- ``training``, ``data``, ``evaluation`` -- import them from here so
they never depend on the remote app layer.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

__all__ = ["atomic_json", "exclusive_run"]


def atomic_json(path: Path, value: Any) -> None:
    """Replace a durable JSON artifact without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def exclusive_run(output: Path) -> Iterator[None]:
    """Hold an OS-released lock so concurrent drivers cannot overwrite a run."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".rl.lock").open("a+b") as stream:
        unlock: Callable[[], None]
        if sys.platform == "win32":
            import msvcrt

            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError(
                    "Another driver owns this RL output directory"
                ) from exc

            def unlock() -> None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError(
                    "Another driver owns this RL output directory"
                ) from exc

            def unlock() -> None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        try:
            yield
        finally:
            unlock()
