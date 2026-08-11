"""The executor contract every compute provider implements.

Deliberately stateless and poll-based: a job is submitted, polled, and its
artifacts fetched. There is no retry logic and no in-memory job registry —
a failed job is rerun by the user. That keeps executors thin and avoids
silently burning GPU budget on retries nobody asked for.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from stateset_agents.core.errors import ErrorCode, StateSetError, wrap_exception
from stateset_agents.remote.job import (
    JobHandle,
    JobStatus,
    RemoteJobResult,
    RemoteJobSpec,
)

__all__ = ["RemoteExecutionError", "RemoteExecutor"]


class RemoteExecutionError(StateSetError):
    """Raised when a remote job cannot be submitted, polled, or retrieved."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.NET_API_ERROR,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code)
        if kwargs:
            self.context.details.update(kwargs)

    @classmethod
    def wrap(
        cls, exc: BaseException, message: str, **context: Any
    ) -> RemoteExecutionError:
        """Wrap a provider SDK exception, preserving the ``.cause`` chain."""
        wrapped = wrap_exception(
            exc, cls, message=message, code=ErrorCode.NET_API_ERROR, **context
        )
        assert isinstance(wrapped, cls)  # wrap_exception constructs `cls`
        return wrapped


class RemoteExecutor(abc.ABC):
    """Runs a :class:`RemoteJobSpec` on some compute provider."""

    #: Provider name, as used by the registry and stamped onto handles.
    name: str = "unknown"

    @abc.abstractmethod
    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        """Start the job and return a handle for polling it."""

    @abc.abstractmethod
    def status(self, handle: JobHandle) -> JobStatus:
        """Return the job's current lifecycle state."""

    @abc.abstractmethod
    def logs(self, handle: JobHandle) -> Iterator[str]:
        """Yield the job's output lines captured so far."""

    @abc.abstractmethod
    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        """Retrieve the trained adapter directory. Errors if not finished."""

    @abc.abstractmethod
    def cancel(self, handle: JobHandle) -> None:
        """Stop the job if it is still running."""

    def wait(self, handle: JobHandle, poll_interval_s: float = 1.0) -> RemoteJobResult:
        """Poll until the job reaches a terminal state, then collect it.

        Shared by every provider — the per-provider work is only in the four
        primitives above.
        """
        while not (status := self.status(handle)).is_terminal:
            time.sleep(poll_interval_s)

        logs = list(self.logs(handle))
        output_dir = self.fetch(handle) if status is JobStatus.SUCCEEDED else None
        return RemoteJobResult(
            handle=handle, status=status, output_dir=output_dir, logs=logs
        )
