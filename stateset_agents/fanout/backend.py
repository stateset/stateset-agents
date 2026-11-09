"""The contract between fan-out orchestration and whatever provides compute.

Errors are the important part. How the orchestrator responds depends entirely on
which kind of failure it is looking at:

=====================  ==================================================
:class:`CapacityError`  No room right now. Wait; this costs no attempt.
:class:`AllocationLostError`  The sandbox is gone. Retry on a fresh one.
:class:`TransientBackendError`  Probably fine. Ride it out while polling.
:class:`FatalBackendError`  Every other run would fail too. Stop launching.
:class:`BackendError`  This attempt failed. Retry on a fresh sandbox.
=====================  ==================================================
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from stateset_agents.core.errors import ErrorCode, StateSetError
from stateset_agents.fanout.manifest import FanoutManifest, Resources

__all__ = [
    "Allocation",
    "AllocationLostError",
    "BackendError",
    "CapacityError",
    "FatalBackendError",
    "FileEntry",
    "LogChunk",
    "ProcessState",
    "RunBackend",
    "TransientBackendError",
]


class BackendError(StateSetError, RuntimeError):
    """An operation failed. The attempt is over; a new sandbox might succeed."""

    message_only_str = True
    default_code = ErrorCode.NET_API_ERROR


class CapacityError(BackendError):
    """No capacity for another sandbox yet. Nothing was allocated."""

    default_code = ErrorCode.RES_EXHAUSTED

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AllocationLostError(BackendError):
    """The sandbox no longer exists: preempted, expired, or stopped."""


class TransientBackendError(BackendError):
    """The control plane did not answer usefully. The sandbox may be fine."""

    default_code = ErrorCode.NET_CONNECTION_FAILED


class FatalBackendError(BackendError):
    """No run can succeed: bad credentials, billing, or a refused configuration."""

    default_code = ErrorCode.CFG_INVALID


@dataclass(frozen=True)
class Allocation:
    """One run's isolated compute. ``root`` is its private working directory."""

    id: str
    root: str
    #: How long the platform actually agreed to keep it. Plans can silently grant
    #: less than was asked for. None means it did not say.
    lifetime_s: float | None = None


@dataclass(frozen=True)
class ProcessState:
    status: str
    running: bool
    exit_code: int | None = None
    detail: str | None = None
    #: Total log bytes written so far, when the backend knows, so that logs need
    #: reading only when they grew.
    stdout_bytes: int | None = None
    stderr_bytes: int | None = None


@dataclass(frozen=True)
class LogChunk:
    data: bytes
    next_offset: int
    eof: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int | None = None


class RunBackend(abc.ABC):
    """Isolated compute for one run at a time, addressed by allocation id."""

    name: str = "unknown"
    #: Whether allocations cost money, and so belong in the cost ledger.
    metered: bool = False
    #: Largest file ``read_file`` can return. None means no limit.
    max_read_bytes: int | None = None

    def validate(  # noqa: B027 - an optional hook, not an abstract requirement
        self, manifest: FanoutManifest
    ) -> None:
        """Refuse a manifest this backend could not run, before allocating."""

    @abc.abstractmethod
    def allocate(
        self, *, run_id: str, resources: Resources, idempotency_key: str
    ) -> Allocation:
        """Create isolated compute, or return what this key already created.

        A transient failure here may hide a create that landed, so the caller asks
        again with the same key rather than risk an orphan.
        """

    @abc.abstractmethod
    def write_file(self, allocation_id: str, path: str, content: bytes) -> None:
        """Write a file, creating parent directories."""

    @abc.abstractmethod
    def worker_argv(
        self, manifest: FanoutManifest, *, root: str, worker_args: list[str]
    ) -> list[str]:
        """The command that installs the package if needed and runs the worker."""

    @abc.abstractmethod
    def start(
        self,
        allocation_id: str,
        argv: list[str],
        *,
        env: dict[str, str],
        timeout_seconds: int,
        name: str,
    ) -> str:
        """Start a background process and return its id.

        Never repeated within an allocation: a request that failed may still
        have reached the server, so the caller abandons the whole sandbox.
        """

    @abc.abstractmethod
    def process(self, allocation_id: str, process_id: str) -> ProcessState:
        """Current process state. Raises AllocationLostError if the sandbox is gone."""

    @abc.abstractmethod
    def logs(
        self, allocation_id: str, process_id: str, stream: str, offset: int
    ) -> LogChunk:
        """Read a log from an absolute byte offset, so readers can resume."""

    @abc.abstractmethod
    def read_file(self, allocation_id: str, path: str) -> bytes | None:
        """File content, or None when the file does not exist."""

    @abc.abstractmethod
    def list_files(self, allocation_id: str, path: str) -> list[FileEntry]:
        """Every file below ``path``, recursively."""

    @abc.abstractmethod
    def cost_cents(self, allocation_id: str) -> float | None:
        """Measured spend so far. None means unknown, never free."""

    @abc.abstractmethod
    def loss_reason(self, allocation_id: str) -> str | None:
        """Why a lost sandbox ended, when known: e.g. ``budget_exhausted``."""

    @abc.abstractmethod
    def release(self, allocation_id: str) -> None:
        """Stop the compute. An already gone sandbox counts as released."""
