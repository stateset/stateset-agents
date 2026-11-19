"""
GRPO State Management Module

Thread-safe state management with TTL support.
"""

import logging
import threading
import time
from collections.abc import ItemsView, KeysView, ValuesView
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar, cast, overload

logger = logging.getLogger(__name__)

T = TypeVar("T")
_DefaultT = TypeVar("_DefaultT")
_MISSING = object()

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})
_JOB_TRANSITIONS = {
    "starting": frozenset({"running", *TERMINAL_JOB_STATUSES}),
    "running": TERMINAL_JOB_STATUSES,
}


class TTLDict(dict[str, T], Generic[T]):
    """
    Dictionary with automatic cleanup of expired entries.

    Thread-safe for concurrent read/write operations.
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 10000,
        cleanup_interval: int = 300,
    ):
        """
        Initialize TTL dictionary.

        Args:
            ttl_seconds: Time-to-live for entries in seconds.
            max_size: Maximum number of entries before eviction.
            cleanup_interval: Seconds between automatic cleanups.
        """
        super().__init__()
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self._timestamps: dict[str, float] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

    def __setitem__(self, key: str, value: T) -> None:
        """Set item with timestamp tracking."""
        with self._lock:
            # Periodic cleanup
            self._maybe_cleanup()

            # Enforce max size by removing oldest entries
            if len(self) >= self.max_size and key not in self:
                self._evict_oldest(len(self) - self.max_size + 1)

            super().__setitem__(key, value)
            self._timestamps[key] = time.time()

    def __getitem__(self, key: str) -> T:
        """Get item, checking expiration."""
        with self._lock:
            if key in self._timestamps:
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    # Entry expired
                    super().pop(key, None)
                    self._timestamps.pop(key, None)
                    raise KeyError(key)
            return super().__getitem__(key)

    def __delitem__(self, key: str) -> None:
        """Delete item and its timestamp."""
        with self._lock:
            super().__delitem__(key)
            self._timestamps.pop(key, None)

    def __contains__(self, key: object) -> bool:
        """Check if key exists and is not expired."""
        if not isinstance(key, str):
            return False
        with self._lock:
            if not super().__contains__(key):
                return False
            if key in self._timestamps:
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    super().pop(key, None)
                    self._timestamps.pop(key, None)
                    return False
            return True

    @overload
    def get(self, key: str, default: None = None) -> T | None: ...

    @overload
    def get(self, key: str, default: _DefaultT) -> T | _DefaultT: ...

    def get(self, key: str, default: _DefaultT | None = None) -> T | _DefaultT | None:
        """Get item with default value."""
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, default: T | None = None) -> T:
        """Set default value if key doesn't exist or is expired."""
        with self._lock:
            if key not in self:
                self[key] = cast(T, default)
            return super().__getitem__(key)

    @overload
    def pop(self, key: str) -> T: ...

    @overload
    def pop(self, key: str, default: _DefaultT) -> T | _DefaultT: ...

    def pop(self, key: str, default: object = _MISSING) -> T | _DefaultT:
        """Remove and return item."""
        with self._lock:
            self._timestamps.pop(key, None)
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, cast(_DefaultT, default))

    def items(self) -> ItemsView[str, T]:  # type: ignore[override]
        """Return non-expired items."""
        with self._lock:
            self.cleanup_expired()
            return super().items()

    def keys(self) -> KeysView[str]:  # type: ignore[override]
        """Return non-expired keys."""
        with self._lock:
            self.cleanup_expired()
            return super().keys()

    def values(self) -> ValuesView[T]:  # type: ignore[override]
        """Return non-expired values."""
        with self._lock:
            self.cleanup_expired()
            return super().values()

    def _evict_oldest(self, count: int = 1) -> int:
        """
        Remove the oldest entries.

        Args:
            count: Number of entries to evict.

        Returns:
            Number of entries actually evicted.
        """
        if not self._timestamps:
            return 0

        sorted_keys = sorted(
            self._timestamps.keys(),
            key=lambda k: self._timestamps[k],
        )

        evicted = 0
        for key in sorted_keys[:count]:
            super().pop(key, None)
            self._timestamps.pop(key, None)
            evicted += 1

        return evicted

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup >= self.cleanup_interval:
            self.cleanup_expired()
            self._last_cleanup = now

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, ts in self._timestamps.items() if now - ts > self.ttl_seconds
            ]

            for key in expired_keys:
                super().pop(key, None)
                self._timestamps.pop(key, None)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def stats(self) -> dict[str, Any]:
        """Get statistics about the dictionary."""
        with self._lock:
            now = time.time()
            expired = sum(
                1 for ts in self._timestamps.values() if now - ts > self.ttl_seconds
            )

            return {
                "total_entries": len(self._timestamps),
                "active_entries": len(self._timestamps) - expired,
                "expired_entries": expired,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
            }


@dataclass
class TrainingJob:
    """Training job state."""

    job_id: str
    status: str
    strategy: str
    user_id: str
    request_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    iterations_completed: int = 0
    total_trajectories: int = 0
    error: str | None = None
    results: list = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Return whether this job has committed a final local outcome."""
        return self.status in TERMINAL_JOB_STATUSES


@dataclass
class ConversationState:
    """Conversation state."""

    conversation_id: str
    user_id: str
    agent_id: str | None = None
    strategy: str = "default"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_message_at: datetime | None = None
    message_count: int = 0
    total_tokens: int = 0
    context: dict[str, Any] = field(default_factory=dict)


class StateManager:
    """
    Centralized state management for the GRPO service.

    Manages training jobs, conversations, and engine state.
    """

    def __init__(
        self,
        conversation_ttl: int = 3600,
        job_ttl: int = 86400,
        max_conversations: int = 10000,
        max_jobs: int = 1000,
    ):
        """
        Initialize state manager.

        Args:
            conversation_ttl: TTL for conversations in seconds (default: 1 hour).
            job_ttl: TTL for jobs in seconds (default: 24 hours).
            max_conversations: Maximum number of active conversations.
            max_jobs: Maximum number of tracked jobs.
        """
        self.conversations: TTLDict[ConversationState] = TTLDict(
            ttl_seconds=conversation_ttl,
            max_size=max_conversations,
        )
        self.jobs: TTLDict[TrainingJob] = TTLDict(
            ttl_seconds=job_ttl,
            max_size=max_jobs,
        )
        self.engines: dict[str, Any] = {}
        self.services: dict[str, Any] = {}

    def create_job(
        self,
        job_id: str,
        strategy: str,
        user_id: str,
        request_id: str,
        config: dict[str, Any] | None = None,
    ) -> TrainingJob:
        """Create and track a new training job."""
        job = TrainingJob(
            job_id=job_id,
            status="starting",
            strategy=strategy,
            user_id=user_id,
            request_id=request_id,
            config=config or {},
        )
        with self.jobs._lock:
            if job_id in self.jobs:
                raise ValueError(f"Training job {job_id} already exists")
            self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> TrainingJob | None:
        """Get a training job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[TrainingJob]:
        """Return current job handles with collection membership read atomically."""
        with self.jobs._lock:
            return list(self.jobs.values())

    def update_job(
        self,
        job_id: str,
        status: str | None = None,
        iterations: int | None = None,
        trajectories: int | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
        *,
        expected_job: TrainingJob | None = None,
    ) -> TrainingJob | None:
        """Atomically update an active job, preserving committed terminal evidence.

        Invalid transitions and updates to terminal jobs return the unchanged
        record without refreshing its TTL. Missing jobs or an expected owner
        that has been replaced return None. Pass expected_job for worker updates
        so an expired/replaced entry cannot receive an older worker's results.
        """
        with self.jobs._lock:
            job = self.jobs.get(job_id)
            if job is None or (expected_job is not None and job is not expected_job):
                return None
            self._update_job_locked(
                job, status, iterations, trajectories, error, result
            )
            return job

    def transition_job(
        self,
        job_id: str,
        status: str,
        *,
        expected_job: TrainingJob | None = None,
        error: str | None = None,
    ) -> bool:
        """Commit a legal status change once, optionally requiring the same owner.

        True means this caller committed the transition; repeated, stale,
        missing, and invalid transitions return False without changing state.
        """
        with self.jobs._lock:
            job = self.jobs.get(job_id)
            if (
                job is None
                or (expected_job is not None and job is not expected_job)
                or job.status == status
            ):
                return False
            return self._update_job_locked(job, status=status, error=error)

    def is_job_running(self, job: TrainingJob) -> bool:
        """Check that a worker's job still owns a running entry."""
        with self.jobs._lock:
            return self.jobs.get(job.job_id) is job and job.status == "running"

    def _update_job_locked(
        self,
        job: TrainingJob,
        status: str | None = None,
        iterations: int | None = None,
        trajectories: int | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Validate the whole update before mutating a record under the TTL lock."""
        if job.is_terminal:
            return False
        if (
            status
            and status != job.status
            and status not in _JOB_TRANSITIONS.get(job.status, ())
        ):
            logger.debug(
                "Ignored training job %s transition: %s -> %s",
                job.job_id,
                job.status,
                status,
            )
            return False

        if status:
            job.status = status
            if status == "running" and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in TERMINAL_JOB_STATUSES:
                job.completed_at = datetime.utcnow()

        if iterations is not None:
            job.iterations_completed = iterations

        if trajectories is not None:
            job.total_trajectories = trajectories

        if error:
            job.error = error

        if result:
            job.results.append(result)

        self.jobs[job.job_id] = job
        return True

    def create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        agent_id: str | None = None,
        strategy: str = "default",
    ) -> ConversationState:
        """Create and track a new conversation."""
        conv = ConversationState(
            conversation_id=conversation_id,
            user_id=user_id,
            agent_id=agent_id,
            strategy=strategy,
        )
        self.conversations[conversation_id] = conv
        return conv

    def get_conversation(self, conversation_id: str) -> ConversationState | None:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def update_conversation(
        self,
        conversation_id: str,
        tokens: int = 0,
    ) -> ConversationState | None:
        """Update conversation state after a message."""
        conv = self.conversations.get(conversation_id)
        if not conv:
            return None

        conv.last_message_at = datetime.utcnow()
        conv.message_count += 1
        conv.total_tokens += tokens

        self.conversations[conversation_id] = conv
        return conv

    def end_conversation(self, conversation_id: str) -> ConversationState | None:
        """End and remove a conversation."""
        return self.conversations.pop(conversation_id, None)

    def register_engine(self, engine_id: str, engine: Any) -> None:
        """Register an active engine."""
        self.engines[engine_id] = engine

    def get_engine(self, engine_id: str) -> Any | None:
        """Get an engine by ID."""
        return self.engines.get(engine_id)

    def unregister_engine(self, engine_id: str) -> Any | None:
        """Unregister and return an engine."""
        return self.engines.pop(engine_id, None)

    def register_service(self, name: str, service: Any) -> None:
        """Register a service."""
        self.services[name] = service

    def get_service(self, name: str) -> Any | None:
        """Get a service by name."""
        return self.services.get(name)

    def cleanup(self) -> dict[str, int]:
        """
        Run cleanup on all state.

        Returns:
            Dictionary with cleanup counts.
        """
        return {
            "conversations_cleaned": self.conversations.cleanup_expired(),
            "jobs_cleaned": self.jobs.cleanup_expired(),
        }

    def stats(self) -> dict[str, Any]:
        """Get state statistics."""
        return {
            "conversations": self.conversations.stats(),
            "jobs": self.jobs.stats(),
            "engines": len(self.engines),
            "services": list(self.services.keys()),
        }


# Global singleton
_state_manager: StateManager | None = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def reset_state_manager() -> None:
    """Reset state manager (for testing)."""
    global _state_manager
    _state_manager = None
