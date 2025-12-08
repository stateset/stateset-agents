"""
GRPO State Management Module

Thread-safe state management with TTL support.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TTLDict(Dict[str, T], Generic[T]):
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
        self._timestamps: Dict[str, float] = {}
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
        with self._lock:
            if not super().__contains__(key):
                return False
            if key in self._timestamps:
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    super().pop(key, None)
                    self._timestamps.pop(key, None)
                    return False
            return True

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get item with default value."""
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, default: Optional[T] = None) -> T:
        """Set default value if key doesn't exist or is expired."""
        with self._lock:
            if key not in self:
                self[key] = default
            return super().__getitem__(key)

    def pop(self, key: str, *args) -> T:
        """Remove and return item."""
        with self._lock:
            self._timestamps.pop(key, None)
            return super().pop(key, *args)

    def items(self) -> Iterator[tuple]:
        """Iterate over non-expired items."""
        with self._lock:
            now = time.time()
            for key in list(super().keys()):
                if key in self._timestamps:
                    if now - self._timestamps[key] <= self.ttl_seconds:
                        yield key, super().__getitem__(key)

    def keys(self) -> Iterator[str]:
        """Iterate over non-expired keys."""
        with self._lock:
            now = time.time()
            for key in list(super().keys()):
                if key in self._timestamps:
                    if now - self._timestamps[key] <= self.ttl_seconds:
                        yield key

    def values(self) -> Iterator[T]:
        """Iterate over non-expired values."""
        for _, value in self.items():
            yield value

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
                k
                for k, ts in self._timestamps.items()
                if now - ts > self.ttl_seconds
            ]

            for key in expired_keys:
                super().pop(key, None)
                self._timestamps.pop(key, None)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """Get statistics about the dictionary."""
        with self._lock:
            now = time.time()
            expired = sum(
                1
                for ts in self._timestamps.values()
                if now - ts > self.ttl_seconds
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
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    iterations_completed: int = 0
    total_trajectories: int = 0
    error: Optional[str] = None
    results: list = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    """Conversation state."""
    conversation_id: str
    user_id: str
    agent_id: Optional[str] = None
    strategy: str = "default"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    total_tokens: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


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
        self.engines: Dict[str, Any] = {}
        self.services: Dict[str, Any] = {}

    def create_job(
        self,
        job_id: str,
        strategy: str,
        user_id: str,
        request_id: str,
        config: Optional[Dict[str, Any]] = None,
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
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        return self.jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        iterations: Optional[int] = None,
        trajectories: Optional[int] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrainingJob]:
        """Update a training job."""
        job = self.jobs.get(job_id)
        if not job:
            return None

        if status:
            job.status = status
            if status == "running" and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in ("completed", "failed", "cancelled"):
                job.completed_at = datetime.utcnow()

        if iterations is not None:
            job.iterations_completed = iterations

        if trajectories is not None:
            job.total_trajectories = trajectories

        if error:
            job.error = error

        if result:
            job.results.append(result)

        self.jobs[job_id] = job
        return job

    def create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        agent_id: Optional[str] = None,
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

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def update_conversation(
        self,
        conversation_id: str,
        tokens: int = 0,
    ) -> Optional[ConversationState]:
        """Update conversation state after a message."""
        conv = self.conversations.get(conversation_id)
        if not conv:
            return None

        conv.last_message_at = datetime.utcnow()
        conv.message_count += 1
        conv.total_tokens += tokens

        self.conversations[conversation_id] = conv
        return conv

    def end_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """End and remove a conversation."""
        return self.conversations.pop(conversation_id, None)

    def register_engine(self, engine_id: str, engine: Any) -> None:
        """Register an active engine."""
        self.engines[engine_id] = engine

    def get_engine(self, engine_id: str) -> Optional[Any]:
        """Get an engine by ID."""
        return self.engines.get(engine_id)

    def unregister_engine(self, engine_id: str) -> Optional[Any]:
        """Unregister and return an engine."""
        return self.engines.pop(engine_id, None)

    def register_service(self, name: str, service: Any) -> None:
        """Register a service."""
        self.services[name] = service

    def get_service(self, name: str) -> Optional[Any]:
        """Get a service by name."""
        return self.services.get(name)

    def cleanup(self) -> Dict[str, int]:
        """
        Run cleanup on all state.

        Returns:
            Dictionary with cleanup counts.
        """
        return {
            "conversations_cleaned": self.conversations.cleanup_expired(),
            "jobs_cleaned": self.jobs.cleanup_expired(),
        }

    def stats(self) -> Dict[str, Any]:
        """Get state statistics."""
        return {
            "conversations": self.conversations.stats(),
            "jobs": self.jobs.stats(),
            "engines": len(self.engines),
            "services": list(self.services.keys()),
        }


# Global singleton
_state_manager: Optional[StateManager] = None


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
