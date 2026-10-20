"""Single-flight background work; only the terminal thread mutates screen state."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TaskResult(Generic[T]):
    """A completed value or exception, consumed by the terminal thread."""

    value: T | None = None
    error: Exception | None = None
    cancelled: bool = False


class BackgroundTask(Generic[T]):
    """Run at most one read-only operation, including after cancellation."""

    def __init__(self) -> None:
        self.busy = False
        self._cancelled = False
        self._results: SimpleQueue[TaskResult[T]] = SimpleQueue()

    def start(self, operation: Callable[[], T]) -> bool:
        """Start without blocking, or reject a concurrent request."""
        if self.busy:
            return False
        self.busy, self._cancelled = True, False

        def run() -> None:
            try:
                result = TaskResult(value=operation())
            except Exception as error:
                result = TaskResult[T](error=error)
            self._results.put(result)

        try:
            threading.Thread(target=run, daemon=True, name="stateset-preview").start()
        except Exception:
            self.busy = False
            raise
        return True

    def cancel(self) -> None:
        """Discard the result; the current read may finish in the background."""
        self._cancelled = True

    def poll(self) -> TaskResult[T] | None:
        """Take a completed result without waiting."""
        try:
            result = self._results.get_nowait()
        except Empty:
            return None
        self.busy = False
        return TaskResult(result.value, result.error, self._cancelled)
