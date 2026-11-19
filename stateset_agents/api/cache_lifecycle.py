"""Own cache callbacks, shared observations, and shutdown admission."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

from stateset_agents.utils.async_calls import drain_owned_operation, run_async_owned

from .resource_lifecycle import ResourceOwnershipError

_T = TypeVar("_T")
_CALLBACKS: ContextVar[tuple[object, ...]] = ContextVar("cache_callbacks", default=())
_ENTRIES: ContextVar[tuple[object, ...]] = ContextVar("cache_entries", default=())


@dataclass(eq=False)
class _EntryTurn:
    key: str | None
    ready: asyncio.Event


class CacheEntryOrdering:
    """Order conflicting local entries; prefix deletions form exclusive barriers."""

    def __init__(self) -> None:
        self._waiting: list[_EntryTurn] = []
        self._active: list[_EntryTurn] = []

    def _wake(self) -> None:
        blocked_keys = {turn.key for turn in self._active}
        barrier = None in blocked_keys
        earlier = False
        for turn in self._waiting:
            if turn.key is None:
                if not self._active and not earlier:
                    turn.ready.set()
                barrier = True
            elif not barrier and turn.key not in blocked_keys:
                turn.ready.set()
            blocked_keys.add(turn.key)
            earlier = True

    @asynccontextmanager
    async def hold(self, key: str | None) -> AsyncIterator[None]:
        """Retain ordering through callback unwind, without serializing other keys."""
        if self in _ENTRIES.get():
            raise ResourceOwnershipError(
                "A cache entry callback cannot reenter its owner"
            )
        turn = _EntryTurn(key, asyncio.Event())
        self._waiting.append(turn)
        self._wake()
        try:
            await turn.ready.wait()
        except BaseException:
            self._waiting.remove(turn)
            self._wake()
            raise
        self._waiting.remove(turn)
        self._active.append(turn)
        token = _ENTRIES.set((*_ENTRIES.get(), self))
        try:
            yield
        finally:
            _ENTRIES.reset(token)
            self._active.remove(turn)
            self._wake()

    def get_status(self) -> dict[str, int | bool]:
        """Expose ordering pressure without retaining or exposing completed keys."""
        return {
            "active_keys": sum(turn.key is not None for turn in self._active),
            "barrier_active": any(turn.key is None for turn in self._active),
            "waiting_operations": len(self._waiting),
        }


class CacheOperations:
    """Retain admitted operations until SDK callbacks and cancellation finish."""

    def __init__(self) -> None:
        self.active = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def check_loop(self) -> None:
        """Keep async client ownership on its first event loop."""
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise ResourceOwnershipError("Cache belongs to another event loop")

    def check_callback(self) -> None:
        """Reject transitions that could wait for their own callback to finish."""
        self.check_loop()
        if self in _CALLBACKS.get():
            raise ResourceOwnershipError("Cache callback cannot wait for itself")

    @contextmanager
    def callback(self) -> Iterator[None]:
        """Propagate callback ownership to inherited child tasks."""
        token = _CALLBACKS.set((*_CALLBACKS.get(), self))
        try:
            yield
        finally:
            _CALLBACKS.reset(token)

    async def run(
        self, operation: Coroutine[Any, Any, _T], fallback: _T, *, available: bool
    ) -> _T:
        """Admit synchronously, then cancel once and drain on interruption."""
        try:
            self.check_loop()
        except BaseException:
            operation.close()
            raise
        if not available:
            operation.close()
            return fallback
        self.active += 1
        self._idle.clear()
        try:
            with self.callback():
                return await run_async_owned(operation)
        finally:
            self.active -= 1
            if not self.active:
                self._idle.set()

    async def wait_idle(self) -> None:
        """Wait for admitted work without cancelling remote writes."""
        await self._idle.wait()


@dataclass
class CacheFlight:
    """Share one connection/probe; retire it when its final observer cancels."""

    task: asyncio.Task[bool] | None = None
    observers: int = 0
    retired: bool = False

    def cancel(self) -> None:
        """Request cancellation once, retaining the task through cleanup."""
        if not self.retired and self.task is not None and not self.task.done():
            self.retired = True
            self.task.cancel()

    async def observe(self) -> bool:
        """Cancellation of one observer cannot cancel another observer's work."""
        assert self.task is not None
        task = self.task
        self.observers += 1
        cancelled = False
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            self.observers -= 1
            if cancelled and not self.observers:
                self.cancel()

                async def join() -> None:
                    await asyncio.gather(task, return_exceptions=True)

                try:
                    await drain_owned_operation(join())
                except asyncio.CancelledError:
                    pass  # Preserve the observer's original cancellation.
