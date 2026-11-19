"""Serialize process-wide API resource ownership and application leases."""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Generic, TypeVar

from stateset_agents.utils.async_calls import drain_owned_operation, run_async_owned

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
_C = TypeVar("_C")
_OPERATIONS: ContextVar[tuple[object, ...]] = ContextVar(
    "api_resource_operations", default=()
)


class ResourceOwnershipError(RuntimeError):
    """A resource transition would discard or interrupt another owner's work."""


class ResourceLease(Generic[_T]):
    """Keep a shared resource alive until this application's cleanup finishes."""

    def __init__(
        self, resource: _T, owner: SharedResource[Any, _T], token: object
    ) -> None:
        self.resource = resource
        self._owner = owner
        self._token = token
        self._close_task: asyncio.Task[None] | None = None

    async def aclose(self) -> None:
        """Release once; repeated cancellation cannot detach final cleanup."""
        self._owner.check_reentry()
        self._owner.check_loop()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._owner.release(self._token))

        async def join() -> None:
            assert self._close_task is not None
            await asyncio.shield(self._close_task)

        await drain_owned_operation(join())


class SharedResource(Generic[_C, _T]):
    """Own initialization, replacement, and cleanup for one global resource.

    Mutations serialize on one event loop. Callbacks cannot reenter their owner,
    including from inherited child tasks. Configuration is captured before any
    await; matching initialization preserves the existing resource. Leases
    prohibit replacement and global close until the last borrower releases.
    Cleanup failure is retained and never automatically retried.
    """

    def __init__(
        self,
        name: str,
        get: Callable[[], _T | None],
        publish: Callable[[_T | None], None],
        construct: Callable[[_C], _T],
        prepare: Callable[[_T], Awaitable[Any]],
        dispose: Callable[[_T], Awaitable[None]],
        reusable: Callable[[_T, _C], bool] = lambda resource, config: True,
        close_rejected: Callable[
            [_T, BaseException], bool
        ] = lambda resource, error: False,
    ) -> None:
        self.name = name
        self._get = get
        self._set = publish
        self._construct = construct
        self._prepare = prepare
        self._dispose = dispose
        self._reusable = reusable
        self._close_rejected = close_rejected
        self._resource: _T | None = None
        self._config: _C | None = None
        self._leases: set[object] = set()
        self._guard = threading.Lock()
        self._gate = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lease_loop: asyncio.AbstractEventLoop | None = None
        self._participants = 0
        self._state = "empty"
        self._phase: str | None = None
        self._generation = 0
        self._error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self._unreleased: list[_T] = []

    def check_reentry(self) -> None:
        """Reject callback cycles before they can wait on their own transition."""
        if self in _OPERATIONS.get():
            raise ResourceOwnershipError(
                f"{self.name} callbacks cannot reenter their resource owner"
            )

    def check_loop(self) -> None:
        """Do not pass leased async resources into a different event loop."""
        if (
            self._lease_loop is not None
            and self._lease_loop is not asyncio.get_running_loop()
        ):
            raise ResourceOwnershipError(
                f"{self.name} leases belong to another event loop"
            )

    @asynccontextmanager
    async def _serialized(self) -> AsyncIterator[None]:
        self.check_reentry()
        self.check_loop()
        loop = asyncio.get_running_loop()
        with self._guard:
            if self._participants and self._loop is not loop:
                raise ResourceOwnershipError(
                    f"{self.name} transition belongs to another event loop"
                )
            if not self._participants:
                self._loop = loop
                self._gate = asyncio.Lock()
            gate = self._gate
            self._participants += 1
        try:
            async with gate:
                token = _OPERATIONS.set((*_OPERATIONS.get(), self))
                try:
                    self._adopt_external()
                    yield
                finally:
                    _OPERATIONS.reset(token)
        finally:
            with self._guard:
                self._participants -= 1

    def _adopt_external(self) -> None:
        # Compatibility for explicitly installed module instances. Public
        # replacement uses initialize(); direct writes cannot retire a lease.
        current = self._get()
        if current is self._resource:
            return
        if self._leases or self._cleanup_error is not None:
            raise ResourceOwnershipError(f"{self.name} owner was replaced externally")
        self._resource = current
        self._config = None
        self._state = "ready" if current is not None else "empty"

    def get(self) -> _T | None:
        """Return the published resource only while it is available to callers."""
        if self._state in {"closing", "failed"} or self._phase == "retiring":
            raise ResourceOwnershipError(f"{self.name} is {self._state}")
        return self._get()

    def status(self) -> dict[str, Any]:
        """Return detached lifecycle evidence without credentials/configuration."""
        return {
            "state": self._state,
            "phase": self._phase,
            "generation": self._generation,
            "leases": len(self._leases),
            "error_type": type(self._error).__name__ if self._error else None,
            "cleanup_error_type": (
                type(self._cleanup_error).__name__ if self._cleanup_error else None
            ),
            "unreleased_resources": len(self._unreleased),
        }

    async def initialize(self, config: _C) -> _T:
        """Reuse matching configuration or replace an unleased owner safely."""
        captured = copy.deepcopy(config)
        async with self._serialized():
            return await self._initialize(captured)

    async def acquire(self, config: _C) -> ResourceLease[_T]:
        """Initialize and claim a lease without a publication/admission gap."""
        captured = copy.deepcopy(config)
        async with self._serialized():
            resource = await self._initialize(captured)
            token = object()
            self._leases.add(token)
            self._lease_loop = asyncio.get_running_loop()
            return ResourceLease(resource, self, token)

    async def _cleanup(self, resource: _T, *, published: bool = False) -> None:
        try:
            await self._dispose(resource)
        except BaseException as error:
            # A published owner can remain available when its own admission
            # guard proves cleanup never began. An unpublished candidate must
            # still be retained if it cannot be released.
            if published and self._close_rejected(resource, error):
                raise
            self._cleanup_error = error
            self._unreleased.append(resource)
            self._state = "failed"
            raise

    async def _initialize(self, config: _C) -> _T:
        if self._cleanup_error is not None:
            raise ResourceOwnershipError(
                f"{self.name} cleanup is unconfirmed; automatic replacement is disabled"
            ) from self._cleanup_error
        previous = self._resource
        if (
            previous is not None
            and config == self._config
            and self._reusable(previous, config)
        ):
            return previous
        if self._leases:
            raise ResourceOwnershipError(
                f"{self.name} configuration cannot change while applications own leases"
            )
        self._state = "replacing" if previous is not None else "initializing"
        self._phase = "preparing"
        self._error = None
        candidate: _T | None = None
        retired = False
        try:
            candidate = self._construct(copy.deepcopy(config))

            async def prepare() -> None:
                assert candidate is not None
                await self._prepare(candidate)

            await run_async_owned(prepare())
            if self._get() is not previous:
                raise ResourceOwnershipError(
                    f"{self.name} changed during initialization"
                )
            if previous is not None:
                self._phase = "retiring"

                async def retire() -> None:
                    nonlocal retired
                    await self._cleanup(previous, published=True)
                    retired = True

                await drain_owned_operation(retire())
            if self._get() is not previous:
                raise ResourceOwnershipError(f"{self.name} changed during replacement")
            self._set(candidate)
            self._resource = candidate
            self._config = config
            self._generation += 1
            self._state = "ready"
            return candidate
        except BaseException as error:
            self._error = error
            if retired and self._get() is previous:
                self._set(None)
                self._resource = None
                self._config = None
            if candidate is not None:
                self._phase = "rollback"
                try:
                    await drain_owned_operation(self._cleanup(candidate))
                except (Exception, asyncio.CancelledError) as cleanup_error:
                    logger.warning(
                        "%s rollback did not complete normally (%s)",
                        self.name,
                        type(cleanup_error).__name__,
                    )
            if self._cleanup_error is None:
                self._state = "ready" if self._get() is not None else "empty"
            raise
        finally:
            self._phase = None

    async def _close(self) -> None:
        if self._cleanup_error is not None:
            raise ResourceOwnershipError(
                f"{self.name} cleanup failed; automatic retry is disabled"
            ) from self._cleanup_error
        resource = self._resource
        if resource is None:
            return
        self._state = "closing"
        self._phase = "closing"
        try:
            await self._cleanup(resource, published=True)
            if self._get() is resource:
                self._set(None)
            self._resource = self._get()
            self._config = None
            self._state = "ready" if self._resource is not None else "empty"
        except BaseException as error:
            self._error = error
            if self._cleanup_error is None:
                self._state = "ready"
            raise
        finally:
            self._phase = None

    async def close(self) -> None:
        """Close an unleased global once, retaining cleanup through cancellation."""
        self.check_reentry()

        async def close() -> None:
            async with self._serialized():
                if self._leases:
                    raise ResourceOwnershipError(
                        f"{self.name} is still owned by applications; release their leases"
                    )
                await self._close()

        await drain_owned_operation(close())

    async def release(self, token: object) -> None:
        """Release a captured owner, closing it after its final lease exits."""
        async with self._serialized():
            if token not in self._leases:
                return
            self._leases.remove(token)
            if not self._leases:
                try:
                    await self._close()
                finally:
                    self._lease_loop = None
