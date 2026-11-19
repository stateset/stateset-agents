"""
Advanced Async Connection Pool and Resource Management

This module provides high-performance async connection pooling, resource management,
and concurrency optimization for the GRPO Agent Framework.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from stateset_agents.utils.async_calls import drain_owned_operation

from .error_handling import ErrorCode, ResourceException

try:  # pragma: no cover - optional dependency
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

logger = logging.getLogger(__name__)

T = TypeVar("T")

POOL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ResourceException,
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    asyncio.TimeoutError,
)
RESOURCE_CLEANUP_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
)


def _require_aiohttp() -> Any:
    """Ensure aiohttp is available before using HTTP pooling utilities."""
    if aiohttp is None:
        raise ImportError(
            "aiohttp is required for async HTTP pooling. "
            "Install with `pip install stateset-agents[training]` or `pip install aiohttp`."
        )
    return aiohttp


class PoolState(Enum):
    """Connection pool states"""

    INITIALIZING = "initializing"
    READY = "ready"
    SCALING = "scaling"
    DRAINING = "draining"
    CLOSED = "closed"


class ResourceState(Enum):
    """Resource states"""

    IDLE = "idle"
    ACTIVE = "active"
    STALE = "stale"
    FAILED = "failed"


@dataclass
class PoolStats:
    """Connection pool statistics"""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    peak_connections: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourceInfo:
    """Resource information"""

    resource_id: str
    created_at: float
    last_used: float
    use_count: int
    state: ResourceState
    metadata: dict[str, Any] = field(default_factory=dict)


class PooledResource(Generic[T]):
    """Wrapper for pooled resources"""

    def __init__(
        self,
        resource: T,
        resource_id: str,
        pool: AsyncResourcePool,
        created_at: float | None = None,
    ):
        self.resource = resource
        self.info = ResourceInfo(
            resource_id=resource_id,
            created_at=created_at or time.time(),
            last_used=time.time(),
            use_count=0,
            state=ResourceState.IDLE,
        )
        self._pool = weakref.ref(pool)
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def acquire(self) -> T:
        """Acquire resource for use"""
        async with self._lock:
            if self.info.state != ResourceState.IDLE:
                raise ValueError(f"Resource {self.info.resource_id} is not idle")

            self.info.state = ResourceState.ACTIVE
            self.info.last_used = time.time()
            self.info.use_count += 1

            return self.resource

    async def release(self) -> None:
        """Release resource back to pool"""
        async with self._lock:
            if self.info.state != ResourceState.ACTIVE:
                logger.warning(f"Releasing non-active resource {self.info.resource_id}")
                return

            self.info.state = ResourceState.IDLE
            self.info.last_used = time.time()

    async def mark_failed(self, error: BaseException) -> None:
        """Mark resource as failed"""
        async with self._lock:
            self.info.state = ResourceState.FAILED
            self.info.metadata["last_error"] = str(error)
            self.info.metadata["error_time"] = time.time()

    def is_stale(self, max_age: float) -> bool:
        """Check if resource is stale"""
        return time.time() - self.info.created_at > max_age

    def is_idle_too_long(self, max_idle: float) -> bool:
        """Check if resource has been idle too long"""
        return (
            self.info.state == ResourceState.IDLE
            and time.time() - self.info.last_used > max_idle
        )


class AsyncResourceFactory(Generic[T]):
    """Factory for creating async resources"""

    @property
    def cleanup_pending(self) -> bool:
        """Whether factory-owned resources remain after pool cleanup."""
        return False

    async def create_resource(self, **kwargs) -> T:
        """Create a new resource"""
        raise NotImplementedError

    async def validate_resource(self, resource: T) -> bool:
        """Validate that resource is still usable"""
        return True

    async def cleanup_resource(self, resource: T) -> None:
        """Release a resource, raising if cleanup cannot be confirmed."""
        if hasattr(resource, "close"):
            try:
                if asyncio.iscoroutinefunction(resource.close):
                    await resource.close()
                else:
                    resource.close()
            except RESOURCE_CLEANUP_EXCEPTIONS as e:
                logger.warning(f"Error closing resource: {e}")
                raise


class HTTPSessionFactory(AsyncResourceFactory[Any]):
    """Factory for HTTP client sessions."""

    def __init__(self, **session_kwargs):
        self.session_kwargs = session_kwargs

    async def create_resource(self, **kwargs) -> Any:
        """Create new HTTP session."""
        aiohttp_mod = _require_aiohttp()
        merged_kwargs = {**self.session_kwargs, **kwargs}
        return aiohttp_mod.ClientSession(**merged_kwargs)

    async def validate_resource(self, session: Any) -> bool:
        """Validate HTTP session."""
        return not bool(session.closed)

    async def cleanup_resource(self, session: Any) -> None:
        """Close HTTP session."""
        if not bool(session.closed):
            await session.close()


class _SharedHTTPSessionFactory(HTTPSessionFactory):
    """Own the global pool's connector until its last session is retired."""

    def __init__(self, **session_kwargs: Any) -> None:
        super().__init__(**session_kwargs)
        self._connector: Any = None
        self._sessions: set[Any] = set()
        self._connector_lock = asyncio.Lock()

    async def create_resource(self, **kwargs) -> Any:
        """Share one connector without giving individual sessions ownership."""
        async with self._connector_lock:
            if self._connector is None or self._connector.closed:
                self._connector = _require_aiohttp().TCPConnector(limit=100)
            try:
                session = await super().create_resource(
                    **kwargs, connector=self._connector, connector_owner=False
                )
            except BaseException:
                if not self._sessions:
                    connector = self._connector
                    await connector.close()
                    self._connector = None
                raise
            self._sessions.add(session)
            return session

    async def cleanup_resource(self, session: Any) -> None:
        """Close a session, then close the shared connector when unused."""
        async with self._connector_lock:
            await super().cleanup_resource(session)
            self._sessions.discard(session)
            if not self._sessions and self._connector is not None:
                connector = self._connector
                await connector.close()
                self._connector = None

    @property
    def cleanup_pending(self) -> bool:
        """Report retained session/connector ownership after attempted cleanup."""
        return bool(self._sessions) or self._connector is not None


class AsyncResourcePool(Generic[T]):
    """High-performance async resource pool"""

    def __init__(
        self,
        factory: AsyncResourceFactory[T],
        min_size: int = 1,
        max_size: int = 10,
        max_idle_time: float = 300.0,  # 5 minutes
        max_age: float = 3600.0,  # 1 hour
        health_check_interval: float = 60.0,  # 1 minute
        acquire_timeout: float = 30.0,
        name: str = "AsyncPool",
    ):
        if not 0 <= min_size <= max_size:
            raise ValueError("Pool sizes must satisfy 0 <= min_size <= max_size")
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.max_age = max_age
        self.health_check_interval = health_check_interval
        self.acquire_timeout = acquire_timeout
        self.name = name

        self._pool: list[PooledResource[T]] = []
        self._available: asyncio.Queue[PooledResource[T]] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._close_lock = asyncio.Lock()
        self._borrowed: set[PooledResource[T]] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_failures: list[PooledResource[T]] = []
        self._state = PoolState.INITIALIZING
        self._stats = PoolStats()
        self._health_check_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._resource_counter = 0

    @property
    def state(self) -> PoolState:
        """Return the lifecycle state without starting any pool operation."""
        return self._state

    @property
    def cleanup_failures(self) -> tuple[PooledResource[T], ...]:
        """Retain failed cleanup handles without retrying or lending them again."""
        return tuple(self._cleanup_failures)

    async def initialize(self) -> None:
        """Initialize once, rolling back owned resources if setup is interrupted."""
        failed = False
        try:
            async with self._condition:
                if self._state == PoolState.READY:
                    return
                if self._state != PoolState.INITIALIZING:
                    raise RuntimeError(f"Pool {self.name} cannot be reinitialized")
                logger.info(
                    f"Initializing pool {self.name} with min_size={self.min_size}"
                )
                try:
                    for _ in range(self.min_size):
                        await drain_owned_operation(
                            self._add_resource_locked(raise_on_fail=True)
                        )
                except BaseException:
                    self._state = PoolState.DRAINING
                    failed = True
                    self._condition.notify_all()
                    raise
                self._state = PoolState.READY
                self._condition.notify_all()
                if self.health_check_interval > 0:
                    self._health_check_task = asyncio.create_task(
                        self._health_check_loop()
                    )
        finally:
            if failed:
                await drain_owned_operation(self._close())

        logger.info(f"Pool {self.name} initialized with {len(self._pool)} resources")

    async def _create_resource(
        self, raise_on_fail: bool = False
    ) -> PooledResource[T] | None:
        """Create a new pooled resource"""
        try:
            self._resource_counter += 1
            resource_id = f"{self.name}_resource_{self._resource_counter}"

            raw_resource = await self.factory.create_resource()
            pooled_resource = PooledResource(raw_resource, resource_id, self)

            logger.debug(f"Created resource {resource_id}")
            return pooled_resource

        except POOL_EXCEPTIONS as e:
            logger.error(f"Failed to create resource: {e}")
            self._stats.failed_connections += 1
            if raise_on_fail:
                raise ResourceException(
                    "Failed to create pooled resource",
                    error_code=ErrorCode.RESOURCE_FAILED,
                    details={"pool": self.name},
                ) from e
            return None

    @asynccontextmanager
    async def acquire(self, timeout: float | None = None) -> AsyncIterator[T]:
        """Borrow through validation and use, then return or retire exactly once.

        Failed or cancelled operations retire their resource. Cleanup finishes
        before cancellation propagates; factory operations must supply their own
        I/O timeouts. Waiting borrowers are rejected when draining starts.
        """
        self._require_ready()
        timeout = timeout or self.acquire_timeout
        start_time = time.monotonic()
        pooled_resource: PooledResource[T] | None = None
        error: BaseException | None = None
        try:
            pooled_resource = await self._checkout(timeout)
            if not await self.factory.validate_resource(pooled_resource.resource):
                raise RuntimeError("Resource validation failed")
            self._stats.total_requests += 1
            yield pooled_resource.resource
            self._stats.successful_requests += 1
            successes = self._stats.successful_requests
            self._stats.average_response_time = (
                self._stats.average_response_time * (successes - 1)
                + time.monotonic()
                - start_time
            ) / successes
        except BaseException as exc:
            error = exc
            if isinstance(exc, Exception):
                self._stats.failed_requests += 1
            raise
        finally:
            if pooled_resource is not None:
                await drain_owned_operation(
                    self._return_resource(pooled_resource, error)
                )

    def _require_ready(self) -> None:
        if self._state != PoolState.READY:
            raise RuntimeError(f"Pool {self.name} is not ready (state: {self._state})")

    def _refresh_stats_locked(self) -> None:
        self._stats.total_connections = len(self._pool)
        self._stats.active_connections = len(self._borrowed)
        self._stats.idle_connections = self._available.qsize()

    async def _add_resource_locked(self, raise_on_fail: bool = False) -> bool:
        """Publish created resources before an owning caller can be cancelled."""
        if not self._has_capacity_locked():
            return False
        resource = await self._create_resource(raise_on_fail=raise_on_fail)
        if resource is None:
            return False
        self._pool.append(resource)
        self._available.put_nowait(resource)
        self._refresh_stats_locked()
        self._condition.notify_all()
        return True

    def _has_capacity_locked(self) -> bool:
        # Retired resources still consume capacity until cleanup finishes.
        return len(self._pool) + len(self._cleanup_tasks) < self.max_size

    async def _checkout(self, timeout: float) -> PooledResource[T]:
        async with self._condition:
            self._require_ready()
            created = False
            if self._available.empty():
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(
                            lambda: self._state != PoolState.READY
                            or not self._available.empty()
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    self._require_ready()
                    # A return may have committed while the waiter timed out.
                    if self._available.empty():
                        if not self._has_capacity_locked():
                            raise RuntimeError(
                                f"Pool {self.name} exhausted (timeout after {timeout}s)"
                            ) from None
                        created = await drain_owned_operation(
                            self._add_resource_locked()
                        )
                        if not created:
                            raise RuntimeError(
                                "Failed to create new resource"
                            ) from None
            self._require_ready()
            resource = self._available.get_nowait()
            self._available.task_done()
            # Queue removal and the ownership claim have no suspension point.
            # Validation belongs to this borrower, not to the idle health check.
            resource.info.state = ResourceState.ACTIVE
            resource.info.last_used = time.time()
            resource.info.use_count += 1
            self._borrowed.add(resource)
            self._refresh_stats_locked()
            self._stats.peak_connections = max(
                self._stats.peak_connections, len(self._borrowed)
            )
            if created:
                self._stats.pool_misses += 1
            else:
                self._stats.pool_hits += 1
            return resource

    async def _return_resource(
        self, resource: PooledResource[T], error: BaseException | None
    ) -> None:
        cleanup = None
        async with self._condition:
            if error is not None:
                await resource.mark_failed(error)
            else:
                await resource.release()
            self._borrowed.discard(resource)
            if (
                self._state == PoolState.READY
                and resource.info.state == ResourceState.IDLE
                and not resource.is_stale(self.max_age)
            ):
                self._available.put_nowait(resource)
            else:
                cleanup = self._retire_locked(resource)
            self._refresh_stats_locked()
            self._condition.notify_all()
        if cleanup is not None:
            await cleanup

    def _retire_locked(self, resource: PooledResource[T]) -> asyncio.Task[None] | None:
        """Detach an unborrowed resource before scheduling one cleanup attempt."""
        if resource in self._borrowed:
            return None
        if resource not in self._pool:
            return resource._cleanup_task
        self._pool.remove(resource)
        retained = []
        while not self._available.empty():
            candidate = self._available.get_nowait()
            self._available.task_done()
            if candidate is not resource:
                retained.append(candidate)
        for candidate in retained:
            self._available.put_nowait(candidate)
        if resource.info.state != ResourceState.FAILED:
            resource.info.state = ResourceState.STALE
        task = asyncio.create_task(self._cleanup_resource(resource))
        resource._cleanup_task = task
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        self._refresh_stats_locked()
        self._condition.notify_all()
        return task

    async def _cleanup_resource(self, resource: PooledResource[T]) -> None:
        try:
            await self.factory.cleanup_resource(resource.resource)
        except asyncio.CancelledError:
            resource.info.metadata["cleanup_error_type"] = "CancelledError"
            self._cleanup_failures.append(resource)
            raise
        except Exception as exc:
            resource.info.metadata["cleanup_error_type"] = type(exc).__name__
            self._cleanup_failures.append(resource)
            logger.warning(
                "Error cleaning up resource %s: %s", resource.info.resource_id, exc
            )

    @staticmethod
    async def _wait_for_cleanup(tasks: list[asyncio.Task[None]]) -> None:
        if tasks:
            await asyncio.gather(*tasks)

    async def _remove_resource(self, pooled_resource: PooledResource[T]) -> None:
        """Retire an unborrowed resource; active ownership always takes precedence."""
        async with self._condition:
            cleanup = self._retire_locked(pooled_resource)
        if cleanup is not None:
            await drain_owned_operation(self._wait_for_cleanup([cleanup]))

    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except POOL_EXCEPTIONS as e:
                logger.error(f"Health check error: {e}")

    async def _perform_health_check(self) -> None:
        """Check only idle resources, atomically retiring unhealthy candidates."""
        cleanup: list[asyncio.Task[None]] = []
        try:
            async with self._condition:
                if self._state != PoolState.READY:
                    return
                for resource in list(self._pool):
                    if resource in self._borrowed:
                        continue
                    retire = resource.is_stale(self.max_age) or (
                        resource.is_idle_too_long(self.max_idle_time)
                        and len(self._pool) > self.min_size
                    )
                    if not retire:
                        try:
                            retire = not await self.factory.validate_resource(
                                resource.resource
                            )
                        except Exception as exc:
                            logger.warning(
                                "Health check failed for %s: %s",
                                resource.info.resource_id,
                                exc,
                            )
                            retire = True
                    if retire:
                        task = self._retire_locked(resource)
                        if task is not None:
                            cleanup.append(task)
        finally:
            if cleanup:
                await drain_owned_operation(self._wait_for_cleanup(cleanup))
        async with self._condition:
            while self._state == PoolState.READY and len(self._pool) < self.min_size:
                if not await drain_owned_operation(self._add_resource_locked()):
                    break

    async def scale(self, target_size: int) -> None:
        """Scale a ready pool, removing only resources with no active borrower."""
        if target_size < self.min_size or target_size > self.max_size:
            raise ValueError(
                f"Target size {target_size} outside valid range [{self.min_size}, {self.max_size}]"
            )

        cleanup: list[asyncio.Task[None]] = []
        try:
            async with self._condition:
                self._require_ready()
                for _ in range(max(0, target_size - len(self._pool))):
                    if not await drain_owned_operation(self._add_resource_locked()):
                        break
                for resource in list(self._pool):
                    if len(self._pool) <= target_size:
                        break
                    if resource not in self._borrowed:
                        task = self._retire_locked(resource)
                        if task is not None:
                            cleanup.append(task)
        finally:
            if cleanup:
                await drain_owned_operation(self._wait_for_cleanup(cleanup))

        logger.info(f"Scaled pool {self.name} to {len(self._pool)} resources")

    def get_stats(self) -> PoolStats:
        """Get current pool statistics"""
        self._stats.timestamp = time.time()
        return self._stats

    async def drain(self) -> None:
        """Stop new checkouts and wait for borrowed resources, including validation."""
        logger.info(f"Draining pool {self.name}")
        async with self._condition:
            if self._state == PoolState.CLOSED:
                return
            self._state = PoolState.DRAINING
            self._condition.notify_all()
            await self._condition.wait_for(lambda: not self._borrowed)

        logger.info(f"Pool {self.name} drained")

    async def close(self) -> None:
        """Drain and close once, finishing owned cleanup before caller cancellation.

        Cleanup errors are logged and handles retained in cleanup_failures;
        retired resources are never reused. Closed means local attempts settled,
        not that every external close succeeded. Borrowers and factory I/O must
        eventually finish.
        """
        await drain_owned_operation(self._close())

    async def _close(self) -> None:
        async with self._close_lock:
            await self._close_once()

    async def _close_once(self) -> None:
        logger.info(f"Closing pool {self.name}")
        # An idle health check may hold the condition during validation. Cancel
        # it before waiting for that condition, so shutdown can interrupt it.
        if self._health_check_task is not None:
            self._health_check_task.cancel()
        async with self._condition:
            if self._state == PoolState.CLOSED:
                cleanup = list(self._cleanup_tasks)
            else:
                self._state = PoolState.DRAINING
                cleanup = []
            self._shutdown_event.set()
            self._condition.notify_all()
        if self._state == PoolState.CLOSED:
            await self._wait_for_cleanup(cleanup)
            return
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        await self.drain()
        async with self._condition:
            for resource in list(self._pool):
                self._retire_locked(resource)
            cleanup = list(self._cleanup_tasks)
        await self._wait_for_cleanup(cleanup)
        async with self._condition:
            self._state = PoolState.CLOSED
            self._refresh_stats_locked()
            self._condition.notify_all()
        logger.info(f"Pool {self.name} closed")


class TaskManagerState(Enum):
    """Admission and drain states for one task manager on one event loop."""

    NEW = "new"
    RUNNING = "running"
    DRAINING = "draining"
    CLOSED = "closed"


class AsyncTaskManager:
    """Bound admission and retain task ownership through cancellation cleanup."""

    def __init__(
        self,
        max_concurrent_tasks: int = 100,
        task_timeout: float = 300.0,
        cleanup_interval: float = 60.0,
        *,
        shutdown_timeout: float = 30.0,
    ) -> None:
        if max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        if cleanup_interval <= 0 or math.isnan(cleanup_interval):
            raise ValueError("cleanup_interval must be positive and not NaN")
        if math.isnan(task_timeout):
            raise ValueError("task_timeout must not be NaN")
        if shutdown_timeout < 0 or math.isnan(shutdown_timeout):
            raise ValueError("shutdown_timeout must be nonnegative and not NaN")
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        self.cleanup_interval = cleanup_interval
        self.shutdown_timeout = shutdown_timeout

        self._active_tasks: dict[asyncio.Task[Any], asyncio.Event] = {}
        self._operations: set[asyncio.Future[Any]] = set()
        self._condition = asyncio.Condition()
        self._waiters: deque[object] = deque()
        self._state = TaskManagerState.NEW
        self._cleanup_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._shutdown_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> TaskManagerState:
        """Return the manager's lifecycle state."""
        return self._state

    async def start(self) -> None:
        """Start maintenance once; a draining or closed manager cannot restart."""
        if self._shutdown_event.is_set():
            raise RuntimeError("Task manager is shutting down or already shut down")
        if self._state == TaskManagerState.RUNNING:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._state = TaskManagerState.RUNNING
        logger.info(
            f"Task manager started with max {self.max_concurrent_tasks} concurrent tasks"
        )

    @staticmethod
    def _close_unstarted(coro: Awaitable[Any]) -> None:
        """Dispose only a fresh coroutine, never a caller's running task/future."""
        if (
            inspect.iscoroutine(coro)
            and inspect.getcoroutinestate(coro) == inspect.CORO_CREATED
        ):
            coro.close()

    async def submit_task(self, coro: Awaitable[T], timeout: float | None = None) -> T:
        """Run admitted work and retain its slot until cancellation cleanup finishes.

        Submission starts the manager lazily. Timeout begins after admission;
        zero expires without starting a fresh coroutine. Rejected or cancelled
        queued coroutine objects are closed. Already running tasks/futures are
        caller-owned until admission and cannot be concurrency-limited retroactively.
        """
        admitted = False
        try:
            timeout = self.task_timeout if timeout is None else timeout
            if math.isnan(timeout):
                raise ValueError("timeout must not be NaN")
            await self.start()
            async with self._condition:
                ticket = object()
                self._waiters.append(ticket)
                try:
                    await self._condition.wait_for(
                        lambda: self._shutdown_event.is_set()
                        or (
                            self._waiters[0] is ticket
                            and len(self._active_tasks) < self.max_concurrent_tasks
                        )
                    )
                    if self._shutdown_event.is_set():
                        raise RuntimeError(
                            "Task manager is shutting down or already shut down"
                        )
                    cancel_requested = asyncio.Event()
                    task = asyncio.create_task(
                        self._run_task(coro, timeout, cancel_requested)
                    )
                    self._active_tasks[task] = cancel_requested
                    admitted = True
                finally:
                    self._waiters.remove(ticket)
                    self._condition.notify_all()

            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                cancel_requested.set()

                async def settle() -> None:
                    await asyncio.gather(task, return_exceptions=True)

                await drain_owned_operation(settle())
                raise
            except asyncio.TimeoutError:
                logger.warning(f"Task timed out after {timeout}s")
                raise
            except POOL_EXCEPTIONS as e:
                logger.error(f"Task failed: {e}")
                raise
        finally:
            if not admitted:
                self._close_unstarted(coro)

    async def _run_task(
        self, coro: Awaitable[T], timeout: float, cancel_requested: asyncio.Event
    ) -> T:
        """Own execution and release admission only after the operation settles."""
        task = asyncio.current_task()
        try:
            return await self._run_operation(coro, timeout, cancel_requested)
        finally:

            async def release() -> None:
                self._close_unstarted(coro)
                async with self._condition:
                    if task is not None:
                        self._active_tasks.pop(task, None)
                    self._condition.notify_all()

            await drain_owned_operation(release())

    async def _run_operation(
        self, coro: Awaitable[T], timeout: float, cancel_requested: asyncio.Event
    ) -> T:
        """Converge timeout and cancellation on one cancellation request."""
        operation = asyncio.ensure_future(coro)
        self._operations.add(operation)
        cancellation_waiter = asyncio.create_task(cancel_requested.wait())
        cancellation_sent = False
        timed_out = False
        try:
            if operation.done():
                return operation.result()
            if cancel_requested.is_set() or timeout <= 0:
                timed_out = not cancel_requested.is_set()
            else:
                done, _ = await asyncio.wait(
                    {operation, cancellation_waiter},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if operation in done:
                    return operation.result()
                timed_out = not done
            operation.cancel()
            cancellation_sent = True
        finally:

            async def settle() -> None:
                if not operation.done() and not cancellation_sent:
                    operation.cancel()
                cancellation_waiter.cancel()
                await asyncio.gather(
                    operation, cancellation_waiter, return_exceptions=True
                )
                self._operations.discard(operation)

            await drain_owned_operation(settle())
        if timed_out and operation.cancelled():
            raise asyncio.TimeoutError
        # Match wait_for: a coroutine that suppresses timeout cancellation can
        # still return its result or raise its own error after finishing cleanup.
        return operation.result()

    async def submit_batch(
        self, coros: list[Awaitable[T]], return_exceptions: bool = False
    ) -> list[T | BaseException]:
        """Run a batch; on failure/cancellation, cancel and join unfinished siblings."""
        tasks: list[asyncio.Task[T]] = []
        try:
            for coro in coros:
                tasks.append(asyncio.create_task(self.submit_task(coro)))
            return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()

            async def settle() -> None:
                await asyncio.gather(*tasks, return_exceptions=True)

            await drain_owned_operation(settle())
            raise
        finally:
            for coro in coros:
                self._close_unstarted(coro)

    async def _cleanup_loop(self) -> None:
        """Background cleanup of completed tasks"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self.cleanup_interval
                )
            except asyncio.TimeoutError:
                async with self._condition:
                    for task in tuple(self._active_tasks):
                        if task.done():
                            self._active_tasks.pop(task, None)
                            if not task.cancelled() and task.exception() is not None:
                                logger.warning(
                                    "Background task failed: %s", task.exception()
                                )
                    self._condition.notify_all()
            except asyncio.CancelledError:
                break
            except POOL_EXCEPTIONS as e:
                logger.error(f"Cleanup loop error: {e}")

    def get_status(self) -> dict[str, int | str]:
        """Observe admission capacity separately from queued and owned work."""
        return {
            "state": self._state.value,
            "active_tasks": len(self._active_tasks),
            "waiting_tasks": len(self._waiters),
            "available_slots": (
                0
                if self._shutdown_event.is_set()
                else self.max_concurrent_tasks - len(self._active_tasks)
            ),
            "max_concurrent_tasks": self.max_concurrent_tasks,
        }

    async def shutdown(self) -> None:
        """Close admission and join one shared drain, even if a caller is cancelled.

        After shutdown_timeout seconds, request cancellation once and wait for
        cleanup to finish. This is cooperative and can exceed the grace period.
        Call from outside managed operations, on the manager's event loop.
        """
        if asyncio.current_task() in self._operations:
            raise RuntimeError("Task manager shutdown must run outside a managed task")
        if self._shutdown_task is None:
            self.close_admission()
            self._shutdown_task = asyncio.create_task(self._drain())
        task = self._shutdown_task

        async def join() -> None:
            await task

        await drain_owned_operation(join())

    def close_admission(self) -> None:
        """Reject new work immediately; shutdown joins and wakes queued callers."""
        if self._state != TaskManagerState.CLOSED:
            self._state = TaskManagerState.DRAINING
        self._shutdown_event.set()

    async def _drain(self) -> None:
        """Wake rejected waiters, then settle all admitted work and maintenance."""
        logger.info("Shutting down task manager")
        async with self._condition:
            self._condition.notify_all()
            tasks = tuple(self._active_tasks)
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=self.shutdown_timeout)
            for task in pending:
                cancel_requested = self._active_tasks.get(task)
                if cancel_requested is not None:
                    cancel_requested.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._state = TaskManagerState.CLOSED
        logger.info("Task manager shutdown complete")


# Global instances for convenience
_http_pool: AsyncResourcePool[aiohttp.ClientSession] | None = None
_task_manager: AsyncTaskManager | None = None


class GlobalResourceState(Enum):
    """Order and outcome of the global resource cleanup sequence."""

    READY = "ready"
    DRAINING_TASKS = "draining_tasks"
    DRAINING_HTTP = "draining_http"
    CLOSED = "closed"
    FAILED = "failed"


_global_resource_state = GlobalResourceState.READY
_global_cleanup_task: asyncio.Task[None] | None = None


def _check_global_access(*, allow_task_drain: bool = False) -> None:
    """Gate new ownership while retaining HTTP access for draining consumers."""
    global _global_cleanup_task, _global_resource_state
    task = _global_cleanup_task
    if task is None:
        return
    if not task.done():
        if (
            allow_task_drain
            and _global_resource_state == GlobalResourceState.DRAINING_TASKS
        ):
            return
        raise RuntimeError("Global resources are shutting down")
    if task.cancelled():
        raise RuntimeError("Global resource cleanup was cancelled; owners retained")
    error = task.exception()
    if error is not None:
        raise RuntimeError("Global resource cleanup failed; owners retained") from error
    # A successful prior lifecycle permits a new generation on first use.
    _global_cleanup_task = None
    _global_resource_state = GlobalResourceState.READY


def _http_cleanup_pending(pool: AsyncResourcePool[Any]) -> bool:
    return bool(pool.cleanup_failures) or bool(
        getattr(pool.factory, "cleanup_pending", False)
    )


def get_global_resource_status() -> dict[str, Any]:
    """Observe cleanup phase and retained ownership without creating resources."""
    return {
        "state": _global_resource_state.value,
        "task_manager_retained": _task_manager is not None,
        "http_pool_retained": _http_pool is not None,
        "http_cleanup_failures": len(_http_pool.cleanup_failures) if _http_pool else 0,
        "http_cleanup_pending": _http_pool is not None
        and (
            bool(_http_pool.cleanup_failures)
            or (
                _http_pool.state == PoolState.CLOSED
                and _http_cleanup_pending(_http_pool)
            )
        ),
    }


async def get_http_pool() -> AsyncResourcePool[aiohttp.ClientSession]:
    """Get global HTTP session pool"""
    global _http_pool

    _check_global_access(allow_task_drain=True)
    aiohttp_mod = _require_aiohttp()
    if (
        _http_pool is not None
        and _http_pool.state == PoolState.CLOSED
        and _http_cleanup_pending(_http_pool)
    ):
        raise RuntimeError("HTTP pool cleanup is unconfirmed; owner retained")
    if _http_pool is None or _http_pool.state == PoolState.CLOSED:
        factory = _SharedHTTPSessionFactory(
            timeout=aiohttp_mod.ClientTimeout(total=30),
        )
        _http_pool = AsyncResourcePool(
            factory=factory, min_size=2, max_size=20, name="HTTPPool"
        )
    pool = _http_pool
    try:
        # Concurrent callers must all await the same initialization boundary.
        await pool.initialize()
    except BaseException:
        if (
            _http_pool is pool
            and pool.state == PoolState.CLOSED
            and not _http_cleanup_pending(pool)
        ):
            _http_pool = None
        raise
    _check_global_access(allow_task_drain=True)
    if _http_pool is not pool or pool.state != PoolState.READY:
        raise RuntimeError("HTTP pool closed during initialization")
    return pool


async def get_task_manager() -> AsyncTaskManager:
    """Get global task manager"""
    global _task_manager

    _check_global_access()
    if _task_manager is None or _task_manager.state == TaskManagerState.CLOSED:
        _task_manager = AsyncTaskManager()
    manager = _task_manager
    await manager.start()
    return manager


async def cleanup_global_resources() -> None:
    """Join one ordered cleanup, retaining failed owners and rejecting replay.

    Managed tasks drain before HTTP closes, so their existing dependencies remain
    usable. Call from outside managed operations on the owners' event loop.
    """
    global _global_cleanup_task, _global_resource_state
    if (
        _task_manager is not None
        and asyncio.current_task() in _task_manager._operations
    ):
        raise RuntimeError("Global cleanup must run outside managed tasks")
    if _global_cleanup_task is None:
        manager = _task_manager
        if manager is not None:
            manager.close_admission()
        _global_resource_state = GlobalResourceState.DRAINING_TASKS
        _global_cleanup_task = asyncio.create_task(_cleanup_global_owners(manager))
    task = _global_cleanup_task

    async def join() -> None:
        await task

    await drain_owned_operation(join())


async def _cleanup_global_owners(manager: AsyncTaskManager | None) -> None:
    """Commit each owner release only after its cleanup is confirmed."""
    global _task_manager, _http_pool, _global_resource_state
    try:
        if manager is not None:
            await manager.shutdown()
            if _task_manager is manager:
                _task_manager = None
        # Capture HTTP only now: draining consumers may have initialized it.
        _global_resource_state = GlobalResourceState.DRAINING_HTTP
        pool = _http_pool
        if pool is not None:
            await pool.close()
            if _http_cleanup_pending(pool):
                raise RuntimeError("HTTP cleanup is unconfirmed; pool retained")
            if _http_pool is pool:
                _http_pool = None
        _global_resource_state = GlobalResourceState.CLOSED
    except BaseException:
        _global_resource_state = GlobalResourceState.FAILED
        raise


# Context manager for automatic cleanup
@asynccontextmanager
async def managed_async_resources():
    """Context manager for automatic resource cleanup"""
    try:
        yield
    finally:
        await cleanup_global_resources()
