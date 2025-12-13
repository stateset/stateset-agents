"""
Advanced Async Connection Pool and Resource Management

This module provides high-performance async connection pooling, resource management,
and concurrency optimization for the GRPO Agent Framework.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncContextManager,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

try:  # pragma: no cover - optional dependency
    import aiohttp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    metadata: Dict[str, Any] = field(default_factory=dict)


class PooledResource(Generic[T]):
    """Wrapper for pooled resources"""

    def __init__(
        self,
        resource: T,
        resource_id: str,
        pool: "AsyncResourcePool",
        created_at: Optional[float] = None,
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

            self.info.state = ResourceState.IDLE
            self.info.last_used = time.time()

    async def mark_failed(self, error: Exception) -> None:
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

    async def create_resource(self, **kwargs) -> T:
        """Create a new resource"""
        raise NotImplementedError

    async def validate_resource(self, resource: T) -> bool:
        """Validate that resource is still usable"""
        return True

    async def cleanup_resource(self, resource: T) -> None:
        """Clean up resource when removed from pool"""
        if hasattr(resource, "close"):
            try:
                if asyncio.iscoroutinefunction(resource.close):
                    await resource.close()
                else:
                    resource.close()
            except Exception as e:
                logger.warning(f"Error closing resource: {e}")


if aiohttp is not None:
    class HTTPSessionFactory(AsyncResourceFactory[aiohttp.ClientSession]):  # type: ignore[name-defined]
        """Factory for HTTP client sessions."""

        def __init__(self, **session_kwargs):
            self.session_kwargs = session_kwargs

        async def create_resource(self, **kwargs) -> aiohttp.ClientSession:  # type: ignore[name-defined]
            """Create new HTTP session."""
            aiohttp_mod = _require_aiohttp()
            merged_kwargs = {**self.session_kwargs, **kwargs}
            return aiohttp_mod.ClientSession(**merged_kwargs)

        async def validate_resource(self, session: aiohttp.ClientSession) -> bool:  # type: ignore[name-defined]
            """Validate HTTP session."""
            return not session.closed

        async def cleanup_resource(self, session: aiohttp.ClientSession) -> None:  # type: ignore[name-defined]
            """Close HTTP session."""
            if not session.closed:
                await session.close()
else:
    class HTTPSessionFactory(AsyncResourceFactory[Any]):
        """Placeholder factory when aiohttp is not installed."""

        def __init__(self, **session_kwargs):
            self.session_kwargs = session_kwargs

        async def create_resource(self, **kwargs) -> Any:
            _require_aiohttp()
            raise AssertionError("unreachable")

        async def validate_resource(self, session: Any) -> bool:
            return False

        async def cleanup_resource(self, session: Any) -> None:
            return None


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
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.max_age = max_age
        self.health_check_interval = health_check_interval
        self.acquire_timeout = acquire_timeout
        self.name = name

        self._pool: List[PooledResource[T]] = []
        self._available = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._state = PoolState.INITIALIZING
        self._stats = PoolStats()
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._resource_counter = 0

    async def initialize(self) -> None:
        """Initialize the pool"""
        logger.info(f"Initializing pool {self.name} with min_size={self.min_size}")

        created: List[PooledResource[T]] = []
        try:
            async with self._lock:
                # Create minimum number of resources
                for _ in range(self.min_size):
                    resource = await self._create_resource(raise_on_fail=True)
                    created.append(resource)
                    self._pool.append(resource)
                    await self._available.put(resource)

                self._state = PoolState.READY
                self._stats.total_connections = len(self._pool)
                self._stats.idle_connections = len(self._pool)
        except Exception:
            # Ensure partially created pools don't leak resources.
            for resource in created:
                try:
                    await self.factory.cleanup_resource(resource.resource)
                except Exception:
                    pass
            self._state = PoolState.CLOSED
            raise

        # Start health check task
        if self.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"Pool {self.name} initialized with {len(self._pool)} resources")

    async def _create_resource(
        self, raise_on_fail: bool = False
    ) -> Optional[PooledResource[T]]:
        """Create a new pooled resource"""
        try:
            self._resource_counter += 1
            resource_id = f"{self.name}_resource_{self._resource_counter}"

            raw_resource = await self.factory.create_resource()
            pooled_resource = PooledResource(raw_resource, resource_id, self)

            logger.debug(f"Created resource {resource_id}")
            return pooled_resource

        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            self._stats.failed_connections += 1
            if raise_on_fail:
                raise
            return None

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None) -> AsyncContextManager[T]:
        """Acquire a resource from the pool"""
        if self._state != PoolState.READY:
            raise RuntimeError(f"Pool {self.name} is not ready (state: {self._state})")

        timeout = timeout or self.acquire_timeout
        start_time = time.time()

        try:
            # Try to get available resource
            try:
                pooled_resource = await asyncio.wait_for(
                    self._available.get(), timeout=timeout
                )
                self._stats.pool_hits += 1

            except asyncio.TimeoutError:
                # Try to create new resource if under max size
                async with self._lock:
                    if len(self._pool) < self.max_size:
                        pooled_resource = await self._create_resource()
                        if pooled_resource:
                            self._pool.append(pooled_resource)
                            self._stats.total_connections += 1
                            self._stats.pool_misses += 1
                        else:
                            raise RuntimeError("Failed to create new resource")
                    else:
                        raise RuntimeError(
                            f"Pool {self.name} exhausted (timeout after {timeout}s)"
                        )

            # Validate resource
            if not await self.factory.validate_resource(pooled_resource.resource):
                logger.warning(
                    f"Resource {pooled_resource.info.resource_id} failed validation"
                )
                await self._remove_resource(pooled_resource)
                raise RuntimeError("Resource validation failed")

            # Acquire resource
            resource = await pooled_resource.acquire()

            async with self._lock:
                self._stats.active_connections += 1
                self._stats.idle_connections -= 1
                self._stats.total_requests += 1

                if self._stats.active_connections > self._stats.peak_connections:
                    self._stats.peak_connections = self._stats.active_connections

            try:
                yield resource

                # Track successful request
                async with self._lock:
                    self._stats.successful_requests += 1
                    response_time = time.time() - start_time

                    # Update average response time
                    if self._stats.average_response_time == 0:
                        self._stats.average_response_time = response_time
                    else:
                        self._stats.average_response_time = (
                            self._stats.average_response_time
                            * (self._stats.successful_requests - 1)
                            + response_time
                        ) / self._stats.successful_requests

            except Exception as e:
                # Mark resource as failed
                await pooled_resource.mark_failed(e)

                async with self._lock:
                    self._stats.failed_requests += 1

                logger.error(
                    f"Error using resource {pooled_resource.info.resource_id}: {e}"
                )
                raise

            finally:
                # Release resource back to pool
                await pooled_resource.release()

                async with self._lock:
                    self._stats.active_connections -= 1
                    self._stats.idle_connections += 1

                # Return to available pool if resource is still good
                if pooled_resource.info.state == ResourceState.IDLE:
                    await self._available.put(pooled_resource)
                else:
                    # Remove failed resource
                    await self._remove_resource(pooled_resource)

        except Exception as e:
            async with self._lock:
                self._stats.failed_requests += 1
            raise

    async def _remove_resource(self, pooled_resource: PooledResource[T]) -> None:
        """Remove resource from pool"""
        try:
            await self.factory.cleanup_resource(pooled_resource.resource)
        except Exception as e:
            logger.warning(f"Error cleaning up resource: {e}")

        async with self._lock:
            if pooled_resource in self._pool:
                self._pool.remove(pooled_resource)
                self._stats.total_connections -= 1

                if pooled_resource.info.state == ResourceState.IDLE:
                    self._stats.idle_connections -= 1
                elif pooled_resource.info.state == ResourceState.ACTIVE:
                    self._stats.active_connections -= 1

        logger.debug(f"Removed resource {pooled_resource.info.resource_id}")

    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _perform_health_check(self) -> None:
        """Perform health check on pool resources"""
        current_time = time.time()
        resources_to_remove = []

        async with self._lock:
            for resource in self._pool:
                # Check for stale resources
                if resource.is_stale(self.max_age):
                    logger.info(f"Resource {resource.info.resource_id} is stale")
                    resources_to_remove.append(resource)
                    continue

                # Check for idle resources
                if (
                    resource.is_idle_too_long(self.max_idle_time)
                    and len(self._pool) > self.min_size
                ):
                    logger.info(f"Resource {resource.info.resource_id} idle too long")
                    resources_to_remove.append(resource)
                    continue

                # Check resource health
                if resource.info.state == ResourceState.IDLE:
                    try:
                        is_valid = await self.factory.validate_resource(
                            resource.resource
                        )
                        if not is_valid:
                            logger.warning(
                                f"Resource {resource.info.resource_id} failed health check"
                            )
                            resources_to_remove.append(resource)
                    except Exception as e:
                        logger.error(
                            f"Health check error for {resource.info.resource_id}: {e}"
                        )
                        resources_to_remove.append(resource)

        # Remove unhealthy resources
        for resource in resources_to_remove:
            await self._remove_resource(resource)

        # Ensure minimum pool size
        async with self._lock:
            while len(self._pool) < self.min_size:
                new_resource = await self._create_resource()
                if new_resource:
                    self._pool.append(new_resource)
                    await self._available.put(new_resource)
                else:
                    break

    async def scale(self, target_size: int) -> None:
        """Scale pool to target size"""
        if target_size < self.min_size or target_size > self.max_size:
            raise ValueError(
                f"Target size {target_size} outside valid range [{self.min_size}, {self.max_size}]"
            )

        async with self._lock:
            current_size = len(self._pool)

            if target_size > current_size:
                # Scale up
                for _ in range(target_size - current_size):
                    resource = await self._create_resource()
                    if resource:
                        self._pool.append(resource)
                        await self._available.put(resource)

            elif target_size < current_size:
                # Scale down - remove idle resources
                resources_to_remove = []
                for resource in self._pool:
                    if (
                        resource.info.state == ResourceState.IDLE
                        and len(self._pool) - len(resources_to_remove) > target_size
                    ):
                        resources_to_remove.append(resource)

                for resource in resources_to_remove:
                    await self._remove_resource(resource)

        logger.info(f"Scaled pool {self.name} to {len(self._pool)} resources")

    def get_stats(self) -> PoolStats:
        """Get current pool statistics"""
        self._stats.timestamp = time.time()
        return self._stats

    async def drain(self) -> None:
        """Drain the pool (wait for active connections to finish)"""
        logger.info(f"Draining pool {self.name}")
        self._state = PoolState.DRAINING

        # Wait for active connections to finish
        while True:
            async with self._lock:
                if self._stats.active_connections == 0:
                    break
            await asyncio.sleep(0.1)

        logger.info(f"Pool {self.name} drained")

    async def close(self) -> None:
        """Close the pool and cleanup all resources"""
        logger.info(f"Closing pool {self.name}")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Drain pool
        await self.drain()

        # Cleanup all resources
        async with self._lock:
            self._state = PoolState.CLOSED

            while self._pool:
                resource = self._pool.pop()
                try:
                    await self.factory.cleanup_resource(resource.resource)
                except Exception as e:
                    logger.warning(f"Error cleaning up resource during close: {e}")

        logger.info(f"Pool {self.name} closed")


class AsyncTaskManager:
    """Advanced async task management with resource limits"""

    def __init__(
        self,
        max_concurrent_tasks: int = 100,
        task_timeout: float = 300.0,
        cleanup_interval: float = 60.0,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        self.cleanup_interval = cleanup_interval

        self._active_tasks: Set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the task manager"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            f"Task manager started with max {self.max_concurrent_tasks} concurrent tasks"
        )

    async def submit_task(
        self, coro: Awaitable[T], timeout: Optional[float] = None
    ) -> T:
        """Submit a task for execution"""
        timeout = timeout or self.task_timeout

        async with self._semaphore:
            task = asyncio.create_task(asyncio.wait_for(coro, timeout=timeout))

            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

            try:
                return await task
            except asyncio.TimeoutError:
                logger.warning(f"Task timed out after {timeout}s")
                raise
            except Exception as e:
                logger.error(f"Task failed: {e}")
                raise

    async def submit_batch(
        self, coros: List[Awaitable[T]], return_exceptions: bool = False
    ) -> List[Union[T, Exception]]:
        """Submit a batch of tasks"""
        tasks = [self.submit_task(coro) for coro in coros]

        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    async def _cleanup_loop(self) -> None:
        """Background cleanup of completed tasks"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)

                # Clean up completed tasks
                completed_tasks = {task for task in self._active_tasks if task.done()}

                for task in completed_tasks:
                    self._active_tasks.discard(task)

                    if task.exception():
                        logger.warning(f"Background task failed: {task.exception()}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    def get_status(self) -> Dict[str, int]:
        """Get task manager status"""
        return {
            "active_tasks": len(self._active_tasks),
            "available_slots": self.max_concurrent_tasks - len(self._active_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
        }

    async def shutdown(self) -> None:
        """Shutdown task manager"""
        logger.info("Shutting down task manager")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Wait for active tasks to complete or cancel them
        if self._active_tasks:
            logger.info(
                f"Waiting for {len(self._active_tasks)} active tasks to complete"
            )

            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within timeout, cancelling")
                for task in self._active_tasks:
                    task.cancel()

                # Wait a bit more for cancellation
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._active_tasks, return_exceptions=True),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.error("Some tasks could not be cancelled")

        logger.info("Task manager shutdown complete")


# Global instances for convenience
_http_pool: Optional[AsyncResourcePool[aiohttp.ClientSession]] = None
_task_manager: Optional[AsyncTaskManager] = None


async def get_http_pool() -> AsyncResourcePool[aiohttp.ClientSession]:
    """Get global HTTP session pool"""
    global _http_pool

    aiohttp_mod = _require_aiohttp()
    if _http_pool is None:
        factory = HTTPSessionFactory(
            timeout=aiohttp_mod.ClientTimeout(total=30),
            connector=aiohttp_mod.TCPConnector(limit=100),
        )
        _http_pool = AsyncResourcePool(
            factory=factory, min_size=2, max_size=20, name="HTTPPool"
        )
        await _http_pool.initialize()

    return _http_pool


async def get_task_manager() -> AsyncTaskManager:
    """Get global task manager"""
    global _task_manager

    if _task_manager is None:
        _task_manager = AsyncTaskManager()
        await _task_manager.start()

    return _task_manager


async def cleanup_global_resources() -> None:
    """Cleanup global resources"""
    global _http_pool, _task_manager

    cleanup_tasks = []

    if _http_pool:
        cleanup_tasks.append(_http_pool.close())
        _http_pool = None

    if _task_manager:
        cleanup_tasks.append(_task_manager.shutdown())
        _task_manager = None

    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)


# Context manager for automatic cleanup
@asynccontextmanager
async def managed_async_resources():
    """Context manager for automatic resource cleanup"""
    try:
        yield
    finally:
        await cleanup_global_resources()
