"""
Distributed Caching Module

Production-ready caching with Redis support, local fallback, and comprehensive
cache management features for high-availability deployments.
"""

import asyncio
import copy
import fnmatch
import hashlib
import inspect
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Generic, TypeVar

from stateset_agents.utils.async_calls import drain_owned_operation, run_async_owned

from .cache_lifecycle import CacheEntryOrdering, CacheFlight, CacheOperations
from .resource_lifecycle import ResourceLease, ResourceOwnershipError, SharedResource

logger = logging.getLogger(__name__)

CACHE_EXCEPTIONS = (
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _redis_errors() -> tuple[type[Exception], ...]:
    """Include the optional SDK's errors without requiring Redis for memory use."""
    try:
        from redis.exceptions import RedisError
    except ImportError:
        return CACHE_EXCEPTIONS
    return (*CACHE_EXCEPTIONS, RedisError)


T = TypeVar("T")


def _json_default_serializer(obj: Any) -> Any:
    """Best-effort JSON serializer for complex objects."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# ============================================================================
# Cache Configuration
# ============================================================================


class CacheBackend(str, Enum):
    """Available cache backends."""

    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"  # Redis primary, memory fallback


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    backend: CacheBackend = CacheBackend.MEMORY
    redis_url: str | None = None
    redis_password: str | None = None
    redis_db: int = 0
    default_ttl_seconds: int = 300
    max_memory_items: int = 10000
    key_prefix: str = "stateset:"
    serializer: str = "json"  # json or pickle
    allow_pickle: bool = False
    compression_enabled: bool = False
    compression_threshold: int = 1024  # bytes
    health_check_interval: int = 30
    retry_attempts: int = 3
    retry_delay: float = 0.1
    #: Bound initial Redis discovery. A missing cache is a degraded-mode event,
    #: not a reason to stall API startup or event-loop shutdown.
    redis_connect_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.redis_connect_timeout_seconds <= 0:
            raise ValueError("redis_connect_timeout_seconds must be positive")
        serializer = str(self.serializer).lower()
        if serializer not in ("json", "pickle"):
            logger.warning(
                "Unknown CACHE_SERIALIZER '%s', defaulting to json", serializer
            )
            serializer = "json"
        if serializer == "pickle" and not self.allow_pickle:
            logger.warning(
                "CACHE_SERIALIZER=pickle is unsafe with untrusted cache backends; "
                "set CACHE_ALLOW_PICKLE=true to override."
            )
            serializer = "json"
        self.serializer = serializer

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create config from environment variables."""
        import os

        backend_str = os.getenv("CACHE_BACKEND", "memory").lower()
        try:
            backend = CacheBackend(backend_str)
        except ValueError:
            logger.warning(
                "Unknown CACHE_BACKEND '%s', defaulting to memory", backend_str
            )
            backend = CacheBackend.MEMORY

        serializer = os.getenv("CACHE_SERIALIZER", "json").lower()
        allow_pickle = os.getenv("CACHE_ALLOW_PICKLE", "false").lower() == "true"
        if serializer not in ("json", "pickle"):
            logger.warning(
                "Unknown CACHE_SERIALIZER '%s', defaulting to json", serializer
            )
            serializer = "json"
        elif serializer == "pickle" and not allow_pickle:
            logger.warning(
                "CACHE_SERIALIZER=pickle is unsafe with untrusted cache backends; "
                "set CACHE_ALLOW_PICKLE=true to override."
            )
            serializer = "json"

        return cls(
            backend=backend,
            redis_url=os.getenv("REDIS_URL"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            default_ttl_seconds=int(os.getenv("CACHE_DEFAULT_TTL", "300")),
            max_memory_items=int(os.getenv("CACHE_MAX_MEMORY_ITEMS", "10000")),
            key_prefix=os.getenv("CACHE_KEY_PREFIX", "stateset:"),
            serializer=serializer,
            allow_pickle=allow_pickle,
            compression_enabled=os.getenv("CACHE_COMPRESSION", "false").lower()
            == "true",
            redis_connect_timeout_seconds=float(
                os.getenv("CACHE_REDIS_CONNECT_TIMEOUT", "2.0")
            ),
        )


# ============================================================================
# Cache Statistics
# ============================================================================


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    avg_latency_ms: float = 0.0
    _latencies: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_latency(self, latency_ms: float) -> None:
        """Record operation latency."""
        self._latencies.append(latency_ms)
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate, 4),
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
        }


# ============================================================================
# Abstract Cache Interface
# ============================================================================


class CacheInterface(ABC, Generic[T]):
    """Abstract interface for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Get a value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: T, ttl: int | None = None) -> bool:
        """Set a value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        pass

    @abstractmethod
    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a pattern. Returns number of keys removed."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all keys. Returns count of deleted keys."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check cache health."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


# ============================================================================
# Memory Cache Implementation
# ============================================================================


@dataclass
class MemoryCacheEntry:
    """Entry in the memory cache."""

    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    monotonic_expires_at: float | None = None

    def is_expired(self) -> bool:
        """Use elapsed time for owned entries, retaining legacy entry construction."""
        if self.monotonic_expires_at is not None:
            return time.monotonic() >= self.monotonic_expires_at
        return time.time() >= self.expires_at


class MemoryCache(CacheInterface):
    """
    High-performance in-memory cache with LRU eviction.

    Features:
    - TTL-based expiration
    - LRU eviction when max size reached
    - Thread-safe operations
    - Comprehensive statistics
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: dict[str, MemoryCacheEntry] = {}
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        start = time.monotonic()
        prefixed_key = f"{self.config.key_prefix}{key}"

        async with self._lock:
            entry = self._cache.get(prefixed_key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                del self._cache[prefixed_key]
                self._stats.misses += 1
                return None

            entry.access_count += 1
            self._stats.hits += 1
            self._stats.bytes_read += entry.size_bytes
            self._stats.record_latency((time.monotonic() - start) * 1000)

            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in cache."""
        return await self._store(key, value, ttl)

    async def _set_until(self, key: str, value: Any, deadline: float) -> bool:
        """Publish a refill without extending its captured monotonic expiry."""
        return await self._store(key, value, deadline=deadline)

    async def _store(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        *,
        deadline: float | None = None,
    ) -> bool:
        start = time.monotonic()
        prefixed_key = f"{self.config.key_prefix}{key}"
        ttl = ttl or self.config.default_ttl_seconds
        expires = start + ttl if deadline is None else deadline

        try:
            # Estimate size
            if self.config.serializer == "pickle":
                size_bytes = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            elif isinstance(value, (dict, list)):
                size_bytes = len(json.dumps(value, default=_json_default_serializer))
            else:
                size_bytes = 100

            async with self._lock:
                if deadline is not None and time.monotonic() >= deadline:
                    self._cache.pop(prefixed_key, None)
                    return False
                # Evict if at capacity
                if len(self._cache) >= self.config.max_memory_items:
                    await self._evict_lru()

                self._cache[prefixed_key] = MemoryCacheEntry(
                    value=value,
                    expires_at=time.time() + (expires - time.monotonic()),
                    size_bytes=size_bytes,
                    monotonic_expires_at=expires,
                )

                self._stats.sets += 1
                self._stats.bytes_written += size_bytes
                self._stats.record_latency((time.monotonic() - start) * 1000)

                return True
        except CACHE_EXCEPTIONS as e:
            logger.error(f"Memory cache set error: {e}")
            self._stats.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        prefixed_key = f"{self.config.key_prefix}{key}"

        async with self._lock:
            if prefixed_key in self._cache:
                del self._cache[prefixed_key]
                self._stats.deletes += 1
                return True
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a pattern."""
        if pattern.startswith(self.config.key_prefix):
            pattern_with_prefix = pattern
        else:
            pattern_with_prefix = f"{self.config.key_prefix}{pattern}"

        async with self._lock:
            matching = [
                key
                for key in list(self._cache.keys())
                if fnmatch.fnmatch(key, pattern_with_prefix)
            ]
            for key in matching:
                del self._cache[key]
                self._stats.deletes += 1
            return len(matching)

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        prefixed_key = f"{self.config.key_prefix}{key}"

        async with self._lock:
            entry = self._cache.get(prefixed_key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[prefixed_key]
                return False
            return True

    async def clear(self) -> int:
        """Clear all keys."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def health_check(self) -> bool:
        """Check cache health."""
        return True

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    async def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._cache:
            return

        # Sort by access_count (ascending) and created_at (ascending)
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: (self._cache[k].access_count, self._cache[k].created_at),
        )

        # Remove 10% of entries
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            del self._cache[key]

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        expired = []

        async with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired.append(key)

            for key in expired:
                del self._cache[key]

        return len(expired)


# ============================================================================
# Redis Cache Implementation
# ============================================================================


class RedisCache(CacheInterface):
    """
    Redis-backed distributed cache.

    Features:
    - Distributed caching across multiple instances
    - Connection pooling
    - Automatic reconnection
    - Compression for large values
    - Cluster support
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._client: Any | None = None
        self._stats = CacheStats()
        self._connected = False
        self._last_health_check = float("-inf")
        self._health_status = False
        self._closed = False
        self._client_cleanup: asyncio.Task[None] | None = None
        self._operations = CacheOperations()
        self._connect_flight: CacheFlight | None = None
        self._health_flight: CacheFlight | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._retiring = False
        self._state = "new"
        self._generation = 0
        self._error: BaseException | None = None
        self._cleanup_error: BaseException | None = None

    async def connect(self) -> bool:
        """Share one connection attempt and retain cancelled setup through cleanup."""
        self._operations.check_callback()
        if self._is_closed():
            raise ResourceOwnershipError("Closed Redis caches cannot reconnect")
        if self._cleanup_error is not None:
            raise ResourceOwnershipError(
                "Redis client cleanup is unconfirmed"
            ) from self._cleanup_error
        if self._retiring:
            return False
        if self._connected:
            return True
        flight = self._connect_flight
        if flight is not None and flight.task is not None and not flight.task.done():
            return False if flight.retired else await flight.observe()
        configuration = copy.deepcopy(self.config)
        self._client_cleanup = None
        self._connect_flight = flight = CacheFlight()
        self._state = "connecting"
        self._error = None
        work = self._connect_once(configuration, flight)
        try:
            flight.task = asyncio.create_task(work)
        except BaseException as error:
            work.close()
            self._state = "unavailable"
            self._error = error
            raise
        return await flight.observe()

    async def _ping(self, client: Any, timeout: float) -> None:
        async def ping() -> Any:
            return await client.ping()

        deadline = time.monotonic() + timeout
        acknowledged = await asyncio.wait_for(run_async_owned(ping()), timeout=timeout)
        if time.monotonic() >= deadline:
            raise TimeoutError("Redis health deadline exceeded")
        if not acknowledged:
            raise ConnectionError("Redis health was not acknowledged")

    async def _connect_once(self, config: CacheConfig, flight: CacheFlight) -> bool:
        published = False
        failure: BaseException | None = None
        try:
            if not config.redis_url:
                return False
            with self._operations.callback():
                import redis.asyncio as redis

                client = redis.from_url(
                    config.redis_url,
                    password=config.redis_password,
                    db=config.redis_db,
                    decode_responses=False,
                    socket_timeout=config.redis_connect_timeout_seconds,
                    socket_connect_timeout=config.redis_connect_timeout_seconds,
                    retry_on_timeout=False,
                )
                self._client = client
                self._generation += 1
                await self._ping(client, config.redis_connect_timeout_seconds)
            if self._closed or flight.retired or self._client is not client:
                return False
            self._connected = True
            self._health_status = True
            self._last_health_check = time.monotonic()
            self._state = "ready"
            published = True
            return True
        except (Exception, asyncio.CancelledError) as error:
            failure = self._error = error
            if isinstance(error, asyncio.CancelledError):
                raise
            logger.warning("Redis connection unavailable (%s)", type(error).__name__)
            return False
        finally:
            if not published:
                try:
                    await self._close_client()
                except (Exception, asyncio.CancelledError) as cleanup_error:
                    if failure is not None:
                        raise failure from cleanup_error
                    raise
                if not self._closed and self._cleanup_error is None:
                    self._state = "unavailable"
                    self._last_health_check = time.monotonic()

    async def _close_client(self) -> None:
        """Retire one client after all its admitted commands have finished."""
        self._operations.check_callback()
        self._connected = False
        self._health_status = False
        client = self._client
        if client is None:
            return
        if self._client_cleanup is None:
            self._retiring = True
            if not self._closed:
                self._state = "retiring"

            async def release() -> None:
                try:
                    await self._operations.wait_idle()
                    with self._operations.callback():
                        close = getattr(client, "aclose", None) or getattr(
                            client, "close", None
                        )
                        if close is not None:
                            result = close()
                            if inspect.isawaitable(result):
                                await result
                    if self._client is client:
                        self._client = None
                    if not self._closed:
                        self._state = "unavailable"
                except BaseException as error:
                    self._cleanup_error = error
                    self._state = "failed"
                    raise
                finally:
                    self._retiring = False

            work = release()
            try:
                self._client_cleanup = asyncio.create_task(work)
            except BaseException as error:
                work.close()
                self._cleanup_error = error
                self._state = "failed"
                raise
        task = self._client_cleanup

        async def join() -> None:
            await asyncio.shield(task)

        await drain_owned_operation(join())

    def _is_closed(self) -> bool:
        return self._closed

    async def _run(self, operation: Coroutine[Any, Any, T], fallback: T) -> T:
        return await self._operations.run(
            operation,
            fallback,
            available=self._connected and not self._closed and not self._retiring,
        )

    async def get(self, key: str) -> Any | None:
        """Get a value from Redis."""
        _, value = await self._get_with_status(key)
        return value

    async def _get_with_status(self, key: str) -> tuple[bool, Any | None]:
        return await self._run(self._read(key), (False, None))

    async def _get_with_expiry(self, key: str) -> tuple[bool, Any | None, float | None]:
        """Read a value and its TTL together for a bounded local fallback copy."""
        return await self._run(self._read_entry(key), (False, None, None))

    async def _read_entry(self, key: str) -> tuple[bool, Any | None, float | None]:
        if not self._connected or self._client is None:
            return False, None, None
        client = self._client
        pipeline_factory = getattr(client, "pipeline", None)
        if not callable(pipeline_factory):
            answered, value = await self._read(key)
            return answered, value, None
        start = time.monotonic()
        prefixed_key = f"{self.config.key_prefix}{key}"
        try:
            async with pipeline_factory(transaction=True) as pipeline:
                pipeline.get(prefixed_key)
                pipeline.pttl(prefixed_key)
                data, ttl_ms = await pipeline.execute(raise_on_error=False)
            if isinstance(data, BaseException):
                raise data
            if data is None or ttl_ms == -2:
                self._stats.misses += 1
                return True, None, start
            value = self._deserialize(data)
            self._stats.hits += 1
            self._stats.bytes_read += len(data)
            self._stats.record_latency((time.monotonic() - start) * 1000)
            # Unknown TTL (for example a PTTL ACL denial) permits the remote
            # answer but never authorizes a new fallback lifetime.
            deadline = None
            if type(ttl_ms) is int and ttl_ms >= -1:
                ttl: float = self.config.default_ttl_seconds
                if ttl_ms >= 0:
                    ttl = min(ttl, ttl_ms / 1000)
                deadline = start + ttl
            return True, value, deadline
        except _redis_errors() as error:
            # Server rejection of MULTI/EXEC/PTTL must not make an otherwise
            # readable cache unavailable. This read-only fallback cannot refill.
            try:
                from redis.exceptions import ResponseError
            except ImportError:
                rejected = False
            else:
                rejected = isinstance(error, ResponseError)
            if rejected:
                answered, value = await self._read(key)
                return answered, value, None
            logger.warning("Redis entry read unavailable (%s)", type(error).__name__)
            self._error = error
            self._stats.errors += 1
            return False, None, None

    async def _read(self, key: str) -> tuple[bool, Any | None]:
        """Return whether Redis answered successfully, even for a cache miss."""
        if not self._connected:
            return False, None
        client = self._client
        if client is None:
            return False, None

        start = time.monotonic()
        prefixed_key = f"{self.config.key_prefix}{key}"

        try:
            data = await client.get(prefixed_key)

            if data is None:
                self._stats.misses += 1
                return True, None

            self._stats.hits += 1
            self._stats.bytes_read += len(data)

            # Decompress if needed
            value = self._deserialize(data)
            self._stats.record_latency((time.monotonic() - start) * 1000)

            return True, value

        except _redis_errors() as e:
            logger.warning("Redis get unavailable (%s)", type(e).__name__)
            self._error = e
            self._stats.errors += 1
            return False, None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value while retaining this client through cancellation cleanup."""
        return await self._run(self._set(key, value, ttl), False)

    async def _set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in Redis."""
        if not self._connected:
            return False
        client = self._client
        if client is None:
            return False

        start = time.monotonic()
        prefixed_key = f"{self.config.key_prefix}{key}"
        ttl = ttl or self.config.default_ttl_seconds

        try:
            data = self._serialize(value)

            if not await client.setex(prefixed_key, ttl, data):
                return False

            self._stats.sets += 1
            self._stats.bytes_written += len(data)
            self._stats.record_latency((time.monotonic() - start) * 1000)

            return True

        except _redis_errors() as e:
            logger.warning("Redis set unavailable (%s)", type(e).__name__)
            self._error = e
            self._stats.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key using the currently admitted client."""
        _, deleted = await self._delete_with_status(key)
        return deleted

    async def _delete_with_status(self, key: str) -> tuple[bool, bool]:
        return await self._run(self._delete(key), (False, False))

    async def _delete(self, key: str) -> tuple[bool, bool]:
        """Delete a key from Redis."""
        if not self._connected:
            return False, False
        client = self._client
        if client is None:
            return False, False

        prefixed_key = f"{self.config.key_prefix}{key}"

        try:
            result = await client.delete(prefixed_key)
            self._stats.deletes += 1
            return True, bool(result > 0)
        except _redis_errors() as e:
            logger.warning("Redis delete unavailable (%s)", type(e).__name__)
            self._error = e
            self._stats.errors += 1
            return False, False

    async def delete_pattern(self, pattern: str) -> int:
        """Retain the same client through every scan/delete page."""
        answered, deleted = await self._delete_pattern_with_status(pattern)
        return deleted if answered else 0

    async def _delete_pattern_with_status(self, pattern: str) -> tuple[bool, int]:
        return await self._run(self._delete_pattern(pattern), (False, 0))

    async def _delete_pattern(self, pattern: str) -> tuple[bool, int]:
        """Delete keys matching a pattern."""
        if not self._connected:
            return False, 0
        client = self._client
        if client is None:
            return False, 0

        if pattern.startswith(self.config.key_prefix):
            pattern_with_prefix = pattern
        else:
            pattern_with_prefix = f"{self.config.key_prefix}{pattern}"

        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await client.scan(
                    cursor, match=pattern_with_prefix, count=100
                )
                if keys:
                    deleted += await client.delete(*keys)
                    self._stats.deletes += len(keys)
                if cursor == 0:
                    break
            return True, deleted
        except _redis_errors() as e:
            logger.warning("Redis delete_pattern unavailable (%s)", type(e).__name__)
            self._error = e
            self._stats.errors += 1
            return False, 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        _, exists = await self._exists_with_status(key)
        return exists

    async def _exists_with_status(self, key: str) -> tuple[bool, bool]:
        return await self._run(self._exists(key), (False, False))

    async def _exists(self, key: str) -> tuple[bool, bool]:
        """Return whether Redis answered successfully and the key exists."""
        if not self._connected:
            return False, False
        client = self._client
        if client is None:
            return False, False

        prefixed_key = f"{self.config.key_prefix}{key}"

        try:
            return True, bool(await client.exists(prefixed_key) > 0)
        except _redis_errors() as e:
            logger.warning("Redis exists unavailable (%s)", type(e).__name__)
            self._error = e
            self._stats.errors += 1
            return False, False

    async def clear(self) -> int:
        """Retain the client until this prefix has finished clearing."""
        answered, deleted = await self._clear_with_status()
        return deleted if answered else 0

    async def _clear_with_status(self) -> tuple[bool, int]:
        return await self._run(self._clear(), (False, 0))

    async def _clear(self) -> tuple[bool, int]:
        """Clear all keys with our prefix."""
        if not self._connected:
            return False, 0
        client = self._client
        if client is None:
            return False, 0

        try:
            pattern = f"{self.config.key_prefix}*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await client.delete(*keys)
                if cursor == 0:
                    break

            return True, deleted
        except _redis_errors() as e:
            logger.warning("Redis clear unavailable (%s)", type(e).__name__)
            self._error = e
            return False, 0

    async def health_check(self) -> bool:
        """Share one probe; cache only its settled observation using a monotonic clock."""
        self._operations.check_callback()
        if self._closed or self._cleanup_error is not None or self._retiring:
            return False
        flight = self._health_flight
        if flight is not None and flight.task is not None and not flight.task.done():
            return False if flight.retired else await flight.observe()
        if (
            time.monotonic() - self._last_health_check
            < self.config.health_check_interval
        ):
            return self._health_status
        self._health_flight = flight = CacheFlight()
        work = self._health_once(flight)
        try:
            flight.task = asyncio.create_task(work)
        except BaseException:
            work.close()
            raise
        return await flight.observe()

    async def _health_once(self, flight: CacheFlight) -> bool:
        healthy = False
        generation = self._generation
        try:
            if not self._connected:
                healthy = await self.connect()
            else:
                client = self._client
                if client is None:
                    self._connected = False
                    return False
                with self._operations.callback():
                    await self._ping(client, self.config.redis_connect_timeout_seconds)
                healthy = (
                    self._client is client and not self._closed and not self._retiring
                )
            return healthy and not flight.retired
        except _redis_errors() as error:
            self._error = error
            logger.warning("Redis health unavailable (%s)", type(error).__name__)
            await self._close_client()
            return False
        finally:
            if (
                not flight.retired
                and not self._closed
                and self._generation == generation
                and self._health_flight is flight
            ):
                self._last_health_check = time.monotonic()
                self._health_status = healthy and self._cleanup_error is None

    def get_status(self) -> dict[str, Any]:
        """Inspect ownership without querying Redis or exposing keys/credentials."""

        def flight_status(flight: CacheFlight | None) -> dict[str, Any] | None:
            if flight is None:
                return None
            return {
                "running": flight.task is not None and not flight.task.done(),
                "observers": flight.observers,
                "cancellation_requested": flight.retired,
            }

        return {
            "state": self._state,
            "generation": self._generation,
            "connected": self._connected,
            "active_operations": self._operations.active,
            "connection": flight_status(self._connect_flight),
            "health": flight_status(self._health_flight),
            "error_type": type(self._error).__name__ if self._error else None,
            "cleanup_error_type": (
                type(self._cleanup_error).__name__ if self._cleanup_error else None
            ),
        }

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.config.serializer == "pickle":
            if not self.config.allow_pickle:
                logger.warning(
                    "Pickle serializer disabled; falling back to json serialization."
                )
            else:
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                # Compress if large
                if (
                    self.config.compression_enabled
                    and len(data) > self.config.compression_threshold
                ):
                    import zlib

                    data = b"ZLIB:" + zlib.compress(data)
                return data
        data = json.dumps(value, default=_json_default_serializer).encode("utf-8")

        # Compress if large
        if (
            self.config.compression_enabled
            and len(data) > self.config.compression_threshold
        ):
            import zlib

            data = b"ZLIB:" + zlib.compress(data)

        return data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if data.startswith(b"ZLIB:"):
            import zlib

            data = zlib.decompress(data[5:])

        if self.config.serializer == "pickle":
            if not self.config.allow_pickle:
                logger.warning(
                    "Pickle serializer disabled; refusing to deserialize pickle payload."
                )
            else:
                # Pickle deserialization is opt-in via config.allow_pickle and
                # only reads from this process's own trusted Redis cache,
                # never externally supplied data.
                return pickle.loads(data)  # nosec: B301
        return json.loads(data.decode("utf-8"))

    async def close(self) -> None:
        """Stop admission, drain commands and shared observations, then close once."""
        self._operations.check_callback()
        if self._close_task is None:
            self._closed = True
            self._connected = False
            self._health_status = False
            self._state = "closing"
            work = self._finish_close()
            try:
                self._close_task = asyncio.create_task(work)
            except BaseException as error:
                work.close()
                self._cleanup_error = error
                self._state = "failed"
                raise
        task = self._close_task

        async def join() -> None:
            await asyncio.shield(task)

        await drain_owned_operation(join())

    async def _finish_close(self) -> None:
        try:
            flights = [self._health_flight, self._connect_flight]
            for flight in flights:
                if flight is not None:
                    flight.cancel()
            tasks = [
                flight.task
                for flight in flights
                if flight is not None and flight.task is not None
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._operations.wait_idle()
            await self._close_client()
            if self._cleanup_error is not None:
                raise self._cleanup_error
            self._state = "closed"
        except BaseException as error:
            self._cleanup_error = error
            self._state = "failed"
            raise


# ============================================================================
# Hybrid Cache (Redis + Memory Fallback)
# ============================================================================


class HybridCache(CacheInterface):
    """
    Hybrid cache with Redis as primary and memory as fallback.

    Provides high availability by falling back to memory cache
    when Redis is unavailable.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._redis = RedisCache(config)
        self._memory = MemoryCache(config)
        self._use_redis = True
        self._operations = CacheOperations()
        self._entries = CacheEntryOrdering()
        self._memory_safe = True
        self._unconfirmed_mutations = 0
        self._last_mutation: dict[str, Any] | None = None
        self._closed = False
        self._state = "ready"
        self._close_task: asyncio.Task[None] | None = None
        self._cleanup_error: BaseException | None = None

    async def _run(self, operation: Coroutine[Any, Any, T], fallback: T) -> T:
        return await self._operations.run(
            operation, fallback, available=not self._closed
        )

    async def _entry(
        self,
        key: str | None,
        operation: Callable[[], Coroutine[Any, Any, T]],
        fallback: T,
    ) -> T:
        async def ordered() -> T:
            async with self._entries.hold(key):
                return await operation()

        return await self._run(ordered(), fallback)

    async def _mutate(
        self,
        name: str,
        operation: Callable[[dict[str, Any]], Coroutine[Any, Any, T]],
        invalidate: Callable[[], Coroutine[Any, Any, Any]],
    ) -> T:
        receipt: dict[str, Any] = {
            "operation": name,
            "completed": False,
            "memory_acknowledged": False,
            "redis_acknowledged": False if self._use_redis else None,
            "error_type": None,
        }
        try:
            result = await operation(receipt)
            receipt["completed"] = True
            return result
        except BaseException as error:
            receipt["error_type"] = type(error).__name__
            # A cancelled write may have reached Redis. Retain entry ordering
            # until local publication is removed; never replay the remote effect.
            try:
                await drain_owned_operation(invalidate())
            except BaseException as cleanup_error:
                self._memory_safe = False
                receipt["invalidation_error_type"] = type(cleanup_error).__name__
            raise
        finally:
            if (
                not receipt["completed"]
                or not receipt["memory_acknowledged"]
                or receipt["redis_acknowledged"] is False
            ):
                self._unconfirmed_mutations += 1
            self._last_mutation = receipt

    async def connect(self) -> bool:
        """Initialize fallback/Redis discovery within this cache's lifetime."""
        if self._closed:
            raise ResourceOwnershipError("Closed hybrid caches cannot reconnect")
        return await self._run(self._connect(), False)

    async def _connect(self) -> bool:
        """Connect to Redis."""
        self._use_redis = bool(self.config.redis_url)
        if self._use_redis and not await self._redis.health_check():
            logger.warning("Redis unavailable, using memory cache only")
        return True

    async def get(self, key: str) -> Any | None:
        """Retain ownership through Redis lookup and memory publication."""
        return await self._entry(key, lambda: self._get(key), None)

    async def _get(self, key: str) -> Any | None:
        """Get from Redis first, then memory."""
        if self._use_redis and await self._redis.health_check():
            answered, value, deadline = await self._redis._get_with_expiry(key)
            if answered:
                if value is None:
                    await self._memory.delete(key)
                    return None
                if deadline is None:
                    await self._memory.delete(key)
                else:
                    await self._memory._set_until(key, copy.deepcopy(value), deadline)
                return value

        if not self._memory_safe:
            return None
        return copy.deepcopy(await self._memory.get(key))

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Retain ownership through both cache writes."""
        self._operations.check_loop()
        if self._closed:
            return False
        snapshot = copy.deepcopy(value)
        return await self._entry(key, lambda: self._set(key, snapshot, ttl), False)

    async def _set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set in both Redis and memory."""

        async def write(receipt: dict[str, Any]) -> bool:
            memory_ok = await self._memory.set(key, value, ttl)
            receipt["memory_acknowledged"] = memory_ok

            if self._use_redis and await self._redis.health_check():
                redis_ok = await self._redis.set(key, value, ttl)
                receipt["redis_acknowledged"] = redis_ok
                return redis_ok or memory_ok

            return memory_ok

        return await self._mutate("set", write, lambda: self._memory.delete(key))

    async def delete(self, key: str) -> bool:
        """Delete from the admitted cache layers."""
        return await self._entry(key, lambda: self._delete(key), False)

    async def _delete(self, key: str) -> bool:
        """Delete from both caches."""

        async def remove(receipt: dict[str, Any]) -> bool:
            memory_ok = await self._memory.delete(key)
            receipt["memory_acknowledged"] = True

            if self._use_redis and await self._redis.health_check():
                answered, redis_ok = await self._redis._delete_with_status(key)
                receipt["redis_acknowledged"] = answered
                return redis_ok or memory_ok

            return memory_ok

        return await self._mutate("delete", remove, lambda: self._memory.delete(key))

    async def delete_pattern(self, pattern: str) -> int:
        """Drain both pattern deletions before releasing this operation."""
        return await self._entry(None, lambda: self._delete_pattern(pattern), 0)

    async def _delete_pattern(self, pattern: str) -> int:
        """Delete matching keys from both caches."""

        async def remove(receipt: dict[str, Any]) -> int:
            memory_deleted = await self._memory.delete_pattern(pattern)
            receipt["memory_acknowledged"] = True

            redis_deleted = 0
            if self._use_redis and await self._redis.health_check():
                answered, redis_deleted = await self._redis._delete_pattern_with_status(
                    pattern
                )
                receipt["redis_acknowledged"] = answered

            return memory_deleted + redis_deleted

        return await self._mutate(
            "delete_pattern", remove, lambda: self._memory.delete_pattern(pattern)
        )

    async def exists(self, key: str) -> bool:
        """Check existence within the cache's admitted lifetime."""
        return await self._entry(key, lambda: self._exists(key), False)

    async def _exists(self, key: str) -> bool:
        """Check if key exists in either cache."""
        if self._use_redis and await self._redis.health_check():
            answered, exists = await self._redis._exists_with_status(key)
            if answered:
                if not exists:
                    await self._memory.delete(key)
                return exists

        return self._memory_safe and await self._memory.exists(key)

    async def clear(self) -> int:
        """Clear both layers without detaching an in-flight deletion."""
        return await self._entry(None, self._clear, 0)

    async def _clear(self) -> int:
        """Clear both caches."""

        async def remove(receipt: dict[str, Any]) -> int:
            memory_count = await self._memory.clear()
            self._memory_safe = True
            receipt["memory_acknowledged"] = True

            if self._use_redis and await self._redis.health_check():
                answered, redis_count = await self._redis._clear_with_status()
                receipt["redis_acknowledged"] = answered
                return redis_count + memory_count

            return memory_count

        return await self._mutate("clear", remove, self._memory.clear)

    async def health_check(self) -> bool:
        """Observe availability only while this cache accepts operations."""
        return await self._run(self._health_check(), False)

    async def _health_check(self) -> bool:
        """Check health of both caches."""
        memory_health = await self._memory.health_check()
        redis_health = False

        if self._use_redis:
            redis_health = await self._redis.health_check()

        # Healthy if at least one cache is working
        return (self._memory_safe and memory_health) or redis_health

    def get_stats(self) -> CacheStats:
        """Get combined statistics."""
        memory_stats = self._memory.get_stats()
        redis_stats = self._redis.get_stats()

        return CacheStats(
            hits=memory_stats.hits + redis_stats.hits,
            misses=memory_stats.misses + redis_stats.misses,
            sets=memory_stats.sets + redis_stats.sets,
            deletes=memory_stats.deletes + redis_stats.deletes,
            errors=memory_stats.errors + redis_stats.errors,
            bytes_read=memory_stats.bytes_read + redis_stats.bytes_read,
            bytes_written=memory_stats.bytes_written + redis_stats.bytes_written,
            avg_latency_ms=(memory_stats.avg_latency_ms + redis_stats.avg_latency_ms)
            / 2,
        )

    def get_status(self) -> dict[str, Any]:
        """Inspect hybrid ownership and the Redis generation without backend I/O."""
        return {
            "state": self._state,
            "active_operations": self._operations.active,
            "entry_ordering": self._entries.get_status(),
            "fallback_safe": self._memory_safe,
            "unconfirmed_mutations": self._unconfirmed_mutations,
            "last_mutation": copy.deepcopy(self._last_mutation),
            "redis": self._redis.get_status(),
            "cleanup_error_type": (
                type(self._cleanup_error).__name__ if self._cleanup_error else None
            ),
        }

    async def close(self) -> None:
        """Stop admission and drain complete hybrid operations before closing Redis."""
        self._operations.check_callback()
        if self._close_task is None:
            self._closed = True
            self._state = "closing"
            work = self._finish_close()
            try:
                self._close_task = asyncio.create_task(work)
            except BaseException as error:
                work.close()
                self._cleanup_error = error
                self._state = "failed"
                raise
        task = self._close_task

        async def join() -> None:
            await asyncio.shield(task)

        await drain_owned_operation(join())

    async def _finish_close(self) -> None:
        try:
            await self._operations.wait_idle()
            with self._operations.callback():
                await self._redis.close()
            self._state = "closed"
        except BaseException as error:
            self._cleanup_error = error
            self._state = "failed"
            raise


# ============================================================================
# Cache Factory
# ============================================================================


async def create_cache(config: CacheConfig | None = None) -> CacheInterface:
    """
    Factory function to create appropriate cache instance.

    Args:
        config: Cache configuration. If None, loads from environment.

    Returns:
        Configured cache instance.
    """
    config = config or CacheConfig.from_env()
    cache: RedisCache | HybridCache

    if config.backend == CacheBackend.REDIS:
        cache = RedisCache(config)
    elif config.backend == CacheBackend.HYBRID:
        cache = HybridCache(config)
    else:
        return MemoryCache(config)
    try:
        await cache.connect()
    except BaseException:
        # The factory owns the unpublished instance, including a client opened
        # before connect failed or was cancelled.
        try:
            await drain_owned_operation(cache.close())
        except Exception as cleanup_error:
            logger.warning(
                "Cache cleanup failed after initialization error (%s)",
                type(cleanup_error).__name__,
            )
        raise
    return cache


# ============================================================================
# Caching Decorators
# ============================================================================


def cached(
    ttl_seconds: int = 300,
    key_prefix: str = "",
    key_builder: Callable[..., str] | None = None,
    cache_none: bool = False,
):
    """
    Decorator to cache function results.

    Args:
        ttl_seconds: Time-to-live for cached results.
        key_prefix: Prefix for cache keys.
        key_builder: Custom function to build cache key from args.
        cache_none: Whether to cache None results.

    Usage:
        @cached(ttl_seconds=60, key_prefix="user")
        async def get_user(user_id: str) -> User:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get cache instance
            cache = get_cache()
            if cache is None:
                return await func(*args, **kwargs)

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key from qualified function name and arguments.
                key_parts = [key_prefix, f"{func.__module__}.{func.__qualname__}"]
                if args:
                    key_parts.append(hashlib.sha256(repr(args).encode()).hexdigest())
                if kwargs:
                    key_parts.append(
                        hashlib.sha256(
                            repr(sorted(kwargs.items())).encode()
                        ).hexdigest()
                    )
                cache_key = ":".join(filter(None, key_parts))

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Cache result
            if result is not None or cache_none:
                await cache.set(cache_key, result, ttl_seconds)

            return result

        return wrapper

    return decorator


def cache_invalidate(key_pattern: str):
    """
    Decorator to invalidate cache entries after function execution.

    Args:
        key_pattern: Pattern of keys to invalidate.

    Usage:
        @cache_invalidate("user:*")
        async def update_user(user_id: str, data: dict):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)

            cache = get_cache()
            if cache is not None:
                has_wildcards = any(ch in key_pattern for ch in ("*", "?", "["))
                if has_wildcards and hasattr(cache, "delete_pattern"):
                    await cache.delete_pattern(key_pattern)
                else:
                    await cache.delete(key_pattern)

            return result

        return wrapper

    return decorator


# ============================================================================
# Global Cache Instance
# ============================================================================

_cache_instance: CacheInterface | None = None


def _set_cache(cache: CacheInterface | None) -> None:
    global _cache_instance
    _cache_instance = cache


def _construct_cache(config: CacheConfig) -> CacheInterface:
    if config.backend == CacheBackend.REDIS:
        return RedisCache(config)
    if config.backend == CacheBackend.HYBRID:
        return HybridCache(config)
    return MemoryCache(config)


async def _prepare_cache(cache: CacheInterface) -> None:
    connect = getattr(cache, "connect", None)
    if connect is not None:
        await connect()


async def _dispose_cache(cache: CacheInterface) -> None:
    close = getattr(cache, "close", None)
    if close is not None:
        await close()


def _cache_reusable(cache: CacheInterface, config: CacheConfig) -> bool:
    return getattr(cache, "config", None) == config and not getattr(
        getattr(cache, "_redis", cache), "_closed", False
    )


def _new_registry() -> SharedResource[CacheConfig, CacheInterface]:
    return SharedResource(
        "cache",
        lambda: _cache_instance,
        _set_cache,
        _construct_cache,
        _prepare_cache,
        _dispose_cache,
        _cache_reusable,
    )


_cache_registry = _new_registry()


async def init_cache(config: CacheConfig | None = None) -> CacheInterface:
    """Reuse matching configuration; replace only an unleased global cache."""
    return await _cache_registry.initialize(config or CacheConfig.from_env())


async def acquire_cache(
    config: CacheConfig | None = None,
) -> ResourceLease[CacheInterface]:
    """Lease the shared cache until the caller releases it with ``aclose()``."""
    return await _cache_registry.acquire(config or CacheConfig.from_env())


def get_cache() -> CacheInterface | None:
    """Get the available global cache, rejecting unconfirmed cleanup."""
    return _cache_registry.get()


def get_cache_status() -> dict[str, Any]:
    """Inspect shared cache transitions and application ownership."""
    return _cache_registry.status()


async def close_cache() -> None:
    """Close once when no applications hold leases; preserve cleanup failures."""
    await _cache_registry.close()
