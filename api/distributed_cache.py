"""
Distributed Caching Module

Production-ready caching with Redis support, local fallback, and comprehensive
cache management features for high-availability deployments.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    redis_url: Optional[str] = None
    redis_password: Optional[str] = None
    redis_db: int = 0
    default_ttl_seconds: int = 300
    max_memory_items: int = 10000
    key_prefix: str = "stateset:"
    serializer: str = "json"  # json or pickle
    compression_enabled: bool = False
    compression_threshold: int = 1024  # bytes
    health_check_interval: int = 30
    retry_attempts: int = 3
    retry_delay: float = 0.1

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create config from environment variables."""
        import os

        backend_str = os.getenv("CACHE_BACKEND", "memory").lower()
        backend = CacheBackend(backend_str) if backend_str in CacheBackend.__members__.values() else CacheBackend.MEMORY

        return cls(
            backend=backend,
            redis_url=os.getenv("REDIS_URL"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            default_ttl_seconds=int(os.getenv("CACHE_DEFAULT_TTL", "300")),
            max_memory_items=int(os.getenv("CACHE_MAX_MEMORY_ITEMS", "10000")),
            key_prefix=os.getenv("CACHE_KEY_PREFIX", "stateset:"),
            compression_enabled=os.getenv("CACHE_COMPRESSION", "false").lower() == "true",
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
    _latencies: List[float] = field(default_factory=list)

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

    def to_dict(self) -> Dict[str, Any]:
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
    async def get(self, key: str) -> Optional[T]:
        """Get a value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
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
        self._cache: Dict[str, MemoryCacheEntry] = {}
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        start = time.monotonic()
        prefixed_key = f"{self.config.key_prefix}{key}"

        async with self._lock:
            entry = self._cache.get(prefixed_key)

            if entry is None:
                self._stats.misses += 1
                return None

            if time.time() > entry.expires_at:
                del self._cache[prefixed_key]
                self._stats.misses += 1
                return None

            entry.access_count += 1
            self._stats.hits += 1
            self._stats.bytes_read += entry.size_bytes
            self._stats.record_latency((time.monotonic() - start) * 1000)

            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        start = time.monotonic()
        prefixed_key = f"{self.config.key_prefix}{key}"
        ttl = ttl or self.config.default_ttl_seconds

        try:
            # Estimate size
            size_bytes = len(json.dumps(value)) if isinstance(value, (dict, list)) else 100

            async with self._lock:
                # Evict if at capacity
                if len(self._cache) >= self.config.max_memory_items:
                    await self._evict_lru()

                self._cache[prefixed_key] = MemoryCacheEntry(
                    value=value,
                    expires_at=time.time() + ttl,
                    size_bytes=size_bytes,
                )

                self._stats.sets += 1
                self._stats.bytes_written += size_bytes
                self._stats.record_latency((time.monotonic() - start) * 1000)

                return True
        except Exception as e:
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

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        prefixed_key = f"{self.config.key_prefix}{key}"

        async with self._lock:
            entry = self._cache.get(prefixed_key)
            if entry is None:
                return False
            if time.time() > entry.expires_at:
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
            key=lambda k: (self._cache[k].access_count, self._cache[k].created_at)
        )

        # Remove 10% of entries
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            del self._cache[key]

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        now = time.time()
        expired = []

        async with self._lock:
            for key, entry in self._cache.items():
                if now > entry.expires_at:
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
        self._client = None
        self._stats = CacheStats()
        self._connected = False
        self._last_health_check = 0.0
        self._health_status = False

    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis

            self._client = redis.from_url(
                self.config.redis_url or "redis://localhost:6379",
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=False,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
            )

            # Test connection
            await self._client.ping()
            self._connected = True
            self._health_status = True
            logger.info("Connected to Redis successfully")
            return True

        except ImportError:
            logger.error("redis package not installed. Install with: pip install redis")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis."""
        if not self._connected:
            return None

        start = time.monotonic()
        prefixed_key = f"{self.config.key_prefix}{key}"

        try:
            data = await self._client.get(prefixed_key)

            if data is None:
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            self._stats.bytes_read += len(data)

            # Decompress if needed
            value = self._deserialize(data)
            self._stats.record_latency((time.monotonic() - start) * 1000)

            return value

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats.errors += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis."""
        if not self._connected:
            return False

        start = time.monotonic()
        prefixed_key = f"{self.config.key_prefix}{key}"
        ttl = ttl or self.config.default_ttl_seconds

        try:
            data = self._serialize(value)

            await self._client.setex(prefixed_key, ttl, data)

            self._stats.sets += 1
            self._stats.bytes_written += len(data)
            self._stats.record_latency((time.monotonic() - start) * 1000)

            return True

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self._stats.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if not self._connected:
            return False

        prefixed_key = f"{self.config.key_prefix}{key}"

        try:
            result = await self._client.delete(prefixed_key)
            self._stats.deletes += 1
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self._stats.errors += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        if not self._connected:
            return False

        prefixed_key = f"{self.config.key_prefix}{key}"

        try:
            return await self._client.exists(prefixed_key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    async def clear(self) -> int:
        """Clear all keys with our prefix."""
        if not self._connected:
            return 0

        try:
            pattern = f"{self.config.key_prefix}*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await self._client.delete(*keys)
                if cursor == 0:
                    break

            return deleted
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0

    async def health_check(self) -> bool:
        """Check Redis health."""
        now = time.time()

        # Use cached result if recent
        if now - self._last_health_check < self.config.health_check_interval:
            return self._health_status

        self._last_health_check = now

        if not self._connected:
            # Try to reconnect
            self._health_status = await self.connect()
            return self._health_status

        try:
            await self._client.ping()
            self._health_status = True
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            self._health_status = False
            self._connected = False
            return False

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        data = json.dumps(value).encode("utf-8")

        # Compress if large
        if self.config.compression_enabled and len(data) > self.config.compression_threshold:
            import zlib
            data = b"ZLIB:" + zlib.compress(data)

        return data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if data.startswith(b"ZLIB:"):
            import zlib
            data = zlib.decompress(data[5:])

        return json.loads(data.decode("utf-8"))

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._connected = False


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

    async def connect(self) -> bool:
        """Connect to Redis."""
        self._use_redis = await self._redis.connect()
        if not self._use_redis:
            logger.warning("Redis unavailable, using memory cache only")
        return True

    async def get(self, key: str) -> Optional[Any]:
        """Get from Redis first, then memory."""
        if self._use_redis and await self._redis.health_check():
            value = await self._redis.get(key)
            if value is not None:
                # Populate memory cache
                await self._memory.set(key, value)
                return value

        return await self._memory.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in both Redis and memory."""
        memory_ok = await self._memory.set(key, value, ttl)

        if self._use_redis and await self._redis.health_check():
            redis_ok = await self._redis.set(key, value, ttl)
            return redis_ok or memory_ok

        return memory_ok

    async def delete(self, key: str) -> bool:
        """Delete from both caches."""
        memory_ok = await self._memory.delete(key)

        if self._use_redis and await self._redis.health_check():
            redis_ok = await self._redis.delete(key)
            return redis_ok or memory_ok

        return memory_ok

    async def exists(self, key: str) -> bool:
        """Check if key exists in either cache."""
        if self._use_redis and await self._redis.health_check():
            if await self._redis.exists(key):
                return True

        return await self._memory.exists(key)

    async def clear(self) -> int:
        """Clear both caches."""
        memory_count = await self._memory.clear()

        if self._use_redis and await self._redis.health_check():
            redis_count = await self._redis.clear()
            return redis_count + memory_count

        return memory_count

    async def health_check(self) -> bool:
        """Check health of both caches."""
        memory_health = await self._memory.health_check()
        redis_health = False

        if self._use_redis:
            redis_health = await self._redis.health_check()

        # Healthy if at least one cache is working
        return memory_health or redis_health

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
            avg_latency_ms=(memory_stats.avg_latency_ms + redis_stats.avg_latency_ms) / 2,
        )


# ============================================================================
# Cache Factory
# ============================================================================

async def create_cache(config: Optional[CacheConfig] = None) -> CacheInterface:
    """
    Factory function to create appropriate cache instance.

    Args:
        config: Cache configuration. If None, loads from environment.

    Returns:
        Configured cache instance.
    """
    config = config or CacheConfig.from_env()

    if config.backend == CacheBackend.REDIS:
        cache = RedisCache(config)
        await cache.connect()
        return cache

    elif config.backend == CacheBackend.HYBRID:
        cache = HybridCache(config)
        await cache.connect()
        return cache

    else:
        return MemoryCache(config)


# ============================================================================
# Caching Decorators
# ============================================================================

def cached(
    ttl_seconds: int = 300,
    key_prefix: str = "",
    key_builder: Optional[Callable[..., str]] = None,
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
                # Default key from function name and args
                key_parts = [key_prefix, func.__name__]
                if args:
                    key_parts.append(hashlib.md5(str(args).encode()).hexdigest()[:8])
                if kwargs:
                    key_parts.append(hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8])
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
                await cache.delete(key_pattern)

            return result

        return wrapper
    return decorator


# ============================================================================
# Global Cache Instance
# ============================================================================

_cache_instance: Optional[CacheInterface] = None


async def init_cache(config: Optional[CacheConfig] = None) -> CacheInterface:
    """Initialize the global cache instance."""
    global _cache_instance
    _cache_instance = await create_cache(config)
    return _cache_instance


def get_cache() -> Optional[CacheInterface]:
    """Get the global cache instance."""
    return _cache_instance


async def close_cache() -> None:
    """Close the global cache instance."""
    global _cache_instance
    if _cache_instance is not None:
        if hasattr(_cache_instance, "close"):
            await _cache_instance.close()
        _cache_instance = None
