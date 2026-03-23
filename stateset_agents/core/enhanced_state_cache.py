"""Cache primitives and backends for enhanced state management."""

import asyncio
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from importlib.util import find_spec
from typing import Any

try:
    import aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

MONGODB_AVAILABLE = find_spec("pymongo") is not None

logger = logging.getLogger(__name__)

STATE_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    OSError,
    asyncio.TimeoutError,
    pickle.PickleError,
)


def _json_default_serializer(obj: Any) -> Any:
    """Best-effort JSON serializer for complex objects."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


class CacheStrategy(Enum):
    """Cache strategies."""

    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"


class ConsistencyLevel(Enum):
    """Consistency levels for distributed state."""

    EVENTUAL = "eventual"
    STRONG = "strong"
    CAUSAL = "causal"
    SESSION = "session"


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    LIFO = "lifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: float | None = None
    version: int = 1
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class StateSnapshot:
    """State snapshot for versioning."""

    snapshot_id: str
    timestamp: float
    state_data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_snapshot: str | None = None


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""


class InMemoryCacheBackend(CacheBackend):
    """High-performance in-memory cache backend."""

    def __init__(
        self,
        max_size: int = 10000,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.cache: dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()
        self.access_frequency = defaultdict(int)
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        async with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]
            if entry.is_expired():
                await self._remove_entry(key)
                return None

            entry.update_access()
            self._update_access_tracking(key)
            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        async with self.lock:
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl,
                size_bytes=self._calculate_size(value),
            )

            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict()

            self.cache[key] = entry
            self._update_access_tracking(key)
            return True

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self.lock:
            return await self._remove_entry(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self.lock:
            if key not in self.cache:
                return False

            entry = self.cache[key]
            if entry.is_expired():
                await self._remove_entry(key)
                return False

            return True

    async def clear(self) -> bool:
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            return True

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return len(str(value).encode("utf-8"))

    def _update_access_tracking(self, key: str):
        """Update access tracking for eviction policies."""
        if self.eviction_policy == EvictionPolicy.LRU:
            self.access_order[key] = time.time()
        elif self.eviction_policy == EvictionPolicy.LFU:
            self.access_frequency[key] += 1

    async def _evict(self):
        """Evict entries based on policy."""
        if not self.cache:
            return

        if self.eviction_policy == EvictionPolicy.LRU:
            oldest_key = min(
                self.access_order.keys(), key=lambda k: self.access_order[k]
            )
            await self._remove_entry(oldest_key)
        elif self.eviction_policy == EvictionPolicy.LFU:
            least_frequent_key = min(
                self.access_frequency.keys(), key=lambda k: self.access_frequency[k]
            )
            await self._remove_entry(least_frequent_key)
        elif self.eviction_policy == EvictionPolicy.FIFO:
            first_key = next(iter(self.cache))
            await self._remove_entry(first_key)
        elif self.eviction_policy == EvictionPolicy.TTL:
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self.cache.items()
                if entry.ttl and current_time > entry.created_at + entry.ttl
            ]

            if expired_keys:
                await self._remove_entry(expired_keys[0])
            else:
                oldest_key = min(
                    self.cache.keys(), key=lambda k: self.cache[k].last_accessed
                )
                await self._remove_entry(oldest_key)

    async def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
            self.access_order.pop(key, None)
            self.access_frequency.pop(key, None)
            return True
        return False


class MultiCacheBackend(CacheBackend):
    """Multi-level cache backend with local + remote backends."""

    def __init__(self, primary: CacheBackend, secondary: CacheBackend):
        self.primary = primary
        self.secondary = secondary

    async def get(self, key: str) -> Any | None:
        value = await self.primary.get(key)
        if value is not None:
            return value

        value = await self.secondary.get(key)
        if value is not None:
            await self.primary.set(key, value)
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        primary_ok = await self.primary.set(key, value, ttl)
        secondary_ok = await self.secondary.set(key, value, ttl)
        return primary_ok or secondary_ok

    async def delete(self, key: str) -> bool:
        primary_ok = await self.primary.delete(key)
        secondary_ok = await self.secondary.delete(key)
        return primary_ok or secondary_ok

    async def exists(self, key: str) -> bool:
        if await self.primary.exists(key):
            return True
        return await self.secondary.exists(key)

    async def clear(self) -> bool:
        primary_ok = await self.primary.clear()
        secondary_ok = await self.secondary.clear()
        return primary_ok or secondary_ok


class RedisCacheBackend(CacheBackend):
    """Redis-based distributed cache backend."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "grpo:",
        serializer: str = "json",
        allow_pickle: bool = False,
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.serializer = serializer.lower()
        self.allow_pickle = allow_pickle
        self.client = None
        self._connection_pool = None

        if self.serializer not in ("json", "pickle"):
            logger.warning(
                "Unknown serializer '%s', defaulting to json", self.serializer
            )
            self.serializer = "json"

        if self.serializer == "pickle" and not self.allow_pickle:
            logger.warning(
                "Pickle serializer disabled by default for security; falling back to json"
            )
            self.serializer = "json"

    async def _ensure_connected(self):
        """Ensure Redis connection."""
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is not available")

        if self.client is None:
            self.client = aioredis.from_url(self.redis_url)

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Any | None:
        """Get value from Redis."""
        await self._ensure_connected()

        try:
            data = await self.client.get(self._make_key(key))
            if data is None:
                return None

            return self._deserialize(data)
        except STATE_EXCEPTIONS as e:
            logger.error("Redis get error: %s", e)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in Redis."""
        await self._ensure_connected()

        try:
            data = self._serialize(value)

            if ttl:
                await self.client.setex(self._make_key(key), ttl, data)
            else:
                await self.client.set(self._make_key(key), data)

            return True
        except STATE_EXCEPTIONS as e:
            logger.error("Redis set error: %s", e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        await self._ensure_connected()

        try:
            result = await self.client.delete(self._make_key(key))
            return result > 0
        except STATE_EXCEPTIONS as e:
            logger.error("Redis delete error: %s", e)
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        await self._ensure_connected()

        try:
            result = await self.client.exists(self._make_key(key))
            return result > 0
        except STATE_EXCEPTIONS as e:
            logger.error("Redis exists error: %s", e)
            return False

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        if self.serializer == "pickle":
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return json.dumps(value, default=_json_default_serializer).encode("utf-8")

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from Redis storage."""
        if self.serializer == "pickle":
            return pickle.loads(data)
        return json.loads(data.decode("utf-8"))

    async def clear(self) -> bool:
        """Clear all cache entries with prefix."""
        await self._ensure_connected()

        try:
            keys = await self.client.keys(f"{self.key_prefix}*")
            if keys:
                await self.client.delete(*keys)
            return True
        except STATE_EXCEPTIONS as e:
            logger.error("Redis clear error: %s", e)
            return False


__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CacheStrategy",
    "ConsistencyLevel",
    "EvictionPolicy",
    "InMemoryCacheBackend",
    "MONGODB_AVAILABLE",
    "MultiCacheBackend",
    "REDIS_AVAILABLE",
    "RedisCacheBackend",
    "STATE_EXCEPTIONS",
    "StateSnapshot",
    "_json_default_serializer",
]
