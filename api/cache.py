"""
API Caching Module

Simple in-memory caching for frequently accessed endpoints.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar
from functools import wraps

from .constants import (
    HEALTH_CACHE_TTL_SECONDS,
    METRICS_CACHE_TTL_SECONDS,
)


T = TypeVar("T")


@dataclass
class CacheEntry:
    """A single cache entry with TTL."""
    value: Any
    expires_at: float

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.monotonic() > self.expires_at


class SimpleCache:
    """
    Simple in-memory cache with TTL support.

    Thread-safe for single-writer, multiple-reader scenarios.
    """

    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Returns None if key doesn't exist or has expired.
        """
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            # Clean up expired entry
            self._cache.pop(key, None)
            self._misses += 1
            return None

        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Set a value in cache with TTL."""
        self._cache[key] = CacheEntry(
            value=value,
            expires_at=time.monotonic() + ttl_seconds,
        )

    def delete(self, key: str) -> bool:
        """Delete a key from cache. Returns True if key existed."""
        return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns the number of entries removed.
        """
        now = time.monotonic()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at < now
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }


# Global cache instance
_cache = SimpleCache()


def get_cache() -> SimpleCache:
    """Get the global cache instance."""
    return _cache


def cached(ttl_seconds: float, key_prefix: str = ""):
    """
    Decorator to cache function results.

    Args:
        ttl_seconds: Time-to-live for cached results.
        key_prefix: Prefix for cache keys.

    Usage:
        @cached(ttl_seconds=30, key_prefix="metrics")
        async def get_metrics():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}"
            if args:
                cache_key += f":{hash(args)}"
            if kwargs:
                cache_key += f":{hash(frozenset(kwargs.items()))}"

            # Try to get from cache
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            _cache.set(cache_key, result, ttl_seconds)

            return result

        return wrapper
    return decorator


def cached_sync(ttl_seconds: float, key_prefix: str = ""):
    """
    Decorator to cache synchronous function results.

    Args:
        ttl_seconds: Time-to-live for cached results.
        key_prefix: Prefix for cache keys.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = f"{key_prefix}:{func.__name__}"
            if args:
                cache_key += f":{hash(args)}"
            if kwargs:
                cache_key += f":{hash(frozenset(kwargs.items()))}"

            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl_seconds)

            return result

        return wrapper
    return decorator


# Pre-configured decorators for common use cases
cache_health = cached(ttl_seconds=HEALTH_CACHE_TTL_SECONDS, key_prefix="health")
cache_metrics = cached(ttl_seconds=METRICS_CACHE_TTL_SECONDS, key_prefix="metrics")
