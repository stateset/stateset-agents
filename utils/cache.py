"""
Cache service for the GRPO Agent Framework
"""

import json
import logging
import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheService:
    """
    Simple in-memory cache service with TTL support
    """

    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry)
        self.default_ttl = default_ttl
        self.lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        with self.lock:
            expiry = time.time() + (ttl if ttl is not None else self.default_ttl)
            self.cache[key] = (value, expiry)

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        with self.lock:
            if key in self.cache:
                _, expiry = self.cache[key]
                if time.time() < expiry:
                    return True
                else:
                    del self.cache[key]
            return False

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self.cache.items() if current_time >= expiry
            ]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)

    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            self.cleanup_expired()  # Clean up before returning size
            return len(self.cache)

    def keys(self) -> list:
        """Get all non-expired keys"""
        with self.lock:
            self.cleanup_expired()
            return list(self.cache.keys())


# Global cache instance
_default_cache = None


def get_cache_service() -> CacheService:
    """Get the default global cache service"""
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheService()
    return _default_cache
