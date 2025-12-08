"""
Distributed Cache Tests

Comprehensive tests for distributed caching including:
- Memory cache implementation
- Cache configuration
- Cache statistics
- Caching decorators
- Cache factory
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.distributed_cache import (
    CacheBackend,
    CacheConfig,
    CacheStats,
    MemoryCache,
    MemoryCacheEntry,
    HybridCache,
    create_cache,
    init_cache,
    get_cache,
    close_cache,
    cached,
    cache_invalidate,
)


# ============================================================================
# Cache Configuration Tests
# ============================================================================

class TestCacheConfig:
    """Tests for cache configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.backend == CacheBackend.MEMORY
        assert config.default_ttl_seconds == 300
        assert config.max_memory_items == 10000
        assert config.key_prefix == "stateset:"

    def test_config_from_env(self):
        """Test loading config from environment."""
        os.environ["CACHE_BACKEND"] = "memory"
        os.environ["CACHE_DEFAULT_TTL"] = "600"
        os.environ["CACHE_KEY_PREFIX"] = "test:"
        os.environ["CACHE_MAX_MEMORY_ITEMS"] = "5000"

        config = CacheConfig.from_env()

        assert config.backend == CacheBackend.MEMORY
        assert config.default_ttl_seconds == 600
        assert config.key_prefix == "test:"
        assert config.max_memory_items == 5000

        # Cleanup
        del os.environ["CACHE_BACKEND"]
        del os.environ["CACHE_DEFAULT_TTL"]
        del os.environ["CACHE_KEY_PREFIX"]
        del os.environ["CACHE_MAX_MEMORY_ITEMS"]

    def test_config_with_redis_url(self):
        """Test config with Redis URL."""
        config = CacheConfig(
            backend=CacheBackend.REDIS,
            redis_url="redis://localhost:6379",
            redis_password="secret",
            redis_db=1,
        )

        assert config.backend == CacheBackend.REDIS
        assert config.redis_url == "redis://localhost:6379"
        assert config.redis_password == "secret"
        assert config.redis_db == 1


# ============================================================================
# Cache Statistics Tests
# ============================================================================

class TestCacheStats:
    """Tests for cache statistics."""

    def test_initial_stats(self):
        """Test initial statistics values."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.deletes == 0
        assert stats.errors == 0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)

        assert stats.hit_rate == 0.8

    def test_hit_rate_zero_requests(self):
        """Test hit rate with no requests."""
        stats = CacheStats()

        assert stats.hit_rate == 0.0

    def test_latency_recording(self):
        """Test latency recording."""
        stats = CacheStats()

        stats.record_latency(10.0)
        stats.record_latency(20.0)
        stats.record_latency(30.0)

        assert stats.avg_latency_ms == 20.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = CacheStats(hits=10, misses=5, sets=15)
        data = stats.to_dict()

        assert data["hits"] == 10
        assert data["misses"] == 5
        assert data["sets"] == 15
        assert "hit_rate" in data


# ============================================================================
# Memory Cache Tests
# ============================================================================

class TestMemoryCache:
    """Tests for memory cache implementation."""

    @pytest.fixture
    def cache(self):
        """Create a fresh memory cache for each test."""
        config = CacheConfig(max_memory_items=100, default_ttl_seconds=300)
        return MemoryCache(config)

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        await cache.set("key1", {"value": "test"})

        result = await cache.get("key1")
        assert result == {"value": "test"}

    @pytest.mark.asyncio
    async def test_get_missing_key(self, cache):
        """Test getting non-existent key."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache):
        """Test setting with custom TTL."""
        await cache.set("expiring", "value", ttl=1)

        # Should exist immediately
        result = await cache.get("expiring")
        assert result == "value"

        # Should expire after TTL
        await asyncio.sleep(1.1)
        result = await cache.get("expiring")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test deleting a key."""
        await cache.set("to_delete", "value")

        result = await cache.delete("to_delete")
        assert result is True

        result = await cache.get("to_delete")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, cache):
        """Test deleting non-existent key."""
        result = await cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, cache):
        """Test checking if key exists."""
        await cache.set("exists", "value")

        assert await cache.exists("exists") is True
        assert await cache.exists("not_exists") is False

    @pytest.mark.asyncio
    async def test_exists_expired(self, cache):
        """Test exists returns False for expired keys."""
        await cache.set("expiring", "value", ttl=0.1)

        await asyncio.sleep(0.15)
        assert await cache.exists("expiring") is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing all keys."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        cleared = await cache.clear()
        assert cleared == 2

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_health_check(self, cache):
        """Test health check."""
        result = await cache.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_stats_tracking(self, cache):
        """Test that statistics are tracked."""
        await cache.set("key", "value")  # 1 set
        await cache.get("key")  # 1 hit
        await cache.get("missing")  # 1 miss
        await cache.delete("key")  # 1 delete

        stats = cache.get_stats()
        assert stats.sets == 1
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.deletes == 1

    @pytest.mark.asyncio
    async def test_key_prefix(self, cache):
        """Test that key prefix is applied."""
        await cache.set("key", "value")

        # Check internal storage uses prefixed key
        prefixed_key = f"{cache.config.key_prefix}key"
        assert prefixed_key in cache._cache

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when max capacity reached."""
        config = CacheConfig(max_memory_items=5)
        cache = MemoryCache(config)

        # Fill cache
        for i in range(5):
            await cache.set(f"key{i}", f"value{i}")

        # Access some keys to update their access count
        await cache.get("key0")
        await cache.get("key1")

        # Add one more to trigger eviction
        await cache.set("new_key", "new_value")

        # Some keys should have been evicted
        total = 0
        for i in range(5):
            if await cache.exists(f"key{i}"):
                total += 1
        assert total < 5  # At least one evicted

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache):
        """Test cleaning up expired entries."""
        await cache.set("expired1", "value1", ttl=0.1)
        await cache.set("expired2", "value2", ttl=0.1)
        await cache.set("valid", "value3", ttl=300)

        await asyncio.sleep(0.15)
        removed = await cache.cleanup_expired()

        assert removed == 2
        assert await cache.exists("valid") is True

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache):
        """Test concurrent access to cache."""
        async def write_task(key: str):
            for i in range(10):
                await cache.set(f"{key}_{i}", f"value_{i}")

        async def read_task(key: str):
            for i in range(10):
                await cache.get(f"{key}_{i}")

        # Run multiple concurrent tasks
        tasks = [
            write_task("a"),
            write_task("b"),
            read_task("a"),
            read_task("b"),
        ]
        await asyncio.gather(*tasks)

        # Should not raise any errors
        stats = cache.get_stats()
        assert stats.sets >= 20
        assert stats.hits + stats.misses >= 20


# ============================================================================
# Hybrid Cache Tests
# ============================================================================

class TestHybridCache:
    """Tests for hybrid cache (memory + Redis fallback)."""

    @pytest.fixture
    def hybrid_cache(self):
        """Create hybrid cache without Redis connection."""
        config = CacheConfig(backend=CacheBackend.HYBRID)
        return HybridCache(config)

    @pytest.mark.asyncio
    async def test_falls_back_to_memory(self, hybrid_cache):
        """Test fallback to memory when Redis unavailable."""
        await hybrid_cache.connect()

        # Should work with memory fallback
        await hybrid_cache.set("key", "value")
        result = await hybrid_cache.get("key")

        assert result == "value"

    @pytest.mark.asyncio
    async def test_health_check_memory_only(self, hybrid_cache):
        """Test health check with memory only."""
        await hybrid_cache.connect()

        result = await hybrid_cache.health_check()
        assert result is True  # Memory is healthy

    @pytest.mark.asyncio
    async def test_combined_stats(self, hybrid_cache):
        """Test combined statistics from both caches."""
        await hybrid_cache.connect()

        await hybrid_cache.set("key", "value")
        await hybrid_cache.get("key")

        stats = hybrid_cache.get_stats()
        assert stats.sets >= 1
        assert stats.hits >= 1


# ============================================================================
# Cache Factory Tests
# ============================================================================

class TestCacheFactory:
    """Tests for cache factory function."""

    @pytest.mark.asyncio
    async def test_create_memory_cache(self):
        """Test creating memory cache."""
        config = CacheConfig(backend=CacheBackend.MEMORY)
        cache = await create_cache(config)

        assert isinstance(cache, MemoryCache)

    @pytest.mark.asyncio
    async def test_create_hybrid_cache(self):
        """Test creating hybrid cache."""
        config = CacheConfig(backend=CacheBackend.HYBRID)
        cache = await create_cache(config)

        assert isinstance(cache, HybridCache)

    @pytest.mark.asyncio
    async def test_create_default_cache(self):
        """Test creating cache with default config."""
        cache = await create_cache()
        assert cache is not None


# ============================================================================
# Global Cache Instance Tests
# ============================================================================

class TestGlobalCache:
    """Tests for global cache instance."""

    @pytest.mark.asyncio
    async def test_init_and_get_cache(self):
        """Test initializing and getting global cache."""
        config = CacheConfig(backend=CacheBackend.MEMORY)
        cache = await init_cache(config)

        assert cache is not None
        assert get_cache() is cache

        await close_cache()
        assert get_cache() is None

    @pytest.mark.asyncio
    async def test_close_without_init(self):
        """Test closing cache that was never initialized."""
        # Should not raise
        await close_cache()


# ============================================================================
# Caching Decorator Tests
# ============================================================================

class TestCachingDecorators:
    """Tests for caching decorators."""

    @pytest.mark.asyncio
    async def test_cached_decorator(self):
        """Test @cached decorator."""
        # Initialize global cache
        await init_cache(CacheConfig(backend=CacheBackend.MEMORY))

        call_count = 0

        @cached(ttl_seconds=60, key_prefix="test")
        async def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should execute function
        result1 = await expensive_operation(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = await expensive_operation(5)
        assert result2 == 10
        assert call_count == 1  # Should NOT increment

        # Different argument should execute function
        result3 = await expensive_operation(10)
        assert result3 == 20
        assert call_count == 2

        await close_cache()

    @pytest.mark.asyncio
    async def test_cached_decorator_no_cache(self):
        """Test @cached decorator when no cache initialized."""
        # Ensure no cache
        await close_cache()

        call_count = 0

        @cached(ttl_seconds=60)
        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        # Should work without cache (just calls function)
        result = await operation()
        assert result == "result"
        assert call_count == 1

        # Should call function again (no caching)
        result = await operation()
        assert result == "result"
        assert call_count == 2


# ============================================================================
# Cache Entry Tests
# ============================================================================

class TestMemoryCacheEntry:
    """Tests for cache entry dataclass."""

    def test_entry_creation(self):
        """Test creating cache entry."""
        entry = MemoryCacheEntry(
            value={"test": "data"},
            expires_at=time.time() + 300,
        )

        assert entry.value == {"test": "data"}
        assert entry.access_count == 0
        assert entry.created_at > 0

    def test_entry_defaults(self):
        """Test entry default values."""
        entry = MemoryCacheEntry(
            value="test",
            expires_at=time.time() + 300,
        )

        assert entry.access_count == 0
        assert entry.size_bytes == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
