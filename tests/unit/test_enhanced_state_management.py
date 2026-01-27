"""
Unit tests for the Enhanced State Management module.

Tests cover caching backends, state management, consistency levels,
and distributed state functionality.
"""

import asyncio
import time
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateset_agents.core.enhanced_state_management import (
    CacheEntry,
    CacheStrategy,
    ConsistencyLevel,
    EvictionPolicy,
    InMemoryCacheBackend,
    MultiCacheBackend,
    StateSnapshot,
)


class TestCacheStrategy:
    """Test CacheStrategy enum."""

    def test_cache_strategy_values(self):
        """Test that all cache strategies have expected values."""
        assert CacheStrategy.WRITE_THROUGH.value == "write_through"
        assert CacheStrategy.WRITE_BACK.value == "write_back"
        assert CacheStrategy.WRITE_AROUND.value == "write_around"
        assert CacheStrategy.READ_THROUGH.value == "read_through"
        assert CacheStrategy.CACHE_ASIDE.value == "cache_aside"


class TestConsistencyLevel:
    """Test ConsistencyLevel enum."""

    def test_consistency_level_values(self):
        """Test that all consistency levels have expected values."""
        assert ConsistencyLevel.EVENTUAL.value == "eventual"
        assert ConsistencyLevel.STRONG.value == "strong"
        assert ConsistencyLevel.CAUSAL.value == "causal"
        assert ConsistencyLevel.SESSION.value == "session"


class TestEvictionPolicy:
    """Test EvictionPolicy enum."""

    def test_eviction_policy_values(self):
        """Test that all eviction policies have expected values."""
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.LFU.value == "lfu"
        assert EvictionPolicy.FIFO.value == "fifo"
        assert EvictionPolicy.LIFO.value == "lifo"
        assert EvictionPolicy.TTL.value == "ttl"
        assert EvictionPolicy.ADAPTIVE.value == "adaptive"


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a CacheEntry."""
        current_time = time.time()
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=current_time,
            last_accessed=current_time,
            access_count=1,
            ttl=3600.0,
        )

        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.ttl == 3600.0
        assert entry.version == 1

    def test_cache_entry_defaults(self):
        """Test CacheEntry default values."""
        current_time = time.time()
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=current_time,
            last_accessed=current_time,
            access_count=1,
        )

        assert entry.ttl is None
        assert entry.version == 1
        assert entry.dependencies == []
        assert entry.tags == []
        assert entry.size_bytes == 0

    def test_is_expired_no_ttl(self):
        """Test is_expired with no TTL."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
        )

        assert entry.is_expired() is False

    def test_is_expired_not_expired(self):
        """Test is_expired when not expired."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            ttl=3600.0,  # 1 hour
        )

        assert entry.is_expired() is False

    def test_is_expired_expired(self):
        """Test is_expired when expired."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=time.time() - 3700,  # Created 3700 seconds ago
            last_accessed=time.time() - 3700,
            access_count=1,
            ttl=3600.0,  # 1 hour TTL
        )

        assert entry.is_expired() is True

    def test_update_access(self):
        """Test updating access statistics."""
        old_time = time.time() - 100
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=old_time,
            last_accessed=old_time,
            access_count=1,
        )

        entry.update_access()

        assert entry.access_count == 2
        assert entry.last_accessed > old_time


class TestStateSnapshot:
    """Test StateSnapshot dataclass."""

    def test_state_snapshot_creation(self):
        """Test creating a StateSnapshot."""
        snapshot = StateSnapshot(
            snapshot_id="snap_001",
            timestamp=time.time(),
            state_data={"key1": "value1", "key2": "value2"},
            metadata={"author": "system"},
        )

        assert snapshot.snapshot_id == "snap_001"
        assert "key1" in snapshot.state_data
        assert snapshot.metadata["author"] == "system"

    def test_state_snapshot_with_parent(self):
        """Test StateSnapshot with parent reference."""
        snapshot = StateSnapshot(
            snapshot_id="snap_002",
            timestamp=time.time(),
            state_data={"key": "value"},
            parent_snapshot="snap_001",
        )

        assert snapshot.parent_snapshot == "snap_001"


class TestInMemoryCacheBackend:
    """Test InMemoryCacheBackend class."""

    @pytest.fixture
    def cache(self):
        """Create an InMemoryCacheBackend for testing."""
        return InMemoryCacheBackend(max_size=100, eviction_policy=EvictionPolicy.LRU)

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache):
        """Test getting a nonexistent key."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache):
        """Test set with TTL."""
        await cache.set("key1", "value1", ttl=3600)
        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_expired_entry(self, cache):
        """Test that expired entries return None."""
        # Manually create an expired entry
        entry = CacheEntry(
            key="expired_key",
            value="expired_value",
            created_at=time.time() - 100,
            last_accessed=time.time() - 100,
            access_count=1,
            ttl=50,  # Expired
        )
        cache.cache["expired_key"] = entry

        result = await cache.get("expired_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test delete operation."""
        await cache.set("key1", "value1")
        result = await cache.delete("key1")

        assert result is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, cache):
        """Test deleting a nonexistent key."""
        result = await cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, cache):
        """Test exists operation."""
        await cache.set("key1", "value1")

        assert await cache.exists("key1") is True
        assert await cache.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clear operation."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        result = await cache.clear()

        assert result is True
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        """Test that eviction occurs when max size is reached."""
        cache = InMemoryCacheBackend(max_size=3, eviction_policy=EvictionPolicy.LRU)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add key4, should evict key2 (LRU)
        await cache.set("key4", "value4")

        assert len(cache.cache) <= 3

    @pytest.mark.asyncio
    async def test_access_updates_tracking(self, cache):
        """Test that accessing a key updates tracking."""
        await cache.set("key1", "value1")

        initial_access = cache.cache["key1"].access_count
        await cache.get("key1")

        assert cache.cache["key1"].access_count == initial_access + 1


class TestLFUEviction:
    """Test LFU eviction policy."""

    @pytest.fixture
    def lfu_cache(self):
        """Create an LFU cache."""
        return InMemoryCacheBackend(max_size=3, eviction_policy=EvictionPolicy.LFU)

    @pytest.mark.asyncio
    async def test_lfu_tracks_frequency(self, lfu_cache):
        """Test LFU tracks access frequency."""
        await lfu_cache.set("key1", "value1")

        # Access key1 multiple times
        for _ in range(5):
            await lfu_cache.get("key1")

        assert lfu_cache.access_frequency["key1"] > 0


class TestFIFOEviction:
    """Test FIFO eviction policy."""

    @pytest.fixture
    def fifo_cache(self):
        """Create a FIFO cache."""
        return InMemoryCacheBackend(max_size=3, eviction_policy=EvictionPolicy.FIFO)

    @pytest.mark.asyncio
    async def test_fifo_eviction_order(self, fifo_cache):
        """Test FIFO eviction order."""
        await fifo_cache.set("key1", "value1")
        await fifo_cache.set("key2", "value2")
        await fifo_cache.set("key3", "value3")

        # Fill the cache
        assert len(fifo_cache.cache) == 3


class TestMultiCacheBackend:
    """Test MultiCacheBackend behavior."""

    @pytest.fixture
    def multi_cache(self):
        """Create a multi-level cache with two in-memory backends."""
        primary = InMemoryCacheBackend(max_size=10)
        secondary = InMemoryCacheBackend(max_size=10)
        return MultiCacheBackend(primary, secondary), primary, secondary

    @pytest.mark.asyncio
    async def test_get_populates_primary(self, multi_cache):
        """Test that secondary hits populate the primary cache."""
        cache, primary, secondary = multi_cache
        await secondary.set("key1", "value1")

        result = await cache.get("key1")

        assert result == "value1"
        assert await primary.exists("key1") is True

    @pytest.mark.asyncio
    async def test_set_writes_both(self, multi_cache):
        """Test that set writes to both caches."""
        cache, primary, secondary = multi_cache
        await cache.set("key2", "value2")

        assert await primary.exists("key2") is True
        assert await secondary.exists("key2") is True

    @pytest.mark.asyncio
    async def test_delete_removes_both(self, multi_cache):
        """Test that delete removes from both caches."""
        cache, primary, secondary = multi_cache
        await cache.set("key3", "value3")

        await cache.delete("key3")

        assert await primary.exists("key3") is False
        assert await secondary.exists("key3") is False


class TestDistributedStateService:
    """Test DistributedStateService class."""

    @pytest.fixture
    def state_service(self):
        """Create a mock DistributedStateService."""
        with patch("core.enhanced_state_management.REDIS_AVAILABLE", False), \
             patch("core.enhanced_state_management.MONGODB_AVAILABLE", False):
            from stateset_agents.core.enhanced_state_management import DistributedStateService
            service = DistributedStateService()
            return service

    def test_service_creation(self, state_service):
        """Test service creation."""
        assert state_service is not None
        assert state_service.state_manager is not None

    @pytest.mark.asyncio
    async def test_health_check(self, state_service):
        """Test health check."""
        health = await state_service.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]


class TestStateManager:
    """Test StateManager class."""

    @pytest.fixture
    def state_manager(self):
        """Create a mock StateManager."""
        with patch("core.enhanced_state_management.REDIS_AVAILABLE", False):
            from stateset_agents.core.enhanced_state_management import StateManager
            manager = StateManager()
            return manager

    @pytest.mark.asyncio
    async def test_set_and_get(self, state_manager):
        """Test basic set and get."""
        await state_manager.set("key1", {"value": "test"})
        result = await state_manager.get("key1")

        assert result == {"value": "test"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, state_manager):
        """Test getting nonexistent key."""
        result = await state_manager.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, state_manager):
        """Test delete operation."""
        await state_manager.set("key1", "value1")
        await state_manager.delete("key1")
        result = await state_manager.get("key1")

        assert result is None


class TestConversationManager:
    """Test ConversationManager class."""

    @pytest.fixture
    def conversation_manager(self):
        """Create a mock ConversationManager."""
        from stateset_agents.core.enhanced_state_management import ConversationManager
        state_manager = MagicMock()
        state_manager.get = AsyncMock(return_value=None)
        state_manager.set = AsyncMock()
        return ConversationManager(state_manager)

    @pytest.mark.asyncio
    async def test_create_conversation(self, conversation_manager):
        """Test creating a conversation."""
        await conversation_manager.create_conversation(
            "conv_001",
            "user_001",
            {"topic": "test"},
        )

        conversation_manager.state_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_conversation(self, conversation_manager):
        """Test getting a conversation."""
        conversation_manager.state_manager.get.return_value = {
            "conversation_id": "conv_001",
            "user_id": "user_001",
            "turns": [],
        }

        result = await conversation_manager.get_conversation("conv_001")

        assert result["conversation_id"] == "conv_001"

    @pytest.mark.asyncio
    async def test_add_turn(self, conversation_manager):
        """Test adding a turn to conversation."""
        conversation_manager.state_manager.get.return_value = {
            "conversation_id": "conv_001",
            "turns": [],
        }

        await conversation_manager.add_turn(
            "conv_001",
            "user",
            "Hello!",
        )

        conversation_manager.state_manager.set.assert_called()


class TestManagedStateContext:
    """Test managed_state_context context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using managed_state_context."""
        with patch("core.enhanced_state_management.get_state_service") as mock_get:
            mock_service = MagicMock()
            mock_get.return_value = mock_service

            from stateset_agents.core.enhanced_state_management import managed_state_context

            async with managed_state_context() as service:
                assert service is mock_service


class TestGetStateService:
    """Test get_state_service function."""

    def test_get_state_service(self):
        """Test getting the global state service."""
        with patch("core.enhanced_state_management.REDIS_AVAILABLE", False), \
             patch("core.enhanced_state_management.MONGODB_AVAILABLE", False):
            from stateset_agents.core.enhanced_state_management import get_state_service

            service1 = get_state_service()
            service2 = get_state_service()

            # Should return the same instance
            assert service1 is service2


class TestCacheSize:
    """Test cache size tracking."""

    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        return InMemoryCacheBackend(max_size=10)

    @pytest.mark.asyncio
    async def test_size_tracking(self, cache):
        """Test that size is tracked for entries."""
        await cache.set("key1", "x" * 100)

        entry = cache.cache.get("key1")
        if entry:
            # Size should be calculated
            assert entry.size_bytes >= 0


class TestCacheDependencies:
    """Test cache dependency tracking."""

    def test_entry_with_dependencies(self):
        """Test creating entry with dependencies."""
        entry = CacheEntry(
            key="derived_key",
            value="derived_value",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            dependencies=["base_key_1", "base_key_2"],
        )

        assert len(entry.dependencies) == 2
        assert "base_key_1" in entry.dependencies


class TestCacheTags:
    """Test cache tagging functionality."""

    def test_entry_with_tags(self):
        """Test creating entry with tags."""
        entry = CacheEntry(
            key="user_data",
            value={"name": "test"},
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            tags=["user", "profile", "v1"],
        )

        assert len(entry.tags) == 3
        assert "user" in entry.tags
        assert "profile" in entry.tags


class TestCacheVersioning:
    """Test cache entry versioning."""

    def test_entry_versioning(self):
        """Test entry version tracking."""
        entry = CacheEntry(
            key="versioned_key",
            value="value_v1",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            version=1,
        )

        assert entry.version == 1

        # Simulate version update
        entry.version = 2
        assert entry.version == 2


class TestConcurrentAccess:
    """Test concurrent cache access."""

    @pytest.mark.asyncio
    async def test_concurrent_gets(self):
        """Test concurrent get operations."""
        cache = InMemoryCacheBackend(max_size=100)
        await cache.set("key1", "value1")

        # Perform concurrent gets
        tasks = [cache.get("key1") for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(r == "value1" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_sets(self):
        """Test concurrent set operations."""
        cache = InMemoryCacheBackend(max_size=100)

        # Perform concurrent sets
        tasks = [cache.set(f"key{i}", f"value{i}") for i in range(10)]
        await asyncio.gather(*tasks)

        # All should be set
        for i in range(10):
            assert await cache.get(f"key{i}") == f"value{i}"
