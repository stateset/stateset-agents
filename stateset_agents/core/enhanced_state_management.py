"""
Enhanced State Management and Caching for GRPO Agent Framework

This module provides advanced state management, distributed caching, consistency guarantees,
and intelligent cache invalidation for production-ready GRPO services.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

try:
    import aioredis
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import pymongo
    from motor.motor_asyncio import AsyncIOMotorClient

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import memcache
    import pymemcache

    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheStrategy(Enum):
    """Cache strategies"""

    WRITE_THROUGH = "write_through"  # Write to cache and storage simultaneously
    WRITE_BACK = "write_back"  # Write to cache first, storage later
    WRITE_AROUND = "write_around"  # Write to storage only, bypass cache
    READ_THROUGH = "read_through"  # Read from cache, fallback to storage
    CACHE_ASIDE = "cache_aside"  # Application manages cache manually


class ConsistencyLevel(Enum):
    """Consistency levels for distributed state"""

    EVENTUAL = "eventual"  # Eventually consistent
    STRONG = "strong"  # Strongly consistent
    CAUSAL = "causal"  # Causally consistent
    SESSION = "session"  # Session consistent


class EvictionPolicy(Enum):
    """Cache eviction policies"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Machine learning based


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    version: int = 1
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class StateSnapshot:
    """State snapshot for versioning"""

    snapshot_id: str
    timestamp: float
    state_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_snapshot: Optional[str] = None


class CacheBackend(ABC):
    """Abstract cache backend"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass


class InMemoryCacheBackend(CacheBackend):
    """High-performance in-memory cache backend"""

    def __init__(
        self,
        max_size: int = 10000,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]

            # Check expiration
            if entry.is_expired():
                await self._remove_entry(key)
                return None

            # Update access statistics
            entry.update_access()
            self._update_access_tracking(key)

            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        async with self.lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl,
                size_bytes=self._calculate_size(value),
            )

            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict()

            self.cache[key] = entry
            self._update_access_tracking(key)

            return True

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        async with self.lock:
            return await self._remove_entry(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        async with self.lock:
            if key not in self.cache:
                return False

            entry = self.cache[key]
            if entry.is_expired():
                await self._remove_entry(key)
                return False

            return True

    async def clear(self) -> bool:
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            return True

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode("utf-8"))

    def _update_access_tracking(self, key: str):
        """Update access tracking for eviction policies"""
        if self.eviction_policy == EvictionPolicy.LRU:
            self.access_order[key] = time.time()
        elif self.eviction_policy == EvictionPolicy.LFU:
            self.access_frequency[key] += 1

    async def _evict(self):
        """Evict entries based on policy"""
        if not self.cache:
            return

        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used
            oldest_key = min(
                self.access_order.keys(), key=lambda k: self.access_order[k]
            )
            await self._remove_entry(oldest_key)

        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            least_frequent_key = min(
                self.access_frequency.keys(), key=lambda k: self.access_frequency[k]
            )
            await self._remove_entry(least_frequent_key)

        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Remove first entry
            first_key = next(iter(self.cache))
            await self._remove_entry(first_key)

        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries first
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self.cache.items()
                if entry.ttl and current_time > entry.created_at + entry.ttl
            ]

            if expired_keys:
                await self._remove_entry(expired_keys[0])
            else:
                # Fallback to LRU
                oldest_key = min(
                    self.cache.keys(), key=lambda k: self.cache[k].last_accessed
                )
                await self._remove_entry(oldest_key)

    async def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache"""
        if key in self.cache:
            del self.cache[key]
            self.access_order.pop(key, None)
            self.access_frequency.pop(key, None)
            return True
        return False


class RedisCacheBackend(CacheBackend):
    """Redis-based distributed cache backend"""

    def __init__(
        self, redis_url: str = "redis://localhost:6379", key_prefix: str = "grpo:"
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.client = None
        self._connection_pool = None

    async def _ensure_connected(self):
        """Ensure Redis connection"""
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is not available")

        if self.client is None:
            self.client = aioredis.from_url(self.redis_url)

    def _make_key(self, key: str) -> str:
        """Create prefixed key"""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        await self._ensure_connected()

        try:
            data = await self.client.get(self._make_key(key))
            if data is None:
                return None

            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis"""
        await self._ensure_connected()

        try:
            data = pickle.dumps(value)

            if ttl:
                await self.client.setex(self._make_key(key), ttl, data)
            else:
                await self.client.set(self._make_key(key), data)

            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis"""
        await self._ensure_connected()

        try:
            result = await self.client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        await self._ensure_connected()

        try:
            result = await self.client.exists(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries with prefix"""
        await self._ensure_connected()

        try:
            keys = await self.client.keys(f"{self.key_prefix}*")
            if keys:
                await self.client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False


class StateManager:
    """Advanced state management with versioning and consistency"""

    def __init__(
        self,
        cache_backend: Optional[CacheBackend] = None,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
        enable_versioning: bool = True,
        max_snapshots: int = 100,
    ):
        self.cache_backend = cache_backend or InMemoryCacheBackend(max_size=10000)
        self.consistency_level = consistency_level
        self.enable_versioning = enable_versioning
        self.max_snapshots = max_snapshots

        # State tracking
        self.state: Dict[str, Any] = {}
        self.state_version: int = 1
        self.snapshots: List[StateSnapshot] = []
        self.watchers: Dict[str, List[Callable]] = defaultdict(list)
        self.dirty_keys: set = set()

        # Consistency and locking
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.global_lock = asyncio.Lock()

        # Change tracking
        self.change_log: List[Dict[str, Any]] = []
        self.max_change_log = 1000

    async def get(self, key: str, default: Any = None, use_cache: bool = True) -> Any:
        """Get state value with optional caching"""
        if use_cache:
            # Try cache first
            cached_value = await self.cache_backend.get(key)
            if cached_value is not None:
                return cached_value

        # Get from local state
        async with self.locks[key]:
            value = self.state.get(key, default)

            # Update cache if value exists
            if use_cache and value is not None:
                await self.cache_backend.set(key, value)

            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        notify_watchers: bool = True,
        create_snapshot: bool = False,
    ) -> bool:
        """Set state value with consistency guarantees"""
        async with self.locks[key]:
            old_value = self.state.get(key)

            # Update local state
            self.state[key] = value
            self.dirty_keys.add(key)

            # Log change
            self._log_change("set", key, old_value, value)

            # Update cache
            await self.cache_backend.set(key, value, ttl)

            # Create snapshot if requested
            if create_snapshot and self.enable_versioning:
                await self._create_snapshot()

            # Notify watchers
            if notify_watchers:
                await self._notify_watchers(key, old_value, value)

            return True

    async def delete(self, key: str, notify_watchers: bool = True) -> bool:
        """Delete state value"""
        async with self.locks[key]:
            if key not in self.state:
                return False

            old_value = self.state[key]
            del self.state[key]
            self.dirty_keys.discard(key)

            # Log change
            self._log_change("delete", key, old_value, None)

            # Remove from cache
            await self.cache_backend.delete(key)

            # Notify watchers
            if notify_watchers:
                await self._notify_watchers(key, old_value, None)

            return True

    async def exists(self, key: str) -> bool:
        """Check if key exists in state"""
        return key in self.state or await self.cache_backend.exists(key)

    async def update(self, updates: Dict[str, Any], atomic: bool = True) -> bool:
        """Update multiple state values atomically"""
        if atomic:
            async with self.global_lock:
                return await self._perform_updates(updates)
        else:
            return await self._perform_updates(updates)

    async def _perform_updates(self, updates: Dict[str, Any]) -> bool:
        """Perform the actual updates"""
        try:
            for key, value in updates.items():
                await self.set(key, value, notify_watchers=False)

            # Notify watchers for all changes
            for key, value in updates.items():
                old_value = self.state.get(key)
                await self._notify_watchers(key, old_value, value)

            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    async def watch(self, key: str, callback: Callable[[str, Any, Any], None]):
        """Watch for changes to a key"""
        self.watchers[key].append(callback)

    async def unwatch(self, key: str, callback: Callable[[str, Any, Any], None]):
        """Stop watching for changes to a key"""
        if key in self.watchers and callback in self.watchers[key]:
            self.watchers[key].remove(callback)

    async def _notify_watchers(self, key: str, old_value: Any, new_value: Any):
        """Notify watchers of state changes"""
        for callback in self.watchers.get(key, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(key, old_value, new_value)
                else:
                    callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Watcher callback failed: {e}")

    def _log_change(self, operation: str, key: str, old_value: Any, new_value: Any):
        """Log state changes"""
        change_entry = {
            "timestamp": time.time(),
            "operation": operation,
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "version": self.state_version,
        }

        self.change_log.append(change_entry)

        # Trim change log if too large
        if len(self.change_log) > self.max_change_log:
            self.change_log = self.change_log[-self.max_change_log :]

    async def _create_snapshot(self) -> str:
        """Create a state snapshot"""
        if not self.enable_versioning:
            return ""

        snapshot_id = str(uuid.uuid4())
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            state_data=dict(self.state),  # Deep copy
            metadata={"version": self.state_version},
        )

        self.snapshots.append(snapshot)
        self.state_version += 1

        # Trim snapshots if too many
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots :]

        return snapshot_id

    async def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore state from snapshot"""
        if not self.enable_versioning:
            return False

        snapshot = next(
            (s for s in self.snapshots if s.snapshot_id == snapshot_id), None
        )
        if not snapshot:
            return False

        async with self.global_lock:
            # Clear current state
            old_state = dict(self.state)
            self.state.clear()
            self.dirty_keys.clear()

            # Restore snapshot state
            self.state.update(snapshot.state_data)
            self.state_version = snapshot.metadata.get("version", 1)

            # Update cache
            for key, value in self.state.items():
                await self.cache_backend.set(key, value)

            # Remove keys that were deleted
            for key in old_state:
                if key not in self.state:
                    await self.cache_backend.delete(key)

            # Log restoration
            self._log_change("restore", "system", old_state, dict(self.state))

            return True

    def get_snapshots(self) -> List[StateSnapshot]:
        """Get list of available snapshots"""
        return self.snapshots.copy()

    def get_change_log(self, since: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get change log since timestamp"""
        if since is None:
            return self.change_log.copy()

        return [change for change in self.change_log if change["timestamp"] >= since]

    async def sync_with_cache(self) -> Dict[str, bool]:
        """Sync dirty state with cache"""
        results = {}

        for key in list(self.dirty_keys):
            try:
                value = self.state.get(key)
                if value is not None:
                    success = await self.cache_backend.set(key, value)
                    results[key] = success
                    if success:
                        self.dirty_keys.discard(key)
                else:
                    success = await self.cache_backend.delete(key)
                    results[key] = success
                    if success:
                        self.dirty_keys.discard(key)
            except Exception as e:
                logger.error(f"Sync failed for key {key}: {e}")
                results[key] = False

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics"""
        return {
            "state_size": len(self.state),
            "dirty_keys": len(self.dirty_keys),
            "snapshots": len(self.snapshots),
            "watchers": sum(len(callbacks) for callbacks in self.watchers.values()),
            "change_log_size": len(self.change_log),
            "state_version": self.state_version,
            "consistency_level": self.consistency_level.value,
        }


class ConversationStateManager:
    """Specialized state manager for conversation contexts"""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.conversation_prefix = "conversation:"
        self.user_prefix = "user:"
        self.session_timeout = 3600  # 1 hour

    async def create_conversation(
        self, conversation_id: str, user_id: str, initial_context: Dict[str, Any] = None
    ) -> bool:
        """Create a new conversation"""
        conversation_key = f"{self.conversation_prefix}{conversation_id}"
        conversation_data = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "turns": [],
            "context": initial_context or {},
            "metadata": {},
        }

        return await self.state_manager.set(
            conversation_key, conversation_data, ttl=self.session_timeout
        )

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation data"""
        conversation_key = f"{self.conversation_prefix}{conversation_id}"
        return await self.state_manager.get(conversation_key)

    async def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Add a turn to conversation"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False

        turn = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        conversation["turns"].append(turn)
        conversation["last_activity"] = time.time()

        conversation_key = f"{self.conversation_prefix}{conversation_id}"
        return await self.state_manager.set(
            conversation_key, conversation, ttl=self.session_timeout
        )

    async def update_context(
        self, conversation_id: str, context_updates: Dict[str, Any]
    ) -> bool:
        """Update conversation context"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False

        conversation["context"].update(context_updates)
        conversation["last_activity"] = time.time()

        conversation_key = f"{self.conversation_prefix}{conversation_id}"
        return await self.state_manager.set(
            conversation_key, conversation, ttl=self.session_timeout
        )

    async def get_user_conversations(self, user_id: str) -> List[str]:
        """Get all conversation IDs for a user"""
        # This is a simplified implementation
        # In practice, you'd maintain an index or use pattern matching
        user_key = f"{self.user_prefix}{user_id}:conversations"
        conversations = await self.state_manager.get(user_key, default=[])
        return conversations

    async def cleanup_expired_conversations(self) -> int:
        """Clean up expired conversations"""
        # This is a simplified implementation
        # In practice, you'd use TTL or periodic cleanup
        current_time = time.time()
        cleaned_count = 0

        # Note: This would need optimization for large-scale deployments
        for key in list(self.state_manager.state.keys()):
            if key.startswith(self.conversation_prefix):
                conversation = await self.state_manager.get(key)
                if (
                    conversation
                    and current_time - conversation["last_activity"]
                    > self.session_timeout
                ):
                    await self.state_manager.delete(key)
                    cleaned_count += 1

        return cleaned_count


class ConversationManager(ConversationStateManager):
    """Compatibility alias for ``ConversationStateManager``."""


class DistributedStateService:
    """Complete distributed state management service"""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        mongodb_url: Optional[str] = None,
        enable_local_cache: bool = True,
        cache_strategy: CacheStrategy = CacheStrategy.READ_THROUGH,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
    ):
        # Setup cache backends
        backends = []

        if enable_local_cache:
            backends.append(InMemoryCacheBackend(max_size=10000))

        if redis_url and REDIS_AVAILABLE:
            backends.append(RedisCacheBackend(redis_url))

        # Use the first available backend
        self.cache_backend = backends[0] if backends else InMemoryCacheBackend()

        # Setup state managers
        self.state_manager = StateManager(
            cache_backend=self.cache_backend, consistency_level=consistency_level
        )

        self.conversation_manager = ConversationStateManager(self.state_manager)

        # Service configuration
        self.cache_strategy = cache_strategy
        self.consistency_level = consistency_level

        # Background tasks
        self._background_tasks = []
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self._background_tasks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover
            # Constructed outside an event loop (tests/import-time); start later.
            return

        self._background_tasks.append(loop.create_task(self._periodic_sync()))
        self._background_tasks.append(loop.create_task(self._periodic_cleanup()))

    async def _periodic_sync(self):
        """Periodically sync state with cache"""
        while True:
            try:
                await self.state_manager.sync_with_cache()
                await asyncio.sleep(30)  # Sync every 30 seconds
            except Exception as e:
                logger.error(f"Periodic sync failed: {e}")
                await asyncio.sleep(60)

    async def _periodic_cleanup(self):
        """Periodically clean up expired data"""
        while True:
            try:
                # Cleanup conversations
                cleaned = (
                    await self.conversation_manager.cleanup_expired_conversations()
                )
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired conversations")

                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                logger.error(f"Periodic cleanup failed: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes

    async def health_check(self) -> Dict[str, Any]:
        """Get service health status"""
        try:
            # Test cache backend
            test_key = "health_check_test"
            test_value = {"timestamp": time.time()}

            await self.cache_backend.set(test_key, test_value, ttl=60)
            retrieved_value = await self.cache_backend.get(test_key)
            await self.cache_backend.delete(test_key)

            cache_healthy = retrieved_value is not None

            # Get stats
            state_stats = self.state_manager.get_stats()

            return {
                "status": "healthy" if cache_healthy else "unhealthy",
                "cache_backend": type(self.cache_backend).__name__,
                "cache_healthy": cache_healthy,
                "state_stats": state_stats,
                "background_tasks": len(self._background_tasks),
                "consistency_level": self.consistency_level.value,
                "cache_strategy": self.cache_strategy.value,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def shutdown(self):
        """Graceful shutdown"""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Final sync
        try:
            await self.state_manager.sync_with_cache()
        except Exception as e:
            logger.error(f"Final sync failed: {e}")

        # Clear cache if needed
        if isinstance(self.cache_backend, InMemoryCacheBackend):
            await self.cache_backend.clear()


# Global state service instance
_state_service: Optional[DistributedStateService] = None


def get_state_service() -> DistributedStateService:
    """Get or create global state service"""
    global _state_service
    if _state_service is None:
        _state_service = DistributedStateService()
    return _state_service


@asynccontextmanager
async def managed_state_context():
    """Context manager for managed state operations"""
    state_service = get_state_service()
    try:
        yield state_service
    finally:
        # Ensure state is synced
        sync_with_cache = getattr(state_service.state_manager, "sync_with_cache", None)
        if callable(sync_with_cache):
            try:
                maybe_awaitable = sync_with_cache()
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
            except TypeError:
                # Non-awaitable mock or sync implementation.
                pass


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create state service
        state_service = DistributedStateService()

        # Basic state operations
        await state_service.state_manager.set(
            "user:123", {"name": "Alice", "preferences": {"theme": "dark"}}
        )
        user_data = await state_service.state_manager.get("user:123")
        print("User data:", user_data)

        # Conversation management
        conversation_id = "conv_123"
        await state_service.conversation_manager.create_conversation(
            conversation_id, "user:123", {"topic": "ai_assistance"}
        )

        await state_service.conversation_manager.add_turn(
            conversation_id, "user", "Hello, I need help with AI"
        )

        await state_service.conversation_manager.add_turn(
            conversation_id,
            "assistant",
            "I'd be happy to help you with AI-related questions!",
        )

        conversation = await state_service.conversation_manager.get_conversation(
            conversation_id
        )
        print("Conversation:", json.dumps(conversation, indent=2))

        # Health check
        health = await state_service.health_check()
        print("Health:", json.dumps(health, indent=2))

        # Cleanup
        await state_service.shutdown()

    asyncio.run(main())
