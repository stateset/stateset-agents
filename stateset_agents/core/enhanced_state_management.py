"""
Enhanced State Management and Caching for GRPO Agent Framework

This module provides advanced state management, distributed caching, consistency guarantees,
and intelligent cache invalidation for production-ready GRPO services.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, TypeVar
from collections.abc import Callable

from .enhanced_state_cache import (
    MONGODB_AVAILABLE,
    REDIS_AVAILABLE,
    STATE_EXCEPTIONS,
    CacheBackend,
    CacheEntry,
    CacheStrategy,
    ConsistencyLevel,
    EvictionPolicy,
    InMemoryCacheBackend,
    MultiCacheBackend,
    RedisCacheBackend,
    StateSnapshot,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StateManager:
    """Advanced state management with versioning and consistency"""

    def __init__(
        self,
        cache_backend: CacheBackend | None = None,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
        enable_versioning: bool = True,
        max_snapshots: int = 100,
    ):
        self.cache_backend = cache_backend or InMemoryCacheBackend(max_size=10000)
        self.consistency_level = consistency_level
        self.enable_versioning = enable_versioning
        self.max_snapshots = max_snapshots

        # State tracking
        self.state: dict[str, Any] = {}
        self.state_version: int = 1
        self.snapshots: list[StateSnapshot] = []
        self.watchers: dict[str, list[Callable]] = defaultdict(list)
        self.dirty_keys: set = set()

        # Consistency and locking
        self.locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.global_lock = asyncio.Lock()

        # Change tracking
        self.change_log: list[dict[str, Any]] = []
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
        ttl: int | None = None,
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

    async def update(self, updates: dict[str, Any], atomic: bool = True) -> bool:
        """Update multiple state values atomically"""
        if atomic:
            async with self.global_lock:
                return await self._perform_updates(updates)
        else:
            return await self._perform_updates(updates)

    async def _perform_updates(self, updates: dict[str, Any]) -> bool:
        """Perform the actual updates"""
        try:
            old_values = {key: self.state.get(key) for key in updates}
            for key, value in updates.items():
                await self.set(key, value, notify_watchers=False)

            # Notify watchers for all changes
            for key, value in updates.items():
                await self._notify_watchers(key, old_values.get(key), value)

            return True
        except STATE_EXCEPTIONS as e:
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
            except STATE_EXCEPTIONS as e:
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

    def get_snapshots(self) -> list[StateSnapshot]:
        """Get list of available snapshots"""
        return self.snapshots.copy()

    def get_change_log(self, since: float | None = None) -> list[dict[str, Any]]:
        """Get change log since timestamp"""
        if since is None:
            return self.change_log.copy()

        return [change for change in self.change_log if change["timestamp"] >= since]

    async def sync_with_cache(self) -> dict[str, bool]:
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
            except STATE_EXCEPTIONS as e:
                logger.error(f"Sync failed for key {key}: {e}")
                results[key] = False

        return results

    def get_stats(self) -> dict[str, Any]:
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
        self, conversation_id: str, user_id: str, initial_context: dict[str, Any] = None
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

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get conversation data"""
        conversation_key = f"{self.conversation_prefix}{conversation_id}"
        return await self.state_manager.get(conversation_key)

    async def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] = None,
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
        self, conversation_id: str, context_updates: dict[str, Any]
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

    async def get_user_conversations(self, user_id: str) -> list[str]:
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
        redis_url: str | None = None,
        mongodb_url: str | None = None,
        enable_local_cache: bool = True,
        cache_strategy: CacheStrategy = CacheStrategy.READ_THROUGH,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
        cache_serializer: str = "json",
        allow_pickle: bool = False,
    ):
        # Setup cache backends
        local_cache = (
            InMemoryCacheBackend(max_size=10000) if enable_local_cache else None
        )
        redis_cache = None

        if redis_url and REDIS_AVAILABLE:
            redis_cache = RedisCacheBackend(
                redis_url,
                serializer=cache_serializer,
                allow_pickle=allow_pickle,
            )

        if local_cache and redis_cache:
            self.cache_backend = MultiCacheBackend(local_cache, redis_cache)
        elif redis_cache:
            self.cache_backend = redis_cache
        elif local_cache:
            self.cache_backend = local_cache
        else:
            self.cache_backend = InMemoryCacheBackend()

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
            except STATE_EXCEPTIONS as e:
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
            except STATE_EXCEPTIONS as e:
                logger.error(f"Periodic cleanup failed: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes

    async def health_check(self) -> dict[str, Any]:
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
        except STATE_EXCEPTIONS as e:
            return {"status": "error", "error": str(e)}

    async def shutdown(self):
        """Graceful shutdown"""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Final sync
        try:
            await self.state_manager.sync_with_cache()
        except STATE_EXCEPTIONS as e:
            logger.error(f"Final sync failed: {e}")

        # Clear cache if needed
        if isinstance(self.cache_backend, InMemoryCacheBackend):
            await self.cache_backend.clear()


# Global state service instance
_state_service: DistributedStateService | None = None


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


__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CacheStrategy",
    "ConsistencyLevel",
    "ConversationManager",
    "ConversationStateManager",
    "DistributedStateService",
    "EvictionPolicy",
    "InMemoryCacheBackend",
    "MONGODB_AVAILABLE",
    "MultiCacheBackend",
    "REDIS_AVAILABLE",
    "RedisCacheBackend",
    "STATE_EXCEPTIONS",
    "StateManager",
    "StateSnapshot",
    "get_state_service",
    "managed_state_context",
]
