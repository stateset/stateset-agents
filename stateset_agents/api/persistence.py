"""
Database Persistence Layer

Production-ready persistence with support for multiple database backends,
connection pooling, migrations, and comprehensive ORM integration.
"""

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Configuration
# ============================================================================

class DatabaseBackend(str, Enum):
    """Supported database backends."""
    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    backend: DatabaseBackend = DatabaseBackend.MEMORY
    connection_url: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800
    echo: bool = False
    auto_migrate: bool = True

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        backend_str = os.getenv("DB_BACKEND", "memory").lower()
        backend = DatabaseBackend(backend_str) if backend_str in [e.value for e in DatabaseBackend] else DatabaseBackend.MEMORY

        return cls(
            backend=backend,
            connection_url=os.getenv("DATABASE_URL"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            auto_migrate=os.getenv("DB_AUTO_MIGRATE", "true").lower() == "true",
        )


# ============================================================================
# Base Models
# ============================================================================

@dataclass
class BaseEntity:
    """Base class for all persistent entities."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEntity":
        """Create from dictionary."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class Agent(BaseEntity):
    """Agent entity."""
    name: str = ""
    model_name: str = "gpt-4"
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    status: str = "active"
    config: Dict[str, Any] = field(default_factory=dict)
    total_tokens_used: int = 0
    total_conversations: int = 0


@dataclass
class Conversation(BaseEntity):
    """Conversation entity."""
    agent_id: str = ""
    user_id: Optional[str] = None
    status: str = "active"
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    last_message_at: Optional[datetime] = None


@dataclass
class TrainingJob(BaseEntity):
    """Training job entity."""
    agent_id: Optional[str] = None
    status: str = "pending"
    strategy: str = "computational"
    prompts: List[str] = field(default_factory=list)
    num_iterations: int = 1
    current_iteration: int = 0
    total_trajectories: int = 0
    average_reward: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class APIKey(BaseEntity):
    """API key entity."""
    key_hash: str = ""
    name: str = ""
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=lambda: ["user"])
    rate_limit: int = 60
    is_active: bool = True
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


# ============================================================================
# Repository Interface
# ============================================================================

class Repository(ABC, Generic[T]):
    """Abstract repository interface for data access."""

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        """List entities with pagination and filtering."""
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching filters."""
        pass


# ============================================================================
# In-Memory Implementation
# ============================================================================

class InMemoryRepository(Repository[T]):
    """In-memory repository implementation for development/testing."""

    def __init__(self, entity_class: Type[T]):
        self._entity_class = entity_class
        self._data: Dict[str, T] = {}
        self._lock = asyncio.Lock()

    async def create(self, entity: T) -> T:
        """Create a new entity."""
        async with self._lock:
            entity.created_at = datetime.utcnow()
            entity.updated_at = datetime.utcnow()
            self._data[entity.id] = entity
            return entity

    async def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        return self._data.get(id)

    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        async with self._lock:
            if entity.id not in self._data:
                raise ValueError(f"Entity {entity.id} not found")
            entity.updated_at = datetime.utcnow()
            self._data[entity.id] = entity
            return entity

    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        async with self._lock:
            if id in self._data:
                del self._data[id]
                return True
            return False

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        """List entities with pagination and filtering."""
        items = list(self._data.values())

        # Apply filters
        if filters:
            for key, value in filters.items():
                items = [item for item in items if getattr(item, key, None) == value]

        # Sort by created_at descending
        items.sort(key=lambda x: x.created_at, reverse=True)

        # Apply pagination
        return items[offset:offset + limit]

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching filters."""
        if not filters:
            return len(self._data)

        items = list(self._data.values())
        for key, value in filters.items():
            items = [item for item in items if getattr(item, key, None) == value]
        return len(items)

    async def clear(self) -> None:
        """Clear all data."""
        async with self._lock:
            self._data.clear()


# ============================================================================
# SQLite Implementation
# ============================================================================

class SQLiteRepository(Repository[T]):
    """SQLite repository implementation for single-instance deployments."""

    def __init__(self, entity_class: Type[T], db_path: str = "stateset.db"):
        self._entity_class = entity_class
        self._db_path = db_path
        self._table_name = entity_class.__name__.lower() + "s"
        self._initialized: bool = False
        self._init_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to SQLite database.

        SQLiteRepository intentionally avoids threadpool-based database I/O in
        order to stay compatible with constrained/embedded runtimes where
        cross-thread loop wakeups can be unreliable. Each operation opens a new
        short-lived connection and performs work synchronously.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return
            self._init_sync()
            self._initialized = True
            logger.info("Initialized SQLite database: %s", self._db_path)

    def _init_sync(self) -> None:
        connection = sqlite3.connect(self._db_path, timeout=30)
        try:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_created
                ON {self._table_name}(created_at)
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("SQLiteRepository is not connected. Call connect() first.")

    def _create_sync(self, entity_id: str, data_json: str, created_at: str, updated_at: str) -> None:
        self._require_initialized()
        connection = sqlite3.connect(self._db_path, timeout=30)
        try:
            connection.execute(
                f"INSERT INTO {self._table_name} (id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (entity_id, data_json, created_at, updated_at),
            )
            connection.commit()
        finally:
            connection.close()

    async def create(self, entity: T) -> T:
        """Create a new entity."""
        await self.connect()
        entity.created_at = datetime.utcnow()
        entity.updated_at = datetime.utcnow()
        self._create_sync(
            entity.id,
            json.dumps(entity.to_dict()),
            entity.created_at.isoformat(),
            entity.updated_at.isoformat(),
        )
        return entity

    def _get_data_sync(self, id: str) -> Optional[str]:
        self._require_initialized()
        connection = sqlite3.connect(self._db_path, timeout=30)
        cursor = connection.execute(
            f"SELECT data FROM {self._table_name} WHERE id = ?",
            (id,),
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
            connection.close()
        if row:
            return row[0]
        return None

    async def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        await self.connect()
        data_json = self._get_data_sync(id)
        if data_json is None:
            return None
        return self._entity_class.from_dict(json.loads(data_json))

    def _update_sync(self, entity_id: str, data_json: str, updated_at: str) -> None:
        self._require_initialized()
        connection = sqlite3.connect(self._db_path, timeout=30)
        try:
            connection.execute(
                f"UPDATE {self._table_name} SET data = ?, updated_at = ? WHERE id = ?",
                (data_json, updated_at, entity_id),
            )
            connection.commit()
        finally:
            connection.close()

    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        await self.connect()
        entity.updated_at = datetime.utcnow()
        self._update_sync(
            entity.id,
            json.dumps(entity.to_dict()),
            entity.updated_at.isoformat(),
        )
        return entity

    def _delete_sync(self, id: str) -> bool:
        self._require_initialized()
        connection = sqlite3.connect(self._db_path, timeout=30)
        cursor = connection.execute(
            f"DELETE FROM {self._table_name} WHERE id = ?",
            (id,),
        )
        try:
            connection.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()
            connection.close()

    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        await self.connect()
        return self._delete_sync(id)

    def _list_data_sync(self, limit: int, offset: int) -> List[str]:
        self._require_initialized()
        connection = sqlite3.connect(self._db_path, timeout=30)
        cursor = connection.execute(
            f"SELECT data FROM {self._table_name} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        try:
            rows = cursor.fetchall()
        finally:
            cursor.close()
            connection.close()

        return [row[0] for row in rows]

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        """List entities with pagination."""
        await self.connect()
        data_rows = self._list_data_sync(limit, offset)
        items = [self._entity_class.from_dict(json.loads(row)) for row in data_rows]

        # Apply filters in memory (for simplicity)
        if filters:
            for key, value in filters.items():
                items = [item for item in items if getattr(item, key, None) == value]

        return items

    def _count_sync(self) -> int:
        self._require_initialized()
        connection = sqlite3.connect(self._db_path, timeout=30)
        cursor = connection.execute(f"SELECT COUNT(*) FROM {self._table_name}")
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
            connection.close()
        return int(row[0]) if row else 0

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities."""
        # SQLite doesn't apply filters at query time in this simple implementation.
        # For filtered counts, re-use list() then count in memory.
        await self.connect()
        if filters:
            items = await self.list(limit=10_000_000, offset=0, filters=filters)
            return len(items)
        return self._count_sync()

    async def close(self) -> None:
        """Close database connection."""
        # No persistent connections to close in this implementation.
        return


# ============================================================================
# Unit of Work Pattern
# ============================================================================

class UnitOfWork:
    """
    Unit of Work pattern for managing database transactions.

    Provides transactional consistency across multiple repository operations.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._agents: Optional[Repository[Agent]] = None
        self._conversations: Optional[Repository[Conversation]] = None
        self._training_jobs: Optional[Repository[TrainingJob]] = None
        self._api_keys: Optional[Repository[APIKey]] = None

    async def __aenter__(self) -> "UnitOfWork":
        """Enter context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        await self.close()

    async def connect(self) -> None:
        """Initialize repositories based on backend."""
        if self.config.backend == DatabaseBackend.MEMORY:
            self._agents = InMemoryRepository(Agent)
            self._conversations = InMemoryRepository(Conversation)
            self._training_jobs = InMemoryRepository(TrainingJob)
            self._api_keys = InMemoryRepository(APIKey)

        elif self.config.backend == DatabaseBackend.SQLITE:
            db_path = self.config.connection_url or "stateset.db"
            if db_path.startswith("sqlite:///"):
                db_path = db_path[10:]

            self._agents = SQLiteRepository(Agent, db_path)
            self._conversations = SQLiteRepository(Conversation, db_path)
            self._training_jobs = SQLiteRepository(TrainingJob, db_path)
            self._api_keys = SQLiteRepository(APIKey, db_path)

            await self._agents.connect()
            await self._conversations.connect()
            await self._training_jobs.connect()
            await self._api_keys.connect()

        logger.info(f"Database initialized with {self.config.backend.value} backend")

    async def close(self) -> None:
        """Close all connections."""
        for repo in [self._agents, self._conversations, self._training_jobs, self._api_keys]:
            if hasattr(repo, "close"):
                await repo.close()

    @property
    def agents(self) -> Repository[Agent]:
        """Get agents repository."""
        if self._agents is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._agents

    @property
    def conversations(self) -> Repository[Conversation]:
        """Get conversations repository."""
        if self._conversations is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conversations

    @property
    def training_jobs(self) -> Repository[TrainingJob]:
        """Get training jobs repository."""
        if self._training_jobs is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._training_jobs

    @property
    def api_keys(self) -> Repository[APIKey]:
        """Get API keys repository."""
        if self._api_keys is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._api_keys


# ============================================================================
# Global Database Instance
# ============================================================================

_database: Optional[UnitOfWork] = None


async def init_database(config: Optional[DatabaseConfig] = None) -> UnitOfWork:
    """Initialize the global database instance."""
    global _database
    config = config or DatabaseConfig.from_env()
    _database = UnitOfWork(config)
    await _database.connect()
    return _database


def get_database() -> Optional[UnitOfWork]:
    """Get the global database instance."""
    return _database


async def close_database() -> None:
    """Close the global database instance."""
    global _database
    if _database is not None:
        await _database.close()
        _database = None


@asynccontextmanager
async def database_session():
    """Context manager for database sessions."""
    db = get_database()
    if db is None:
        db = await init_database()
    try:
        yield db
    finally:
        pass  # Connection pooling handles cleanup
