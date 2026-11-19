"""Memory and SQLite repositories with explicit, task-owned transactions."""

import asyncio
import copy
import json
import logging
import os
import re
import sqlite3
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import get_ident
from typing import Any, Generic, TypeVar, cast, get_args, get_type_hints

from stateset_agents.utils.async_calls import drain_owned_operation

from .persistence_transactions import TransactionAborted as TransactionAborted
from .persistence_transactions import TransactionConflict, TransactionOwner
from .resource_lifecycle import ResourceLease, SharedResource

logger = logging.getLogger(__name__)

_BaseEntityT = TypeVar("_BaseEntityT", bound="BaseEntity")
T = TypeVar("T", bound="BaseEntity")


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
    connection_url: str | None = None
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
        backend = (
            DatabaseBackend(backend_str)
            if backend_str in [e.value for e in DatabaseBackend]
            else DatabaseBackend.MEMORY
        )

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = copy.deepcopy(value)
        return result

    @classmethod
    def from_dict(cls: type[_BaseEntityT], data: dict[str, Any]) -> _BaseEntityT:
        """Create from dictionary."""
        prepared = copy.deepcopy(data)
        for key, annotation in get_type_hints(cls).items():
            if (
                annotation is datetime or datetime in get_args(annotation)
            ) and isinstance(prepared.get(key), str):
                prepared[key] = datetime.fromisoformat(prepared[key])
        return cls(**prepared)


@dataclass
class Agent(BaseEntity):
    """Agent entity."""

    name: str = ""
    model_name: str = "gpt-4"
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    status: str = "active"
    config: dict[str, Any] = field(default_factory=dict)
    total_tokens_used: int = 0
    total_conversations: int = 0


@dataclass
class Conversation(BaseEntity):
    """Conversation entity."""

    agent_id: str = ""
    user_id: str | None = None
    status: str = "active"
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    last_message_at: datetime | None = None


@dataclass
class TrainingJob(BaseEntity):
    """Training job entity."""

    agent_id: str | None = None
    status: str = "pending"
    strategy: str = "computational"
    prompts: list[str] = field(default_factory=list)
    num_iterations: int = 1
    current_iteration: int = 0
    total_trajectories: int = 0
    average_reward: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class APIKey(BaseEntity):
    """API key entity."""

    key_hash: str = ""
    name: str = ""
    user_id: str | None = None
    roles: list[str] = field(default_factory=lambda: ["user"])
    rate_limit: int = 60
    is_active: bool = True
    last_used_at: datetime | None = None
    expires_at: datetime | None = None


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
    async def get(self, id: str) -> T | None:
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
        filters: dict[str, Any] | None = None,
    ) -> list[T]:
        """List entities with pagination and filtering."""
        pass

    @abstractmethod
    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities matching filters."""
        pass


# ============================================================================
# In-Memory Implementation
# ============================================================================


class InMemoryRepository(Repository[T]):
    """Detached in-memory records, optionally owned by a shared UnitOfWork."""

    def __init__(
        self, entity_class: type[T], *, _owner: TransactionOwner | None = None
    ):
        self._entity_class = entity_class
        self._data: dict[str, T] = {}
        self._owner = _owner or TransactionOwner()

    async def create(self, entity: T) -> T:
        """Create a record without retaining references to the caller's object."""
        with self._owner.operation():
            if entity.id in self._data:
                raise ValueError(f"Entity {entity.id} already exists")
            stored = copy.deepcopy(entity)
            stored.created_at = stored.updated_at = datetime.utcnow()
            result = copy.deepcopy(stored)
            self._data[stored.id] = stored
            entity.created_at, entity.updated_at = stored.created_at, stored.updated_at
            return result

    async def get(self, id: str) -> T | None:
        """Get a detached record by ID."""
        with self._owner.operation():
            return copy.deepcopy(self._data.get(id))

    async def update(self, entity: T) -> T:
        """Replace an existing record with a detached candidate."""
        with self._owner.operation():
            if entity.id not in self._data:
                raise ValueError(f"Entity {entity.id} not found")
            stored = copy.deepcopy(entity)
            stored.updated_at = datetime.utcnow()
            result = copy.deepcopy(stored)
            self._data[stored.id] = stored
            entity.updated_at = stored.updated_at
            return result

    async def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        with self._owner.operation():
            return self._data.pop(id, None) is not None

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[T]:
        """Filter, order, and then paginate detached records."""
        with self._owner.operation():
            _validate_page(limit, offset)
            items = _filter_entities(list(self._data.values()), filters)
            items.sort(key=lambda item: item.created_at, reverse=True)
            return copy.deepcopy(items[offset : offset + limit])

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count matching records without pagination."""
        with self._owner.operation():
            return len(_filter_entities(list(self._data.values()), filters))

    async def clear(self) -> None:
        """Clear all records within the current ownership scope."""
        with self._owner.operation():
            self._data.clear()


def _validate_page(limit: int, offset: int) -> None:
    if limit < 0 or offset < 0:
        raise ValueError("Pagination limit and offset must be nonnegative")


def _filter_entities(items: list[T], filters: dict[str, Any] | None) -> list[T]:
    for key, value in (filters or {}).items():
        items = [item for item in items if getattr(item, key, None) == value]
    return items


class _SQLiteStore:
    """One thread-confined connection shared by a UnitOfWork's repositories."""

    def __init__(self, path: str, owner: TransactionOwner) -> None:
        self.path, self.owner = path, owner
        self.connection: sqlite3.Connection | None = None
        self.thread: int | None = None
        self.closed = False

    def connect(self) -> sqlite3.Connection:
        if self.closed:
            raise RuntimeError("SQLite store is closed")
        if self.thread is not None and self.thread != get_ident():
            raise TransactionConflict("Use a SQLite UnitOfWork on its creating thread")
        if self.connection is None:
            # Fail lock contention promptly: waiting here would block the event
            # loop that may own the other transaction. No automatic write replay.
            self.connection = sqlite3.connect(
                self.path, timeout=0, isolation_level=None
            )
            self.thread = get_ident()
        return self.connection

    def close(self) -> None:
        if self.closed:
            return
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.closed = True


class SQLiteRepository(Repository[T]):
    """SQLite records with autocommit outside an explicit transaction."""

    def __init__(
        self,
        entity_class: type[T],
        db_path: str = "stateset.db",
        *,
        _store: _SQLiteStore | None = None,
    ):
        self._entity_class = entity_class
        self._db_path = db_path
        self._table_name = entity_class.__name__.lower() + "s"
        if re.fullmatch(r"[a-z_][a-z0-9_]*", self._table_name) is None:
            raise ValueError(
                "Entity class name must form a safe SQLite table identifier"
            )
        self._owns_store = _store is None
        self._store = _store or _SQLiteStore(db_path, TransactionOwner())
        self._owner = self._store.owner
        self._initialized = False

    async def connect(self) -> None:
        """Initialize the schema once; keep the connection for transaction ownership."""
        with self._owner.operation():
            if self._initialized:
                return
            connection = self._store.connect()
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """).close()
            connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_created
                ON {self._table_name}(created_at)
            """).close()
            self._initialized = True
            if self._owns_store:
                self._owner.state = "ready"

    async def create(self, entity: T) -> T:
        """Insert a serialized candidate; duplicate IDs never overwrite records."""
        await self.connect()
        with self._owner.operation():
            candidate = copy.deepcopy(entity)
            candidate.created_at = candidate.updated_at = datetime.utcnow()
            data = json.dumps(candidate.to_dict(), allow_nan=False)
            result = copy.deepcopy(candidate)
            self._store.connect().execute(
                f"INSERT INTO {self._table_name} (id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",  # nosec: B608
                (
                    candidate.id,
                    data,
                    candidate.created_at.isoformat(),
                    candidate.updated_at.isoformat(),
                ),
            ).close()
            entity.created_at, entity.updated_at = (
                candidate.created_at,
                candidate.updated_at,
            )
            return result

    async def get(self, id: str) -> T | None:
        """Get a detached record by ID."""
        await self.connect()
        with self._owner.operation():
            cursor = self._store.connect().execute(
                f"SELECT data FROM {self._table_name} WHERE id = ?",  # nosec: B608
                (id,),
            )
            try:
                row = cursor.fetchone()
            finally:
                cursor.close()
            return self._entity_class.from_dict(json.loads(row[0])) if row else None

    async def update(self, entity: T) -> T:
        """Update an existing record; missing IDs fail consistently across backends."""
        await self.connect()
        with self._owner.operation():
            candidate = copy.deepcopy(entity)
            candidate.updated_at = datetime.utcnow()
            data = json.dumps(candidate.to_dict(), allow_nan=False)
            result = copy.deepcopy(candidate)
            cursor = self._store.connect().execute(
                f"UPDATE {self._table_name} SET data = ?, updated_at = ? WHERE id = ?",  # nosec: B608
                (data, candidate.updated_at.isoformat(), candidate.id),
            )
            try:
                if cursor.rowcount != 1:
                    raise ValueError(f"Entity {candidate.id} not found")
            finally:
                cursor.close()
            entity.updated_at = candidate.updated_at
            return result

    async def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        await self.connect()
        with self._owner.operation():
            cursor = self._store.connect().execute(
                f"DELETE FROM {self._table_name} WHERE id = ?",  # nosec: B608
                (id,),
            )
            try:
                return cursor.rowcount > 0
            finally:
                cursor.close()

    def _read_all(self) -> list[T]:
        cursor = self._store.connect().execute(
            f"SELECT data FROM {self._table_name} ORDER BY created_at DESC, rowid ASC"  # nosec: B608
        )
        try:
            return [self._entity_class.from_dict(json.loads(row[0])) for row in cursor]
        finally:
            cursor.close()

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[T]:
        """Filter before pagination, including fields inside serialized records."""
        await self.connect()
        with self._owner.operation():
            _validate_page(limit, offset)
            if filters:
                return _filter_entities(self._read_all(), filters)[
                    offset : offset + limit
                ]
            cursor = self._store.connect().execute(
                f"SELECT data FROM {self._table_name} ORDER BY created_at DESC, rowid ASC LIMIT ? OFFSET ?",  # nosec: B608
                (limit, offset),
            )
            try:
                return [
                    self._entity_class.from_dict(json.loads(row[0])) for row in cursor
                ]
            finally:
                cursor.close()

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count matches without a hidden pagination limit."""
        await self.connect()
        with self._owner.operation():
            if filters:
                return len(_filter_entities(self._read_all(), filters))
            cursor = self._store.connect().execute(
                f"SELECT COUNT(*) FROM {self._table_name}"  # nosec: B608
            )
            try:
                return int(cursor.fetchone()[0])
            finally:
                cursor.close()

    async def close(self) -> None:
        """Close the retained store only after its transaction has finished."""
        with self._owner.lock:
            self._owner.close_admission()
            try:
                self._store.close()
            except BaseException:
                self._owner.state = "failed"
                raise
            if self._owns_store:
                self._owner.state = "closed"


# ============================================================================
# Unit of Work Pattern
# ============================================================================


class UnitOfWork:
    """Own repositories; use ``transaction()`` for atomic multi-repository writes.

    ``async with UnitOfWork(...)`` still opens and closes resources. Ordinary
    repository calls retain autocommit behavior outside an explicit transaction.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = copy.deepcopy(config)
        self._agents: Repository[Agent] | None = None
        self._conversations: Repository[Conversation] | None = None
        self._training_jobs: Repository[TrainingJob] | None = None
        self._api_keys: Repository[APIKey] | None = None
        self._owner = TransactionOwner()
        self._store: _SQLiteStore | None = None
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._memory_before: (
            list[tuple[InMemoryRepository[Any], dict[str, Any]]] | None
        ) = None

    async def __aenter__(self) -> "UnitOfWork":
        """Open resources without implicitly changing existing autocommit behavior."""
        try:
            await self.connect()
        except BaseException:
            try:
                await self.close()
            except (Exception, asyncio.CancelledError) as cleanup_error:
                logger.warning(
                    "Database cleanup failed after connection error (%s)",
                    type(cleanup_error).__name__,
                )
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close owned connections after the resource scope exits."""
        try:
            await self.close()
        except Exception as cleanup_error:
            if exc_val is None:
                raise
            logger.warning(
                "Database cleanup failed while preserving the scope error (%s)",
                type(cleanup_error).__name__,
            )

    async def connect(self) -> None:
        """Initialize once; repeated calls preserve repository handles and records."""
        with self._owner.operation():
            if self._is_connected():
                return
        async with self._connect_lock:
            with self._owner.operation():
                if self._is_connected():
                    return
                if self.config.backend == DatabaseBackend.MEMORY:
                    self._agents = InMemoryRepository(Agent, _owner=self._owner)
                    self._conversations = InMemoryRepository(
                        Conversation, _owner=self._owner
                    )
                    self._training_jobs = InMemoryRepository(
                        TrainingJob, _owner=self._owner
                    )
                    self._api_keys = InMemoryRepository(APIKey, _owner=self._owner)
                elif self.config.backend == DatabaseBackend.SQLITE:
                    db_path = self.config.connection_url or "stateset.db"
                    if db_path.startswith("sqlite:///"):
                        db_path = db_path[10:]
                    if self._store is None:
                        self._store = _SQLiteStore(db_path, self._owner)
                    self._agents = SQLiteRepository(Agent, db_path, _store=self._store)
                    self._conversations = SQLiteRepository(
                        Conversation, db_path, _store=self._store
                    )
                    self._training_jobs = SQLiteRepository(
                        TrainingJob, db_path, _store=self._store
                    )
                    self._api_keys = SQLiteRepository(
                        APIKey, db_path, _store=self._store
                    )
                else:
                    raise NotImplementedError(
                        f"Database backend {self.config.backend.value} is not implemented"
                    )
            for repository in self._repositories():
                if isinstance(repository, SQLiteRepository):
                    await repository.connect()
            with self._owner.operation():
                self._connected = True
                self._owner.state = "ready"
        logger.info("Database initialized with %s backend", self.config.backend.value)

    def _is_connected(self) -> bool:
        return self._connected

    def _repositories(self) -> list[Repository[Any] | None]:
        return [self._agents, self._conversations, self._training_jobs, self._api_keys]

    def _begin(self) -> None:
        if self._store is not None:
            self._store.connect().execute("BEGIN IMMEDIATE").close()
        else:
            self._memory_before = [
                (repository, copy.deepcopy(repository._data))
                for repository in self._repositories()
                if isinstance(repository, InMemoryRepository)
            ]

    def _commit(self) -> None:
        if self._store is not None:
            self._store.connect().commit()
        self._memory_before = None

    def _rollback(self) -> bool:
        if self._store is not None:
            connection = self._store.connect()
            if not connection.in_transaction:
                return False
            connection.rollback()
            return True
        if self._memory_before is None:
            return False
        for repository, data in self._memory_before:
            repository._data = data
        self._memory_before = None
        return True

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["UnitOfWork"]:
        """Commit on success and roll back errors/cancellation across all repositories.

        A scope belongs to its creating task. Nested scopes and access by other
        tasks fail before effects; repository failures make it rollback-only.
        No failed or uncertain transaction is retried automatically.
        """
        await self.connect()
        async with self._owner.transaction(self._begin, self._commit, self._rollback):
            yield self

    def get_status(self) -> dict[str, Any]:
        """Inspect connection state and detached active/latest transaction receipts."""
        return self._owner.inspect()

    async def close(self) -> None:
        """Join one cleanup; cancellation cannot abandon remaining connections."""
        with self._owner.lock:
            if self._close_task is None:
                self._owner.close_admission()
                work = self._close_repositories()
                try:
                    self._close_task = asyncio.create_task(work)
                except BaseException:
                    work.close()
                    self._owner.state = "failed"
                    raise
            task = self._close_task
            if asyncio.current_task() is task:
                raise TransactionConflict("Database cleanup cannot wait for itself")

        async def join() -> None:
            await asyncio.shield(task)

        await drain_owned_operation(join())

    async def _close_repositories(self) -> None:
        failure: BaseException | None = None
        for repository in self._repositories():
            if (
                isinstance(repository, SQLiteRepository)
                and repository._store is self._store
            ):
                # These are views of one connection. Its uncertain close must
                # not be retried through each repository's alias.
                continue
            if repository is not None and hasattr(repository, "close"):
                try:
                    await cast(Any, repository).close()
                except (Exception, asyncio.CancelledError) as error:
                    if failure is None:
                        failure = error
        if self._store is not None and not self._store.closed:
            try:
                self._store.close()
            except Exception as error:
                if failure is None:
                    failure = error
        with self._owner.lock:
            self._owner.state = "failed" if failure is not None else "closed"
        if failure is not None:
            raise failure

    @property
    def agents(self) -> Repository[Agent]:
        """Get agents repository."""
        if not self._connected or self._agents is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._agents

    @property
    def conversations(self) -> Repository[Conversation]:
        """Get conversations repository."""
        if not self._connected or self._conversations is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conversations

    @property
    def training_jobs(self) -> Repository[TrainingJob]:
        """Get training jobs repository."""
        if not self._connected or self._training_jobs is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._training_jobs

    @property
    def api_keys(self) -> Repository[APIKey]:
        """Get API keys repository."""
        if not self._connected or self._api_keys is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._api_keys


# ============================================================================
# Global Database Instance
# ============================================================================

_database: UnitOfWork | None = None


def _set_database(database: UnitOfWork | None) -> None:
    global _database
    _database = database


async def _prepare_database(database: UnitOfWork) -> None:
    await database.connect()


async def _dispose_database(database: UnitOfWork) -> None:
    await database.close()


def _database_reusable(database: UnitOfWork, config: DatabaseConfig) -> bool:
    return database.config == config and database.get_status()["state"] not in {
        "closing",
        "closed",
        "failed",
    }


def _database_close_rejected(database: UnitOfWork, error: BaseException) -> bool:
    # The built-in close admission guard runs before creating any cleanup task.
    # A repository can also raise TransactionConflict after cleanup begins; that
    # remains an unconfirmed failure and must not be retried automatically.
    return (
        type(database) is UnitOfWork
        and isinstance(error, TransactionConflict)
        and database._close_task is None
    )


def _new_registry() -> SharedResource[DatabaseConfig, UnitOfWork]:
    return SharedResource(
        "database",
        lambda: _database,
        _set_database,
        lambda config: UnitOfWork(config),
        _prepare_database,
        _dispose_database,
        _database_reusable,
        _database_close_rejected,
    )


_database_registry = _new_registry()


async def init_database(config: DatabaseConfig | None = None) -> UnitOfWork:
    """Reuse matching configuration; replace only an unleased global database."""
    return await _database_registry.initialize(config or DatabaseConfig.from_env())


async def acquire_database(
    config: DatabaseConfig | None = None,
) -> ResourceLease[UnitOfWork]:
    """Lease repositories until the caller releases ownership with ``aclose()``."""
    return await _database_registry.acquire(config or DatabaseConfig.from_env())


def get_database() -> UnitOfWork | None:
    """Get the available global database, rejecting unconfirmed cleanup."""
    return _database_registry.get()


def get_database_status() -> dict[str, Any]:
    """Inspect shared database transitions and application ownership."""
    return _database_registry.status()


async def close_database() -> None:
    """Close once when no applications hold leases; preserve cleanup failures."""
    await _database_registry.close()


@asynccontextmanager
async def database_session() -> AsyncIterator[UnitOfWork]:
    """Borrow the global database; use its transaction() for atomic writes."""
    db = get_database()
    if db is None:
        db = await init_database()
    yield db  # The application's database owner is responsible for cleanup.
