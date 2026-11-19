"""Versioned v1 conversation journal with durable receipts and process ownership.

Short SQLite transactions publish records. A separate advisory lock owns a
request through provider cleanup without keeping a database transaction open.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from stateset_agents.core.multiturn_context import ConversationConflict
from stateset_agents.utils.async_calls import drain_owned_operation
from stateset_agents.utils.snapshot_files import (
    canonical_json,
    open_regular,
    writer_lock,
)

from .conversation_lifecycle import ConversationOperation, GatewayConversations


class ConversationStoreUnavailable(RuntimeError):
    """Durable conversation state cannot currently be acknowledged."""


class StoredConversationNotFound(LookupError):
    """No accessible, unexpired conversation exists for this principal."""


_FORMAT = "stateset.gateway.conversation"
_RECEIPT_STATES = {"running", "completed", "failed", "cancelled", "interrupted"}
_SCOPES: ContextVar[tuple[ConversationStore, ...]] = ContextVar(
    "durable_conversation_scopes", default=()
)


def _validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {
        "messages",
        "user_id",
        "_lifecycle",
    }:
        raise ValueError("Invalid stored conversation shape")
    if not isinstance(record["user_id"], str) or not record["user_id"]:
        raise ValueError("Stored conversations require an explicit principal")
    messages = record["messages"]
    if not isinstance(messages, list) or any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError("Invalid stored conversation history")
    state = record["_lifecycle"]
    if not isinstance(state, dict) or set(state) - {
        "state",
        "revision",
        "active",
        "last_finished",
        "recovery_required",
    }:
        raise ValueError("Invalid stored lifecycle")
    if (
        state.get("state") not in {"active", "ended"}
        or type(state.get("revision")) is not int
        or state["revision"] < 0
    ):
        raise ValueError("Invalid stored lifecycle state/revision")
    if type(state.get("recovery_required", False)) is not bool:
        raise ValueError("Invalid recovery requirement")
    for name in ("active", "last_finished"):
        if name not in state:
            raise ValueError("Missing stored operation receipt")
        receipt = state[name]
        if receipt is None:
            continue
        if not isinstance(receipt, dict) or set(receipt) != {
            "id",
            "state",
            "phase",
            "user_recorded",
            "response_recorded",
            "generation_status",
            "cancellation_requested",
            "error_type",
            "work_error_type",
        }:
            raise ValueError("Invalid stored operation receipt")
        if (
            not isinstance(receipt["id"], str)
            or not receipt["id"]
            or receipt["state"] not in _RECEIPT_STATES
        ):
            raise ValueError("Invalid stored operation identity/state")
        if receipt["phase"] not in {
            "preparing",
            "generating",
            "committing",
            "finished",
        }:
            raise ValueError("Invalid stored operation phase")
        if receipt["generation_status"] not in {
            "not_started",
            "unconfirmed",
            "returned",
        }:
            raise ValueError("Invalid stored generation outcome")
        if any(
            type(receipt[key]) is not bool
            for key in ("user_recorded", "response_recorded", "cancellation_requested")
        ):
            raise ValueError("Invalid stored publication flags")
        if any(
            receipt[key] is not None and not isinstance(receipt[key], str)
            for key in ("error_type", "work_error_type")
        ):
            raise ValueError("Invalid stored error classification")
    return record


def _encode(record: dict[str, Any]) -> str:
    return canonical_json(
        {"format": _FORMAT, "version": 1, "record": _validate_record(record)}
    ).decode("utf-8")


def _decode(raw: str, principal: str) -> dict[str, Any]:
    try:
        envelope = json.loads(raw)
        if not isinstance(envelope, dict) or set(envelope) != {
            "format",
            "version",
            "record",
        }:
            raise ValueError("Invalid conversation envelope")
        if (
            envelope["format"] != _FORMAT
            or type(envelope["version"]) is not int
            or envelope["version"] != 1
        ):
            raise ValueError("Unsupported conversation format")
        record = _validate_record(envelope["record"])
        if record["user_id"] != principal:
            raise ValueError("Stored principal does not match the index")
        # Reject non-finite or otherwise noncanonical data, including nested fields.
        canonical_json(record)
        return record
    except (ValueError, TypeError, KeyError) as error:
        raise ConversationStoreUnavailable(
            "Stored conversation requires repair"
        ) from error


class ConversationStore:
    """Own durable v1 histories on a shared local SQLite file and lock directory."""

    @property
    def closed(self) -> bool:
        """Whether this owner has finished draining its requests."""
        return self._closed

    def __init__(
        self, path: str | Path, *, ttl_seconds: int = 86400, max_items: int = 500
    ) -> None:
        requested = Path(path).expanduser()
        if str(path) == ":memory:" or requested.is_symlink():
            raise ValueError("Durable conversation storage requires a regular file")
        self.path = requested.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = open_regular(self.path, os.O_RDWR | os.O_CREAT)
        os.close(descriptor)
        self.lock_directory = self.path.with_name(
            f".{self.path.name}.conversation-locks"
        )
        if self.lock_directory.is_symlink():
            raise ValueError("Conversation lock directory must not be a symlink")
        self.lock_directory.mkdir(mode=0o700, exist_ok=True)
        if ttl_seconds < 1 or max_items < 1:
            raise ValueError("Conversation retention limits must be positive")
        self.ttl_seconds, self.max_items = ttl_seconds, max_items
        self.owner = GatewayConversations()
        self._closing = False
        self._closed = False
        self._tasks: dict[asyncio.Task[Any], int] = {}
        self._close_task: asyncio.Task[None] | None = None
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS gateway_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO gateway_metadata VALUES ('schema_version', '1')"
            )
            if connection.execute(
                "SELECT value FROM gateway_metadata WHERE key = 'schema_version'"
            ).fetchone() != ("1",):
                raise ConversationStoreUnavailable(
                    "Unsupported gateway database schema"
                )
            connection.execute("""CREATE TABLE IF NOT EXISTS gateway_conversations (
                id TEXT PRIMARY KEY, principal TEXT NOT NULL, revision INTEGER NOT NULL,
                accessed REAL NOT NULL, data TEXT NOT NULL
            )""")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS gateway_conversations_accessed ON gateway_conversations(accessed)"
            )

    @contextmanager
    def _transaction(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise ConversationStoreUnavailable("Conversation storage is closed")
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path, timeout=0, isolation_level=None)
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
        except sqlite3.Error as error:
            raise ConversationStoreUnavailable(
                "Conversation storage is unavailable"
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def _lock_path(self, identifier: str) -> Path:
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        return self.lock_directory / digest

    @contextmanager
    def _claim(self, identifier: str) -> Iterator[None]:
        with ExitStack() as locks:
            try:
                locks.enter_context(
                    writer_lock(self._lock_path(identifier), ConversationConflict)
                )
            except ConversationConflict as error:
                raise ConversationConflict(
                    "Another request owns this conversation; inspect its status"
                ) from error
            yield

    def _load(
        self, connection: sqlite3.Connection, identifier: str, principal: str
    ) -> tuple[int, dict[str, Any]]:
        row = connection.execute(
            "SELECT revision, data FROM gateway_conversations WHERE id = ? AND principal = ? AND accessed >= ?",
            (identifier, principal, time.time() - self.ttl_seconds - 1),
        ).fetchone()
        if row is None:
            raise StoredConversationNotFound("Conversation not found")
        return row[0], _decode(row[1], principal)

    def _write(
        self,
        connection: sqlite3.Connection,
        identifier: str,
        principal: str,
        revision: int,
        record: dict[str, Any],
    ) -> int:
        cursor = connection.execute(
            "UPDATE gateway_conversations SET data = ?, revision = revision + 1 WHERE id = ? AND principal = ? AND revision = ?",
            (_encode(record), identifier, principal, revision),
        )
        if cursor.rowcount != 1:
            raise ConversationConflict("Conversation changed during this request")
        return revision + 1

    def _recover(
        self,
        connection: sqlite3.Connection,
        identifier: str,
        principal: str,
        revision: int,
        record: dict[str, Any],
    ) -> int:
        state = record["_lifecycle"]
        receipt = state["active"]
        if receipt is not None:
            receipt.update(state="interrupted", phase="finished")
            state["active"] = None
            state["last_finished"] = receipt
            state["recovery_required"] = (
                receipt["generation_status"] != "not_started"
                and not receipt["response_recorded"]
            )
            state["revision"] += 1
            revision = self._write(connection, identifier, principal, revision, record)
        return revision

    def read(self, identifier: str, principal: str) -> dict[str, Any]:
        """Read an authorized record; settle abandoned ownership without replay."""
        with self._transaction() as connection:
            _, record = self._load(connection, identifier, principal)
            connection.execute(
                "UPDATE gateway_conversations SET accessed = ? WHERE id = ?",
                (time.time(), identifier),
            )
        if record["_lifecycle"]["active"] is not None:
            with ExitStack() as locks:
                try:
                    locks.enter_context(self._claim(identifier))
                except ConversationConflict:
                    return record  # A live process still owns this request.
                with self._transaction() as connection:
                    revision, record = self._load(connection, identifier, principal)
                    self._recover(connection, identifier, principal, revision, record)
        return record

    @contextmanager
    def operation(
        self, identifier: str, principal: str, current: Callable[[], bool]
    ) -> Iterator[ConversationOperation]:
        """Own one request across workers, journaling before provider dispatch."""
        if self._closing or self._closed:
            raise ConversationStoreUnavailable("Conversation storage is closing")
        if not principal:
            raise ValueError("Durable conversations require a principal")
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Conversation operations require an async task")
        # Authorize before exposing whether another process owns the identifier.
        with self._transaction(write=False) as connection:
            existing = connection.execute(
                "SELECT principal, accessed FROM gateway_conversations WHERE id = ?",
                (identifier,),
            ).fetchone()
            if (
                existing
                and existing[1] >= time.time() - self.ttl_seconds - 1
                and existing[0] != principal
            ):
                raise StoredConversationNotFound("Conversation not found")
        with self._claim(identifier):
            with self._transaction() as connection:
                connection.execute(
                    "DELETE FROM gateway_conversations WHERE accessed < ?",
                    (time.time() - self.ttl_seconds - 1,),
                )
                found = connection.execute(
                    "SELECT principal FROM gateway_conversations WHERE id = ?",
                    (identifier,),
                ).fetchone()
                if found is None:
                    record: dict[str, Any] = {"messages": [], "user_id": principal}
                    self.owner.inspect(record)
                    record["_lifecycle"]["recovery_required"] = False
                    connection.execute(
                        "INSERT INTO gateway_conversations VALUES (?, ?, 1, ?, ?)",
                        (identifier, principal, time.time(), _encode(record)),
                    )
                elif found[0] != principal:
                    raise StoredConversationNotFound("Conversation not found")
                revision, record = self._load(connection, identifier, principal)
                revision = self._recover(
                    connection, identifier, principal, revision, record
                )
                connection.execute(
                    "UPDATE gateway_conversations SET accessed = ? WHERE id = ?",
                    (time.time(), identifier),
                )
                connection.execute(
                    """DELETE FROM gateway_conversations WHERE id IN (
                    SELECT id FROM gateway_conversations WHERE id != ? ORDER BY accessed DESC, id LIMIT -1 OFFSET ?
                )""",
                    (identifier, self.max_items - 1),
                )
            if record["_lifecycle"].get("recovery_required"):
                raise ConversationConflict(
                    "Interrupted generation requires acknowledgement; inspect conversation status"
                )
            poisoned = False

            def is_current() -> bool:
                if poisoned or self._closed or not current():
                    return False
                with self._transaction(write=False) as connection:
                    row = connection.execute(
                        "SELECT revision FROM gateway_conversations WHERE id = ? AND principal = ? AND accessed >= ?",
                        (identifier, principal, time.time() - self.ttl_seconds - 1),
                    ).fetchone()
                    return bool(row == (revision,))

            def persist(candidate: dict[str, Any]) -> None:
                nonlocal revision, poisoned
                if poisoned:
                    raise ConversationStoreUnavailable(
                        "Conversation publication is unconfirmed"
                    )
                try:
                    with self._transaction() as connection:
                        next_revision = self._write(
                            connection, identifier, principal, revision, candidate
                        )
                        receipt = (
                            candidate["_lifecycle"]["active"]
                            or candidate["_lifecycle"]["last_finished"]
                        )
                        if receipt and receipt["response_recorded"]:
                            connection.execute(
                                "UPDATE gateway_conversations SET accessed = ? WHERE id = ?",
                                (time.time(), identifier),
                            )
                    revision = next_revision
                except BaseException:
                    poisoned = True
                    raise

            self._tasks[task] = self._tasks.get(task, 0) + 1
            token = _SCOPES.set((*_SCOPES.get(), self))
            try:
                with self.owner.operation(
                    (identifier,), record, is_current, persist=persist
                ) as operation:
                    yield operation
            finally:
                _SCOPES.reset(token)
                self._tasks[task] -= 1
                if not self._tasks[task]:
                    del self._tasks[task]

    def acknowledge(
        self, identifier: str, principal: str, operation_id: str
    ) -> dict[str, Any]:
        """Acknowledge one interrupted outcome without resubmitting any work."""
        self.read(identifier, principal)
        with self._claim(identifier):
            with self._transaction() as connection:
                revision, record = self._load(connection, identifier, principal)
                state = record["_lifecycle"]
                receipt = state["last_finished"]
                if (
                    state["active"] is not None
                    or receipt is None
                    or receipt["id"] != operation_id
                    or receipt["state"] != "interrupted"
                ):
                    raise ConversationConflict(
                        "The interrupted operation changed; inspect status again"
                    )
                if state.get("recovery_required"):
                    state["recovery_required"] = False
                    state["revision"] += 1
                    self._write(connection, identifier, principal, revision, record)
                return copy.deepcopy(state)

    def delete(self, identifier: str, principal: str) -> None:
        """Delete durably; old request revisions cannot recreate this record."""
        with self._transaction() as connection:
            self._load(connection, identifier, principal)
            connection.execute(
                "DELETE FROM gateway_conversations WHERE id = ? AND principal = ?",
                (identifier, principal),
            )

    async def aclose(self) -> None:
        """Stop admission, cancel requests once, and drain cleanup before closing."""
        if self in _SCOPES.get() or asyncio.current_task() in self._tasks:
            raise ConversationConflict(
                "An active conversation cannot close its own store"
            )
        if self._close_task is None:
            self._closing = True

            async def close() -> None:
                pending = list(self._tasks)
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                self._closed = True

            self._close_task = asyncio.create_task(close())
        task = self._close_task

        async def join() -> None:
            await asyncio.shield(task)

        await drain_owned_operation(join())
