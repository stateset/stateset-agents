"""Task-owned local transactions with observable commit and rollback outcomes."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock, get_ident
from typing import Any
from uuid import uuid4


class TransactionConflict(RuntimeError):
    """An operation cannot share, nest, or outlive another task's transaction."""


class TransactionAborted(RuntimeError):
    """A failed repository operation made the transaction rollback-only."""


@dataclass
class _Transaction:
    task: asyncio.Task[Any] | None
    thread: int = field(default_factory=get_ident)
    rollback_only: bool = False
    receipt: dict[str, Any] = field(
        default_factory=lambda: {
            "id": uuid4().hex,
            "state": "active",
            "phase": "beginning",
            "commit_status": "not_started",
            "rollback_status": "not_started",
            "error_type": None,
            "operation_error_type": None,
            "rollback_error_type": None,
        }
    )


_SCOPES: ContextVar[tuple[tuple[TransactionOwner, _Transaction], ...]] = ContextVar(
    "persistence_transactions", default=()
)


def _current_task() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


class TransactionOwner:
    """Serialize short effects and fence an explicit transaction to its task."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.state = "new"
        self.active: _Transaction | None = None
        self.last_transaction: dict[str, Any] | None = None

    def _check(self) -> _Transaction | None:
        if self.state in {"closing", "closed", "failed"}:
            raise RuntimeError("Database is closed; create a new UnitOfWork")
        scope = next(
            (scope for owner, scope in reversed(_SCOPES.get()) if owner is self),
            None,
        )
        if scope is not None and (
            self.active is not scope
            or scope.task is not _current_task()
            or scope.thread != get_ident()
        ):
            raise TransactionConflict("Transaction belongs to another or finished task")
        if self.active is not None and self.active is not scope:
            raise TransactionConflict("Another task owns this database transaction")
        if scope is not None and scope.rollback_only:
            raise TransactionAborted(
                "Transaction must roll back after an operation failure"
            )
        return scope

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Guard a synchronous repository effect; never await while holding this."""
        with self.lock:
            scope = self._check()
            try:
                yield
            except BaseException as error:
                if scope is not None:
                    scope.rollback_only = True
                    scope.receipt["operation_error_type"] = type(error).__name__
                raise

    @asynccontextmanager
    async def transaction(
        self,
        begin: Callable[[], None],
        commit: Callable[[], None],
        rollback: Callable[[], bool],
    ) -> AsyncIterator[None]:
        """Commit on successful exit; retain failure facts without replaying work."""
        with self.lock:
            self._check()
            if self.active is not None:
                raise TransactionConflict("Nested transactions are not supported")
            scope = _Transaction(_current_task())
            self.active = scope
        token = _SCOPES.set((*_SCOPES.get(), (self, scope)))
        try:
            with self.lock:
                begin()
                scope.receipt["phase"] = "working"
            yield
            with self.lock:
                self._check()
                scope.receipt.update(phase="committing", commit_status="unconfirmed")
                commit()
                scope.receipt.update(state="committed", commit_status="acknowledged")
        except BaseException as error:
            with self.lock:
                scope.receipt.update(
                    phase="rolling_back",
                    error_type=type(error).__name__,
                    rollback_status="unconfirmed",
                )
                try:
                    rolled_back = rollback()
                except BaseException as cleanup_error:
                    scope.receipt.update(
                        state="failed", rollback_error_type=type(cleanup_error).__name__
                    )
                    self.state = "failed"
                    # Preserve the body/commit error; expose cleanup uncertainty.
                else:
                    scope.receipt.update(
                        state="rolled_back" if rolled_back else "failed",
                        rollback_status="acknowledged" if rolled_back else "not_active",
                    )
                    if (
                        not rolled_back
                        and scope.receipt["commit_status"] == "unconfirmed"
                    ):
                        self.state = "failed"
            raise
        finally:
            with self.lock:
                scope.receipt["phase"] = "finished"
                self.last_transaction = copy.deepcopy(scope.receipt)
                if self.active is scope:
                    self.active = None
            _SCOPES.reset(token)

    def close_admission(self) -> None:
        """Reject close during a transaction instead of abandoning its writes."""
        with self.lock:
            if self.active is not None:
                raise TransactionConflict(
                    "Finish the active transaction before closing"
                )
            if self.state in {"new", "ready"}:
                self._check()
            if self.state not in {"closed", "closing"}:
                self.state = "closing"

    def inspect(self) -> dict[str, Any]:
        """Return detached lifecycle receipts without entity data or error text."""
        with self.lock:
            return copy.deepcopy(
                {
                    "state": self.state,
                    "active_transaction": self.active.receipt if self.active else None,
                    "last_transaction": self.last_transaction,
                }
            )
