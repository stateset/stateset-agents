"""Gateway request ownership independent of provider completion and record lifetime."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Any, TypeVar
from uuid import uuid4

from stateset_agents.core.multiturn_context import (
    ConversationClosedError,
    ConversationConflict,
)
from stateset_agents.utils.async_calls import drain_owned_operation

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


class GatewayConversations:
    """Own active requests; settled JSON receipts live with their conversation."""

    def __init__(self) -> None:
        self.lock = RLock()
        self._active: dict[tuple[str, ...], ConversationOperation] = {}

    def __getstate__(self) -> dict[str, Any]:
        with self.lock:
            if self._active:
                raise RuntimeError("Cannot copy a gateway with active conversations")
            return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.lock = RLock()
        self._active = {}

    def _state(self, record: dict[str, Any]) -> dict[str, Any]:
        if "_lifecycle" not in record:
            record["_lifecycle"] = {
                "state": "active",
                "revision": 0,
                "active": None,
                "last_finished": None,
            }
        state: dict[str, Any] = record["_lifecycle"]
        return state

    def inspect(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return detached content-free receipts for an authorized record."""
        with self.lock:
            return copy.deepcopy(self._state(record))

    def retire(self, record: dict[str, Any]) -> None:
        """Retire a record without releasing an in-flight request's ownership."""
        with self.lock:
            state = self._state(record)
            if state["state"] == "active":
                state["state"] = "ended"
                state["revision"] += 1

    @contextmanager
    def operation(
        self,
        key: tuple[str, ...],
        record: dict[str, Any],
        current: Callable[[], bool],
        *,
        persist: Callable[[dict[str, Any]], None] | None = None,
    ) -> Iterator[ConversationOperation]:
        """Admit one request per key and retain its outcome through cleanup."""
        with self.lock:
            if key in self._active:
                raise ConversationConflict(
                    "Another request owns this conversation; inspect its status"
                )
            state = self._state(record)
            if state.get("active") is not None:
                raise ConversationConflict(
                    "This conversation has unfinished request ownership"
                )
            if state["state"] != "active" or not current():
                raise ConversationClosedError("Conversation is no longer active")
            operation = ConversationOperation(
                self, key, record, current, persist=persist
            )
            self._active[key] = operation
            state["active"] = operation.receipt
        failure: BaseException | None = None
        try:
            operation.checkpoint()
            yield operation
            operation.check()
        except BaseException as error:
            failure = error
            with self.lock:
                operation.receipt.update(
                    state=(
                        "cancelled"
                        if isinstance(error, asyncio.CancelledError)
                        else "failed"
                    ),
                    error_type=type(error).__name__,
                )
            raise
        else:
            with self.lock:
                operation.receipt.update(state="completed", phase="finished")
        finally:
            with self.lock:
                try:
                    if state.get("active") is operation.receipt:
                        state["last_finished"] = copy.deepcopy(operation.receipt)
                        state["active"] = None
                        try:
                            operation.checkpoint()
                        except Exception as persistence_error:
                            if failure is None:
                                raise
                            logger.warning(
                                "Conversation outcome publication failed (%s)",
                                type(persistence_error).__name__,
                            )
                finally:
                    if self._active.get(key) is operation:
                        self._active.pop(key)


class ConversationOperation:
    """One gateway request with detached input and retained callback outcome."""

    def __init__(
        self,
        owner: GatewayConversations,
        key: tuple[str, ...],
        record: dict[str, Any],
        current: Callable[[], bool],
        *,
        persist: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.owner, self.key, self.record, self.current = owner, key, record, current
        self._persist = persist
        self.messages = copy.deepcopy(record.get("messages", []))
        self.receipt: dict[str, Any] = {
            "id": uuid4().hex,
            "state": "running",
            "phase": "preparing",
            "user_recorded": False,
            "response_recorded": False,
            "generation_status": "not_started",
            "cancellation_requested": False,
            "error_type": None,
            "work_error_type": None,
        }

    def checkpoint(self) -> None:
        """Acknowledge durable state before crossing the next effect boundary."""
        if self._persist is not None:
            with self.owner.lock:
                if self.owner._active.get(self.key) is not self:
                    raise ConversationClosedError(
                        "This conversation request has finished"
                    )
                self._persist(self.record)

    def check(self) -> None:
        """Reject retired/replaced records, superseded work and conflicting edits."""
        with self.owner.lock:
            if self.receipt["cancellation_requested"]:
                raise asyncio.CancelledError
            if (
                self.owner._active.get(self.key) is not self
                or self.owner._state(self.record).get("active") is not self.receipt
            ):
                raise ConversationClosedError(
                    "This conversation request has already finished"
                )
            if (
                self.owner._state(self.record)["state"] != "active"
                or not self.current()
            ):
                self.owner.retire(self.record)
                raise ConversationClosedError(
                    "Conversation ended, expired, or changed during this request"
                )
            if self.record.get("messages", []) != self.messages:
                raise ConversationConflict(
                    "Conversation history changed during this request"
                )

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Guard a short publication; never call an agent while holding this lock."""
        with self.owner.lock:
            self.check()
            yield

    def publish(self, fields: dict[str, Any], *, response: bool = False) -> None:
        """Publish already prepared response data without exposing nested aliases."""
        prepared = copy.deepcopy(fields)
        baseline = copy.deepcopy(prepared.get("messages", self.messages))
        with self.guard():
            self.receipt["phase"] = "committing" if response else "preparing"
            self.record.update(prepared)
            self.messages = baseline
            self.owner._state(self.record)["revision"] += 1
            self.receipt["user_recorded"] = True
            self.receipt["response_recorded"] = response
            self.checkpoint()

    async def run(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Cancel a callback once, fence later work, and drain repeated cancellation."""

        async def perform() -> _T:
            with self.guard():
                self.receipt.update(phase="generating", generation_status="unconfirmed")
                self.checkpoint()
            try:
                result = await operation()
            except BaseException as error:
                with self.owner.lock:
                    self.receipt["work_error_type"] = type(error).__name__
                raise
            with self.owner.lock:
                self.receipt["generation_status"] = "returned"
                self.checkpoint()
            return result

        work = perform()
        try:
            task = asyncio.create_task(work)
        except BaseException:
            work.close()
            raise
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            with self.owner.lock:
                self.receipt["cancellation_requested"] = True
                try:
                    self.checkpoint()
                except Exception as persistence_error:
                    logger.warning(
                        "Conversation cancellation publication failed (%s)",
                        type(persistence_error).__name__,
                    )
            task.cancel()

            async def join() -> None:
                try:
                    await task
                except BaseException:
                    pass

            try:
                await drain_owned_operation(join())
            except asyncio.CancelledError:
                pass
            raise
        self.check()
        return result
