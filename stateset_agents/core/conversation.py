"""Conversation generations fence work that outlives a reset."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import Any, Concatenate, ParamSpec, TypeVar, cast
from uuid import uuid4

from ..utils.async_calls import closing_async_generator

_P = ParamSpec("_P")
_T = TypeVar("_T")
_SCOPES: ContextVar[tuple[tuple[_ConversationState, _Attempt], ...]] = ContextVar(
    "conversation_operations", default=()
)
_HOST_GUARDS: ContextVar[
    tuple[tuple[_ConversationState, Callable[[], AbstractContextManager[Any]]], ...]
] = ContextVar("conversation_host_guards", default=())


class ConversationResetError(RuntimeError):
    """Work belongs to a retired conversation and cannot record a new result."""


@dataclass
class _Attempt:
    mode: str
    epoch: int
    id: str = field(default_factory=lambda: uuid4().hex)
    state: str = "running"
    error_type: str | None = None
    cancellation_requested: bool = False


class _ConversationState:
    def __init__(self) -> None:
        self.epoch = 0
        self.active: dict[str, _Attempt] = {}
        self.last_finished: dict[str, Any] | None = None
        self.lock = RLock()

    def __getstate__(self) -> dict[str, Any]:
        with self.lock:
            if self.active:
                raise RuntimeError(
                    "Cannot copy an agent during a conversation operation"
                )
            return {key: value for key, value in vars(self).items() if key != "lock"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = RLock()

    def _current(self) -> _Attempt | None:
        return next(
            (attempt for owner, attempt in reversed(_SCOPES.get()) if owner is self),
            None,
        )

    def check(self, epoch: int | None = None) -> None:
        with self.lock, self._host_scope():
            attempt = self._current()
            if attempt is not None and attempt.cancellation_requested:
                raise asyncio.CancelledError
            if epoch is not None and epoch != self.epoch:
                raise ConversationResetError(
                    "Conversation was reset; previous work cannot be reused"
                )
            if attempt is not None and (
                attempt.epoch != self.epoch
                or self.active.get(attempt.id) is not attempt
            ):
                raise ConversationResetError(
                    "Conversation operation was reset or has already finished"
                )

    @contextmanager
    def commit(self) -> Iterator[None]:
        """Guard a short local mutation; never await or invoke user hooks here."""
        with self.lock, self._host_scope():
            self.check()
            yield

    @contextmanager
    def _host_scope(self) -> Iterator[None]:
        with ExitStack() as stack:
            for owner, guard in _HOST_GUARDS.get():
                if owner is self:
                    stack.enter_context(guard())
            yield

    def reset(self, agent: Any) -> None:
        with self.lock:
            # Reset is an explicit control action even when invoked by old work.
            self.epoch += 1
            agent.conversation_history = []
            agent._last_reasoning = None
            if "turn_count" in vars(agent):
                agent.turn_count = 0

    def _snapshot(self, attempt: _Attempt) -> dict[str, Any]:
        return {**vars(attempt), "invalidated": attempt.epoch != self.epoch}

    def inspect(self) -> dict[str, Any]:
        with self.lock:
            last = (
                {
                    **self.last_finished,
                    "invalidated": self.last_finished["epoch"] != self.epoch,
                }
                if self.last_finished is not None
                else None
            )
            return copy.deepcopy(
                {
                    "epoch": self.epoch,
                    "active": [
                        self._snapshot(attempt) for attempt in self.active.values()
                    ],
                    "last_finished": last,
                }
            )

    @contextmanager
    def operation(self, mode: str) -> Iterator[None]:
        with self.lock:
            self.check()
            inherited = self._current()
            attempt = inherited or _Attempt(mode, self.epoch)
            if inherited is None:
                self.active[attempt.id] = attempt
        if inherited is not None:
            yield
            self.check()
            return
        token = _SCOPES.set((*_SCOPES.get(), (self, attempt)))
        try:
            yield
            self.check()
        except BaseException as error:
            attempt.error_type = type(error).__name__
            attempt.state = (
                "cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "closed" if isinstance(error, GeneratorExit) else "failed"
            )
            raise
        else:
            attempt.state = "completed"
        finally:
            with self.lock:
                self.active.pop(attempt.id, None)
                self.last_finished = self._snapshot(attempt)
            _SCOPES.reset(token)


def _owner(agent: Any) -> _ConversationState | None:
    # Standalone mixins keep working without an Agent lifecycle owner.
    factory = getattr(agent, "_conversation_owner", None)
    return factory() if callable(factory) else None


def check_conversation(agent: Any) -> None:
    """Fence a callback boundary when a mixin has a conversation owner."""
    owner = _owner(agent)
    if owner is not None:
        owner.check()


@contextmanager
def bind_conversation_guard(
    agent: Any, guard: Callable[[], AbstractContextManager[Any]]
) -> Iterator[None]:
    """Bind a host record's guard to nested agent work without changing its APIs.

    The host guard must not acquire an agent lock: ordering is agent, host, then
    planning. Host retirement must release its lock before calling agent methods.
    """
    owner = _owner(agent)
    if owner is None:
        yield
        return
    token = _HOST_GUARDS.set((*_HOST_GUARDS.get(), (owner, guard)))
    try:
        yield
    finally:
        _HOST_GUARDS.reset(token)


def conversation_call(
    function: Callable[Concatenate[Any, _P], Coroutine[Any, Any, _T]],
) -> Callable[Concatenate[Any, _P], Coroutine[Any, Any, _T]]:
    """Keep nested preparation, generation and tool calls in one conversation."""

    @wraps(function)
    async def call(agent: Any, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        owner = _owner(agent)
        with owner.operation(function.__name__) if owner is not None else nullcontext():
            return await function(agent, *args, **kwargs)

    return call


def conversation_stream(
    function: Callable[Concatenate[Any, _P], AsyncIterator[str]],
) -> Callable[Concatenate[Any, _P], AsyncGenerator[str, None]]:
    """Own streaming context through cleanup without leaking it into the consumer."""

    @wraps(function)
    async def stream(
        agent: Any, *args: _P.args, **kwargs: _P.kwargs
    ) -> AsyncGenerator[str, None]:
        owner = _owner(agent)
        epoch = owner.epoch if owner is not None else None
        attempt: _Attempt | None = None

        async def drive() -> AsyncGenerator[str, None]:
            nonlocal attempt
            if owner is not None:
                owner.check(epoch)
            with (
                owner.operation(function.__name__)
                if owner is not None
                else nullcontext()
            ):
                attempt = owner._current() if owner is not None else None
                source = cast(
                    AsyncGenerator[str, None], function(agent, *args, **kwargs)
                )
                async with closing_async_generator(
                    source, forward_errors=True
                ) as chunks:
                    async for chunk in chunks:
                        if owner is not None:
                            owner.check()
                        yield chunk

        async with closing_async_generator(drive(), forward_errors=True) as chunks:
            try:
                async for chunk in chunks:
                    if owner is not None:
                        owner.check(epoch)
                    yield chunk
            except asyncio.CancelledError:
                if owner is not None and attempt is not None:
                    with owner.lock:
                        attempt.cancellation_requested = True
                raise

    return stream
