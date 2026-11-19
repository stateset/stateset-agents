"""Named conversation turn ownership and staged context publication."""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..utils.async_calls import drain_owned_operation
from .multiturn_context import (
    ConversationClosedError,
    ConversationConflict,
    ConversationContext,
    context_data,
    publish_context,
)

if TYPE_CHECKING:
    from .multiturn_agent import MultiTurnAgent


@dataclass
class _Turn:
    agent: Any
    context: ConversationContext
    receipt: dict[str, Any]
    revision: int


_TURNS: ContextVar[tuple[_Turn, ...]] = ContextVar(
    "named_conversation_turns", default=()
)


def check_turn(agent: MultiTurnAgent) -> None:
    """Reject callbacks belonging to a reset, ended, replaced or finished turn."""
    agent._conversation_owner().check()
    for turn in reversed(_TURNS.get()):
        if turn.agent is agent:
            context, receipt = turn.context, turn.receipt
            owner = context._owner()
            with agent._conversation_owner().lock, owner.lock:
                if receipt["cancellation_requested"]:
                    raise asyncio.CancelledError
                if (
                    owner.state != "active"
                    or agent.active_conversations.get(context.conversation_id)
                    is not context
                    or owner.operation is not receipt
                ):
                    raise ConversationClosedError(
                        "Conversation ended, changed, or its turn finished"
                    )
            return


@contextmanager
def planning_commit(agent: MultiTurnAgent) -> Iterator[None]:
    """Fence short planning publications by the host's epoch and context revision."""
    with agent._conversation_owner().commit():
        turn = next(
            (item for item in reversed(_TURNS.get()) if item.agent is agent), None
        )
        if turn is None:
            yield
        else:
            with turn.context._owner().lock:
                check_turn(agent)
                if turn.context._owner().revision != turn.revision:
                    raise ConversationConflict("Conversation changed during planning")
                yield


def consume_planning_commands(
    agent: MultiTurnAgent, candidate: ConversationContext
) -> None:
    """Consume accepted one-shot planning intent before invoking a planner.

    A later failed generation must not leave the command queued for another turn.
    Plan/callback outcomes remain separately observable in the planning manager.
    """
    for turn in reversed(_TURNS.get()):
        if turn.agent is not agent:
            continue
        context = turn.context
        owner = context._owner()
        with agent._conversation_owner().lock, owner.lock:
            check_turn(agent)
            if (
                candidate.conversation_id != context.conversation_id
                or owner.revision != turn.revision
            ):
                raise ConversationConflict("Conversation changed before planning")
            changed = False
            for key in ("plan_update", "plan_goal"):
                if candidate.metadata.get(key) is not None and key in context.metadata:
                    context.metadata.pop(key)
                    changed = True
            if changed:
                owner.revision += 1
                turn.revision = owner.revision
        return


async def _stage(
    agent: MultiTurnAgent,
    receipt: dict[str, Any],
    phase: str,
    operation: Awaitable[Any],
    outcome: str | None = None,
) -> Any:
    owner = next(
        turn.context._owner()
        for turn in reversed(_TURNS.get())
        if turn.agent is agent and turn.receipt is receipt
    )

    async def perform() -> Any:
        check_turn(agent)
        with owner.lock:
            receipt["phase"] = phase
            if outcome is not None:
                receipt[outcome] = "unconfirmed"
        result = await operation
        with owner.lock:
            if outcome is not None:
                receipt[outcome] = "returned"
        return result

    try:
        work = perform()
        try:
            task = asyncio.create_task(work)
        except BaseException:
            work.close()
            raise
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with owner.lock:
                receipt["cancellation_requested"] = True
            task.cancel()

            async def join() -> None:
                try:
                    await task
                except BaseException:
                    pass  # Caller cancellation takes precedence over cleanup.

            try:
                await drain_owned_operation(join())
            except asyncio.CancelledError:
                pass
            raise
    finally:
        if inspect.iscoroutine(operation):
            operation.close()


async def run_turn(
    agent: MultiTurnAgent,
    conversation_id: str,
    user_message: str,
    strategy: str,
    context_update: dict[str, Any] | None,
    *,
    analyze: bool,
    use_tools: bool,
) -> ConversationContext:
    """Serialize one named conversation while allowing other IDs to progress."""
    if not isinstance(user_message, str):
        raise TypeError("user_message must be text")
    agent._ensure_backend_usable()
    with agent._conversation_owner().lock:
        context = agent.active_conversations.get(conversation_id)
        if context is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        owner = context._owner()
        with owner.lock:
            if owner.state != "active":
                raise ConversationClosedError("Conversation has ended")
            if owner.operation is not None:
                raise ConversationConflict(
                    "Another turn is active for this conversation"
                )
            candidate = ConversationContext(**context_data(context))
            revision = owner.revision
            receipt = {
                "id": uuid4().hex,
                "state": "running",
                "phase": "preparing",
                "strategy": strategy,
                "user_recorded": False,
                "generation_status": "not_started",
                "tool_status": "not_started",
                "error_type": None,
                "cancellation_requested": False,
            }
            owner.operation = receipt
    turn = _Turn(agent, context, receipt, revision)
    token = _TURNS.set((*_TURNS.get(), turn))
    try:
        agent._apply_context_update(candidate, context_update)
        if analyze and len(candidate.history) > agent.max_conversation_turns:
            await _stage(
                agent, receipt, "compressing", agent._compress_context(candidate)
            )
        candidate.add_turn({"role": "user", "content": user_message})
        accepted = context_data(candidate)
        with agent._conversation_owner().lock, owner.lock:
            check_turn(agent)
            if owner.revision != revision:
                raise ConversationConflict(
                    "Conversation changed before accepting the user message"
                )
            publish_context(context, accepted)
            turn.revision = owner.revision
            receipt["user_recorded"] = True
        if analyze:
            await _stage(
                agent,
                receipt,
                "analyzing",
                agent._analyze_user_input(candidate, user_message),
            )
        check_turn(agent)
        strategy_func = agent.strategies.get(strategy, agent._default_strategy)
        response = await _stage(
            agent, receipt, "generating", strategy_func(candidate), "generation_status"
        )
        check_turn(agent)
        if not isinstance(response, str):
            raise TypeError("Conversation strategy must return text")
        if use_tools:
            response = await _stage(
                agent,
                receipt,
                "tools",
                agent._apply_tools(candidate, response),
                "tool_status",
            )
        check_turn(agent)
        if not isinstance(response, str):
            raise TypeError("Conversation tools must return text")
        candidate.add_turn({"role": "assistant", "content": response})
        prepared = context_data(candidate)
        with agent._conversation_owner().lock, owner.lock:
            check_turn(agent)
            receipt["phase"] = "committing"
            if owner.revision != turn.revision:
                raise ConversationConflict("Conversation changed during generation")
            publish_context(context, prepared)
            receipt.update(state="completed", phase="finished")
        return candidate
    except BaseException as error:
        with owner.lock:
            receipt["state"] = (
                "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            )
            receipt["error_type"] = type(error).__name__
        raise
    finally:
        with owner.lock:
            owner.last_operation = copy.deepcopy(receipt)
            owner.operation = None
        _TURNS.reset(token)
