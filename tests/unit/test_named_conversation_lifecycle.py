"""Named turns retain ownership through cancellation, end and context conflicts."""

import asyncio
import copy
import pickle

import pytest

from stateset_agents.core.conversation import ConversationResetError
from stateset_agents.core.multiturn_agent import (
    ConversationClosedError,
    ConversationConflict,
    MultiTurnAgent,
)


def make_agent():
    return MultiTurnAgent({}, response_backend=lambda prompt: "answer")


@pytest.mark.asyncio
async def test_same_id_rejects_overlap_while_other_conversation_progresses():
    agent = make_agent()
    first, second = await agent.start_conversation(), await agent.start_conversation()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(context):
        entered.set()
        await release.wait()
        return "first answer"

    agent.strategies["blocked"] = blocked
    task = asyncio.create_task(
        agent.continue_conversation(first.conversation_id, "first", strategy="blocked")
    )
    try:
        await entered.wait()
        with pytest.raises(ConversationConflict, match="Another turn"):
            await agent.continue_conversation(first.conversation_id, "overlap")
        other = await agent.continue_conversation(second.conversation_id, "second")
        assert other[-1]["content"] == "answer"
        assert [turn["content"] for turn in first.history] == ["first"]
        status = agent.get_conversation_status(first.conversation_id)
        assert status["active"]["generation_status"] == "unconfirmed"
        status["active"]["state"] = "mutated"
        assert first.get_status()["active"]["state"] == "running"
    finally:
        release.set()
        await task
    assert first.get_status()["last_finished"]["state"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("retire", ["end", "reset"])
async def test_retired_backend_cannot_fallback_use_tools_or_commit(retire):
    agent = make_agent()
    context = await agent.start_conversation()
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    async def backend(prompt):
        entered.set()
        await release.wait()
        return "search for reference"

    async def search(query):
        effects.append(query)
        return "result"

    agent.response_backend = backend
    agent.register_tool("search", search)
    task = asyncio.create_task(
        agent.generate_multiturn_response(context.conversation_id, "look up reference")
    )
    await entered.wait()
    if retire == "end":
        assert agent.end_conversation(context.conversation_id) is context
        error = ConversationClosedError
    else:
        await agent.reset()
        error = ConversationResetError
    release.set()
    with pytest.raises(error):
        await task
    assert effects == []
    assert [turn["role"] for turn in context.history] == ["user"]
    assert context.get_status()["last_finished"]["state"] == "failed"
    if retire == "end":
        with pytest.raises(ConversationClosedError):
            context.add_turn({"role": "assistant", "content": "late"})


@pytest.mark.asyncio
async def test_managed_context_mutation_conflicts_with_pending_candidate():
    agent = make_agent()
    context = await agent.start_conversation(initial_context={"topic": "original"})
    entered, release = asyncio.Event(), asyncio.Event()

    async def strategy(candidate):
        candidate.topic = "uncommitted"
        entered.set()
        await release.wait()
        return "stale answer"

    agent.strategies["test"] = strategy
    task = asyncio.create_task(
        agent.continue_conversation(
            context.conversation_id, "question", strategy="test"
        )
    )
    await entered.wait()
    context.add_turn({"role": "system", "content": "external update"})
    release.set()
    with pytest.raises(ConversationConflict, match="changed during generation"):
        await task
    assert context.topic == "original"
    assert [turn["content"] for turn in context.history] == [
        "question",
        "external update",
    ]


@pytest.mark.asyncio
async def test_failed_strategy_preserves_accepted_user_but_not_candidate_mutations():
    agent = make_agent()
    context = await agent.start_conversation()
    update = {"entities": {"nested": ["accepted"]}}

    async def fail(candidate):
        candidate.entities["nested"].append("uncommitted")
        candidate.topic = "uncommitted"
        raise LookupError("strategy failed")

    agent.strategies["fail"] = fail
    with pytest.raises(LookupError):
        await agent.continue_conversation(
            context.conversation_id, "question", strategy="fail", context_update=update
        )
    update["entities"]["nested"].append("caller mutation")
    assert context.entities == {"nested": ["accepted"]}
    assert context.topic is None
    assert [turn["role"] for turn in context.history] == ["user"]
    receipt = context.get_status()["last_finished"]
    assert receipt["user_recorded"] is True
    assert receipt["generation_status"] == "unconfirmed"
    assert receipt["error_type"] == "LookupError"


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_cleanup_and_fences_nested_generation():
    agent = make_agent()
    context = await agent.start_conversation()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def strategy(candidate):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            with pytest.raises(asyncio.CancelledError):
                await agent.generate_response("must not reach backend")
            return "suppressed cancellation"

    agent.response_backend = lambda prompt: calls.append(prompt) or "late"
    agent.strategies["test"] = strategy
    task = asyncio.create_task(
        agent.continue_conversation(
            context.conversation_id, "question", strategy="test"
        )
    )
    try:
        await entered.wait()
        task.cancel()
        await cleaning.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(ConversationConflict):
            await agent.continue_conversation(context.conversation_id, "overlap")
        for value in (context, agent):
            with pytest.raises(RuntimeError, match="Cannot copy"):
                copy.deepcopy(value)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == []
    assert len(context.history) == 1
    receipt = context.get_status()["last_finished"]
    assert receipt["state"] == "cancelled"
    assert receipt["generation_status"] == "returned"
    assert receipt["cancellation_requested"] is True
    assert context.get_status()["active"] is None


@pytest.mark.asyncio
async def test_future_backend_detached_results_and_idle_context_copy():
    agent = make_agent()
    context = await agent.start_conversation()

    def backend(prompt):
        future = asyncio.get_running_loop().create_future()
        future.set_result("future answer")
        return future

    agent.response_backend = backend
    result = await agent.continue_conversation(context.conversation_id, "question")
    assert result[-1]["content"] == "future answer"
    result[-1]["content"] = "caller mutation"
    for clone in (copy.deepcopy(context), pickle.loads(pickle.dumps(context))):
        clone.add_turn({"role": "user", "content": "clone only"})
        assert len(clone.history) == 3
    assert len(context.history) == 2
    assert context.history[-1]["content"] == "future answer"


@pytest.mark.asyncio
async def test_escaped_task_cannot_generate_after_its_turn_finished():
    agent = make_agent()
    context = await agent.start_conversation()
    release = asyncio.Event()
    children = []

    async def escaped():
        await release.wait()
        return await agent.generate_response("late")

    async def strategy(candidate):
        children.append(asyncio.create_task(escaped()))
        return "answer"

    agent.strategies["test"] = strategy
    await agent.continue_conversation(
        context.conversation_id, "question", strategy="test"
    )
    release.set()
    with pytest.raises(ConversationResetError):
        await children[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["end", "cancel"])
async def test_tool_effect_settles_but_cannot_start_next_tool_or_commit(interrupt):
    agent = make_agent()
    agent.response_backend = lambda prompt: "search and read doc_12"
    context = await agent.start_conversation()
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    async def search(query):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            pass  # A provider may complete an effect despite local cancellation.
        effects.append("search completed")
        return "search result"

    async def read(document):
        effects.append("read started")
        return "document"

    agent.register_tool("search", search)
    agent.register_tool("read", read)
    task = asyncio.create_task(
        agent.generate_multiturn_response(context.conversation_id, "question")
    )
    await entered.wait()
    if interrupt == "end":
        agent.end_conversation(context.conversation_id)
    else:
        task.cancel()
    release.set()
    with pytest.raises(
        ConversationClosedError if interrupt == "end" else asyncio.CancelledError
    ):
        await task
    assert effects == ["search completed"]
    assert len(context.history) == 1
    assert context.get_status()["last_finished"]["tool_status"] == "unconfirmed"
