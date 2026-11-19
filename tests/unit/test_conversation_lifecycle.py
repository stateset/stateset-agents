"""Old conversation work cannot record, emit, or select after reset."""

import asyncio
import copy
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

from stateset_agents.core.agent import (
    AgentConfig,
    ConversationResetError,
    MultiTurnAgent,
    ToolAgent,
)
from stateset_agents.core.enhanced import EnhancedMultiTurnAgent
from stateset_agents.core.function_calling import (
    create_function_calling_agent_class,
    tool,
)
from stateset_agents.core.structured_output import StructuredOutputMixin

from .test_enhanced_agent_lifecycle import CountingMemory


async def make_agent(enhanced=False):
    config = AgentConfig(
        model_name="stub://conversation", stub_responses=["first second", "new answer"]
    )
    agent = (
        EnhancedMultiTurnAgent(
            config, memory_system=CountingMemory(), reasoning_enabled=False
        )
        if enhanced
        else MultiTurnAgent(config)
    )
    await agent.initialize()
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("enhanced", [False, True])
@pytest.mark.parametrize("mode", ["response", "turn", "batch", "process"])
async def test_reset_fences_old_generation_and_allows_new_work(
    monkeypatch, enhanced, mode
):
    agent = await make_agent(enhanced)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def generate(prompt, context=None, **kwargs):
        calls.append(prompt)
        if "old question" in prompt:
            entered.set()
            await release.wait()
            return {"response": "old answer"}
        return {"response": "new answer"}

    monkeypatch.setattr(agent, "_generate_with_model_details", generate)
    operation = {
        "response": lambda: agent.generate_response("old question"),
        "turn": lambda: agent.generate_turn("old question"),
        "batch": lambda: agent.generate_turns("old question", 3),
        "process": lambda: agent.process_turn([], "old question"),
    }[mode]
    pending = asyncio.create_task(operation())
    await asyncio.wait_for(entered.wait(), 5)
    before = agent.get_conversation_status()
    assert len(before["active"]) == 1  # Nested base/enhanced calls share ownership.
    with pytest.raises(RuntimeError, match="conversation operation"):
        copy.deepcopy(agent)
    await agent.reset()
    after = agent.get_conversation_status()
    assert after["epoch"] == before["epoch"] + 1
    assert after["active"][0]["invalidated"] is True
    assert after["active"][0]["state"] == "running"
    assert agent.conversation_history == []
    assert agent.turn_count == 0
    assert await agent.generate_response("new question") == "new answer"
    release.set()
    with pytest.raises(ConversationResetError):
        await pending
    assert agent.conversation_history == [
        {"role": "assistant", "content": "new answer"}
    ]
    assert agent.turn_count == 1
    assert len(calls) == 2  # A retired sequential batch never starts its next sample.
    status = agent.get_conversation_status()
    assert status["active"] == []
    assert status["last_finished"]["invalidated"] is True
    assert status["last_finished"]["error_type"] == "ConversationResetError"
    if enhanced:
        assert agent.performance_metrics["total_responses"] == 1
        assert len(agent.memory_system.entries) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("enhanced", [False, True])
async def test_stream_reset_stops_later_chunks_without_leaking_context(enhanced):
    agent = await make_agent(enhanced)
    stream = agent.generate_response_stream("old question")
    assert await anext(stream) == "first "
    assert agent._conversation_owner()._current() is None
    await agent.reset()
    assert await agent.generate_response("new question") == "new answer"
    with pytest.raises(ConversationResetError):
        await anext(stream)
    assert agent.conversation_history == [
        {"role": "assistant", "content": "new answer"}
    ]
    assert agent.turn_count == 1
    assert agent.get_conversation_status()["active"] == []
    if enhanced:
        assert len(agent.memory_system.entries) == 1


@pytest.mark.asyncio
async def test_stream_closes_in_another_task_without_contextvar_errors():
    agent = await make_agent()
    stream = agent.generate_response_stream("hello")
    await anext(stream)
    await asyncio.create_task(stream.aclose())
    assert agent.get_conversation_status()["active"] == []
    assert agent.get_conversation_status()["last_finished"]["state"] == "closed"
    assert agent.conversation_history == []


@pytest.mark.asyncio
@pytest.mark.parametrize("already_selected", [False, True])
async def test_batch_selection_rights_do_not_survive_reset(already_selected):
    agent = await make_agent(True)
    candidate = (await agent.generate_turns("question", 1))[0]
    if already_selected:
        await agent.remember_candidate(candidate)
    count = len(agent.memory_system.entries)
    await agent.reset()
    with pytest.raises(ConversationResetError):
        await agent.remember_candidate(candidate)
    assert len(agent.memory_system.entries) == count
    assert agent.get_generation_status()["last_finished"]["conversation_epoch"] == 0


@pytest.mark.asyncio
async def test_reset_during_memory_write_preserves_acknowledgement_without_new_bookkeeping(
    monkeypatch,
):
    agent = await make_agent(True)
    entered, release = asyncio.Event(), asyncio.Event()
    add = agent.memory_system.add_entry

    async def blocked(content, context=None, importance=1.0):
        if "old question" in content:
            entered.set()
            await release.wait()
        await add(content, context, importance)

    monkeypatch.setattr(agent.memory_system, "add_entry", blocked)
    pending = asyncio.create_task(agent.generate_response("old question"))
    await asyncio.wait_for(entered.wait(), 5)
    await agent.reset()
    await agent.generate_response("new question")
    release.set()
    with pytest.raises(ConversationResetError):
        await pending
    assert agent.conversation_history == [
        {"role": "assistant", "content": "new answer"}
    ]
    assert agent.performance_metrics["total_responses"] == 1
    assert len(agent.memory_system.entries) == 2  # Already admitted write was drained.
    receipt = agent.get_generation_status()["last_finished"]
    assert receipt["memory_status"] == "stored"
    assert receipt["state"] == "failed"
    assert receipt["error_type"] == "ConversationResetError"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_fails", [False, True])
async def test_reset_during_rollout_never_falls_back_or_records(
    monkeypatch, backend_fails
):
    agent = await make_agent()
    entered, release = asyncio.Event(), asyncio.Event()

    class Backend:
        async def generate_with_logprobs(self, *args, **kwargs):
            entered.set()
            await release.wait()
            if backend_fails:
                raise ValueError("old backend failure")
            return [{"response": "old", "token_ids": [], "log_probs": []}]

    agent.set_rollout_backend(Backend())
    calls = []
    monkeypatch.setattr(
        agent.model, "generate", lambda *args: calls.append(args) or "fallback"
    )
    pending = asyncio.create_task(agent.generate_response("old question"))
    await asyncio.wait_for(entered.wait(), 5)
    await agent.reset()
    release.set()
    with pytest.raises(ConversationResetError):
        await pending
    assert calls == []
    assert agent.conversation_history == []


@pytest.mark.asyncio
async def test_tool_agent_does_not_start_a_second_tool_after_reset():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def first():
        calls.append("first started")
        entered.set()
        await release.wait()
        calls.append("first finished")
        return "done"

    def second():
        calls.append("second")
        return "unexpected"

    response = '```json\n{"tool":"first","parameters":{}}\n```\n```json\n{"tool":"second","parameters":{}}\n```'
    agent = ToolAgent(
        AgentConfig(model_name="stub://tools-reset", stub_responses=[response]),
        tools=[
            {"name": "first", "function": first},
            {"name": "second", "function": second},
        ],
    )
    await agent.initialize()
    pending = asyncio.create_task(agent.generate_response("calculate result"))
    await asyncio.wait_for(entered.wait(), 5)
    await agent.reset()
    assert not pending.done()
    release.set()
    with pytest.raises(ConversationResetError):
        await pending
    assert calls == ["first started", "first finished"]
    assert agent.conversation_history == []


@pytest.mark.asyncio
async def test_parallel_function_tools_settle_before_retired_round_exits():
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    finished = []

    @tool(description="Test controlled work")
    async def work(index: int) -> str:
        entered[index].set()
        await release[index].wait()
        finished.append(index)
        return "done"

    response = '{"tool_calls":[{"id":"a","type":"function","function":{"name":"work","arguments":"{\\"index\\":0}"}},{"id":"b","type":"function","function":{"name":"work","arguments":"{\\"index\\":1}"}}]}'
    cls = create_function_calling_agent_class()
    agent = cls(
        AgentConfig(model_name="stub://functions-reset", stub_responses=[response]),
        tools=[work],
    )
    await agent.initialize()
    pending = asyncio.create_task(agent.chat_with_tools("do work"))
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 5)
    await agent.reset()
    release[0].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not pending.done()
    assert agent.get_conversation_status()["active"][0]["invalidated"] is True
    release[1].set()
    with pytest.raises(ConversationResetError):
        await pending
    assert sorted(finished) == [0, 1]
    assert agent.conversation_history == []
    assert agent.get_conversation_status()["active"] == []


@pytest.mark.asyncio
async def test_structured_retry_does_not_cross_reset_even_with_custom_generator():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    class StructuredAgent(MultiTurnAgent, StructuredOutputMixin):
        async def generate_response(self, *args, **kwargs):
            calls.append(True)
            entered.set()
            await release.wait()
            return "invalid JSON"

    agent = StructuredAgent(AgentConfig(model_name="stub://structured-reset"))
    await agent.initialize()
    pending = asyncio.create_task(
        agent.generate_structured("extract", json_schema={"type": "object"})
    )
    await asyncio.wait_for(entered.wait(), 5)
    await agent.reset()
    release.set()
    with pytest.raises(ConversationResetError):
        await pending
    assert calls == [True]


@pytest.mark.asyncio
async def test_finished_context_cannot_authorize_background_child(monkeypatch):
    agent = await make_agent()
    release = asyncio.Event()
    child = None

    async def generate(*args, **kwargs):
        nonlocal child

        async def later():
            await release.wait()
            return await agent.generate_response("detached child")

        child = asyncio.create_task(later())
        return {"response": "parent"}

    monkeypatch.setattr(agent, "_generate_with_model_details", generate)
    await agent.generate_response("parent")
    release.set()
    with pytest.raises(ConversationResetError, match="finished"):
        await child
    assert agent.turn_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepare", "complete"])
async def test_threaded_reset_and_status_do_not_deadlock_user_callbacks(
    monkeypatch, phase
):
    agent = await make_agent(True)
    calls = []
    with ThreadPoolExecutor(max_workers=1) as executor:

        def adapt(persona, context):
            calls.append(True)
            if phase == "complete" and len(calls) == 1:
                return
            status = executor.submit(agent.get_conversation_status).result(timeout=2)
            assert len(status["active"]) == 1
            executor.submit(lambda: asyncio.run(agent.reset())).result(timeout=2)
            persona.traits["late"] = 1.0

        monkeypatch.setattr(agent, "_adapt_persona_profile", adapt)
        with pytest.raises(ConversationResetError):
            await agent.generate_response("hello")
    assert "late" not in agent.persona.traits
    assert agent.conversation_history == []


@pytest.mark.asyncio
async def test_idle_serialization_legacy_objects_and_detached_history():
    agent = await make_agent()
    await agent.generate_response("hello")
    agent.conversation_history[0]["extra"] = {"values": [1]}
    history = agent.get_history()
    history[0]["extra"]["values"].append(2)
    assert agent.conversation_history[0]["extra"]["values"] == [1]
    for clone in (copy.deepcopy(agent), pickle.loads(pickle.dumps(agent))):
        await clone.reset()
        assert clone.get_conversation_status()["epoch"] == 1
        assert agent.get_conversation_status()["epoch"] == 0
    del agent._conversation
    await agent.reset()
    assert agent.get_conversation_status()["epoch"] == 1
    assert agent.turn_count == 0


@pytest.mark.asyncio
async def test_concurrent_current_conversations_are_independent(monkeypatch):
    agent = await make_agent()
    other = await make_agent()
    entered, release = asyncio.Event(), asyncio.Event()

    async def generate(*args, **kwargs):
        entered.set()
        await release.wait()
        return {"response": "answer"}

    monkeypatch.setattr(agent, "_generate_with_model_details", generate)
    tasks = [
        asyncio.create_task(agent.generate_response(str(index))) for index in range(2)
    ]
    await asyncio.wait_for(entered.wait(), 5)
    await asyncio.sleep(0)
    assert len(agent.get_conversation_status()["active"]) == 2
    await other.reset()
    release.set()
    assert await asyncio.gather(*tasks) == ["answer", "answer"]
    assert agent.turn_count == 2
    assert agent.get_conversation_status()["epoch"] == 0


@pytest.mark.asyncio
async def test_cancelled_stream_cannot_complete_memory_when_pending_source_ends(
    monkeypatch,
):
    agent = await make_agent(True)
    entered, release = asyncio.Event(), asyncio.Event()
    closed = []

    async def source(self, messages, context=None):
        try:
            yield "partial"
            entered.set()
            await release.wait()
        finally:
            closed.append(True)

    monkeypatch.setattr(MultiTurnAgent, "generate_response_stream", source)
    stream = agent.generate_response_stream("hello")
    assert await anext(stream) == "partial"
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), 5)
    pending.cancel()
    await asyncio.sleep(0)
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert closed == [True]
    assert agent.memory_system.entries == []
    assert agent.get_generation_status()["last_finished"]["state"] == "cancelled"
    assert agent.get_conversation_status()["last_finished"]["state"] == "cancelled"


@pytest.mark.asyncio
async def test_reset_inside_persona_feedback_prevents_late_adaptation_and_memory():
    from stateset_agents.core.enhanced import PersonaProfile

    agent = await make_agent(True)
    with ThreadPoolExecutor(max_workers=1) as executor:

        class ResetPersona(PersonaProfile):
            def adapt(self, feedback):
                executor.submit(lambda: asyncio.run(agent.reset())).result(timeout=2)
                self.traits["late"] = 1.0

        agent.persona = ResetPersona("test")
        with pytest.raises(ConversationResetError):
            await agent.self_improve({"persona_feedback": {}})
    assert "late" not in agent.persona.traits
    assert agent.memory_system.entries == []


@pytest.mark.asyncio
async def test_reset_retires_completed_status_without_erasing_completion():
    agent = await make_agent()
    await agent.generate_response("hello")
    await agent.reset()
    status = agent.get_conversation_status()
    assert status["last_finished"]["state"] == "completed"
    assert status["last_finished"]["invalidated"] is True
    status["last_finished"]["state"] = "modified"
    assert agent.get_conversation_status()["last_finished"]["state"] == "completed"
