"""Enhanced readiness includes optional memory initialization and its effects."""

import asyncio
import copy
import pickle

import pytest

from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.enhanced.enhanced_agent import (
    EnhancedMultiTurnAgent,
    VectorMemory,
    create_enhanced_agent,
)


class CountingMemory(VectorMemory):
    def __init__(self, error=None):
        super().__init__()
        self.calls = 0
        self.error = error

    async def initialize(self):
        self.calls += 1
        if self.error is not None:
            raise self.error


def enhanced(memory):
    return EnhancedMultiTurnAgent(
        AgentConfig(model_name="stub://enhanced", stub_responses=["first", "second"]),
        memory_system=memory,
    )


@pytest.mark.asyncio
async def test_repeat_initialization_reuses_memory_and_stub_progress():
    memory = CountingMemory()
    agent = enhanced(memory)
    await agent.initialize()
    assert await agent.generate_response("hello") == "first"
    await agent.initialize()
    assert memory.calls == 1
    assert len(memory.entries) == 1
    assert await agent.generate_response("again") == "second"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_failed_memory_setup_is_not_ready_or_replayed(error_type):
    memory = CountingMemory(error_type("injected memory failure"))
    agent = enhanced(memory)
    with pytest.raises(error_type):
        await agent.initialize()
    assert agent.get_initialization_status()["state"] == "failed"
    assert agent.get_initialization_status()["phase"] == "extensions"
    with pytest.raises(RuntimeError, match="initialization is failed"):
        await agent.generate_response("why?")
    with pytest.raises(RuntimeError, match="initialization is failed"):
        await agent.self_improve({"insight": "must not be stored"})
    assert agent.reasoning_history == []
    assert agent.self_improvement_data == []
    assert memory.entries == []
    memory.error = None
    with pytest.raises(RuntimeError, match="fresh memory"):
        await agent.initialize()
    assert memory.calls == 1
    replacement = CountingMemory()
    agent.memory_system = replacement
    await agent.initialize()
    assert agent.get_initialization_status()["state"] == "ready"
    assert replacement.calls == 1


@pytest.mark.asyncio
async def test_pending_memory_setup_fences_generation_and_duplicate_initializers():
    started, release = asyncio.Event(), asyncio.Event()

    class WaitingMemory(CountingMemory):
        async def initialize(self):
            self.calls += 1
            started.set()
            await release.wait()

    memory = WaitingMemory()
    agent = enhanced(memory)
    pending = asyncio.create_task(agent.initialize())
    try:
        await started.wait()
        assert agent.get_initialization_status()["state"] == "initializing"
        with pytest.raises(RuntimeError, match="initialization is initializing"):
            await agent.generate_response("hello")
        with pytest.raises(RuntimeError, match="reenter"):
            await agent.initialize()
        assert memory.calls == 1
    finally:
        release.set()
        await pending
    assert agent.get_initialization_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_factory_disable_flags_prevent_component_creation_and_effects(
    monkeypatch,
):
    import stateset_agents.core.enhanced.enhanced_agent as implementation

    def unexpected(*args, **kwargs):
        pytest.fail("disabled component was constructed")

    monkeypatch.setattr(implementation, "VectorMemory", unexpected)
    monkeypatch.setattr(implementation, "ReasoningEngine", unexpected)
    agent = create_enhanced_agent(
        model_name="stub://disabled", memory_enabled=False, reasoning_enabled=False
    )
    await agent.initialize()
    await agent.generate_response("why is the sky blue?")
    await agent.self_improve({"insight": "no memory effects"})
    status = agent.get_agent_status()
    assert status["memory_entries"] == 0
    assert status["reasoning_sessions"] == 0
    assert "vector_memory" not in status["capabilities"]
    assert "chain_of_thought" not in status["capabilities"]


@pytest.mark.asyncio
async def test_generation_rejects_uninitialized_memory_replacement():
    agent = enhanced(CountingMemory())
    await agent.initialize()
    agent.memory_system = CountingMemory()
    with pytest.raises(RuntimeError, match="memory.*changed"):
        await agent.generate_response("hello")
    await agent.initialize()
    assert await agent.generate_response("hello") == "first"


def test_status_is_detached_from_agent_configuration_and_persona():
    agent = enhanced(CountingMemory())
    status = agent.get_agent_status()
    status["persona"]["traits"]["injected"] = 1
    status["performance_metrics"]["total_responses"] = 100
    status["base_config"]["model_name"] = "different"
    assert "injected" not in agent.persona.traits
    assert agent.performance_metrics == {}
    assert agent.config.model_name == "stub://enhanced"


@pytest.mark.asyncio
async def test_retrieved_memory_reaches_model_prompt_without_mutating_inputs(
    monkeypatch,
):
    agent = enhanced(CountingMemory())
    await agent.initialize()
    await agent.memory_system.add_entry("Refund window is thirty days.")
    captured = []

    async def generate(prompt, context):
        captured.append(prompt)
        return "Thirty days."

    monkeypatch.setattr(agent, "_generate_with_model", generate)
    messages = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Explain the refund window."},
    ]
    context = {"request_id": "r1"}
    previous = copy.deepcopy((messages, context))
    await agent.generate_response(messages, context)
    assert "Refund window is thirty days." in captured[0]
    assert "Explain the refund window." in captured[0]
    assert (messages, context) == previous


@pytest.mark.asyncio
async def test_actual_task_cancellation_fences_custom_memory_retry():
    started = asyncio.Event()

    class WaitingMemory(CountingMemory):
        async def initialize(self):
            self.calls += 1
            started.set()
            await asyncio.Event().wait()

    memory = WaitingMemory()
    agent = enhanced(memory)
    pending = asyncio.create_task(agent.initialize())
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert agent.get_initialization_status()["state"] == "failed"
    with pytest.raises(RuntimeError, match="fresh memory"):
        await agent.initialize()
    assert memory.calls == 1


@pytest.mark.asyncio
async def test_memory_setup_can_use_a_helper_thread_without_holding_agent_lock():
    class ThreadMemory(CountingMemory):
        async def initialize(self):
            self.calls += 1

            def inspect_from_thread():
                # A competing initializer must see the live attempt instead of
                # waiting on a lock held by this task while it awaits the thread.
                with pytest.raises(RuntimeError, match="reenter"):
                    asyncio.run(agent.initialize())

            await asyncio.to_thread(inspect_from_thread)

    agent = enhanced(ThreadMemory())
    await asyncio.wait_for(agent.initialize(), timeout=3)
    assert agent.get_initialization_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_suppressed_memory_cancellation_does_not_publish_readiness():
    started = asyncio.Event()

    class SuppressingMemory(CountingMemory):
        async def initialize(self):
            self.calls += 1
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return

    memory = SuppressingMemory()
    agent = enhanced(memory)
    pending = asyncio.create_task(agent.initialize())
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert agent.get_initialization_status()["state"] == "failed"
    with pytest.raises(RuntimeError, match="initialization is failed"):
        await agent.generate_response("hello")
    # Memory did finish, so an explicit retry can reuse that completed effect.
    await agent.initialize()
    assert memory.calls == 1
    assert agent.get_initialization_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_interrupt_memory_cleanup():
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cleaned = []

    class CleaningMemory(CountingMemory):
        async def initialize(self):
            self.calls += 1
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                cleaned.append(True)

    agent = enhanced(CleaningMemory())
    pending = asyncio.create_task(agent.initialize())
    await started.wait()
    pending.cancel()
    await cleaning.wait()
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    assert agent.get_initialization_status()["phase"] == "extensions_cleanup"
    with pytest.raises(RuntimeError, match="reenter"):
        await agent.initialize()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cleaned == [True]
    assert agent.get_initialization_status()["state"] == "failed"


@pytest.mark.asyncio
async def test_completed_and_legacy_enhanced_agents_remain_pickleable():
    agent = enhanced(CountingMemory())
    await agent.initialize()
    await agent.generate_response("hello")
    restored = pickle.loads(pickle.dumps(agent))
    await restored.initialize()
    assert restored.memory_system.calls == 1
    assert await restored.generate_response("again") == "second"
    assert len(agent.memory_system.entries) == 1
    for key in ("memory_enabled", "reasoning_enabled", "_memory_setup"):
        del agent.__dict__[key]
    restored = pickle.loads(pickle.dumps(agent))
    await restored.initialize()
    assert restored.memory_enabled is True
    assert restored.reasoning_enabled is True
    assert await restored.generate_response("again") == "second"
