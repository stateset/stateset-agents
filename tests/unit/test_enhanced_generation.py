"""Shared enhancement preparation and completion across generation surfaces."""

import asyncio
import copy
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.enhanced.enhanced_agent import EnhancedMultiTurnAgent

from .test_agent_checkpoint_roundtrip import tiny_base as tiny_base
from .test_enhanced_agent_lifecycle import CountingMemory


class ObservedMemory(CountingMemory):
    def __init__(self):
        super().__init__()
        self.queries = []
        self.writes = []

    async def retrieve_relevant(self, query, top_k=5):
        self.queries.append(query)
        return await super().retrieve_relevant(query, top_k=top_k)

    async def add_entry(self, content, context=None, importance=1.0):
        self.writes.append(content)
        await super().add_entry(content, context, importance)


async def prepared_agent():
    memory = ObservedMemory()
    agent = EnhancedMultiTurnAgent(
        AgentConfig(
            model_name="stub://enhanced-generation",
            stub_responses=["first answer", "second answer"],
        ),
        memory_system=memory,
    )
    await agent.initialize()
    await memory.add_entry("Refund window is thirty days.")
    memory.writes.clear()
    return agent, memory


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["response", "turn", "stream"])
async def test_every_single_response_surface_prepares_and_remembers_once(
    monkeypatch, mode
):
    agent, memory = await prepared_agent()
    prompts = []
    adaptations = []
    adapt = agent._adapt_persona
    generate = agent.model.generate

    def adapt_once(context):
        adaptations.append(copy.deepcopy(context))
        adapt(context)

    def capture(prompt, context=None):
        prompts.append((prompt, copy.deepcopy(context)))
        return generate(prompt, context)

    monkeypatch.setattr(agent.model, "generate", capture)
    monkeypatch.setattr(agent, "_adapt_persona", adapt_once)
    messages = [{"role": "user", "content": "Explain the refund window"}]
    context = {"urgency": "high", "nested": {"values": [1]}}
    original = copy.deepcopy((messages, context))
    if mode == "response":
        answer = await agent.generate_response(messages, context)
    elif mode == "turn":
        answer = (await agent.generate_turn(messages, context)).content
    else:
        answer = "".join(
            [part async for part in agent.generate_response_stream(messages, context)]
        )
    assert answer == "first answer"
    assert "Refund window is thirty days." in prompts[0][0]
    assert len(memory.queries) == 1
    assert len(memory.writes) == 1
    assert len(agent.reasoning_history) == 1
    assert agent.performance_metrics["total_responses"] == 1
    assert agent.turn_count == 1
    assert len(agent.conversation_history) == 1
    assert (messages, context) == original
    assert adaptations == [context]
    assert agent.persona.traits["responsiveness"] == pytest.approx(0.6)
    status = agent.get_generation_status()["last_finished"]
    assert status["state"] == "completed"
    assert status["memory_status"] == "stored"
    assert status["outputs"] == ["first answer"]


@pytest.mark.asyncio
async def test_batch_shares_preparation_preserves_tokens_and_requires_selection(
    monkeypatch,
):
    agent, memory = await prepared_agent()
    prompts = []

    async def batch(prompt, n, context):
        prompts.append(prompt)
        return [
            {
                "response": f"candidate {i}",
                "prompt_token_ids": [1, 2],
                "token_ids": [i + 3],
                "sampler_log_probs": [-0.3],
            }
            for i in range(n)
        ]

    monkeypatch.setattr(agent, "_generate_with_model_details_batch", batch)
    turns = await agent.generate_turns("Explain refund policy", 2)
    assert "Refund window is thirty days." in prompts[0]
    assert len(memory.queries) == 1
    assert len(agent.reasoning_history) == 1
    assert memory.writes == []
    assert agent.performance_metrics["total_responses"] == 2
    assert turns[1].metadata["token_ids"] == [4]
    assert turns[1].metadata["sampler_log_probs"] == [-0.3]
    assert agent.get_generation_status()["last_finished"]["memory_status"] == "deferred"
    await agent.remember_candidate(turns[1])
    await agent.remember_candidate(turns[1])
    assert len(memory.writes) == 1
    assert memory.entries[-1].context == {
        "query": "Explain refund policy",
        "response": "candidate 1",
    }
    with pytest.raises(ValueError, match="different candidate"):
        await agent.remember_candidate(turns[0])
    assert len(memory.writes) == 1


@pytest.mark.asyncio
async def test_empty_batch_has_no_enhancement_effects():
    agent, memory = await prepared_agent()
    assert await agent.generate_turns("Explain refunds", 0) == []
    assert memory.queries == []
    assert memory.writes == []
    assert agent.reasoning_history == []


@pytest.mark.asyncio
async def test_partial_stream_close_does_not_commit_memory_or_metrics():
    agent, memory = await prepared_agent()
    stream = agent.generate_response_stream("Explain refunds")
    assert await stream.__anext__() == "first "
    await stream.aclose()
    assert len(memory.queries) == 1
    assert memory.writes == []
    assert agent.performance_metrics == {}
    assert agent.reasoning_history == []
    assert agent.conversation_history == []
    status = agent.get_generation_status()["last_finished"]
    assert status["state"] == "closed"
    assert status["partial_text"] == "first "


@pytest.mark.asyncio
async def test_stream_cancellation_drains_source_before_finishing(monkeypatch):
    agent, memory = await prepared_agent()
    entered, release = asyncio.Event(), asyncio.Event()
    closed = []

    async def source(self, messages, context=None):
        try:
            yield "partial "
            entered.set()
            await release.wait()
            yield "late"
        finally:
            closed.append(True)

    monkeypatch.setattr(MultiTurnAgent, "generate_response_stream", source)
    stream = agent.generate_response_stream("Explain refunds")
    assert await stream.__anext__() == "partial "
    pending = asyncio.create_task(stream.__anext__())
    await entered.wait()
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert closed == [True]
    assert memory.writes == []
    assert agent.get_generation_status()["last_finished"]["state"] == "cancelled"


@pytest.mark.asyncio
async def test_failed_memory_write_retains_generated_output(monkeypatch):
    agent, memory = await prepared_agent()
    original = memory.add_entry

    async def fail_after_effect(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("injected after write")

    monkeypatch.setattr(memory, "add_entry", fail_after_effect)
    with pytest.raises(RuntimeError, match="after write"):
        await agent.generate_response("refund")
    status = agent.get_generation_status()["last_finished"]
    assert status["outputs"] == ["first answer"]
    assert status["memory_status"] == "unconfirmed"
    assert status["state"] == "failed"
    assert len(memory.writes) == 1


@pytest.mark.asyncio
async def test_uncertain_candidate_write_cannot_be_replayed(monkeypatch):
    agent, memory = await prepared_agent()
    turns = await agent.generate_turns("refund", 2)
    original = memory.add_entry

    async def fail_after_effect(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("injected after write")

    monkeypatch.setattr(memory, "add_entry", fail_after_effect)
    with pytest.raises(RuntimeError, match="after write"):
        await agent.remember_candidate(turns[0])
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await agent.remember_candidate(turns[0])
    assert len(memory.writes) == 1


@pytest.mark.asyncio
async def test_cancelled_selection_drains_one_write_and_blocks_concurrent_replays(
    monkeypatch,
):
    agent, memory = await prepared_agent()
    turns = await agent.generate_turns("refund", 2)
    entered, release = asyncio.Event(), asyncio.Event()
    original = memory.add_entry

    async def wait_then_write(*args, **kwargs):
        entered.set()
        await release.wait()
        await original(*args, **kwargs)

    monkeypatch.setattr(memory, "add_entry", wait_then_write)
    pending = asyncio.create_task(agent.remember_candidate(turns[1]))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for _ in range(3):
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
        with pytest.raises(RuntimeError, match="writing"):
            await agent.remember_candidate(turns[1])
        with pytest.raises(ValueError, match="different candidate"):
            await agent.remember_candidate(turns[0])
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
    await agent.remember_candidate(turns[1])
    assert len(memory.writes) == 1
    assert agent.get_generation_status()["last_finished"]["memory_status"] == "stored"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["retrieval", "reasoning", "sampling"])
async def test_suppressed_cancellation_cannot_continue_to_memory_write(
    monkeypatch, stage
):
    agent, memory = await prepared_agent()
    entered, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def suppress(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup.set()
            await release.wait()
        return "late answer" if stage == "sampling" else []

    if stage == "retrieval":
        monkeypatch.setattr(memory, "retrieve_relevant", suppress)
    elif stage == "reasoning":
        monkeypatch.setattr(agent.reasoning_engine, "reason", suppress)
    else:
        monkeypatch.setattr(MultiTurnAgent, "generate_response", suppress)
    pending = asyncio.create_task(agent.generate_response("Explain refunds"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        pending.cancel()
        await asyncio.wait_for(cleanup.wait(), 2)
        for _ in range(3):
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
        with pytest.raises(RuntimeError, match="during a conversation operation"):
            copy.deepcopy(agent)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
    status = agent.get_generation_status()["last_finished"]
    assert status["state"] == "cancelled"
    assert status["memory_status"] == "skipped"
    assert status["outputs"] == (["late answer"] if stage == "sampling" else [])
    assert memory.writes == []
    assert agent.performance_metrics == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["replacement", "encoder"])
async def test_changed_memory_during_sampling_prevents_late_write(monkeypatch, change):
    agent, memory = await prepared_agent()
    original = MultiTurnAgent.generate_response

    async def sample_then_change(self, messages, context=None):
        response = await original(self, messages, context)
        if change == "replacement":
            self.memory_system = ObservedMemory()
            await self.initialize()
        else:
            memory.embedding_model = object()
        return response

    monkeypatch.setattr(MultiTurnAgent, "generate_response", sample_then_change)
    with pytest.raises(RuntimeError, match="changed"):
        await agent.generate_response("refund")
    assert memory.writes == []
    assert agent.get_generation_status()["last_finished"]["outputs"] == ["first answer"]


@pytest.mark.asyncio
async def test_candidate_preflight_failure_remains_safe_to_select(monkeypatch):
    agent, memory = await prepared_agent()
    turns = await agent.generate_turns("refund", 2)
    calculate = agent._calculate_importance

    def fail(*args):
        raise ValueError("invalid importance")

    monkeypatch.setattr(agent, "_calculate_importance", fail)
    with pytest.raises(ValueError, match="importance"):
        await agent.remember_candidate(turns[0])
    assert memory.writes == []
    assert agent.get_generation_status()["last_finished"]["memory_status"] == "deferred"
    monkeypatch.setattr(agent, "_calculate_importance", calculate)
    await agent.remember_candidate(turns[1])
    assert len(memory.writes) == 1


@pytest.mark.asyncio
async def test_enhancement_hooks_can_inspect_status_from_helper_thread(monkeypatch):
    agent, memory = await prepared_agent()
    observed = []
    adapt = agent._adapt_persona
    importance = agent._calculate_importance
    with ThreadPoolExecutor(max_workers=1) as executor:

        def inspect():
            observed.append(
                executor.submit(agent.get_generation_status).result(timeout=2)
            )

        def adapt_with_status(context):
            inspect()
            adapt(context)

        def importance_with_status(query, response):
            inspect()
            return importance(query, response)

        monkeypatch.setattr(agent, "_adapt_persona", adapt_with_status)
        monkeypatch.setattr(agent, "_calculate_importance", importance_with_status)
        turns = await agent.generate_turns("refund", 2)
        await agent.remember_candidate(turns[0])
    assert len(observed) == 2
    assert observed[1]["last_finished"]["memory_status"] == "writing"
    assert len(memory.writes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["turn", "batch", "stream"])
async def test_disabled_components_have_no_effects_for_all_surfaces(mode):
    agent, memory = await prepared_agent()
    agent.memory_enabled = agent.reasoning_enabled = False
    if mode == "turn":
        await agent.generate_turn("Explain refunds")
    elif mode == "batch":
        await agent.generate_turns("Explain refunds", 2)
    else:
        assert [
            part async for part in agent.generate_response_stream("Explain refunds")
        ]
    assert memory.queries == memory.writes == agent.reasoning_history == []
    assert agent.get_generation_status()["last_finished"]["memory_status"] == "disabled"


@pytest.mark.asyncio
async def test_candidate_validation_and_retention_bound():
    agent, memory = await prepared_agent()
    other, _ = await prepared_agent()
    turns = await agent.generate_turns("refund", 2)
    with pytest.raises(ValueError, match="not retained"):
        await other.remember_candidate(turns[0])
    changed = copy.deepcopy(turns[0])
    changed.content = "edited answer"
    with pytest.raises(ValueError, match="differs"):
        await agent.remember_candidate(changed)
    for _ in range(32):
        await agent.generate_turns("refund", 1)
    assert len(agent.get_generation_status()["retained_batches"]) == 32
    with pytest.raises(ValueError, match="not retained"):
        await agent.remember_candidate(turns[0])
    assert memory.writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "clone", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
async def test_copy_retains_output_evidence_but_not_candidate_selection_rights(clone):
    agent, memory = await prepared_agent()
    turns = await agent.generate_turns("refund", 2)
    copied = clone(agent)
    status = copied.get_generation_status()
    assert status["last_finished"]["outputs"] == [turn.content for turn in turns]
    assert status["retained_batches"] == []
    with pytest.raises(ValueError, match="not retained"):
        await copied.remember_candidate(turns[0])
    await agent.remember_candidate(turns[0])
    assert len(memory.writes) == 1


@pytest.mark.asyncio
async def test_concurrent_generations_have_distinct_detached_receipts(monkeypatch):
    agent, memory = await prepared_agent()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def sample(self, messages, context=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
        await release.wait()
        return messages[-1]["content"]

    monkeypatch.setattr(MultiTurnAgent, "generate_response", sample)
    tasks = [
        asyncio.create_task(agent.generate_response(query)) for query in ("one", "two")
    ]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        snapshot = agent.get_generation_status()
        assert len(snapshot["active"]) == 2
        assert len({receipt["id"] for receipt in snapshot["active"]}) == 2
        snapshot["active"][0]["outputs"].append("forged")
        assert all(
            not receipt["outputs"]
            for receipt in agent.get_generation_status()["active"]
        )
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert len(memory.writes) == 2
    assert agent.get_generation_status()["active"] == []


@pytest.mark.asyncio
async def test_real_local_batch_preserves_sampler_evidence(tiny_base):
    agent = EnhancedMultiTurnAgent(
        AgentConfig(
            model_name=str(tiny_base),
            system_prompt="hello",
            torch_dtype="float32",
            device_map=None,
            attn_implementation="eager",
            max_new_tokens=2,
        ),
        memory_enabled=False,
        reasoning_enabled=False,
    )
    await agent.initialize()
    # The fixture tokenizer has no published context limit; bound it to its tiny model.
    agent.tokenizer.model_max_length = 24
    # Ensure actual text tokens, rather than a random all-padding output.
    agent.generation_config.suppress_tokens = [0, 1, 2]
    turns = await agent.generate_turns("hello", 2)
    assert len(turns) == 2
    for turn in turns:
        assert turn.metadata["prompt_token_ids"]
        assert turn.metadata["token_ids"]
        assert len(turn.metadata["token_ids"]) == len(
            turn.metadata["sampler_log_probs"]
        )
        assert turn.metadata["stateset_enhancement"]["memory_status"] == "disabled"
