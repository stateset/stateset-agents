"""Memory backend changes publish one coherent embedding space or no change."""

import asyncio
import copy
import inspect
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stateset_agents.core.enhanced.enhanced_agent import VectorMemory


@pytest.fixture
def encoders(monkeypatch):
    instances = []
    failures = {}

    class Encoder:
        def __init__(self, name):
            self.name = name
            self.calls = []
            instances.append(self)

        def to(self, device):
            if failures.get("to"):
                raise RuntimeError("device setup failed")
            return self

        def eval(self):
            return self

        def encode(self, text, convert_to_numpy):
            self.calls.append(text)
            if text == failures.get("encode"):
                raise RuntimeError("embedding failed")
            size = 3 if self.name == "second" else 2
            result = np.zeros(size)
            for position, word in enumerate(("refund", "shipping", "product")[:size]):
                result[position] = text.lower().split().count(word)
            return result

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=Encoder),
    )
    return instances, failures


@pytest.mark.asyncio
async def test_repeated_initialization_preserves_encoder_and_embeddings(encoders):
    instances, _ = encoders
    memory = VectorMemory("first")
    await memory.initialize()
    await memory.add_entry("refund")
    model = memory.model
    embedding = memory.entries[0].embedding
    await memory.initialize()
    assert memory.model is model
    assert memory.entries[0].embedding is embedding
    assert len(instances) == 1


@pytest.mark.asyncio
async def test_optional_dependency_fallback_retrieves_by_word_overlap(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    memory = VectorMemory()
    await memory.initialize()
    await memory.add_entry("Refund policy details")
    await memory.add_entry("Shipping delivery estimate")
    assert [
        entry.content for entry in await memory.retrieve_relevant("refund policy")
    ] == ["Refund policy details"]
    assert await memory.retrieve_relevant("unrelated") == []
    assert await memory.retrieve_relevant(" ") == []
    assert memory.get_initialization_status()["backend"] == "lexical"
    assert memory.get_initialization_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_backend_change_reindexes_existing_memory_before_use(encoders):
    memory = VectorMemory("first")
    await memory.initialize()
    await memory.add_entry("refund product")
    entry = memory.entries[0]
    memory.embedding_model = "second"
    with pytest.raises(RuntimeError, match="changed"):
        await memory.retrieve_relevant("refund")
    await memory.initialize()
    assert memory.entries[0] is entry
    assert entry.embedding.shape == (3,)
    assert await memory.retrieve_relevant("product") == [entry]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["to", "encode"])
async def test_failed_backend_change_keeps_original_model_and_index(encoders, failure):
    _, failures = encoders
    memory = VectorMemory("first")
    await memory.initialize()
    await memory.add_entry("refund")
    await memory.add_entry("shipping")
    previous = memory.model
    previous_embeddings = [entry.embedding for entry in memory.entries]
    memory.embedding_model = "second"
    failures[failure] = True if failure == "to" else "shipping"
    with pytest.raises(RuntimeError):
        await memory.initialize()
    assert memory.model is previous
    assert all(
        entry.embedding is expected
        for entry, expected in zip(memory.entries, previous_embeddings, strict=True)
    )
    assert memory.get_initialization_status()["state"] == "failed"
    with pytest.raises(RuntimeError, match="initialization is failed"):
        await memory.add_entry("must not be added")
    failures.clear()
    await memory.initialize()
    assert memory.model is not previous
    assert all(entry.embedding.shape == (3,) for entry in memory.entries)
    assert len(memory.entries) == 2


@pytest.mark.asyncio
async def test_entries_added_before_initialization_are_indexed(encoders):
    memory = VectorMemory("first")
    await memory.add_entry("refund")
    entry = memory.entries[0]
    assert entry.embedding is None
    await memory.initialize()
    assert entry.embedding is not None
    assert await memory.retrieve_relevant("refund") == [entry]


@pytest.mark.asyncio
async def test_zero_vectors_and_empty_result_limits_do_not_update_access(encoders):
    memory = VectorMemory("first")
    await memory.initialize()
    await memory.add_entry("unrelated")
    await memory.add_entry("refund")
    assert await memory.retrieve_relevant("refund", top_k=0) == []
    assert await memory.retrieve_relevant("unknown") == []
    assert all(entry.access_count == 0 for entry in memory.entries)
    assert await memory.retrieve_relevant("refund") == [memory.entries[1]]
    assert memory.entries[0].access_count == 0


@pytest.mark.asyncio
async def test_explicit_promotion_reindexes_lexical_entries(encoders, monkeypatch):
    module = sys.modules["sentence_transformers"]
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    memory = VectorMemory("first")
    await memory.initialize()
    await memory.add_entry("refund")
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    await memory.initialize()
    assert memory.model is None
    await memory.initialize(force=True)
    assert memory.model is not None
    assert memory.entries[0].embedding is not None
    assert memory.get_initialization_status()["backend"] == "sentence_transformers"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["insert", "retrieve"])
async def test_late_embedding_cannot_commit_across_backend_change(
    encoders, monkeypatch, operation
):
    memory = VectorMemory("first")
    await memory.initialize()
    await memory.add_entry("refund")
    started, release = asyncio.Event(), asyncio.Event()
    original = memory._generate_embedding

    async def delayed(text):
        value = await original(text)
        started.set()
        await release.wait()
        return value

    monkeypatch.setattr(memory, "_generate_embedding", delayed)
    pending = asyncio.create_task(
        memory.add_entry("shipping")
        if operation == "insert"
        else memory.retrieve_relevant("refund")
    )
    try:
        await started.wait()
        memory.embedding_model = "second"
        await memory.initialize()
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="changed"):
        await pending
    assert len(memory.entries) == 1
    assert memory.entries[0].access_count == 0
    assert memory.entries[0].embedding.shape == (3,)


@pytest.mark.asyncio
async def test_async_sdk_embedding_is_closed_and_not_stored(encoders, monkeypatch):
    memory = VectorMemory("first")
    await memory.initialize()
    coroutines = []

    async def invalid():
        return np.ones(2)

    def encode(*args, **kwargs):
        result = invalid()
        coroutines.append(result)
        return result

    monkeypatch.setattr(memory.model, "encode", encode)
    with pytest.raises(TypeError, match="synchronous"):
        await memory.add_entry("refund")
    assert memory.entries == []
    assert inspect.getcoroutinestate(coroutines[0]) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_lexical_memory_remains_copyable_and_pickleable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    memory = VectorMemory()
    await memory.initialize()
    await memory.add_entry("refund")
    for clone in (copy.deepcopy(memory), pickle.loads(pickle.dumps(memory))):
        await clone.initialize()
        assert (await clone.retrieve_relevant("refund"))[0].content == "refund"
        assert clone.entries is not memory.entries
    assert memory.entries[0].access_count == 0


@pytest.mark.asyncio
async def test_failed_builtin_memory_setup_can_retry_through_agent(encoders):
    from stateset_agents.core.agent import AgentConfig
    from stateset_agents.core.enhanced.enhanced_agent import EnhancedMultiTurnAgent

    _, failures = encoders
    failures["to"] = True
    memory = VectorMemory("first")
    agent = EnhancedMultiTurnAgent(
        AgentConfig(model_name="stub://retry"), memory_system=memory
    )
    with pytest.raises(RuntimeError, match="device setup"):
        await agent.initialize()
    assert agent.get_initialization_status()["state"] == "failed"
    failures.clear()
    await agent.initialize()
    assert agent.get_initialization_status()["state"] == "ready"
    failures["to"] = True
    with pytest.raises(RuntimeError, match="device setup"):
        await memory.initialize(force=True)
    with pytest.raises(RuntimeError, match="Memory initialization is failed"):
        await agent.self_improve({"insight": "must not be stored"})
    assert agent.self_improvement_data == []


@pytest.mark.asyncio
async def test_async_embedding_override_can_use_a_worker_thread(encoders):
    class ThreadMemory(VectorMemory):
        async def _generate_embedding(self, text):
            return await asyncio.to_thread(
                lambda: asyncio.run(VectorMemory._generate_embedding(self, text))
            )

    memory = ThreadMemory("first")
    await memory.initialize()
    await asyncio.wait_for(memory.add_entry("refund"), timeout=3)
    assert (
        await asyncio.wait_for(memory.retrieve_relevant("refund"), timeout=3)
        == memory.entries
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [[float("nan")], [], [[1.0, 2.0]]])
async def test_invalid_embedding_does_not_mutate_entries(encoders, monkeypatch, vector):
    memory = VectorMemory("first")
    await memory.initialize()
    monkeypatch.setattr(memory.model, "encode", lambda *args, **kwargs: vector)
    with pytest.raises(ValueError, match="finite vectors"):
        await memory.add_entry("refund")
    assert memory.entries == []
