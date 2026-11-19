"""Local sentence-transformers integration without downloaded model weights."""

import numpy as np
import pytest

from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.enhanced.enhanced_agent import (
    EnhancedMultiTurnAgent,
    VectorMemory,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_embedding_backend_loads_retrieves_and_reindexes(
    tmp_path, monkeypatch
):
    sentence_transformers = pytest.importorskip("sentence_transformers")
    from sentence_transformers.models import Pooling, WordEmbeddings

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def save_model(size):
        words = tmp_path / f"words-{size}.txt"
        positive = " ".join(["1", *(["0"] * (size - 1))])
        other = " ".join(["0", "1", *(["0"] * (size - 2))])
        words.write_text(f"refund {positive}\nreturn {positive}\nshipping {other}\n")
        embeddings = WordEmbeddings.from_text_file(str(words))
        model = sentence_transformers.SentenceTransformer(
            modules=[embeddings, Pooling(size)], device="cpu", local_files_only=True
        )
        path = tmp_path / f"encoder-{size}"
        model.save(str(path), create_model_card=False)
        return path

    first, second = save_model(2), save_model(3)
    memory = VectorMemory(str(first))
    memory.device = "cpu"
    await memory.initialize()
    await memory.add_entry("refund")
    await memory.add_entry("shipping")
    # No word overlaps: retrieval must come from the actual encoded vectors.
    assert [entry.content for entry in await memory.retrieve_relevant("return")] == [
        "refund"
    ]
    loaded = memory.model
    snapshot = tmp_path / "memory.json"
    await memory.save_snapshot(snapshot)
    await memory.initialize()
    assert memory.model is loaded
    memory.embedding_model = str(second)
    await memory.initialize()
    assert memory.model is not loaded
    assert all(entry.embedding.shape == (3,) for entry in memory.entries)
    np.testing.assert_allclose(memory.entries[0].embedding, [1, 0, 0])
    assert [entry.content for entry in await memory.retrieve_relevant("return")] == [
        "refund"
    ]

    recovered = VectorMemory(str(second))
    recovered.device = "cpu"
    await recovered.initialize()
    await recovered.restore_snapshot(snapshot)
    assert recovered.embedding_model == str(second)
    assert all(entry.embedding.shape == (3,) for entry in recovered.entries)
    assert [entry.content for entry in await recovered.retrieve_relevant("return")] == [
        "refund"
    ]

    agent = EnhancedMultiTurnAgent(
        AgentConfig(model_name="stub://semantic-batch", stub_responses=["refund"]),
        memory_system=memory,
        reasoning_enabled=False,
    )
    await agent.initialize()
    prompts = []
    generate = agent.model.generate

    def capture(prompt, context=None):
        prompts.append(prompt)
        return generate(prompt, context)

    monkeypatch.setattr(agent.model, "generate", capture)
    before = len(memory.entries)
    turns = await agent.generate_turns("return", 2)
    assert all("refund" in prompt for prompt in prompts)
    assert len(memory.entries) == before
    await agent.remember_candidate(turns[1])
    assert len(memory.entries) == before + 1
    assert memory.entries[-1].context == {"query": "return", "response": "refund"}
