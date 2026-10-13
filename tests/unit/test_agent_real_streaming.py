"""Real-model streaming behavior without loading a model or tokenizer."""

import asyncio
import queue
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent


class FakeStreamer:
    """A blocking iterator with the completion behavior of transformers."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.chunks: queue.Queue[str | None] = queue.Queue()

    def put_text(self, text: str) -> None:
        self.chunks.put(text)

    def end(self) -> None:
        self.chunks.put(None)

    def __iter__(self) -> "FakeStreamer":
        return self

    def __next__(self) -> str:
        chunk = self.chunks.get()
        if chunk is None:
            raise StopIteration
        return chunk


class FakeTokenizer:
    chat_template = None
    model_max_length = 100

    def __call__(self, *_args: Any, **_kwargs: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [[1, 2]]}

    def encode(self, *_args: Any, **_kwargs: Any) -> list[int]:
        return [1]


class FakeModel:
    def __init__(self, chunks: list[str], error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.training = True
        self.config = SimpleNamespace(use_cache=False)
        self.mode_during_generation: tuple[bool, bool, bool] | None = None

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def generate(self, **kwargs: Any) -> None:
        self.mode_during_generation = (
            self.training,
            self.config.use_cache,
            torch.is_grad_enabled(),
        )
        if self.error is not None:
            raise self.error
        streamer = kwargs["streamer"]
        for chunk in self.chunks:
            streamer.put_text(chunk)
        streamer.end()


def make_agent(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> MultiTurnAgent:
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(TextIteratorStreamer=FakeStreamer)
    )
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://stream-test", max_new_tokens=10)
    )
    agent.model = model
    agent.tokenizer = FakeTokenizer()
    agent.generation_config = object()
    return agent


@pytest.mark.asyncio
async def test_stream_hides_stop_phrase_split_across_chunks(monkeypatch):
    """No part of a split role delimiter reaches the client."""
    model = FakeModel(["Hello Us", "er: hidden"])
    agent = make_agent(monkeypatch, model)

    chunks = [chunk async for chunk in agent.generate_response_stream("Question")]

    assert "".join(chunks) == "Hello "
    assert agent.conversation_history == [{"role": "assistant", "content": "Hello"}]
    assert agent.turn_count == 1
    assert model.mode_during_generation == (False, True, False)
    assert model.training is True
    assert model.config.use_cache is False


@pytest.mark.asyncio
async def test_stream_releases_partial_prefix_when_it_is_not_a_stop_phrase(monkeypatch):
    """Withheld text appears once a later chunk rules out a delimiter."""
    agent = make_agent(monkeypatch, FakeModel(["The Us", "age is fine"]))

    chunks = [chunk async for chunk in agent.generate_response_stream("Question")]

    assert "".join(chunks) == "The Usage is fine"
    assert agent.conversation_history == [
        {"role": "assistant", "content": "The Usage is fine"}
    ]


@pytest.mark.asyncio
async def test_stream_propagates_model_failure(monkeypatch):
    """A generation worker failure unblocks the stream and reaches the caller."""
    model = FakeModel([], error=ValueError("generation failed"))
    agent = make_agent(monkeypatch, model)

    with pytest.raises(RuntimeError, match="Model generation failed during streaming"):
        async for _ in agent.generate_response_stream("Question"):
            pass

    assert agent.conversation_history == []
    assert agent.turn_count == 0
    assert model.training is True
    assert model.config.use_cache is False


@pytest.mark.asyncio
async def test_stream_wait_does_not_block_event_loop(monkeypatch):
    """Another coroutine can run while the model waits to emit a chunk."""
    release = threading.Event()
    fallback_used = threading.Event()

    class WaitingModel(FakeModel):
        def generate(self, **kwargs: Any) -> None:
            release.wait(timeout=2)
            super().generate(**kwargs)

    model = WaitingModel(["Ready"])
    agent = make_agent(monkeypatch, model)

    def fallback_release() -> None:
        fallback_used.set()
        release.set()

    timer = threading.Timer(1, fallback_release)
    timer.start()

    async def unblock() -> None:
        await asyncio.sleep(0)
        release.set()

    try:
        task = asyncio.create_task(unblock())
        chunks = [chunk async for chunk in agent.generate_response_stream("Question")]
        await task
    finally:
        timer.cancel()
        release.set()

    assert "".join(chunks) == "Ready"
    assert not fallback_used.is_set()
