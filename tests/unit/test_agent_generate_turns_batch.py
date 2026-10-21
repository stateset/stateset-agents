"""Batched generation over different conversations matches one-at-a-time."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("transformers")

from tests.learning._toy import make_model, make_tokenizer  # noqa: E402

CONVERSATIONS = [
    [{"role": "user", "content": "hello"}],
    [
        {"role": "user", "content": "say a number please"},
        {"role": "assistant", "content": "12"},
        {"role": "user", "content": "again"},
    ],
    [{"role": "user", "content": "ok"}],
]


def _greedy_agent(seed: int = 0) -> Any:
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    agent = MultiTurnAgent(
        AgentConfig(
            model_name="gpt2",
            max_new_tokens=6,
            temperature=1.0,
            do_sample=False,
            use_chat_template=False,
        )
    )
    agent.model = make_model(seed)
    agent.tokenizer = make_tokenizer()
    agent.generation_config = agent._build_generation_config()
    return agent


def _view(turn: Any) -> tuple[Any, ...]:
    meta = turn.metadata
    return (
        turn.content,
        meta.get("prompt_token_ids"),
        meta.get("token_ids"),
        [round(x, 4) for x in meta.get("sampler_log_probs", [])],
    )


def test_batch_of_distinct_conversations_matches_single_generation() -> None:
    agent = _greedy_agent()

    async def run() -> tuple[list[Any], list[Any]]:
        single = [await agent.generate_turn(c) for c in CONVERSATIONS]
        batched = await agent.generate_turns_batch(CONVERSATIONS)
        return single, batched

    single, batched = asyncio.run(run())

    assert [_view(t) for t in batched] == [_view(t) for t in single]
    for turn in batched:
        assert turn.metadata["token_ids"], "token ids must be captured"
        decoded = agent.tokenizer.decode(
            turn.metadata["token_ids"], skip_special_tokens=True
        )
        assert turn.content == decoded


def test_contexts_must_align_with_histories() -> None:
    agent = _greedy_agent()

    with pytest.raises(ValueError, match="contexts"):
        asyncio.run(agent.generate_turns_batch(CONVERSATIONS, contexts=[{}]))


def test_empty_batch_returns_no_turns() -> None:
    assert asyncio.run(_greedy_agent().generate_turns_batch([])) == []


class _RecordingBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_with_logprobs(self, prompts: list[str], **_: Any) -> list[Any]:
        self.prompts = list(prompts)
        raise RuntimeError("engine down")


def test_rollout_backend_receives_every_distinct_prompt_then_falls_back() -> None:
    agent = _greedy_agent()
    backend = _RecordingBackend()
    agent._rollout_backend = backend

    turns = asyncio.run(agent.generate_turns_batch(CONVERSATIONS))

    assert len(backend.prompts) == 3 and len(set(backend.prompts)) == 3
    assert len(turns) == 3
    assert all("engine down" in t.metadata["rollout_backend_error"] for t in turns)
