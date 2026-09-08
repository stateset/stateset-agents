"""An attachable rollout backend (e.g. vLLM) supplies generate_turn's tokens.

The backend only has to return objects shaped like
``training.vllm_backend.GenerationResult``: response text, prompt and
response token ids, and per-token log-probs. No vLLM is needed to test the
contract; the GRPO trainers then get per-token rollouts from whichever engine
is attached.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.trajectory import ConversationTurn


class _FakeRolloutBackend:
    def __init__(self, response: str = "hello from vllm") -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.response = response

    async def generate_with_logprobs(self, prompts, **kwargs):
        self.calls.append((list(prompts), dict(kwargs)))
        return [
            SimpleNamespace(
                prompt=p,
                response=self.response,
                full_text=p + self.response,
                prompt_token_ids=[11, 12, 13],
                response_token_ids=[21, 22],
                token_logprobs=[-0.5, -1.5],
                cumulative_logprob=-2.0,
                sequence_length=2,
                finish_reason="stop",
            )
            for p in prompts
        ]


def _stub_agent() -> MultiTurnAgent:
    return MultiTurnAgent(
        AgentConfig(model_name="stub://x", use_stub_model=True, stub_responses=["ok"])
    )


@pytest.mark.asyncio
async def test_generate_turn_uses_attached_rollout_backend():
    agent = _stub_agent()
    await agent.initialize()
    backend = _FakeRolloutBackend()
    agent.set_rollout_backend(backend)
    turn = await agent.generate_turn([{"role": "user", "content": "hi"}])
    assert isinstance(turn, ConversationTurn) and turn.role == "assistant"
    assert turn.content == "hello from vllm"
    assert turn.metadata["prompt_token_ids"] == [11, 12, 13]
    assert turn.metadata["token_ids"] == [21, 22]
    assert turn.metadata["sampler_log_probs"] == [-0.5, -1.5]
    assert turn.metadata["rollout_backend"] == "_FakeRolloutBackend"
    assert len(backend.calls) == 1 and backend.calls[0][0][0]  # rendered prompt


@pytest.mark.asyncio
async def test_generate_response_also_uses_attached_backend():
    agent = _stub_agent()
    await agent.initialize()
    agent.set_rollout_backend(_FakeRolloutBackend("plain text"))
    assert await agent.generate_response("hi") == "plain text"


@pytest.mark.asyncio
async def test_backend_passes_generation_settings():
    agent = _stub_agent()
    await agent.initialize()
    agent.config.temperature = 0.3
    agent.config.top_p = 0.8
    agent.config.max_new_tokens = 17
    backend = _FakeRolloutBackend()
    agent.set_rollout_backend(backend)
    await agent.generate_turn("hi")
    kwargs = backend.calls[0][1]
    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.8
    assert kwargs["max_tokens"] == 17


@pytest.mark.asyncio
async def test_detaching_backend_restores_native_generation():
    agent = _stub_agent()
    await agent.initialize()
    agent.set_rollout_backend(_FakeRolloutBackend())
    agent.set_rollout_backend(None)
    turn = await agent.generate_turn("hi")
    assert turn.content == "ok" and "token_ids" not in turn.metadata


@pytest.mark.asyncio
async def test_backend_failure_falls_back_to_native_generation():
    class Broken:
        async def generate_with_logprobs(self, prompts, **kwargs):
            raise RuntimeError("engine down")

    agent = _stub_agent()
    await agent.initialize()
    agent.set_rollout_backend(Broken())
    turn = await agent.generate_turn("hi")
    assert turn.content == "ok"
    assert turn.metadata.get("rollout_backend_error", "").startswith("RuntimeError")


@pytest.mark.asyncio
async def test_vllm_generator_exposes_rollout_backend_entry_point(monkeypatch):
    from stateset_agents.training import vllm_backend

    gen = object.__new__(vllm_backend.VLLMGenerator)
    seen = {}

    async def fake_generate(prompts, sampling_params=None, **kwargs):
        seen["prompts"], seen["kwargs"] = list(prompts), dict(kwargs)
        return ["r" for _ in prompts]

    monkeypatch.setattr(gen, "generate", fake_generate)
    out = await gen.generate_with_logprobs(["a", "b"], temperature=0.2, max_tokens=9)
    assert out == ["r", "r"]
    assert seen["prompts"] == ["a", "b"]
    assert seen["kwargs"] == {"temperature": 0.2, "max_tokens": 9}
