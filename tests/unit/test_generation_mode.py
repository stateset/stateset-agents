"""Rollout generation runs in inference mode even when a trainer left the
model in train(); the trainer's mode is restored afterwards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stateset_agents.core.generation_mode import inference_mode
from tests._tiny_tokenizer import tiny_tokenizer


class _Model:
    def __init__(self, training: bool, use_cache: bool | None):
        self.training = training
        self.config = (
            SimpleNamespace(use_cache=use_cache)
            if use_cache is not None
            else SimpleNamespace()
        )
        self.calls: list[str] = []

    def eval(self):
        self.training = False
        self.calls.append("eval")

    def train(self):
        self.training = True
        self.calls.append("train")


def test_inference_mode_switches_to_eval_with_cache_and_restores():
    model = _Model(training=True, use_cache=False)
    with inference_mode(model):
        assert model.training is False
        assert model.config.use_cache is True
    assert model.training is True
    assert model.config.use_cache is False
    assert model.calls == ["eval", "train"]


def test_inference_mode_leaves_an_eval_model_in_eval():
    model = _Model(training=False, use_cache=True)
    with inference_mode(model):
        assert model.training is False
    assert model.training is False and model.calls == ["eval"]


def test_inference_mode_tolerates_models_without_config_or_modes():
    with inference_mode(object()):
        pass
    model = _Model(training=True, use_cache=None)
    with inference_mode(model):
        assert model.training is False
    assert model.training is True


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _tiny_hf_agent():
    from transformers import GPT2Config, GPT2LMHeadModel

    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(n_embd=32, n_layer=2, n_head=2, vocab_size=256, n_positions=128)
    )
    tokenizer = tiny_tokenizer()
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="gpt2",
            max_new_tokens=5,
            temperature=1.0,
            use_chat_template=False,
        )
    )
    agent.model = model
    agent.tokenizer = tokenizer
    agent.generation_config = agent._build_generation_config()
    return agent


@pytest.mark.asyncio
async def test_agent_generates_in_eval_mode_and_restores_train_mode():
    agent = _tiny_hf_agent()
    model = agent.model
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    seen: list[tuple[bool, bool]] = []
    real_generate = model.generate

    def spy(*args, **kwargs):
        seen.append((model.training, bool(model.config.use_cache)))
        return real_generate(*args, **kwargs)

    model.generate = spy  # type: ignore[method-assign]
    await agent.generate_turn("hello there")
    await agent.generate_turns("hello there", 2)
    assert seen == [(False, True), (False, True)]  # eval + cache at every generate
    assert model.training is True  # the trainer's mode is back
    assert model.config.use_cache is False
