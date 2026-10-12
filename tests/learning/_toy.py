"""Shared toy task for the learning suite.

Task: a character-level tiny GPT-2 policy should learn to emit *digits*.
The reward of a completion is the fraction of its characters that are digits
(dense, verifiable, ~0.1 for a random policy). Learning is measured
deterministically as the probability mass the policy puts on the ten digit
tokens at the first response position of held-out prompts, so tests are not
at the mercy of sampling noise.
"""

from __future__ import annotations

import os
import string
from collections.abc import Sequence
from typing import Any

import torch

from tests._tiny_tokenizer import tiny_tokenizer

#: Smallest vocabulary ``tiny_tokenizer`` accepts: specials + string.printable.
VOCAB_SIZE = 103
TARGET_CHARS = string.digits

TRAIN_PROMPTS = ["hello", "say a number", "count", "abc def"]
HELDOUT_PROMPTS = ["xyz", "what now?", "ok go", "tell me"]


def seeds() -> list[int]:
    """Seeds to run each learning test with (``LEARNING_SEEDS=0,1,2``)."""
    raw = os.environ.get("LEARNING_SEEDS", "0")
    return [int(s) for s in raw.split(",") if s.strip()]


def make_tokenizer() -> Any:
    return tiny_tokenizer(VOCAB_SIZE)


def make_model(
    seed: int, *, n_layer: int = 1, n_embd: int = 32, n_positions: int = 128
) -> Any:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=2,
            vocab_size=VOCAB_SIZE,
            n_positions=n_positions,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def make_agent(
    model: Any, tokenizer: Any, *, max_new_tokens: int = 6, **config: Any
) -> Any:
    """A real ``MultiTurnAgent`` wrapping the tiny model (sampling, T=1)."""
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    agent = MultiTurnAgent(
        AgentConfig(
            model_name="gpt2",
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=True,
            use_chat_template=False,
            **config,
        )
    )
    agent.model = model
    agent.tokenizer = tokenizer
    agent.generation_config = agent._build_generation_config()
    return agent


def digit_fraction(text: str) -> float:
    text = str(text)
    if not text:
        return 0.0
    return sum(ch in TARGET_CHARS for ch in text) / len(text)


def _last_assistant_content(turns: Sequence[Any]) -> str:
    for turn in reversed(list(turns)):
        role = turn.get("role") if isinstance(turn, dict) else turn.role
        if role == "assistant":
            return str(turn.get("content") if isinstance(turn, dict) else turn.content)
    return ""


class DigitReward:
    """Reward = ``sign`` * fraction of digit characters in the last reply."""

    def __init__(self, sign: float = 1.0) -> None:
        self.sign = float(sign)
        self.weight = 1.0
        self.name = "digit_reward"

    async def compute_reward(self, turns: Sequence[Any], context: Any = None) -> Any:
        from stateset_agents.core.reward_base import RewardResult

        return RewardResult(
            score=self.sign * digit_fraction(_last_assistant_content(turns))
        )

    async def compute_turn_reward(self, turn: Any, context: Any = None) -> Any:
        return await self.compute_reward([turn], context)


def target_token_ids(tokenizer: Any) -> list[int]:
    return [int(tokenizer.convert_tokens_to_ids(ch)) for ch in TARGET_CHARS]


def agent_prompt(prompt: str) -> str:
    """What ``MultiTurnAgent`` conditions on without a chat template."""
    return f"User: {prompt}\nAssistant: "


@torch.no_grad()
def target_mass(
    model: Any, tokenizer: Any, contexts: Sequence[str] | None = None
) -> float:
    """Mean probability of emitting a digit right after each context."""
    if contexts is None:
        contexts = [agent_prompt(p) for p in HELDOUT_PROMPTS]
    was_training = model.training
    model.eval()
    ids = target_token_ids(tokenizer)
    try:
        device = next(model.parameters()).device
        values = []
        for text in contexts:
            enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(device)).logits[0, -1]
            probs = torch.softmax(logits.float(), dim=-1)
            values.append(float(probs[ids].sum()))
    finally:
        model.train(was_training)
    return sum(values) / len(values)
