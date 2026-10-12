"""The enhanced DPOTrainer learns a digits-over-letters preference.

Training pairs prefer a random 4-digit reply over a random 4-letter reply;
the held-out pairs use different prompts and strings. The policy's
log-prob margin log pi(digits) - log pi(letters) must grow, and must shrink
when the preference labels are swapped.
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("transformers")

from tests.learning._toy import (  # noqa: E402
    HELDOUT_PROMPTS,
    TRAIN_PROMPTS,
    make_model,
    make_tokenizer,
    seeds,
)

pytestmark = pytest.mark.learning

STEPS = 10
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _digit_letter_pairs(prompts: list[str], rng: random.Random) -> list[tuple]:
    return [
        (
            prompt,
            "".join(rng.choice(DIGITS) for _ in range(4)),
            "".join(rng.choice(LETTERS) for _ in range(4)),
        )
        for prompt in prompts
    ]


def _digit_margin(trainer: Any, pairs: list[tuple]) -> float:
    import torch

    model = trainer.agent.model
    model.eval()
    with torch.no_grad():
        margins = [
            float(
                trainer._response_log_prob(model, *trainer._encode_pair(p, digits))
                - trainer._response_log_prob(model, *trainer._encode_pair(p, letters))
            )
            for p, digits, letters in pairs
        ]
    model.train()
    return sum(margins) / len(margins)


def _train(seed: int, *, prefer_digits: bool) -> tuple[float, float]:
    from stateset_agents.core.enhanced.advanced_rl_algorithms import (
        DPOConfig,
        DPOTrainer,
    )

    tokenizer = make_tokenizer()
    agent = SimpleNamespace(model=make_model(seed), tokenizer=tokenizer)
    trainer = DPOTrainer(
        agent, DPOConfig(learning_rate=1e-2, beta=0.1, max_length=64), device="cpu"
    )
    heldout = _digit_letter_pairs(HELDOUT_PROMPTS, random.Random(1000))
    before = _digit_margin(trainer, heldout)
    rng = random.Random(seed)

    async def run() -> None:
        for _ in range(STEPS):
            batch = [
                {
                    "prompt": p,
                    "chosen": digits if prefer_digits else letters,
                    "rejected": letters if prefer_digits else digits,
                }
                for p, digits, letters in _digit_letter_pairs(TRAIN_PROMPTS, rng)
            ]
            await trainer.train_step(batch)

    asyncio.run(run())
    return before, _digit_margin(trainer, heldout)


@pytest.mark.parametrize("seed", seeds())
def test_dpo_increases_chosen_over_rejected_margin(seed: int) -> None:
    # Observed (seeds 0/1/2): margin 0.25 / -0.09 / 0.40 -> 14.9 / 14.0 / 15.1.
    before, after = _train(seed, prefer_digits=True)
    assert after > before + 3.0, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_dpo_swapped_preference_decreases_margin(seed: int) -> None:
    # Observed (seeds 0/1/2): margin 0.25 / -0.09 / 0.40 -> -13.2 / -13.2 / -11.3.
    before, after = _train(seed, prefer_digits=False)
    assert after < before - 3.0, (before, after)
