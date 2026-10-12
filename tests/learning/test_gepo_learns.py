"""GEPO learns the digit task through its real HF rollout + train_step path."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("transformers")

from tests.learning._toy import (  # noqa: E402
    HELDOUT_PROMPTS,
    TRAIN_PROMPTS,
    digit_fraction,
    make_model,
    make_tokenizer,
    seeds,
    target_mass,
)

pytestmark = pytest.mark.learning

STEPS = 12


def _train(seed: int, *, sign: float) -> tuple[float, float]:
    import torch

    from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer

    tokenizer = make_tokenizer()
    model = make_model(seed)
    config = GEPOConfig(
        model_name="gpt2",
        use_lora=False,
        group_size=8,
        num_gradient_updates=1,
        num_episodes=STEPS,
        num_epochs=1,
        warmup_ratio=0.0,
        max_prompt_length=32,
        max_completion_length=6,
        learning_rate=1e-2,
        temperature=1.0,
        top_p=1.0,
        seed=seed,
    )

    def reward_fn(prompt: str, response: str) -> float:
        return sign * digit_fraction(response)

    trainer = GEPOTrainer(
        config=config, model=model, tokenizer=tokenizer, reward_fn=reward_fn
    )
    # GEPO conditions on the raw prompt (no chat rendering).
    before = target_mass(model, tokenizer, HELDOUT_PROMPTS)

    async def run() -> None:
        torch.manual_seed(seed)  # fixed sampling stream
        for step in range(STEPS):
            offset = (2 * step) % len(TRAIN_PROMPTS)
            await trainer.train_step((TRAIN_PROMPTS * 2)[offset : offset + 2])

    asyncio.run(run())
    return before, target_mass(model, tokenizer, HELDOUT_PROMPTS)


@pytest.mark.parametrize("seed", seeds())
def test_gepo_increases_target_probability(seed: int) -> None:
    before, after = _train(seed, sign=1.0)
    assert after > before + 0.04, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_gepo_negated_reward_decreases_target_probability(seed: int) -> None:
    before, after = _train(seed, sign=-1.0)
    assert after < before - 0.01, (before, after)
