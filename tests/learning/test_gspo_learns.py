"""GSPO and GSPO-token learn the digit task through real HF rollouts.

With sequence-level rewards and one on-policy update per batch, GSPO-token's
gradient is mathematically identical to GSPO's (paper, eq. 13-14); the two
paths are separate code, and in practice produce matching trajectories.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("transformers")

from tests.learning._toy import (  # noqa: E402
    TRAIN_PROMPTS,
    DigitReward,
    make_agent,
    make_model,
    make_tokenizer,
    seeds,
    target_mass,
)

pytestmark = pytest.mark.learning

STEPS = 10


def _train(seed: int, *, sign: float, token_level: bool) -> tuple[float, float]:
    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    tokenizer = make_tokenizer()
    model = make_model(seed)
    agent = make_agent(model, tokenizer)
    config = GSPOConfig(
        model_name="gpt2",
        num_generations=8,
        num_outer_iterations=STEPS,
        num_iterations=1,
        max_prompt_length=32,
        max_completion_length=8,
        learning_rate=1e-2,
        beta=0.0,
        seed=seed,
    )
    trainer_cls = GSPOTokenTrainer if token_level else GSPOTrainer
    trainer = trainer_cls(
        config=config,
        model=model,
        tokenizer=tokenizer,
        agent=agent,
        environment=None,
        reward_model=DigitReward(sign),
        ref_model=None,
    )
    step = trainer.train_step_token_level if token_level else trainer.train_step
    before = target_mass(model, tokenizer)

    async def run() -> None:
        import torch

        torch.manual_seed(seed)  # fixed sampling stream
        for _ in range(STEPS):
            await step(TRAIN_PROMPTS, num_groups=2)

    asyncio.run(run())
    return before, target_mass(model, tokenizer)


@pytest.mark.parametrize("token_level", [False, True], ids=["gspo", "gspo_token"])
@pytest.mark.parametrize("seed", seeds())
def test_gspo_increases_target_probability(seed: int, token_level: bool) -> None:
    before, after = _train(seed, sign=1.0, token_level=token_level)
    # Observed deltas over seeds 0-4: +0.058 .. +0.101.
    assert after > before + 0.03, (before, after)


@pytest.mark.parametrize("token_level", [False, True], ids=["gspo", "gspo_token"])
@pytest.mark.parametrize("seed", seeds())
def test_gspo_negated_reward_decreases_target_probability(
    seed: int, token_level: bool
) -> None:
    before, after = _train(seed, sign=-1.0, token_level=token_level)
    # Observed deltas over seeds 0-4: -0.020 .. -0.030.
    assert after < before - 0.008, (before, after)
