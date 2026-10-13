"""PPOTrainer.train() learns the digit task from its own sampled completions."""

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

EPISODES = 10


def _train(seed: int, *, sign: float) -> tuple[float, float]:
    import torch

    from stateset_agents.experimental.ppo import PPOConfig, PPOTrainer

    tokenizer = make_tokenizer()
    model = make_model(seed)
    config = PPOConfig(
        model_name="gpt2",
        max_prompt_length=32,
        max_completion_length=6,
        num_ppo_epochs=2,
        mini_batch_size=8,
        learning_rate=1e-2,
        use_cpu=True,
        num_generations=8,
        per_device_train_batch_size=2,
        temperature=1.0,
        top_p=1.0,
        beta=0.0,
        entropy_coef=0.0,
        num_episodes=EPISODES,
        use_lora=False,
        gradient_checkpointing=False,
        use_reference_model=False,
        warmup_ratio=0.0,
        seed=seed,
    )
    trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_fn=lambda _prompt, completion: sign * digit_fraction(completion),
    )
    # PPO conditions on the raw prompt text (no chat rendering).
    before = target_mass(model, tokenizer, HELDOUT_PROMPTS)
    torch.manual_seed(seed)  # fixed sampling stream
    asyncio.run(trainer.train(TRAIN_PROMPTS, num_episodes=EPISODES))
    return before, target_mass(model, tokenizer, HELDOUT_PROMPTS)


@pytest.mark.parametrize("seed", seeds())
def test_ppo_increases_target_probability(seed: int) -> None:
    # Observed (seeds 0/1/2): 0.10 -> 0.26 / 0.46 / 0.56.
    before, after = _train(seed, sign=1.0)
    assert after > before + 0.08, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_ppo_negated_reward_decreases_target_probability(seed: int) -> None:
    # Observed (seeds 0/1/2): 0.10 -> 0.045 / 0.034 / 0.041.
    before, after = _train(seed, sign=-1.0)
    assert after < before - 0.02, (before, after)
