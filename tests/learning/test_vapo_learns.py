"""VAPO learns the digit task through its real HF rollout + train_step path.

``train_step`` first spends its calls on value warmup (returning early), then
runs joint actor/critic updates; the test drives both phases.
"""

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

WARMUP_STEPS = 2
STEPS = 12
NEGATIVE_STEPS = 20


def _train(seed: int, *, sign: float, steps: int = STEPS) -> tuple[float, float]:
    import numpy as np
    import torch

    from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer

    tokenizer = make_tokenizer()
    model = make_model(seed)
    config = VAPOConfig(
        model_name="gpt2",
        use_lora=False,
        group_size=8,
        per_device_train_batch_size=2,
        num_episodes=steps,
        num_epochs=1,
        warmup_ratio=0.0,
        value_warmup_steps=WARMUP_STEPS,
        value_hidden_size=64,
        # Paper alpha=0.05 gives lambda_policy = 1 - 1/(alpha*len) <= 0 for
        # completions under 20 tokens, which VAPO clamps to 0 (pure TD(0) on a
        # terminal-only reward, so credit flows only through a barely-warmed
        # critic). Scale alpha to the 6-token toy so lambda_policy ~ 0.83.
        lambda_policy_alpha=1.0,
        max_prompt_length=32,
        max_completion_length=6,
        actor_learning_rate=1e-2,
        critic_learning_rate=1e-2,
        temperature=1.0,
        top_p=1.0,
        seed=seed,
    )

    def reward_fn(prompt: str, response: str) -> float:
        return sign * digit_fraction(response)

    trainer = VAPOTrainer(
        config=config, model=model, tokenizer=tokenizer, reward_fn=reward_fn
    )
    # VAPO conditions on the raw prompt (no chat rendering).
    before = target_mass(model, tokenizer, HELDOUT_PROMPTS)

    async def run() -> None:
        torch.manual_seed(seed)  # fixed sampling stream
        np.random.seed(seed)  # warmup samples prompts with numpy
        await trainer.train_step(TRAIN_PROMPTS)  # value warmup phase
        assert trainer.value_warmup_complete
        for step in range(steps):
            offset = (2 * step) % len(TRAIN_PROMPTS)
            await trainer.train_step((TRAIN_PROMPTS * 2)[offset : offset + 2])

    asyncio.run(run())
    return before, target_mass(model, tokenizer, HELDOUT_PROMPTS)


@pytest.mark.parametrize("seed", seeds())
def test_vapo_increases_target_probability(seed: int) -> None:
    before, after = _train(seed, sign=1.0)
    assert after > before + 0.04, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_vapo_negated_reward_decreases_target_probability(seed: int) -> None:
    # Digits are rare (~10%) in samples, so pushing them down is slower than
    # pushing them up: at 12 steps seed 2 moved only -0.004, so the control
    # trains longer. Observed deltas at 20 steps, seeds 0-4: -0.015 .. -0.056.
    before, after = _train(seed, sign=-1.0, steps=NEGATIVE_STEPS)
    assert after < before - 0.006, (before, after)
