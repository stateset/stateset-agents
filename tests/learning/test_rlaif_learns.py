"""RLAIFTrainer.train() learns from its judge's scores.

The LLM judge is replaced by a deterministic local judge (same
``evaluate(prompt, completion)`` contract) scoring the digit fraction, so the
test exercises the real sampling, advantage and update path offline.
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

EPISODES = 10


class _DigitJudge:
    def __init__(self, sign: float) -> None:
        self.sign = sign

    async def evaluate(self, prompt: str, completion: str) -> float:
        return self.sign * digit_fraction(completion)


def _train(seed: int, *, sign: float) -> tuple[float, float]:
    import torch

    from stateset_agents.training.rlaif_trainer import RLAIFConfig, RLAIFTrainer

    tokenizer = make_tokenizer()
    model = make_model(seed)
    config = RLAIFConfig(
        use_constitutional=False,
        use_cpu=True,
        max_prompt_length=32,
        max_completion_length=6,
        num_generations=8,
        per_device_train_batch_size=2,
        temperature=1.0,
        top_p=1.0,
        # RLAIF applies a plain SGD update with this step size.
        learning_rate=0.5,
        num_episodes=EPISODES,
        use_lora=False,
        gradient_checkpointing=False,
        seed=seed,
    )
    trainer = RLAIFTrainer(config, model=model, tokenizer=tokenizer)
    trainer.judge = _DigitJudge(sign)
    before = target_mass(model, tokenizer, HELDOUT_PROMPTS)
    torch.manual_seed(seed)  # fixed sampling stream
    asyncio.run(trainer.train(TRAIN_PROMPTS, num_episodes=EPISODES))
    return before, target_mass(model, tokenizer, HELDOUT_PROMPTS)


@pytest.mark.parametrize("seed", seeds())
def test_rlaif_increases_target_probability(seed: int) -> None:
    # Observed (seeds 0/1/2): 0.10 -> 0.99 / 0.99 / 0.99.
    before, after = _train(seed, sign=1.0)
    assert after > before + 0.3, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_rlaif_negated_reward_decreases_target_probability(seed: int) -> None:
    # Observed (seeds 0/1/2): 0.10 -> 0.072 / 0.076 / 0.072.
    before, after = _train(seed, sign=-1.0)
    assert after < before - 0.01, (before, after)
