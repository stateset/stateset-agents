"""GSPO's trust region is exercised when one rollout feeds several updates.

With ``num_iterations=1`` every update is on-policy, so the sequence ratio is
exactly 1 and the clip never activates. The GSPO paper's clip exists for the
multi-update regime: rollouts are scored once under a frozen old policy and
then reused for ``num_iterations`` optimizer steps. This test checks that the
trainer honours ``num_iterations`` (one optimizer step per iteration), that
later iterations see ratios away from 1 and some get clipped, and that the
policy still learns the rewarded direction.
"""

from __future__ import annotations

import asyncio
from typing import Any

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

STEPS = 6
ITERATIONS = 3


def _train(
    seed: int, *, num_iterations: int, token_level: bool = False
) -> tuple[float, list[dict[str, Any]]]:
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
        num_iterations=num_iterations,
        max_prompt_length=32,
        max_completion_length=8,
        learning_rate=1e-2,
        beta=0.0,
        zero_signal_patience=0,
        seed=seed,
    )
    trainer_cls = GSPOTokenTrainer if token_level else GSPOTrainer
    trainer = trainer_cls(
        config=config,
        model=model,
        tokenizer=tokenizer,
        agent=agent,
        environment=None,
        reward_model=DigitReward(1.0),
        ref_model=None,
    )
    before = target_mass(model, tokenizer)
    metrics: list[dict[str, Any]] = []

    async def run() -> None:
        import torch

        torch.manual_seed(seed)
        for _ in range(STEPS):
            step = trainer.train_step_token_level if token_level else trainer.train_step
            metrics.append(await step(TRAIN_PROMPTS, num_groups=2))

    asyncio.run(run())
    return target_mass(model, tokenizer) - before, metrics


def test_num_iterations_is_no_longer_reported_as_ignored() -> None:
    from stateset_agents.training.gspo_config import GSPOConfig

    warnings = GSPOConfig(model_name="gpt2", num_iterations=3).validate()
    assert not any("num_iterations" in w for w in warnings), warnings


@pytest.mark.parametrize("token_level", [False, True], ids=["gspo", "gspo_token"])
@pytest.mark.parametrize("seed", seeds())
def test_single_iteration_is_on_policy(seed: int, token_level: bool) -> None:
    _, metrics = _train(seed, num_iterations=1, token_level=token_level)
    assert all(m["optimizer_updates"] == 1 for m in metrics)
    assert all(m["clipping_fraction"] == 0.0 for m in metrics)


@pytest.mark.parametrize("token_level", [False, True], ids=["gspo", "gspo_token"])
@pytest.mark.parametrize("seed", seeds())
def test_multiple_iterations_engage_the_clip_and_learn(
    seed: int, token_level: bool
) -> None:
    drift, metrics = _train(seed, num_iterations=ITERATIONS, token_level=token_level)

    assert all(m["optimizer_updates"] == ITERATIONS for m in metrics), metrics
    # Later iterations are off-policy relative to the frozen rollout scores.
    assert any(m["clipping_fraction"] > 0.0 for m in metrics), [
        m["clipping_fraction"] for m in metrics
    ]
    assert drift > 0.03, drift
