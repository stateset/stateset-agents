"""The KL penalty measurably restrains policy drift on the real token path.

Same seed, same rollouts budget: a run with ``beta > 0`` against a frozen
reference must move the held-out target probability less than an unregularized
run, while still moving it in the rewarded direction. This guards the whole KL
wiring (reference model creation, k3 token objective, loss composition), not
just the formula.
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
from tests.learning.test_grpo_multi_turn_learns import (  # noqa: E402
    PromptEnvironment,
    _contexts,
)

pytestmark = pytest.mark.learning

STEPS = 10


def _drift(seed: int, *, beta: float) -> tuple[float, list[dict[str, Any]]]:
    import torch

    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    tokenizer = make_tokenizer()
    model = make_model(seed)
    agent = make_agent(model, tokenizer)
    scenarios = [{"id": f"s{i}", "prompt": p} for i, p in enumerate(TRAIN_PROMPTS)]
    config = TrainingConfig(
        num_episodes=STEPS,
        num_generations=8,
        learning_rate=1e-2,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        max_prompt_length=64,
        max_completion_length=8,
        beta=beta,
        use_reference_model=beta > 0,
        # Fixed-budget learning and negative controls may saturate this toy
        # sampler. Production zero-signal stopping is tested independently.
        zero_signal_patience=0,
        bf16=False,
        fp16=False,
        report_to="none",
        seed=seed,
        lr_scheduler_type="constant",
    )
    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=PromptEnvironment(scenarios),
        reward_fn=DigitReward(1.0),
        config=config,
    )
    contexts = _contexts(agent)
    before = target_mass(model, tokenizer, contexts)
    metrics: list[dict[str, Any]] = []

    async def run() -> None:
        await trainer.initialize()
        torch.manual_seed(seed)  # fixed sampling stream
        for step in range(STEPS):
            batch = [scenarios[(2 * step) % 4], scenarios[(2 * step + 1) % 4]]
            groups = await trainer.generate_trajectories(batch, num_generations=8)
            metrics.append(await trainer.training_step(groups))

    asyncio.run(run())
    if beta > 0:
        assert trainer.reference_model is not None
    return target_mass(model, tokenizer, contexts) - before, metrics


@pytest.mark.parametrize("seed", seeds())
def test_kl_penalty_restrains_drift_but_keeps_direction(seed: int) -> None:
    free, _ = _drift(seed, beta=0.0)
    restrained, metrics = _drift(seed, beta=1.0)

    assert all(m["path"] == "token" for m in metrics), [m.get("path") for m in metrics]
    assert all(m["kl_penalty"] > 0 for m in metrics[1:]), [
        m["kl_penalty"] for m in metrics
    ]
    # Observed over seeds 0-2: free +0.30 .. +0.43, restrained/free 0.34 .. 0.37.
    assert free > 0.15, free
    assert 0.03 < restrained < 0.6 * free, (restrained, free)
