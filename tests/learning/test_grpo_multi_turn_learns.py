"""MultiTurnGRPOTrainer learns the digit task on its batched per-token path.

Rollouts are real: ``trainer.generate_trajectories`` drives a tiny
``Environment`` whose episodes call the real ``MultiTurnAgent.generate_turn``
(which records ``prompt_token_ids``/``token_ids``), the trainer applies the
reward function, and ``trainer.training_step`` takes the optimizer step.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

pytest.importorskip("transformers")

from stateset_agents.core.environment_base import (  # noqa: E402
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from tests.learning._toy import (  # noqa: E402
    HELDOUT_PROMPTS,
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


class PromptEnvironment(Environment):
    """One-turn episodes: the scenario prompt opens the conversation."""

    def __init__(self, scenarios: list[dict[str, Any]]) -> None:
        super().__init__(max_turns=1)
        self.scenarios = scenarios

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        return EnvironmentState(
            episode_id=str(uuid.uuid4()),
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={"scenario": scenario or {}},
        )

    async def step(self, state: EnvironmentState, action: Any) -> Any:
        new_state = state.copy()
        new_state.turn_count += 1
        new_state.status = EpisodeStatus.COMPLETED
        return new_state, 0.0, True, {}

    async def get_initial_prompt(self, scenario: dict[str, Any] | None = None) -> str:
        return str((scenario or {}).get("prompt", ""))


def _contexts(agent: Any) -> list[str]:
    """Held-out prompts rendered exactly as the agent renders an episode."""
    return [
        agent._render_prompt([{"role": "system", "content": p}])[0]
        for p in HELDOUT_PROMPTS
    ]


def _train(seed: int, *, sign: float) -> tuple[float, float, list[dict[str, Any]]]:
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
        beta=0.0,
        bf16=False,
        fp16=False,
        report_to="none",
        seed=seed,
        # Thresholds were calibrated at a constant learning rate; initialize()
        # now builds the configured (default cosine + warmup) schedule.
        lr_scheduler_type="constant",
    )
    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=PromptEnvironment(scenarios),
        reward_fn=DigitReward(sign),
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
    return before, target_mass(model, tokenizer, contexts), metrics


@pytest.mark.parametrize("seed", seeds())
def test_multi_turn_grpo_increases_target_probability(seed: int) -> None:
    before, after, metrics = _train(seed, sign=1.0)
    assert all(m["path"] == "token" for m in metrics), [m.get("path") for m in metrics]
    assert all(m["optimizer_step"] for m in metrics)
    assert after > before + 0.15, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_multi_turn_grpo_negated_reward_decreases_target_probability(
    seed: int,
) -> None:
    before, after, metrics = _train(seed, sign=-1.0)
    assert all(m["path"] == "token" for m in metrics)
    assert after < before - 0.025, (before, after)
