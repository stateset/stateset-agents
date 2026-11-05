"""SingleTurnGRPOTrainer learns the digit task through its real ``train()`` loop.

Each episode resets a tiny environment, samples a group with the real
``MultiTurnAgent.generate_turn`` (token metadata included), scores it with the
reward function and takes a GRPO step -- exactly the user-facing loop.
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
    TRAIN_PROMPTS,
    DigitReward,
    make_agent,
    make_model,
    make_tokenizer,
    seeds,
    target_mass,
)

pytestmark = pytest.mark.learning

EPISODES = 16


class PromptEnvironment(Environment):
    """One-step episodes over a fixed list of prompt scenarios."""

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


def _train(
    seed: int, *, sign: float, monkeypatch: pytest.MonkeyPatch
) -> tuple[float, float, list[str]]:
    import torch

    from stateset_agents.training import single_turn_trainer as stt
    from stateset_agents.training.config import TrainingConfig

    paths: list[str] = []
    real_loss = stt.compute_grpo_loss

    def spy_loss(*args: Any, **kwargs: Any) -> Any:
        out = real_loss(*args, **kwargs)
        paths.append(str(out.get("path")))
        return out

    monkeypatch.setattr(stt, "compute_grpo_loss", spy_loss)

    tokenizer = make_tokenizer()
    model = make_model(seed)
    agent = make_agent(model, tokenizer)
    config = TrainingConfig(
        num_episodes=EPISODES,
        max_steps_per_episode=1,
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
        # Keep the fixed-budget negative control running after sampling
        # saturates; do not change production signal-guard defaults.
        zero_signal_patience=0,
        seed=seed,
    )
    trainer = stt.SingleTurnGRPOTrainer(
        agent=agent,
        environment=PromptEnvironment([{"prompt": p} for p in TRAIN_PROMPTS]),
        reward_fn=DigitReward(sign),
        config=config,
    )
    before = target_mass(model, tokenizer)

    async def run() -> None:
        await trainer.initialize()
        torch.manual_seed(seed)  # fixed sampling stream
        await trainer.train()

    asyncio.run(run())
    return before, target_mass(model, tokenizer), paths


@pytest.mark.parametrize("seed", seeds())
def test_single_turn_grpo_increases_target_probability(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after, paths = _train(seed, sign=1.0, monkeypatch=monkeypatch)
    assert paths and set(paths) == {"token"}, paths
    assert after > before + 0.06, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_single_turn_grpo_negated_reward_decreases_target_probability(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after, paths = _train(seed, sign=-1.0, monkeypatch=monkeypatch)
    assert paths and set(paths) == {"token"}, paths
    assert after < before - 0.015, (before, after)
