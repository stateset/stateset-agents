"""
Integration tests for the end-to-end training pipeline using the stub backend.

These tests cover gaps in the unit suite: composite reward flows, checkpoint
roundtrips, agent reset semantics, and multi-episode stability. They stay
stub-only (no GPU / no model weights) so they can run in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.agent_backends import StubModel
from stateset_agents.core.basic_rewards import (
    ConcisenessReward,
    EngagementReward,
    HelpfulnessReward,
    SafetyReward,
)
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward_base import CompositeReward
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer


def _customer_scenarios(count: int = 3) -> list[dict]:
    return [
        {
            "id": f"scenario_{i}",
            "context": f"Customer needs help with issue #{i}",
            "user_responses": [
                f"I have a problem with my order {i}.",
                "It hasn't arrived yet.",
                "Please check and let me know.",
            ],
        }
        for i in range(count)
    ]


def _stub_agent(responses: list[str] | None = None) -> MultiTurnAgent:
    config = AgentConfig(
        model_name="stub://integration",
        use_stub_model=True,
        max_new_tokens=32,
        stub_responses=responses
        or [
            "I can help with that.",
            "Let me take a look.",
            "Here is what I found.",
        ],
    )
    return MultiTurnAgent(config)


@pytest.mark.asyncio
async def test_pipeline_with_composite_reward_completes():
    agent = _stub_agent()
    env = ConversationEnvironment(scenarios=_customer_scenarios(2), max_turns=3)
    reward_fn = CompositeReward(
        reward_functions=[
            HelpfulnessReward(weight=0.6),
            SafetyReward(weight=0.2),
            ConcisenessReward(weight=0.2),
        ],
        combination_method="weighted_sum",
        normalize_weights=True,
    )
    config = TrainingConfig(
        num_episodes=2,
        max_steps_per_episode=3,
        num_generations=1,
        learning_rate=1e-5,
        bf16=False,
        seed=7,
    )
    trainer = SingleTurnGRPOTrainer(
        agent=agent, environment=env, reward_fn=reward_fn, config=config
    )

    await trainer.initialize()
    result = await trainer.train()

    assert result is agent
    assert trainer.global_step >= 1


@pytest.mark.asyncio
async def test_composite_reward_survives_component_failure():
    """If one reward fails, the composite must still score the rest."""

    class BrokenReward(HelpfulnessReward):
        async def compute_reward(self, turns, context=None):
            raise RuntimeError("intentional failure")

    reward_fn = CompositeReward(
        reward_functions=[
            BrokenReward(weight=0.5),
            HelpfulnessReward(weight=0.5),
        ]
    )

    from stateset_agents.core.trajectory import ConversationTurn

    turns = [
        ConversationTurn(
            role="assistant",
            content="I'd be glad to help you with that, please share details.",
        )
    ]
    result = await reward_fn.compute_reward(turns)

    assert result.score >= 0.0
    assert "helpfulness" in result.components
    assert "broken" not in result.components


@pytest.mark.asyncio
async def test_multi_turn_agent_reset_clears_state():
    agent = _stub_agent()
    await agent.initialize()

    await agent.generate_response("First message")
    await agent.reset()

    # Fresh state → still produces a string without raising
    response = await agent.generate_response("Hello again")
    assert isinstance(response, str) and response


@pytest.mark.asyncio
async def test_stub_backend_is_detected_via_property():
    agent = _stub_agent()
    await agent.initialize()

    assert agent._is_stub_backend is True
    assert isinstance(agent.model, StubModel)


@pytest.mark.asyncio
async def test_checkpoint_path_resolution_and_directory_created(tmp_path):
    """save_checkpoint should create an artifact directory under output_dir."""
    agent = _stub_agent()
    env = ConversationEnvironment(scenarios=_customer_scenarios(1), max_turns=2)
    reward_fn = HelpfulnessReward()
    config = TrainingConfig(
        num_episodes=1,
        max_steps_per_episode=2,
        num_generations=1,
        learning_rate=1e-5,
        bf16=False,
        output_dir=str(tmp_path / "ckpt"),
    )
    trainer = SingleTurnGRPOTrainer(
        agent=agent, environment=env, reward_fn=reward_fn, config=config
    )
    await trainer.initialize()
    await trainer.train()

    await trainer.save_checkpoint(checkpoint_name="smoke")
    checkpoint_root = Path(config.output_dir)
    assert checkpoint_root.exists()
    # At least one subdirectory / file was written
    assert any(checkpoint_root.iterdir())


@pytest.mark.asyncio
async def test_pipeline_stable_across_multiple_episodes():
    """Run the trainer twice; state should remain consistent."""
    agent = _stub_agent()
    env = ConversationEnvironment(scenarios=_customer_scenarios(3), max_turns=2)
    reward_fn = HelpfulnessReward()
    config = TrainingConfig(
        num_episodes=3,
        max_steps_per_episode=2,
        num_generations=1,
        learning_rate=1e-5,
        bf16=False,
    )
    trainer = SingleTurnGRPOTrainer(
        agent=agent, environment=env, reward_fn=reward_fn, config=config
    )
    await trainer.initialize()
    await trainer.train()
    first_step = trainer.global_step

    # Running again should continue to advance, not crash
    await trainer.train()
    assert trainer.global_step >= first_step


@pytest.mark.asyncio
async def test_environment_episode_returns_trajectory_turns():
    agent = _stub_agent(["Sure, I can help."])
    await agent.initialize()

    scenarios = _customer_scenarios(1)
    env = ConversationEnvironment(scenarios=scenarios, max_turns=2)

    trajectory = await env.run_episode(
        agent_fn=agent.generate_response, scenario=scenarios[0]
    )

    assert trajectory.turns, "Expected at least one turn"


@pytest.mark.asyncio
async def test_composite_reward_serializes_components_for_logging():
    reward_fn = CompositeReward(
        reward_functions=[
            HelpfulnessReward(weight=1.0),
            EngagementReward(weight=0.5),
        ]
    )
    from stateset_agents.core.trajectory import ConversationTurn

    result = await reward_fn.compute_reward(
        [ConversationTurn(role="assistant", content="Absolutely — happy to help!")]
    )

    # Metadata is JSON-serializable (common sink for W&B / telemetry)
    payload = json.dumps(result.metadata, default=str)
    assert "component_names" in payload


@pytest.mark.asyncio
async def test_training_config_bf16_false_runs_on_cpu_backend():
    """Guard against accidental CUDA/bf16 requirements in the stub path."""
    agent = _stub_agent()
    env = ConversationEnvironment(scenarios=_customer_scenarios(1), max_turns=2)
    reward_fn = HelpfulnessReward()
    config = TrainingConfig(
        num_episodes=1,
        max_steps_per_episode=2,
        num_generations=1,
        learning_rate=1e-5,
        bf16=False,
        fp16=False,
    )
    trainer = SingleTurnGRPOTrainer(
        agent=agent, environment=env, reward_fn=reward_fn, config=config
    )
    await trainer.initialize()
    result = await trainer.train()
    assert result is agent


@pytest.mark.asyncio
async def test_stub_responses_cycle_across_turns():
    agent = _stub_agent(["A", "B", "C"])
    await agent.initialize()

    collected = [await agent.generate_response(f"q{i}") for i in range(5)]
    joined = " | ".join(collected)
    # Cycle guarantees every configured response is observed at least once
    for expected in ("A", "B", "C"):
        assert expected in joined


@pytest.mark.asyncio
async def test_helpfulness_reward_scales_with_content():
    reward_fn = HelpfulnessReward()
    from stateset_agents.core.trajectory import ConversationTurn

    terse = await reward_fn.compute_reward(
        [ConversationTurn(role="assistant", content="ok")]
    )
    helpful = await reward_fn.compute_reward(
        [
            ConversationTurn(
                role="assistant",
                content=(
                    "Absolutely — I can help. Let me walk you through the steps "
                    "to resolve this and confirm the details on your account."
                ),
            )
        ]
    )
    assert helpful.score >= terse.score


@pytest.mark.asyncio
async def test_safety_reward_penalizes_unsafe_content():
    reward_fn = SafetyReward()
    from stateset_agents.core.trajectory import ConversationTurn

    safe = await reward_fn.compute_reward(
        [ConversationTurn(role="assistant", content="Happy to help.")]
    )
    unsafe = await reward_fn.compute_reward(
        [
            ConversationTurn(
                role="assistant",
                content="I'll help you hack into that account right now.",
            )
        ]
    )
    assert safe.score >= unsafe.score


@pytest.mark.asyncio
async def test_environment_respects_max_turns():
    agent = _stub_agent(["hi"])
    await agent.initialize()
    scenarios = [
        {
            "id": "loop",
            "context": "checking max_turns",
            "user_responses": ["a", "b", "c", "d", "e"],
        }
    ]
    env = ConversationEnvironment(scenarios=scenarios, max_turns=2)
    trajectory = await env.run_episode(
        agent_fn=agent.generate_response, scenario=scenarios[0]
    )
    assistant_turns = [t for t in trajectory.turns if getattr(t, "role", None) == "assistant"]
    # Max_turns caps assistant turns at 2
    assert len(assistant_turns) <= 2


@pytest.mark.asyncio
async def test_trainer_initialize_is_idempotent():
    agent = _stub_agent()
    env = ConversationEnvironment(scenarios=_customer_scenarios(1), max_turns=2)
    reward_fn = HelpfulnessReward()
    config = TrainingConfig(
        num_episodes=1, max_steps_per_episode=2, num_generations=1, bf16=False
    )
    trainer = SingleTurnGRPOTrainer(
        agent=agent, environment=env, reward_fn=reward_fn, config=config
    )
    await trainer.initialize()
    await trainer.initialize()  # second call must not raise
    result = await trainer.train()
    assert result is agent


@pytest.mark.asyncio
async def test_composite_min_combination_picks_lowest_component():
    reward_fn = CompositeReward(
        reward_functions=[HelpfulnessReward(), SafetyReward()],
        combination_method="min",
    )
    from stateset_agents.core.trajectory import ConversationTurn

    result = await reward_fn.compute_reward(
        [ConversationTurn(role="assistant", content="Sure, here's the info.")]
    )
    component_scores = result.metadata.get("component_scores", [])
    assert result.score == min(component_scores)


@pytest.mark.asyncio
async def test_composite_max_combination_picks_highest_component():
    reward_fn = CompositeReward(
        reward_functions=[HelpfulnessReward(), SafetyReward()],
        combination_method="max",
    )
    from stateset_agents.core.trajectory import ConversationTurn

    result = await reward_fn.compute_reward(
        [ConversationTurn(role="assistant", content="Of course, here you go.")]
    )
    component_scores = result.metadata.get("component_scores", [])
    assert result.score == max(component_scores)
