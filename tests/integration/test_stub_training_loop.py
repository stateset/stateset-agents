"""
Integration test: end-to-end training loop using the stub backend.

This test validates the entire pipeline — agent → environment → reward →
trainer — without requiring a GPU or real model weights.  If this test
breaks, it means the training plumbing has a real interface mismatch.
"""

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import HelpfulnessReward
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.trainer import SingleTurnGRPOTrainer


@pytest.mark.asyncio
async def test_stub_training_loop_completes():
    """Run a minimal training loop end-to-end with the stub backend."""
    # 1. Agent
    agent_config = AgentConfig(
        model_name="stub://integration-test",
        use_stub_model=True,
        max_new_tokens=32,
        stub_responses=[
            "I'd be happy to help you with that!",
            "Let me look into this for you.",
        ],
    )
    agent = MultiTurnAgent(agent_config)

    # 2. Environment with simple scenarios
    scenarios = [
        {
            "id": "scenario_1",
            "context": "User needs help with a product",
            "user_responses": [
                "Hi, I need help with my order.",
                "Order number 12345.",
                "Thanks!",
            ],
        },
        {
            "id": "scenario_2",
            "context": "User has a technical question",
            "user_responses": [
                "How do I reset my password?",
                "I don't have access to my email.",
                "Okay, I'll try that.",
            ],
        },
    ]
    env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

    # 3. Reward function (real, not mocked)
    reward_fn = HelpfulnessReward()

    # 4. Training config (minimal)
    training_config = TrainingConfig(
        num_episodes=2,
        max_steps_per_episode=3,
        num_generations=1,
        learning_rate=1e-5,
        seed=42,
        bf16=False,
    )

    # 5. Trainer
    trainer = SingleTurnGRPOTrainer(
        agent=agent,
        environment=env,
        reward_fn=reward_fn,
        config=training_config,
    )

    await trainer.initialize()
    result_agent = await trainer.train()

    # Verify the training loop actually ran
    assert trainer.global_step >= 1
    assert result_agent is agent

    # Verify the agent still works after training
    response = await agent.generate_response("Hello, can you help me?")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_stub_agent_generates_deterministic_responses():
    """Verify stub agent returns configured responses in order."""
    responses = ["Alpha", "Beta", "Gamma"]
    config = AgentConfig(
        model_name="stub://deterministic",
        use_stub_model=True,
        stub_responses=responses,
    )
    agent = MultiTurnAgent(config)
    await agent.initialize()

    collected = []
    for i in range(3):
        resp = await agent.generate_response(f"Message {i}")
        collected.append(resp)

    # Stub should cycle through responses
    for i, expected in enumerate(responses):
        assert (
            expected in collected[i]
        ), f"Turn {i}: expected '{expected}' in '{collected[i]}'"


@pytest.mark.asyncio
async def test_conversation_environment_episode():
    """Run a full episode in ConversationEnvironment with a stub agent."""
    config = AgentConfig(
        model_name="stub://episode",
        use_stub_model=True,
        stub_responses=["I can help with that."],
    )
    agent = MultiTurnAgent(config)
    await agent.initialize()

    scenarios = [
        {
            "id": "test_ep",
            "context": "Simple test",
            "user_responses": [
                "Hello",
                "Thanks",
            ],
        }
    ]
    env = ConversationEnvironment(scenarios=scenarios, max_turns=2)

    trajectory = await env.run_episode(
        agent_fn=agent.generate_response,
        scenario=scenarios[0],
    )

    assert len(trajectory.turns) >= 1
    # Find the first user/assistant exchange (skip system turns)
    conversation_turns = [
        t
        for t in trajectory.turns
        if t.user_message is not None or t.assistant_response is not None
    ]
    assert len(conversation_turns) >= 1 or len(trajectory.turns) >= 2
