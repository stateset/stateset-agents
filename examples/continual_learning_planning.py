"""
Continual learning + long-term planning example.

This script shows how to:
1) Enable long-term planning in the agent.
2) Train with replay + LwF across multiple tasks using task_id.
"""

import asyncio
import logging

from stateset_agents import ConversationEnvironment, MultiTurnAgent, train
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    config = AgentConfig(
        model_name="gpt2",
        system_prompt="You are a helpful assistant.",
        enable_planning=True,
        planning_config={"max_steps": 4, "update_interval": 1},
    )
    agent = MultiTurnAgent(config)
    await agent.initialize()

    scenarios = [
        {
            "id": "travel_1",
            "task_id": "travel",
            "topic": "travel",
            "context": "User needs help planning a trip.",
            "user_responses": [
                "I want to visit Japan next month.",
                "I like food and temples.",
                "Can you suggest an itinerary?",
            ],
        },
        {
            "id": "billing_1",
            "task_id": "billing",
            "topic": "billing",
            "context": "User has a billing question.",
            "user_responses": [
                "I was double-charged for my order.",
                "Can you help me get a refund?",
                "Thanks for looking into it.",
            ],
        },
    ]

    environment = ConversationEnvironment(scenarios=scenarios, max_turns=6)
    reward_fn = CompositeReward(
        [HelpfulnessReward(weight=0.7), SafetyReward(weight=0.3)]
    )

    trained_agent = await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=20,
        profile="balanced",
        config_overrides={
            "continual_strategy": "replay_lwf",
            "continual_kl_beta": 0.1,
            "replay_buffer_size": 200,
            "replay_ratio": 0.3,
            "replay_sampling": "balanced",
            "task_id_key": "task_id",
            "task_schedule": ["travel", "billing"],
            "task_switch_steps": 5,
        },
    )

    context = {
        "conversation_id": "demo-trip",
        "goal": "Plan a 4-day trip to Tokyo and Kyoto",
    }
    messages = [{"role": "user", "content": "Can you draft a trip plan?"}]
    response = await trained_agent.generate_response(messages, context=context)
    logger.info("Response: %s", response)

    update_context = {
        "conversation_id": "demo-trip",
        "plan_update": {"action": "advance"},
    }
    followup_messages = [
        {"role": "user", "content": "Great. What should we do next?"}
    ]
    followup = await trained_agent.generate_response(
        followup_messages, context=update_context
    )
    logger.info("Updated response: %s", followup)


if __name__ == "__main__":
    asyncio.run(main())
