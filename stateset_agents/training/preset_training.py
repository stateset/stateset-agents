"""
Scenario-specific training convenience helpers.
"""

from __future__ import annotations

from typing import Any

from stateset_agents.core.agent import Agent


async def train_customer_service_agent(
    model_name: str = "stub://default",
    scenarios_file: str | None = None,
    num_episodes: int = 500,
    **kwargs,
) -> Agent:
    """Train a customer service agent."""
    from ..core.agent import AGENT_CONFIGS, create_agent
    from ..core.environment import CONVERSATION_CONFIGS, ConversationEnvironment
    from ..core.reward import create_customer_service_reward
    from .train import train

    agent_config = AGENT_CONFIGS["customer_service"].copy()
    agent_config.update(kwargs.get("agent_config", {}))

    agent = create_agent(agent_type="multi_turn", model_name=model_name, **agent_config)

    env_config = CONVERSATION_CONFIGS["customer_service"].copy()
    if scenarios_file:
        import json

        with open(scenarios_file) as handle:
            env_config["scenarios"] = json.load(handle)

    environment = ConversationEnvironment(**env_config)
    reward_fn = create_customer_service_reward()

    return await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=num_episodes,
        **kwargs,
    )


async def train_tutoring_agent(
    model_name: str = "stub://default",
    subject: str = "general",
    num_episodes: int = 800,
    **kwargs,
) -> Agent:
    """Train a tutoring agent."""
    from ..core.agent import AGENT_CONFIGS, create_agent
    from ..core.environment import CONVERSATION_CONFIGS, ConversationEnvironment
    from ..core.reward import create_tutoring_reward
    from .train import train

    agent_config = AGENT_CONFIGS["tutor"].copy()
    agent_config.update(kwargs.get("agent_config", {}))

    agent = create_agent(agent_type="multi_turn", model_name=model_name, **agent_config)

    env_config = CONVERSATION_CONFIGS["tutoring"].copy()
    environment = ConversationEnvironment(**env_config)
    reward_fn = create_tutoring_reward()

    return await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=num_episodes,
        **kwargs,
    )


async def train_task_agent(
    model_name: str = "stub://default",
    task_type: str = "data_analysis",
    num_episodes: int = 600,
    **kwargs,
) -> Agent:
    """Train a task-oriented agent."""
    from ..core.agent import AGENT_CONFIGS, create_agent
    from ..core.environment import TASK_CONFIGS, TaskEnvironment
    from ..core.reward import create_task_agent_reward
    from .train import train

    agent_config = AGENT_CONFIGS["helpful_assistant"].copy()
    agent_config.update(kwargs.get("agent_config", {}))

    agent = create_agent(agent_type="multi_turn", model_name=model_name, **agent_config)

    env_config = TASK_CONFIGS.get(task_type, TASK_CONFIGS["data_analysis"])

    def success_criteria(turns: Any, context: dict[str, Any]) -> bool:
        return float(context.get("task_progress", 0.0)) >= 1.0

    environment = TaskEnvironment(success_criteria=success_criteria, **env_config)
    task_criteria = env_config["tasks"][0] if env_config["tasks"] else {}
    reward_fn = create_task_agent_reward(task_criteria)

    return await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=num_episodes,
        **kwargs,
    )


__all__ = [
    "train_customer_service_agent",
    "train_tutoring_agent",
    "train_task_agent",
]
