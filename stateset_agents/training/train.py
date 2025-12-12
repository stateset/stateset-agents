"""
High-level training interface for GRPO Agent Framework
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from stateset_agents.core.agent import Agent, MultiTurnAgent
from stateset_agents.core.environment import Environment
from stateset_agents.core.reward import RewardFunction
from .config import TrainingConfig, TrainingProfile
from .diagnostics import DiagnosticsMonitor
from .trainer import MultiTurnGRPOTrainer

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """Training modes"""

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    AUTO = "auto"


async def train(
    agent: Agent,
    environment: Environment,
    reward_fn: Optional[RewardFunction] = None,
    num_episodes: int = 1000,
    profile: str = "balanced",
    training_mode: str = "auto",
    config_overrides: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> Agent:
    """
    High-level training function for agents

    Args:
        agent: Agent to train
        environment: Environment for training
        reward_fn: Reward function (if None, uses environment's reward)
        num_episodes: Number of training episodes
        profile: Training profile ("conservative", "balanced", "aggressive")
        training_mode: Training mode ("single_turn", "multi_turn", "auto")
        config_overrides: Custom configuration overrides
        save_path: Path to save trained agent
        **kwargs: Additional arguments

    Returns:
        Trained agent
    """

    logger.info(f"Starting training with profile: {profile}")

    # Determine training mode
    if training_mode == "auto":
        training_mode = (
            "multi_turn" if isinstance(agent, MultiTurnAgent) else "single_turn"
        )

    # Create training configuration
    config = TrainingConfig.from_profile(
        profile=TrainingProfile(profile),
        num_episodes=num_episodes,
        **(config_overrides or {}),
    )

    # Set up reward function
    if reward_fn is None:
        reward_fn = environment.reward_fn

    # Create trainer based on mode
    if training_mode == "multi_turn":
        trainer = MultiTurnGRPOTrainer(
            agent=agent, environment=environment, reward_fn=reward_fn, config=config
        )
    else:
        from training.trainer import SingleTurnGRPOTrainer
        trainer = SingleTurnGRPOTrainer(
            agent=agent, environment=environment, reward_fn=reward_fn, config=config
        )

    # Add diagnostics
    diagnostics = DiagnosticsMonitor(config)
    trainer.add_callback(diagnostics)

    # Initialize and train
    await trainer.initialize()
    trained_agent = await trainer.train()

    # Save if requested
    if save_path:
        await trainer.save_checkpoint(save_path)
        logger.info(f"Trained agent saved to {save_path}")

    return trained_agent


class AutoTrainer:
    """
    Automatic trainer that optimizes hyperparameters and handles training
    """

    def __init__(
        self, auto_adjust: bool = True, early_stopping: bool = True, patience: int = 50
    ):
        self.auto_adjust = auto_adjust
        self.early_stopping = early_stopping
        self.patience = patience

    async def train(
        self,
        agent: Agent,
        environment: Environment,
        reward_fn: Optional[RewardFunction] = None,
        **kwargs,
    ) -> Agent:
        """
        Automatically train agent with optimization
        """

        # Analyze task difficulty
        logger.info("Analyzing task characteristics...")
        task_analysis = await self._analyze_task(agent, environment, reward_fn)

        # Select optimal profile based on analysis
        profile = self._select_profile(task_analysis)
        logger.info(f"Selected training profile: {profile}")

        # Configure auto-adjustments
        config_overrides = {
            "auto_adjust": self.auto_adjust,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
        }

        # Train with selected profile
        return await train(
            agent=agent,
            environment=environment,
            reward_fn=reward_fn,
            profile=profile,
            config_overrides=config_overrides,
            **kwargs,
        )

    async def _analyze_task(
        self,
        agent: Agent,
        environment: Environment,
        reward_fn: Optional[RewardFunction],
    ) -> Dict[str, Any]:
        """Analyze task characteristics"""

        logger.info("Running baseline evaluation...")

        # Run a few episodes to assess baseline performance
        baseline_rewards = []
        reward_diversities = []

        for _ in range(5):  # Small sample for analysis
            try:
                # Generate trajectory
                async def agent_wrapper(history, context):
                    return await agent.generate_response(history, context)

                trajectory = await environment.run_episode(agent_wrapper)
                baseline_rewards.append(trajectory.total_reward)

                # Calculate reward diversity if we have turn rewards
                if trajectory.turn_rewards:
                    import numpy as np

                    reward_diversities.append(np.std(trajectory.turn_rewards))

            except Exception as e:
                logger.warning(f"Baseline evaluation episode failed: {e}")
                continue

        if not baseline_rewards:
            logger.warning("No successful baseline episodes, using default analysis")
            return {"performance": 0.0, "diversity": 0.0, "difficulty": "unknown"}

        import numpy as np

        avg_performance = np.mean(baseline_rewards)
        avg_diversity = np.mean(reward_diversities) if reward_diversities else 0.0

        # Determine difficulty
        if avg_performance < 0.1:
            difficulty = "very_hard"
        elif avg_performance < 0.3:
            difficulty = "hard"
        elif avg_performance < 0.7:
            difficulty = "moderate"
        else:
            difficulty = "easy"

        analysis = {
            "performance": avg_performance,
            "diversity": avg_diversity,
            "difficulty": difficulty,
            "num_episodes": len(baseline_rewards),
        }

        logger.info(f"Task analysis: {analysis}")
        return analysis

    def _select_profile(self, task_analysis: Dict[str, Any]) -> str:
        """Select optimal training profile based on task analysis"""

        difficulty = task_analysis.get("difficulty", "moderate")
        performance = task_analysis.get("performance", 0.5)
        diversity = task_analysis.get("diversity", 0.1)

        # Profile selection logic
        if difficulty in ["very_hard", "hard"] or performance < 0.2:
            return "conservative"  # Prioritize stability
        elif difficulty == "easy" and performance > 0.8:
            return "aggressive"  # Can afford to be aggressive
        elif diversity < 0.05:
            return "conservative"  # Low diversity needs stability
        else:
            return "balanced"  # Default for moderate cases


# Convenience functions for common training scenarios


async def train_customer_service_agent(
    model_name: str = "openai/gpt-oss-120b",
    scenarios_file: Optional[str] = None,
    num_episodes: int = 500,
    **kwargs,
) -> Agent:
    """Train a customer service agent"""

    from ..core.agent import AGENT_CONFIGS, create_agent
    from ..core.environment import CONVERSATION_CONFIGS, ConversationEnvironment
    from ..core.reward import create_customer_service_reward

    # Create agent
    agent_config = AGENT_CONFIGS["customer_service"].copy()
    agent_config.update(kwargs.get("agent_config", {}))

    agent = create_agent(agent_type="multi_turn", model_name=model_name, **agent_config)

    # Create environment
    env_config = CONVERSATION_CONFIGS["customer_service"].copy()
    if scenarios_file:
        import json

        with open(scenarios_file, "r") as f:
            env_config["scenarios"] = json.load(f)

    environment = ConversationEnvironment(**env_config)

    # Create reward function
    reward_fn = create_customer_service_reward()

    # Train
    return await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=num_episodes,
        **kwargs,
    )


async def train_tutoring_agent(
    model_name: str = "openai/gpt-oss-120b",
    subject: str = "general",
    num_episodes: int = 800,
    **kwargs,
) -> Agent:
    """Train a tutoring agent"""

    from ..core.agent import AGENT_CONFIGS, create_agent
    from ..core.environment import CONVERSATION_CONFIGS, ConversationEnvironment
    from ..core.reward import create_tutoring_reward

    # Create agent
    agent_config = AGENT_CONFIGS["tutor"].copy()
    agent_config.update(kwargs.get("agent_config", {}))

    agent = create_agent(agent_type="multi_turn", model_name=model_name, **agent_config)

    # Create environment
    env_config = CONVERSATION_CONFIGS["tutoring"].copy()
    environment = ConversationEnvironment(**env_config)

    # Create reward function
    reward_fn = create_tutoring_reward()

    # Train
    return await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=num_episodes,
        **kwargs,
    )


async def train_task_agent(
    model_name: str = "openai/gpt-oss-120b",
    task_type: str = "data_analysis",
    num_episodes: int = 600,
    **kwargs,
) -> Agent:
    """Train a task-oriented agent"""

    from ..core.agent import AGENT_CONFIGS, create_agent
    from ..core.environment import TASK_CONFIGS, TaskEnvironment
    from ..core.reward import create_task_agent_reward

    # Create agent
    agent_config = AGENT_CONFIGS["helpful_assistant"].copy()
    agent_config.update(kwargs.get("agent_config", {}))

    agent = create_agent(agent_type="multi_turn", model_name=model_name, **agent_config)

    # Create environment
    env_config = TASK_CONFIGS.get(task_type, TASK_CONFIGS["data_analysis"])

    def success_criteria(turns, context):
        return context.get("task_progress", 0.0) >= 1.0

    environment = TaskEnvironment(success_criteria=success_criteria, **env_config)

    # Create reward function
    task_criteria = env_config["tasks"][0] if env_config["tasks"] else {}
    reward_fn = create_task_agent_reward(task_criteria)

    # Train
    return await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=num_episodes,
        **kwargs,
    )
