"""
CartPole Quick Start - Minimal Example

The simplest possible example of using StateSet-Agents with a Gym environment.
Perfect for getting started in just 5 minutes!

This example shows how 3 lines of code can wrap any Gym environment for GRPO training.

Usage:
    python examples/rl_environments/cartpole_quickstart.py
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Train GRPO on CartPole with minimal code."""

    logger.info("🚀 CartPole Quick Start - Minimal Example\n")

    # Step 1: Import
    logger.info("Step 1: Import framework components")
    from core.gym import GymEnvironmentAdapter, create_gym_agent
    from training.config import TrainingConfig
    from training.multi_turn_trainer import MultiTurnGRPOTrainer

    try:
        import gymnasium as gym
    except ImportError:
        import gym

    # Step 2: Create Gym Environment
    logger.info("Step 2: Create CartPole environment")
    gym_env = gym.make("CartPole-v1")

    # Step 3: Wrap for Framework (auto-creates processors!)
    logger.info("Step 3: Wrap with GymEnvironmentAdapter")
    env = GymEnvironmentAdapter(gym_env, auto_create_processors=True)

    # Step 4: Create Agent
    logger.info("Step 4: Create GymAgent")
    agent = create_gym_agent(
        model_name="gpt2",
        use_stub=True,  # Use stub for quick demo (no model download)
        temperature=0.8
    )
    await agent.initialize()

    # Step 5: Train!
    logger.info("Step 5: Train with GRPO (10 episodes)\n")

    config = TrainingConfig(
        num_episodes=10,  # Just 10 for quick demo
        num_generations=4,
        learning_rate=3e-5,
        log_interval=2
    )

    trainer = MultiTurnGRPOTrainer(agent, env, config)
    await trainer.train()

    logger.info("\n✅ Done! That's how easy it is!")
    logger.info("\nNext steps:")
    logger.info("  1. Set use_stub=False to use real GPT-2")
    logger.info("  2. Increase num_episodes to 100+ for real learning")
    logger.info("  3. Try other environments: MountainCar-v0, Pendulum-v1")

    env.close()


if __name__ == "__main__":
    asyncio.run(main())
