"""
CartPole-v1 with GRPO Training

This example demonstrates using the stateset-agents framework for classic RL tasks.
We train an agent to balance a pole on a cart using Group Relative Policy Optimization (GRPO).

This proves the framework is not just for conversational AI, but a general-purpose RL framework!

Expected Performance:
- Random baseline: ~20-30 reward
- After 50 episodes: ~50-100 reward
- After 200 episodes: ~150-300 reward
- Optimal: 500 (episode length limit)

Usage:
    python examples/rl_environments/cartpole_grpo.py
"""

import asyncio
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Train GRPO agent on CartPole-v1."""

    # Import framework components
    from core.gym import GymEnvironmentAdapter, create_gym_agent
    from core.agent import AgentConfig
    from training.config import TrainingConfig
    from training.multi_turn_trainer import MultiTurnGRPOTrainer

    logger.info("=" * 60)
    logger.info("CartPole-v1 with GRPO - Classic RL meets Modern LLMs")
    logger.info("=" * 60)

    # 1. Create Gym Environment
    logger.info("\n[1/5] Creating CartPole-v1 environment...")
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    gym_env = gym.make("CartPole-v1")
    logger.info(f"✓ Created {gym_env.spec.id if hasattr(gym_env, 'spec') else 'CartPole-v1'}")
    logger.info(f"  Action space: {gym_env.action_space}")
    logger.info(f"  Observation space: {gym_env.observation_space}")

    # 2. Wrap with GymEnvironmentAdapter
    logger.info("\n[2/5] Wrapping environment for framework...")
    env_adapter = GymEnvironmentAdapter(
        gym_env,
        auto_create_processors=True,  # Automatically creates CartPoleObservationProcessor
        max_steps=500
    )
    logger.info(f"✓ Created {env_adapter}")

    # 3. Create GymAgent
    logger.info("\n[3/5] Creating GymAgent...")
    agent_config = AgentConfig(
        model_name="gpt2",  # Small, fast model
        max_new_tokens=5,  # Very short for CartPole actions (0 or 1)
        temperature=0.8,  # Moderate exploration
        do_sample=True,  # Enable sampling for exploration
        use_stub_model=False  # Set to True for quick testing
    )

    agent = create_gym_agent(
        model_name="gpt2",
        use_stub=False,
        temperature=0.8
    )

    logger.info("  Initializing agent...")
    await agent.initialize()
    logger.info(f"✓ Created and initialized {agent}")

    # 4. Configure GRPO Training
    logger.info("\n[4/5] Configuring GRPO training...")
    training_config = TrainingConfig(
        num_episodes=100,  # Start with 100 episodes
        num_generations=4,  # Generate 4 trajectories per episode for group advantages
        learning_rate=3e-5,
        batch_size=8,
        max_turns_per_episode=500,  # CartPole max steps
        use_kl_penalty=True,
        kl_coef=0.02,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_ratio=0.2,  # PPO-style clipping
        value_loss_coef=0.5,
        entropy_coef=0.01,  # Small entropy bonus for exploration
        normalize_advantages=True,
        log_interval=10  # Log every 10 episodes
    )

    logger.info(f"✓ Training configuration:")
    logger.info(f"  Episodes: {training_config.num_episodes}")
    logger.info(f"  Generations per episode: {training_config.num_generations}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Discount factor (gamma): {training_config.gamma}")

    # 5. Create Trainer and Train!
    logger.info("\n[5/5] Creating GRPO trainer and starting training...")
    logger.info("=" * 60)

    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=env_adapter,
        config=training_config,
        output_dir=Path("outputs/cartpole_grpo")
    )

    logger.info("\n🚀 Starting GRPO training on CartPole-v1...")
    logger.info("Watch the agent learn to balance the pole!")
    logger.info("-" * 60)

    # Train!
    metrics = await trainer.train()

    # 6. Report Results
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    if metrics:
        final_reward = metrics.get("final_average_reward", 0)
        best_reward = metrics.get("best_reward", 0)
        logger.info(f"\nFinal Results:")
        logger.info(f"  Final Average Reward: {final_reward:.2f}")
        logger.info(f"  Best Reward: {best_reward:.2f}")
        logger.info(f"  Random Baseline: ~22")
        logger.info(f"  Optimal Performance: 500")

        if final_reward > 100:
            logger.info("\n🎉 Great! Agent is learning to balance the pole!")
        elif final_reward > 50:
            logger.info("\n📈 Good progress! Agent is improving.")
        else:
            logger.info("\n💡 Try training longer or adjusting hyperparameters.")

    # 7. Save model
    logger.info("\n💾 Saving trained model...")
    output_dir = Path("outputs/cartpole_grpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(agent.model, 'save_pretrained'):
        agent.model.save_pretrained(output_dir / "final_model")
        logger.info(f"✓ Model saved to {output_dir / 'final_model'}")

    # 8. Cleanup
    env_adapter.close()
    logger.info("\n✨ Done!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
