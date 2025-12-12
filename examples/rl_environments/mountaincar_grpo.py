"""
MountainCar-v0 with GRPO Training

Train an agent to drive up a steep mountain by building momentum.
This is harder than CartPole because the car must rock back and forth
to build enough speed.

The environment:
- Observation: [position, velocity]
- Actions: 0 (push left), 1 (no push), 2 (push right)
- Goal: Reach the flag at position 0.5
- Challenge: Car's engine is too weak to drive straight up!

Expected Performance:
- Random baseline: ~-200 (times out at 200 steps)
- After 200 episodes: ~-150 to -100
- Good performance: ~-110 (reaching goal in 110 steps)
- Optimal: ~-90 to -80

Usage:
    python examples/rl_environments/mountaincar_grpo.py
"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Train GRPO agent on MountainCar-v0."""

    from stateset_agents.core.gym import GymEnvironmentAdapter, create_gym_agent
    from stateset_agents.core.agent import AgentConfig
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    logger.info("=" * 60)
    logger.info("MountainCar-v0 with GRPO - Building Momentum to Reach the Goal")
    logger.info("=" * 60)

    # 1. Create Gym Environment
    logger.info("\n[1/5] Creating MountainCar-v0 environment...")
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    gym_env = gym.make("MountainCar-v0")
    logger.info(f"✓ Created {gym_env.spec.id if hasattr(gym_env, 'spec') else 'MountainCar-v0'}")
    logger.info(f"  Action space: {gym_env.action_space} (0=left, 1=nothing, 2=right)")
    logger.info(f"  Observation space: {gym_env.observation_space} ([position, velocity])")
    logger.info(f"  Goal: Reach position 0.5 (flag on the right mountain)")

    # 2. Wrap with Adapter (auto-creates MountainCarObservationProcessor!)
    logger.info("\n[2/5] Wrapping environment for framework...")
    env_adapter = GymEnvironmentAdapter(
        gym_env,
        auto_create_processors=True,  # Automatically uses MountainCarObservationProcessor
        max_steps=200  # MountainCar default
    )
    logger.info(f"✓ Created {env_adapter}")
    logger.info(f"  Auto-selected: MountainCarObservationProcessor")

    # 3. Create GymAgent
    logger.info("\n[3/5] Creating GymAgent...")
    agent_config = AgentConfig(
        model_name="gpt2",
        max_new_tokens=5,  # Actions are 0, 1, or 2
        temperature=0.9,  # Higher exploration for harder task
        do_sample=True,
        use_stub_model=False  # Set to True for quick testing
    )

    agent = create_gym_agent(
        model_name="gpt2",
        use_stub=False,
        temperature=0.9  # More exploration
    )

    logger.info("  Initializing agent...")
    await agent.initialize()
    logger.info(f"✓ Created and initialized {agent}")

    # 4. Configure GRPO Training
    logger.info("\n[4/5] Configuring GRPO training...")
    training_config = TrainingConfig(
        num_episodes=200,  # MountainCar needs more episodes
        num_generations=8,  # More trajectories for better group advantages
        learning_rate=5e-5,  # Slightly higher for harder task
        batch_size=16,
        max_turns_per_episode=200,
        use_kl_penalty=True,
        kl_coef=0.02,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.02,  # Higher entropy for more exploration
        normalize_advantages=True,
        log_interval=20
    )

    logger.info(f"✓ Training configuration:")
    logger.info(f"  Episodes: {training_config.num_episodes}")
    logger.info(f"  Generations per episode: {training_config.num_generations}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Entropy bonus: {training_config.entropy_coef} (exploration)")

    # 5. Create Trainer and Train!
    logger.info("\n[5/5] Creating GRPO trainer and starting training...")
    logger.info("=" * 60)

    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=env_adapter,
        config=training_config,
        output_dir=Path("outputs/mountaincar_grpo")
    )

    logger.info("\n🚀 Starting GRPO training on MountainCar-v0...")
    logger.info("Watch the agent learn to rock back and forth!")
    logger.info("Challenge: The car must build momentum - it can't drive straight up.")
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
        logger.info(f"  Random Baseline: ~-200 (timeout)")
        logger.info(f"  Good Performance: ~-110 (110 steps to goal)")
        logger.info(f"  Optimal Performance: ~-90")

        if final_reward > -110:
            logger.info("\n🎉 Excellent! Agent learned to reach the goal efficiently!")
        elif final_reward > -150:
            logger.info("\n📈 Good progress! Agent is learning to build momentum.")
        else:
            logger.info("\n💡 MountainCar is hard! Try more episodes or higher exploration.")

    # 7. Save model
    logger.info("\n💾 Saving trained model...")
    output_dir = Path("outputs/mountaincar_grpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(agent.model, 'save_pretrained'):
        agent.model.save_pretrained(output_dir / "final_model")
        logger.info(f"✓ Model saved to {output_dir / 'final_model'}")

    # 8. Tips
    logger.info("\n💡 Tips for MountainCar:")
    logger.info("  - This task is harder than CartPole (sparse rewards)")
    logger.info("  - Agent must learn to rock back and forth")
    logger.info("  - Increase entropy_coef for more exploration")
    logger.info("  - Try reward shaping (give small rewards for gaining velocity)")
    logger.info("  - Or try GSPO instead of GRPO (better for sparse rewards)")

    # Cleanup
    env_adapter.close()
    logger.info("\n✨ Done!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
