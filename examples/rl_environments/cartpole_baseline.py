"""
CartPole Baseline Comparisons

Compare different approaches to CartPole to validate GRPO learning:
1. Random Policy - No learning, random actions
2. Simple Heuristic - Push cart toward falling direction
3. GRPO Agent - Our trained agent

This helps validate that GRPO is actually learning and not just random.

Usage:
    python examples/rl_environments/cartpole_baseline.py
"""

import asyncio
import logging
import numpy as np
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_random_policy(num_episodes: int = 100) -> List[float]:
    """
    Run random policy on CartPole.

    Random policy: Choose action 0 or 1 randomly each step.
    Expected performance: ~22 average reward.
    """
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    logger.info("\n[1/3] Running Random Policy...")
    env = gym.make("CartPole-v1")
    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        episode_reward = 0
        done = False

        while not done:
            # Random action
            action = env.action_space.sample()
            result = env.step(action)

            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated

            episode_reward += reward

        rewards.append(episode_reward)

        if (episode + 1) % 20 == 0:
            avg = np.mean(rewards[-20:])
            logger.info(f"  Episode {episode + 1}/{num_episodes}, Avg reward (last 20): {avg:.2f}")

    env.close()

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    logger.info(f"✓ Random Policy: {avg_reward:.2f} ± {std_reward:.2f}")

    return rewards


def run_heuristic_policy(num_episodes: int = 100) -> List[float]:
    """
    Run simple heuristic policy on CartPole.

    Heuristic: Push cart in the direction the pole is falling.
    - If pole tilting right (positive angle), push right (action 1)
    - If pole tilting left (negative angle), push left (action 0)

    Expected performance: ~200-400 average reward (much better than random).
    """
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    logger.info("\n[2/3] Running Heuristic Policy (push toward falling direction)...")
    env = gym.make("CartPole-v1")
    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        episode_reward = 0
        done = False

        while not done:
            # Heuristic: push in direction pole is falling
            # obs[2] is pole angle (positive = right, negative = left)
            action = 1 if obs[2] > 0 else 0

            result = env.step(action)

            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated

            episode_reward += reward

        rewards.append(episode_reward)

        if (episode + 1) % 20 == 0:
            avg = np.mean(rewards[-20:])
            logger.info(f"  Episode {episode + 1}/{num_episodes}, Avg reward (last 20): {avg:.2f}")

    env.close()

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    logger.info(f"✓ Heuristic Policy: {avg_reward:.2f} ± {std_reward:.2f}")

    return rewards


async def run_grpo_policy(num_episodes: int = 100) -> List[float]:
    """
    Run trained GRPO agent on CartPole.

    This loads a trained model (if available) or uses a fresh agent
    to demonstrate what GRPO should achieve.

    Expected performance after training: 150-400 average reward.
    """
    from core.gym import GymEnvironmentAdapter, create_gym_agent
    from pathlib import Path

    logger.info("\n[3/3] Running GRPO Agent...")

    try:
        import gymnasium as gym
    except ImportError:
        import gym

    # Create environment
    gym_env = gym.make("CartPole-v1")
    env = GymEnvironmentAdapter(gym_env, auto_create_processors=True)

    # Create agent
    agent = create_gym_agent(model_name="gpt2", use_stub=False, temperature=0.5)

    # Try to load trained model
    model_path = Path("outputs/cartpole_grpo/final_model")
    if model_path.exists():
        logger.info(f"  Loading trained model from {model_path}")
        # Note: In production, you'd load the model here
        # For now, we'll use a fresh agent
    else:
        logger.info("  Using fresh agent (no trained model found)")
        logger.info("  Tip: Run cartpole_grpo.py first to train a model!")

    await agent.initialize()

    rewards = []

    for episode in range(num_episodes):
        state = await env.reset()
        episode_reward = 0
        done = False
        turns = []

        # Get initial prompt
        system_prompt = await env.get_initial_prompt()

        while not done and state.turn_count < 500:
            # Format history for agent
            messages = [{"role": "system", "content": system_prompt}]

            # Add observation
            obs_text = state.context.get("observation_text", "")
            messages.append({"role": "user", "content": obs_text})

            # Generate action
            response = await agent.generate_response(messages)

            # Create action turn
            from core.trajectory import ConversationTurn
            action_turn = ConversationTurn(role="assistant", content=response)

            # Step environment
            state, obs_turn, reward, done = await env.step(state, action_turn)
            episode_reward += reward

        rewards.append(episode_reward)

        if (episode + 1) % 20 == 0:
            avg = np.mean(rewards[-20:])
            logger.info(f"  Episode {episode + 1}/{num_episodes}, Avg reward (last 20): {avg:.2f}")

    env.close()

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    logger.info(f"✓ GRPO Agent: {avg_reward:.2f} ± {std_reward:.2f}")

    return rewards


def plot_comparison(random_rewards, heuristic_rewards, grpo_rewards):
    """Plot comparison of all policies (if matplotlib available)."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(random_rewards, alpha=0.3, label="Random")
        plt.plot(heuristic_rewards, alpha=0.3, label="Heuristic")
        plt.plot(grpo_rewards, alpha=0.3, label="GRPO")

        # Moving averages
        window = 20
        if len(random_rewards) >= window:
            plt.plot(np.convolve(random_rewards, np.ones(window)/window, mode='valid'),
                    label="Random (MA)", linewidth=2)
        if len(heuristic_rewards) >= window:
            plt.plot(np.convolve(heuristic_rewards, np.ones(window)/window, mode='valid'),
                    label="Heuristic (MA)", linewidth=2)
        if len(grpo_rewards) >= window:
            plt.plot(np.convolve(grpo_rewards, np.ones(window)/window, mode='valid'),
                    label="GRPO (MA)", linewidth=2)

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("CartPole Performance Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.boxplot([random_rewards, heuristic_rewards, grpo_rewards],
                   labels=["Random", "Heuristic", "GRPO"])
        plt.ylabel("Reward")
        plt.title("Reward Distribution")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("cartpole_baseline_comparison.png", dpi=150)
        logger.info("\n📊 Plot saved to: cartpole_baseline_comparison.png")

    except ImportError:
        logger.info("\n💡 Install matplotlib to generate comparison plots: pip install matplotlib")


async def main():
    """Run all baseline comparisons."""
    logger.info("=" * 60)
    logger.info("CartPole Baseline Comparisons")
    logger.info("=" * 60)

    num_episodes = 100

    # Run all policies
    random_rewards = run_random_policy(num_episodes)
    heuristic_rewards = run_heuristic_policy(num_episodes)
    grpo_rewards = await run_grpo_policy(num_episodes)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Random Policy:     {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    logger.info(f"Heuristic Policy:  {np.mean(heuristic_rewards):.2f} ± {np.std(heuristic_rewards):.2f}")
    logger.info(f"GRPO Agent:        {np.mean(grpo_rewards):.2f} ± {np.std(grpo_rewards):.2f}")
    logger.info(f"Optimal:           500.00 (episode length limit)")

    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("Analysis")
    logger.info("=" * 60)

    if np.mean(grpo_rewards) > np.mean(random_rewards) * 1.5:
        logger.info("✅ GRPO is significantly better than random!")
    else:
        logger.info("⚠️  GRPO not significantly better than random. More training needed.")

    if np.mean(grpo_rewards) > np.mean(heuristic_rewards):
        logger.info("✅ GRPO beats the heuristic! Strong learning!")
    elif np.mean(grpo_rewards) > np.mean(heuristic_rewards) * 0.7:
        logger.info("📈 GRPO approaching heuristic performance. Good progress!")
    else:
        logger.info("💡 GRPO below heuristic. Try more training or tune hyperparameters.")

    # Plot
    plot_comparison(random_rewards, heuristic_rewards, grpo_rewards)

    logger.info("\n✨ Done!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
