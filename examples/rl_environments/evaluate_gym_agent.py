"""
Evaluate Trained Gym Agents

Load a trained GRPO agent and evaluate its performance on a Gym environment.
Useful for testing after training and comparing different checkpoints.

Features:
- Load trained models
- Run multiple evaluation episodes
- Compute statistics (mean, std, min, max)
- Optional visualization (render episodes)
- Save evaluation results

Usage:
    # Evaluate CartPole agent
    python examples/rl_environments/evaluate_gym_agent.py --env CartPole-v1 --model outputs/cartpole_grpo/final_model

    # Evaluate with rendering
    python examples/rl_environments/evaluate_gym_agent.py --env CartPole-v1 --model outputs/cartpole_grpo/final_model --render

    # More episodes for better statistics
    python examples/rl_environments/evaluate_gym_agent.py --env CartPole-v1 --model outputs/cartpole_grpo/final_model --num-episodes 100
"""

import asyncio
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def evaluate_agent(
    env_id: str,
    agent,
    env_adapter,
    num_episodes: int = 50,
    render: bool = False,
    max_steps: int = 500
) -> Dict[str, float]:
    """
    Evaluate agent on environment.

    Args:
        env_id: Environment ID (e.g., "CartPole-v1")
        agent: Initialized agent
        env_adapter: GymEnvironmentAdapter
        num_episodes: Number of episodes to run
        render: Whether to render episodes
        max_steps: Max steps per episode

    Returns:
        Dict with evaluation metrics
    """
    from stateset_agents.core.trajectory import ConversationTurn

    logger.info(f"\n🧪 Evaluating agent on {env_id}...")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Render: {render}")

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        # Reset environment
        state = await env_adapter.reset()
        episode_reward = 0
        done = False
        turns = []

        # Get system prompt
        system_prompt = await env_adapter.get_initial_prompt()

        while not done and state.turn_count < max_steps:
            # Format history
            messages = [{"role": "system", "content": system_prompt}]

            # Add current observation
            obs_text = state.context.get("observation_text", "")
            messages.append({"role": "user", "content": obs_text})

            # Generate action
            response = await agent.generate_response(messages)
            action_turn = ConversationTurn(role="assistant", content=response)

            # Step environment
            state, obs_turn, reward, done = await env_adapter.step(state, action_turn)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_lengths.append(state.turn_count)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"  Episode {episode + 1}/{num_episodes}, "
                       f"Avg reward (last 10): {avg_reward:.2f}")

    # Compute statistics
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "num_episodes": num_episodes,
    }

    return results


async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained Gym agent")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                       help="Gym environment ID")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model (optional)")
    parser.add_argument("--num-episodes", type=int, default=50,
                       help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Max steps per episode")
    parser.add_argument("--use-stub", action="store_true",
                       help="Use stub model for testing")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Gym Agent Evaluation - {args.env}")
    logger.info("=" * 60)

    # Import framework components
    from stateset_agents.core.gym import GymEnvironmentAdapter, create_gym_agent

    try:
        import gymnasium as gym
    except ImportError:
        import gym

    # Create environment
    logger.info(f"\n[1/4] Creating {args.env} environment...")
    gym_env = gym.make(args.env, render_mode="human" if args.render else None)
    env_adapter = GymEnvironmentAdapter(
        gym_env,
        auto_create_processors=True,
        max_steps=args.max_steps
    )
    logger.info(f"✓ Environment created")

    # Create agent
    logger.info(f"\n[2/4] Creating agent...")
    agent = create_gym_agent(
        model_name="gpt2",
        use_stub=args.use_stub,
        temperature=0.3  # Lower temperature for evaluation (less random)
    )

    # Load model if path provided
    if args.model:
        model_path = Path(args.model)
        if model_path.exists():
            logger.info(f"  Loading model from {model_path}")
            # Note: In a full implementation, you'd load the model here
            # For now, we initialize a fresh agent
            logger.warning("  Model loading not implemented yet - using fresh agent")
        else:
            logger.warning(f"  Model path not found: {model_path}")
            logger.info("  Using fresh agent")

    await agent.initialize()
    logger.info(f"✓ Agent initialized")

    # Evaluate
    logger.info(f"\n[3/4] Running evaluation...")
    results = await evaluate_agent(
        env_id=args.env,
        agent=agent,
        env_adapter=env_adapter,
        num_episodes=args.num_episodes,
        render=args.render,
        max_steps=args.max_steps
    )

    # Report results
    logger.info(f"\n[4/4] Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Episodes: {results['num_episodes']}")
    logger.info(f"\nReward Statistics:")
    logger.info(f"  Mean:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    logger.info(f"  Min:    {results['min_reward']:.2f}")
    logger.info(f"  Max:    {results['max_reward']:.2f}")
    logger.info(f"\nEpisode Length:")
    logger.info(f"  Mean:   {results['mean_length']:.2f} ± {results['std_length']:.2f}")

    # Interpretation
    logger.info(f"\n📊 Interpretation:")
    if args.env.startswith("CartPole"):
        if results['mean_reward'] > 400:
            logger.info("  🏆 Excellent! Agent is performing near-optimally.")
        elif results['mean_reward'] > 200:
            logger.info("  ✅ Good! Agent has learned to balance well.")
        elif results['mean_reward'] > 50:
            logger.info("  📈 Decent. Agent is learning but needs more training.")
        else:
            logger.info("  ⚠️  Weak performance. More training needed.")
    elif args.env.startswith("MountainCar"):
        if results['mean_reward'] > -90:
            logger.info("  🏆 Excellent! Agent reaches goal very efficiently.")
        elif results['mean_reward'] > -110:
            logger.info("  ✅ Good! Agent consistently reaches the goal.")
        elif results['mean_reward'] > -150:
            logger.info("  📈 Progress! Agent sometimes reaches goal.")
        else:
            logger.info("  ⚠️  Struggling. Agent rarely reaches goal.")

    # Cleanup
    env_adapter.close()

    logger.info("\n✨ Evaluation complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
