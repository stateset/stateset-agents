"""
Interactive Gym Environment Testing

Test the gym adapter, processors, and mappers interactively.
Great for debugging and understanding how the components work.

Features:
- Test any Gym environment
- See observation processing in action
- Test action parsing
- Run manual episodes step-by-step
- Compare with raw gym API

Usage:
    # Test CartPole
    python examples/rl_environments/gym_environment_test.py CartPole-v1

    # Test MountainCar
    python examples/rl_environments/gym_environment_test.py MountainCar-v0

    # Test with custom environment
    python examples/rl_environments/gym_environment_test.py Pendulum-v1
"""

import asyncio
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_observation_processor(env_id: str, gym_env):
    """Test observation processor."""
    from stateset_agents.core.gym.processors import create_observation_processor
    import numpy as np

    logger.info("\n" + "=" * 60)
    logger.info("Test 1: Observation Processor")
    logger.info("=" * 60)

    processor = create_observation_processor(env_id)
    logger.info(f"✓ Created: {processor.__class__.__name__}")

    # Get sample observation
    obs, _ = gym_env.reset() if isinstance(gym_env.reset(), tuple) else (gym_env.reset(), {})

    logger.info(f"\nRaw Observation:")
    logger.info(f"  Type: {type(obs)}")
    logger.info(f"  Shape: {obs.shape if isinstance(obs, np.ndarray) else 'N/A'}")
    logger.info(f"  Values: {obs}")

    # Process observation
    processed = processor.process(obs)
    logger.info(f"\nProcessed Observation:")
    logger.info(f"  {processed}")

    # Get system prompt
    system_prompt = processor.get_system_prompt(gym_env)
    logger.info(f"\nSystem Prompt:")
    logger.info(f"  {system_prompt[:200]}..." if len(system_prompt) > 200 else f"  {system_prompt}")

    return processor


def test_action_mapper(env_id: str, gym_env):
    """Test action mapper."""
    from stateset_agents.core.gym.mappers import create_action_mapper

    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Action Mapper")
    logger.info("=" * 60)

    mapper = create_action_mapper(gym_env)
    logger.info(f"✓ Created: {mapper.__class__.__name__}")

    logger.info(f"\nAction Space: {gym_env.action_space}")

    # Test various action formats
    test_responses = [
        "0",
        "1",
        "Action: 0",
        "I choose 1",
        "The best move is action 0",
        "LEFT" if hasattr(mapper, 'name_to_action') else "invalid",
    ]

    logger.info(f"\nTesting action parsing:")
    for response in test_responses:
        try:
            action = mapper.parse_action(response)
            logger.info(f"  '{response}' → {action}")
        except Exception as e:
            logger.info(f"  '{response}' → Error: {e}")

    return mapper


async def test_gym_adapter(env_id: str, gym_env, processor, mapper):
    """Test GymEnvironmentAdapter."""
    from stateset_agents.core.gym.adapter import GymEnvironmentAdapter
    from stateset_agents.core.trajectory import ConversationTurn

    logger.info("\n" + "=" * 60)
    logger.info("Test 3: GymEnvironmentAdapter")
    logger.info("=" * 60)

    adapter = GymEnvironmentAdapter(
        gym_env,
        observation_processor=processor,
        action_mapper=mapper
    )
    logger.info(f"✓ Created: {adapter}")

    # Test reset
    logger.info(f"\nTesting reset()...")
    state = await adapter.reset()
    logger.info(f"  Episode ID: {state.episode_id}")
    logger.info(f"  Turn count: {state.turn_count}")
    logger.info(f"  Status: {state.status}")
    logger.info(f"  Observation text: {state.context.get('observation_text', '')[:100]}...")

    # Test step
    logger.info(f"\nTesting step()...")
    action_turn = ConversationTurn(role="assistant", content="0")
    new_state, obs_turn, reward, done = await adapter.step(state, action_turn)

    logger.info(f"  Action: {action_turn.content}")
    logger.info(f"  New turn count: {new_state.turn_count}")
    logger.info(f"  Reward: {reward}")
    logger.info(f"  Done: {done}")
    logger.info(f"  New observation: {obs_turn.content[:100]}...")

    # Test full episode
    logger.info(f"\nRunning full episode...")
    state = await adapter.reset()
    total_reward = 0
    step_count = 0
    max_steps = 50

    while not done and step_count < max_steps:
        # Random action
        import random
        action_str = str(random.randint(0, gym_env.action_space.n - 1))
        action_turn = ConversationTurn(role="assistant", content=action_str)

        state, obs_turn, reward, done = await adapter.step(state, action_turn)
        total_reward += reward
        step_count += 1

    logger.info(f"  Episode completed!")
    logger.info(f"  Steps: {step_count}")
    logger.info(f"  Total reward: {total_reward}")
    logger.info(f"  Final status: {state.status}")

    adapter.close()


async def test_with_agent(env_id: str, gym_env):
    """Test with actual agent."""
    from stateset_agents.core.gym import GymEnvironmentAdapter, create_gym_agent
    from stateset_agents.core.trajectory import ConversationTurn

    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Full Integration with Agent")
    logger.info("=" * 60)

    # Create adapter
    adapter = GymEnvironmentAdapter(gym_env, auto_create_processors=True)
    logger.info(f"✓ Created adapter")

    # Create agent (stub mode for speed)
    agent = create_gym_agent(model_name="gpt2", use_stub=True, temperature=0.8)
    await agent.initialize()
    logger.info(f"✓ Created agent (stub mode)")

    # Run episode
    logger.info(f"\nRunning episode with agent...")
    state = await adapter.reset()
    system_prompt = await adapter.get_initial_prompt()
    total_reward = 0
    step_count = 0
    max_steps = 20  # Just 20 steps for demo

    while step_count < max_steps:
        # Generate action with agent
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state.context.get("observation_text", "")}
        ]

        response = await agent.generate_response(messages)
        action_turn = ConversationTurn(role="assistant", content=response)

        # Step environment
        state, obs_turn, reward, done = await adapter.step(state, action_turn)
        total_reward += reward
        step_count += 1

        if step_count <= 3:  # Show first 3 steps
            logger.info(f"  Step {step_count}: action='{response.strip()}', reward={reward:.2f}")

        if done:
            break

    logger.info(f"\n  Episode completed!")
    logger.info(f"  Steps: {step_count}")
    logger.info(f"  Total reward: {total_reward:.2f}")

    adapter.close()


async def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test Gym environment integration")
    parser.add_argument("env_id", type=str, default="CartPole-v1", nargs="?",
                       help="Gym environment ID (default: CartPole-v1)")
    args = parser.parse_args()

    try:
        import gymnasium as gym
    except ImportError:
        import gym

    logger.info("=" * 60)
    logger.info(f"Gym Environment Testing - {args.env_id}")
    logger.info("=" * 60)

    # Create environment
    logger.info(f"\nCreating {args.env_id}...")
    gym_env = gym.make(args.env_id)
    logger.info(f"✓ Environment created")
    logger.info(f"  Action space: {gym_env.action_space}")
    logger.info(f"  Observation space: {gym_env.observation_space}")

    # Run tests
    processor = test_observation_processor(args.env_id, gym_env)
    mapper = test_action_mapper(args.env_id, gym_env)
    await test_gym_adapter(args.env_id, gym_env, processor, mapper)

    # Create fresh env for agent test (previous one is closed)
    gym_env = gym.make(args.env_id)
    await test_with_agent(args.env_id, gym_env)

    logger.info("\n" + "=" * 60)
    logger.info("✅ All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
