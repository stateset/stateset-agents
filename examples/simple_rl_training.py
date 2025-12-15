"""
Simple RL Training Example

This script demonstrates the minimal setup required to train an agent using
Group Sequence Policy Optimization (GSPO). It uses a small model (GPT-2)
and a simple environment to show the core training loop.

Usage:
    python examples/simple_rl_training.py
"""

import asyncio
import logging
from stateset_agents.core.agent import MultiTurnAgent, AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import create_domain_reward
from stateset_agents.training.gspo_trainer import GSPOConfig, train_with_gspo
from stateset_agents.training.config import TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 1. Configure the Agent
    # We use GPT-2 as it's small and can run on most machines for demonstration.
    logger.info("Initializing Agent...")
    agent_config = AgentConfig(
        model_name="gpt2",
        system_prompt="You are a helpful assistant.",
        max_new_tokens=64,  # Keep responses short for speed
        temperature=0.7,
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    # 2. Configure the Environment
    # We use a simple conversation environment with a basic scenario.
    logger.info("Setting up Environment...")
    scenarios = [
        {
            "id": "scenario_1",
            "topic": "greeting",
            "user_goal": "Exchange pleasantries",
            "context": "The user is just saying hello.",
            "user_responses": ["Hello!", "How are you?", "Nice to meet you."]
        }
    ]
    environment = ConversationEnvironment(
        scenarios=scenarios,
        max_turns=5,  # Short episodes
        persona="You are a friendly bot."
    )

    # 3. Configure the Reward Model
    # We use a pre-built domain reward for general helpfulness.
    logger.info("Initializing Reward Model...")
    reward_model = create_domain_reward("customer_service") 

    # 4. Configure Training
    logger.info("Configuring Training...")
    # Base configuration
    base_config = TrainingConfig(
        run_name="simple_rl_demo",
        output_dir="./outputs/simple_rl",
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )
    
    # GSPO specific configuration
    gspo_config = GSPOConfig.from_training_config(
        base_config,
        num_outer_iterations=2,  # Very few iterations for demo
        generations_per_iteration=4,
        num_generations=2,       # Group size (K)
        learning_rate=1e-5,
        save_steps=10,
        logging_steps=1,
        report_to="none"         # Disable wandb for simple demo
    )

    # 5. Run Training
    logger.info("Starting Training Loop...")
    trained_agent = await train_with_gspo(
        config=gspo_config,
        agent=agent,
        environment=environment,
        reward_model=reward_model
    )

    logger.info("Training Complete!")
    
    # 6. Test the trained agent
    logger.info("Testing the agent...")
    response = await trained_agent.generate_response([{"role": "user", "content": "Hello!"}])
    logger.info(f"Agent response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
