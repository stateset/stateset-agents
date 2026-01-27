"""
GSPO Training Demo

This example demonstrates how to train conversational agents using the
Group Sequence Policy Optimization (GSPO) algorithm.

GSPO provides superior training stability, efficiency, and performance compared
to GRPO, especially for large MoE models and long responses.

Usage:
    python examples/train_with_gspo.py --task customer_service --model gpt2
    python examples/train_with_gspo.py --task technical_support --use-gspo-token
"""

import argparse
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def train_gspo_demo(
    task: str = "customer_service",
    model_name: str = "gpt2",
    use_gspo_token: bool = False,
    num_iterations: int = 10,
    output_dir: str = "./outputs/gspo_demo",
):
    """
    Demonstrate GSPO training for conversational agents.

    Args:
        task: Task type (customer_service, technical_support, sales)
        model_name: Model to use (gpt2 recommended for demo)
        use_gspo_token: Use GSPO-token variant
        num_iterations: Number of training iterations
        output_dir: Output directory for checkpoints
    """
    from stateset_agents import MultiTurnAgent
    from stateset_agents.core.agent import AgentConfig
    from stateset_agents.core.environment import ConversationEnvironment, CONVERSATION_CONFIGS
    from stateset_agents.rewards.multi_objective_reward import (
        create_customer_service_reward,
        create_domain_reward,
    )
    from stateset_agents.training.config import get_config_for_task
    from stateset_agents.training.gspo_trainer import GSPOConfig, train_with_gspo
    from stateset_agents.training.gspo_token_trainer import train_with_gspo_token

    logger.info(f"🚀 Starting GSPO Training Demo")
    logger.info(f"Task: {task}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Variant: {'GSPO-token' if use_gspo_token else 'GSPO'}")

    # Create agent
    logger.info("Initializing agent...")
    system_prompts = {
        "customer_service": "You are a helpful and empathetic customer service representative.",
        "technical_support": "You are a knowledgeable technical support specialist.",
        "sales": "You are a friendly and persuasive sales representative.",
    }

    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt=system_prompts.get(
            task, "You are a helpful assistant."
        ),
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()
    logger.info("✅ Agent initialized")

    # Create environment
    logger.info("Setting up environment...")
    if task in CONVERSATION_CONFIGS:
        env_config = CONVERSATION_CONFIGS[task].copy()
    else:
        env_config = CONVERSATION_CONFIGS["customer_service"].copy()

    environment = ConversationEnvironment(**env_config)
    logger.info(f"✅ Environment configured with {len(environment.scenarios)} scenarios")

    # Create reward model
    logger.info("Initializing reward model...")
    if task == "customer_service":
        reward_model = create_customer_service_reward()
    else:
        reward_model = create_domain_reward(task)
    logger.info("✅ Reward model initialized")

    # Get base training config
    base_config = get_config_for_task(task, model_name=model_name)

    # Create GSPO config
    gspo_config = GSPOConfig.from_training_config(
        base_config,
        num_outer_iterations=num_iterations,
        output_dir=output_dir,
        # GSPO specific parameters
        num_generations=4,  # Group size
        clip_range_left=3e-4,  # Sequence-level clipping
        clip_range_right=4e-4,
        generations_per_iteration=20,  # Queries per iteration
        # Use smaller model settings for demo
        use_lora=True,
        lora_r=8,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        # Logging
        report_to="none",  # Set to "wandb" to enable W&B logging
        logging_steps=1,
        save_steps=5,
        use_gspo_token=use_gspo_token,
    )

    # Run training
    logger.info("Starting training...")
    logger.info(f"Configuration: {gspo_config.to_dict()}")

    if use_gspo_token:
        trained_agent = await train_with_gspo_token(
            config=gspo_config,
            agent=agent,
            environment=environment,
            reward_model=reward_model,
        )
    else:
        trained_agent = await train_with_gspo(
            config=gspo_config,
            agent=agent,
            environment=environment,
            reward_model=reward_model,
        )

    logger.info("✅ Training completed!")

    # Test the trained agent
    logger.info("\n" + "=" * 60)
    logger.info("Testing trained agent...")
    logger.info("=" * 60)

    test_messages = {
        "customer_service": [
            {"role": "user", "content": "My order hasn't arrived and I need it urgently."}
        ],
        "technical_support": [
            {"role": "user", "content": "My application keeps crashing. Can you help?"}
        ],
        "sales": [
            {"role": "user", "content": "Tell me about your premium features."}
        ],
    }

    test_message = test_messages.get(
        task,
        [{"role": "user", "content": "Hello, can you help me?"}],
    )

    response = await trained_agent.generate_response(test_message)
    logger.info(f"\nUser: {test_message[0]['content']}")
    logger.info(f"Agent: {response}")
    logger.info("=" * 60)

    return trained_agent


async def compare_gspo_vs_grpo():
    """
    Compare GSPO and GRPO training characteristics.

    This demonstration shows the key differences:
    1. GSPO uses sequence-level importance ratios
    2. GSPO has much higher clipping fractions but better efficiency
    3. GSPO is more stable for MoE models
    """
    logger.info("\n" + "=" * 80)
    logger.info("GSPO vs GRPO Comparison")
    logger.info("=" * 80)

    comparison = """
    Key Differences:

    1. Importance Ratio:
       - GRPO: Token-level π_θ(y_t|x,y_<t) / π_θ_old(y_t|x,y_<t)
       - GSPO: Sequence-level (π_θ(y|x) / π_θ_old(y|x))^(1/|y|)

    2. Clipping:
       - GRPO: Per-token clipping, typical range: [0.8, 1.2]
       - GSPO: Per-sequence clipping, typical range: [1-3e-4, 1+4e-4]

    3. Advantages:
       - GRPO: Token-level, but all tokens share same advantage
       - GSPO: Sequence-level, normalized by group
       - GSPO-token: Token-level with sequence-level clipping

    4. Stability:
       - GRPO: Can suffer from token-level noise accumulation
       - GSPO: More stable, especially for long sequences and MoE models

    5. Training Efficiency:
       - GRPO: Lower clipping fraction (~0.15)
       - GSPO: Higher clipping fraction (~15%) but better sample efficiency

    6. MoE Training:
       - GRPO: Requires Routing Replay strategy
       - GSPO: No special strategies needed

    Reference: https://arxiv.org/abs/2507.18071v2
    """

    logger.info(comparison)
    logger.info("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train conversational agents with GSPO"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        choices=["customer_service", "technical_support", "sales"],
        help="Task type",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (gpt2 recommended for demo)",
    )
    parser.add_argument(
        "--use-gspo-token",
        action="store_true",
        help="Use GSPO-token variant",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/gspo_demo",
        help="Output directory",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison between GSPO and GRPO",
    )

    args = parser.parse_args()

    if args.compare:
        asyncio.run(compare_gspo_vs_grpo())
    else:
        asyncio.run(
            train_gspo_demo(
                task=args.task,
                model_name=args.model,
                use_gspo_token=args.use_gspo_token,
                num_iterations=args.iterations,
                output_dir=args.output_dir,
            )
        )


if __name__ == "__main__":
    main()
