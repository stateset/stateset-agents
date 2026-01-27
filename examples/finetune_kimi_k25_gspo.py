"""
Fine-tune Kimi-K2.5 with GSPO

This script demonstrates how to fine-tune Kimi-K2.5 models using GSPO
(Group Sequence Policy Optimization), the algorithm optimized for large MoE models.

Kimi-K2.5 Model Details:
- Architecture: Mixture-of-Experts (MoE)
- Total Parameters: 1T
- Activated Parameters: 32B
- Context Length: 256K
- Native Multimodality: Vision + Language

Usage:
    # Quick start with small batch
    python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task customer_service

    # With W&B logging
    python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task conversational_agent --wandb

    # Custom task with system prompt
    python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task custom --system-prompt "You are a helpful coding assistant."

    # Production training with more iterations
    python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task customer_service --iterations 100
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_kimi_k25_config(
    model_name: str,
    task: str,
    use_lora: bool = True,
    use_8bit: bool = False,
    use_4bit: bool = False,
    output_dir: str = "./outputs/kimi_k25_gspo",
):
    """
    Get optimized GSPO configuration for Kimi-K2.5 models.

    These configurations are optimized for Kimi-K2.5's MoE architecture
    and GSPO's sequence-level optimization.

    Args:
        model_name: Kimi-K2.5 model name
        task: Task type
        use_lora: Use LoRA for efficient fine-tuning (strongly recommended for MoE)
        use_8bit: Use 8-bit quantization
        use_4bit: Use 4-bit quantization
        output_dir: Output directory for checkpoints
    """
    from stateset_agents.training.gspo_trainer import GSPOConfig
    from stateset_agents.training.config import get_config_for_task

    # Get base config for task
    base_config = get_config_for_task(task, model_name=model_name)

    # Kimi-K2.5 specific configuration optimized for MoE + GSPO
    config = GSPOConfig.from_training_config(
        base_config,
        # GSPO parameters optimized for MoE models
        num_generations=6,  # Larger group for MoE stability
        clip_range_left=2e-4,
        clip_range_right=3e-4,
        # Training parameters
        beta=0.01,  # Small KL penalty for MoE stability
        learning_rate=5e-6,
        num_outer_iterations=50,
        generations_per_iteration=20,
        per_device_train_batch_size=1,  # MoE requires smaller batch
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        # Model optimization - LoRA is strongly recommended for MoE
        use_lora=use_lora,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        gradient_checkpointing=True,
        use_4bit=use_4bit if use_4bit else use_8bit,
        use_8bit=use_8bit,
        # Generation settings
        max_prompt_length=2048,
        max_completion_length=2048,
        temperature=1.0,  # Kimi-K2.5 recommends 1.0 for thinking mode
        top_p=0.95,  # Kimi-K2.5 recommends 0.95
        top_k=50,
        repetition_penalty=1.1,
        # Output
        output_dir=output_dir,
        save_steps=10,
        logging_steps=1,
        save_total_limit=3,
        # Report to W&B if enabled
        report_to="none",
        # Enable reference model for KL penalty (important for MoE)
        use_reference_model=True,
    )

    return config


async def finetune_kimi_k25(
    model_name: str,
    task: str = "customer_service",
    system_prompt: Optional[str] = None,
    use_lora: bool = True,
    use_8bit: bool = False,
    use_4bit: bool = False,
    output_dir: str = "./outputs/kimi_k25_gspo",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    num_iterations: int = 50,
):
    """
    Fine-tune a Kimi-K2.5 model using GSPO.

    Args:
        model_name: Kimi-K2.5 model name (e.g., "moonshotai/Kimi-K2.5")
        task: Task type (customer_service, conversational_agent, technical_support)
        system_prompt: Custom system prompt (overrides default if provided)
        use_lora: Use LoRA for efficient fine-tuning (strongly recommended)
        use_8bit: Use 8-bit quantization
        use_4bit: Use 4-bit quantization
        output_dir: Output directory for checkpoints
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        num_iterations: Number of outer training iterations
    """
    from stateset_agents import MultiTurnAgent
    from stateset_agents.core.agent import AgentConfig
    from stateset_agents.core.environment import (
        ConversationEnvironment,
        CONVERSATION_CONFIGS,
    )
    from stateset_agents.rewards.multi_objective_reward import (
        create_customer_service_reward,
        create_domain_reward,
    )
    from stateset_agents.training.gspo_trainer import train_with_gspo

    logger.info("=" * 80)
    logger.info("🚀 Fine-tuning Kimi-K2.5 with GSPO")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Task: {task}")
    logger.info(f"LoRA: {use_lora}")
    logger.info(
        f"Quantization: {'4-bit' if use_4bit else '8-bit' if use_8bit else 'None'}"
    )
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # System prompts for different tasks
    default_system_prompts = {
        "customer_service": "You are Kimi, a helpful and empathetic customer service representative created by Moonshot AI. You assist customers with their inquiries professionally and efficiently.",
        "conversational_agent": "You are Kimi, a helpful and intelligent conversational AI assistant created by Moonshot AI. You engage in natural, helpful conversations.",
        "technical_support": "You are Kimi, a knowledgeable technical support specialist created by Moonshot AI. You help users troubleshoot technical issues with clear, detailed explanations.",
        "sales": "You are Kimi, a friendly and persuasive sales representative created by Moonshot AI. You help customers discover products that meet their needs.",
    }

    final_system_prompt = system_prompt or default_system_prompts.get(
        task, "You are Kimi, a helpful AI assistant created by Moonshot AI."
    )

    # Create agent
    logger.info("Initializing Kimi-K2.5 agent...")
    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt=final_system_prompt,
        max_new_tokens=2048,
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()
    logger.info("✅ Agent initialized")

    # Create environment
    logger.info("Setting up environment...")
    if task in CONVERSATION_CONFIGS:
        env_config = CONVERSATION_CONFIGS[task].copy()
    else:
        # For custom tasks, use customer_service as base
        env_config = CONVERSATION_CONFIGS["customer_service"].copy()

    environment = ConversationEnvironment(**env_config)
    logger.info(
        f"✅ Environment configured with {len(environment.scenarios)} scenarios"
    )

    # Create reward model
    logger.info("Initializing reward model...")
    if task == "customer_service":
        reward_model = create_customer_service_reward()
    else:
        reward_model = create_domain_reward(task)
    logger.info("✅ Reward model initialized")

    # Get GSPO configuration
    logger.info("Creating GSPO configuration...")
    gspo_config = get_kimi_k25_config(
        model_name=model_name,
        task=task,
        use_lora=use_lora,
        use_8bit=use_8bit,
        use_4bit=use_4bit,
        output_dir=output_dir,
    )

    # Override number of iterations if specified
    if num_iterations:
        gspo_config.num_outer_iterations = num_iterations

    # Enable W&B if requested
    if use_wandb:
        gspo_config.report_to = "wandb"
        gspo_config.wandb_project = wandb_project or f"kimi-k25-gspo-{task}"
        gspo_config.wandb_tags = ["kimi-k25", "gspo", task, "moe"]
        logger.info(f"✅ W&B enabled: {gspo_config.wandb_project}")

    logger.info("✅ GSPO configuration ready")
    logger.info(f"   - Group size: {gspo_config.num_generations}")
    logger.info(
        f"   - Clipping: [{gspo_config.clip_range_left}, {gspo_config.clip_range_right}]"
    )
    logger.info(f"   - Learning rate: {gspo_config.learning_rate}")
    logger.info(f"   - Iterations: {gspo_config.num_outer_iterations}")
    logger.info(f"   - LoRA rank: {gspo_config.lora_r}")

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting GSPO training...")
    logger.info("=" * 80 + "\n")

    trained_agent = await train_with_gspo(
        config=gspo_config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )

    logger.info("\n" + "=" * 80)
    logger.info("✅ Training completed!")
    logger.info("=" * 80)

    # Test the trained agent
    logger.info("\n" + "=" * 80)
    logger.info("Testing trained Kimi-K2.5 model...")
    logger.info("=" * 80)

    test_queries = {
        "customer_service": "My order hasn't arrived yet and it's been 2 weeks. Can you help?",
        "conversational_agent": "Hi! How are you doing today?",
        "technical_support": "The application crashes every time I try to export a file. What should I do?",
        "sales": "I'm looking for a cloud computing solution for my small business. What do you recommend?",
    }

    test_query = test_queries.get(task, "Hello! Can you help me?")
    messages = [{"role": "user", "content": test_query}]

    logger.info(f"\nUser: {test_query}")
    response = await trained_agent.generate_response(messages)
    logger.info(f"Kimi-K2.5: {response}\n")
    logger.info("=" * 80)

    return trained_agent


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Kimi-K2.5 models with GSPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with customer service task
  python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task customer_service

  # Train with W&B logging
  python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task conversational_agent --wandb

  # Custom system prompt
  python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task custom --system-prompt "You are a helpful coding assistant."

  # Production training with more iterations
  python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task customer_service --iterations 100 --wandb
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="moonshotai/Kimi-K2.5",
        help="Kimi-K2.5 model name (default: moonshotai/Kimi-K2.5)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        choices=[
            "customer_service",
            "conversational_agent",
            "technical_support",
            "sales",
            "custom",
        ],
        help="Task type",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (overrides default)",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning (recommended for MoE)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning, NOT recommended for 1T MoE)",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/kimi_k25_gspo",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project name",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of outer training iterations",
    )

    args = parser.parse_args()

    # Handle no-lora flag
    use_lora = args.use_lora and not args.no_lora

    # Warn if not using LoRA
    if not use_lora:
        logger.warning(
            "⚠️  Warning: Training without LoRA on a 1T MoE model is extremely memory intensive!"
        )
        response = input("Are you sure you want to proceed? (y/N): ")
        if response.lower() != "y":
            logger.info("Training cancelled.")
            return

    # Run training
    asyncio.run(
        finetune_kimi_k25(
            model_name=args.model,
            task=args.task,
            system_prompt=args.system_prompt,
            use_lora=use_lora,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            num_iterations=args.iterations,
        )
    )


if __name__ == "__main__":
    main()
