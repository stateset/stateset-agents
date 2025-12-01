"""
Fine-tune Qwen3 Models with GSPO

This script demonstrates how to fine-tune Qwen3 models using GSPO,
the same algorithm used by Alibaba to train the production Qwen3 models.

Supported Qwen3 Models:
- Qwen/Qwen2.5-0.5B
- Qwen/Qwen2.5-1.5B
- Qwen/Qwen2.5-3B
- Qwen/Qwen2.5-7B
- Qwen/Qwen2.5-14B
- Qwen/Qwen2.5-32B
- Qwen/Qwen2.5-72B

Usage:
    # Small model (0.5B) - Great for testing
    python examples/finetune_qwen3_gspo.py --model Qwen/Qwen2.5-0.5B --task customer_service

    # Medium model (7B) - Balanced performance
    python examples/finetune_qwen3_gspo.py --model Qwen/Qwen2.5-7B --task technical_support --use-lora

    # Large model (32B) - High quality
    python examples/finetune_qwen3_gspo.py --model Qwen/Qwen2.5-32B --task sales --use-lora --use-4bit

    # MoE model (A3B) - Efficient and powerful
    python examples/finetune_qwen3_gspo.py --model Qwen/Qwen2.5-A3B --task customer_service
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


def get_qwen3_config(
    model_name: str,
    task: str,
    use_lora: bool = True,
    use_4bit: bool = False,
    use_8bit: bool = False,
    output_dir: str = "./outputs/qwen3_gspo",
):
    """
    Get optimized GSPO configuration for Qwen3 models.

    These configurations are based on the original GSPO paper
    and optimized for different model sizes.
    """
    from training.gspo_trainer import GSPOConfig
    from training.config import get_config_for_task

    # Get base config for task
    base_config = get_config_for_task(task, model_name=model_name)

    # Determine model size and adjust hyperparameters
    if "0.5B" in model_name or "0.5b" in model_name:
        # Small model config
        config = GSPOConfig.from_training_config(
            base_config,
            # GSPO parameters
            num_generations=4,
            clip_range_left=3e-4,
            clip_range_right=4e-4,
            # Training
            learning_rate=1e-5,
            num_outer_iterations=100,
            generations_per_iteration=50,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            # Model optimization
            use_lora=use_lora,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            gradient_checkpointing=True,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            # Generation
            max_prompt_length=512,
            max_completion_length=512,
            temperature=0.7,
            # Output
            output_dir=output_dir,
            save_steps=10,
            logging_steps=1,
        )
    elif "1.5B" in model_name or "1.5b" in model_name or "3B" in model_name or "3b" in model_name:
        # Medium model config
        config = GSPOConfig.from_training_config(
            base_config,
            num_generations=4,
            clip_range_left=2.5e-4,
            clip_range_right=3.5e-4,
            learning_rate=8e-6,
            num_outer_iterations=100,
            generations_per_iteration=40,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            use_lora=use_lora,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            gradient_checkpointing=True,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            max_prompt_length=1024,
            max_completion_length=1024,
            temperature=0.7,
            output_dir=output_dir,
            save_steps=10,
            logging_steps=1,
        )
    elif "7B" in model_name or "7b" in model_name or "14B" in model_name or "14b" in model_name:
        # Large model config
        config = GSPOConfig.from_training_config(
            base_config,
            num_generations=6,  # Larger group for stability
            clip_range_left=2e-4,
            clip_range_right=3e-4,
            learning_rate=5e-6,
            num_outer_iterations=100,
            generations_per_iteration=30,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            use_lora=True,  # Always use LoRA for large models
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            gradient_checkpointing=True,
            use_4bit=use_4bit if use_4bit else use_8bit,  # Use quantization for large models
            use_8bit=use_8bit if not use_4bit else False,
            max_prompt_length=2048,
            max_completion_length=2048,
            temperature=0.7,
            output_dir=output_dir,
            save_steps=5,
            logging_steps=1,
        )
    else:
        # Very large model config (32B, 72B, MoE)
        config = GSPOConfig.from_training_config(
            base_config,
            num_generations=8,  # Even larger group for MoE stability
            clip_range_left=1.5e-4,
            clip_range_right=2.5e-4,
            learning_rate=3e-6,
            num_outer_iterations=100,
            generations_per_iteration=20,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            use_lora=True,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            gradient_checkpointing=True,
            use_4bit=True,  # Always use quantization for very large models
            max_prompt_length=4096,
            max_completion_length=2048,
            temperature=0.7,
            output_dir=output_dir,
            save_steps=5,
            logging_steps=1,
        )

    return config


async def finetune_qwen3(
    model_name: str,
    task: str = "customer_service",
    use_lora: bool = True,
    use_4bit: bool = False,
    use_8bit: bool = False,
    output_dir: str = "./outputs/qwen3_gspo",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
):
    """
    Fine-tune a Qwen3 model using GSPO.

    Args:
        model_name: Qwen3 model name (e.g., "Qwen/Qwen2.5-7B")
        task: Task type (customer_service, technical_support, sales)
        use_lora: Use LoRA for efficient fine-tuning
        use_4bit: Use 4-bit quantization
        use_8bit: Use 8-bit quantization
        output_dir: Output directory for checkpoints
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
    """
    from stateset_agents import MultiTurnAgent
    from stateset_agents.core.agent import AgentConfig
    from stateset_agents.core.environment import ConversationEnvironment, CONVERSATION_CONFIGS
    from stateset_agents.rewards.multi_objective_reward import create_domain_reward
    from training.gspo_trainer import train_with_gspo

    logger.info("=" * 80)
    logger.info("🚀 Fine-tuning Qwen3 with GSPO")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Task: {task}")
    logger.info(f"LoRA: {use_lora}")
    logger.info(f"Quantization: {'4-bit' if use_4bit else '8-bit' if use_8bit else 'None'}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # System prompts for different tasks
    system_prompts = {
        "customer_service": "You are Qwen, a helpful and empathetic customer service representative created by Alibaba Cloud. You assist customers with their inquiries professionally and efficiently.",
        "technical_support": "You are Qwen, a knowledgeable technical support specialist created by Alibaba Cloud. You help users troubleshoot technical issues with clear, detailed explanations.",
        "sales": "You are Qwen, a friendly and persuasive sales representative created by Alibaba Cloud. You help customers discover products that meet their needs.",
    }

    # Create agent
    logger.info("Initializing Qwen3 agent...")
    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt=system_prompts.get(task, system_prompts["customer_service"]),
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
        env_config = CONVERSATION_CONFIGS["customer_service"].copy()

    environment = ConversationEnvironment(**env_config)
    logger.info(f"✅ Environment configured with {len(environment.scenarios)} scenarios")

    # Create reward model
    logger.info("Initializing reward model...")
    reward_model = create_domain_reward(task)
    logger.info("✅ Reward model initialized")

    # Get GSPO configuration
    logger.info("Creating GSPO configuration...")
    gspo_config = get_qwen3_config(
        model_name=model_name,
        task=task,
        use_lora=use_lora,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        output_dir=output_dir,
    )

    # Enable W&B if requested
    if use_wandb:
        gspo_config.report_to = "wandb"
        gspo_config.wandb_project = wandb_project or f"qwen3-gspo-{task}"
        gspo_config.wandb_tags = ["qwen3", "gspo", task, model_name.split("/")[-1]]
        logger.info(f"✅ W&B enabled: {gspo_config.wandb_project}")

    logger.info("✅ GSPO configuration ready")
    logger.info(f"   - Group size: {gspo_config.num_generations}")
    logger.info(f"   - Clipping: [{gspo_config.clip_range_left}, {gspo_config.clip_range_right}]")
    logger.info(f"   - Learning rate: {gspo_config.learning_rate}")
    logger.info(f"   - Iterations: {gspo_config.num_outer_iterations}")

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
    logger.info("Testing trained Qwen3 model...")
    logger.info("=" * 80)

    test_queries = {
        "customer_service": "My order hasn't arrived yet and it's been 2 weeks. Can you help?",
        "technical_support": "The application crashes every time I try to export a file. What should I do?",
        "sales": "I'm looking for a cloud computing solution for my small business. What do you recommend?",
    }

    test_query = test_queries.get(task, test_queries["customer_service"])
    messages = [{"role": "user", "content": test_query}]

    logger.info(f"\nUser: {test_query}")
    response = await trained_agent.generate_response(messages)
    logger.info(f"Qwen3: {response}\n")
    logger.info("=" * 80)

    return trained_agent


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3 models with GSPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Small model for testing
  python examples/finetune_qwen3_gspo.py --model Qwen/Qwen2.5-0.5B

  # 7B model with LoRA
  python examples/finetune_qwen3_gspo.py --model Qwen/Qwen2.5-7B --use-lora

  # 32B model with 4-bit quantization
  python examples/finetune_qwen3_gspo.py --model Qwen/Qwen2.5-32B --use-4bit

  # Enable W&B logging
  python examples/finetune_qwen3_gspo.py --model Qwen/Qwen2.5-7B --wandb --wandb-project my-project
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Qwen3 model name (e.g., Qwen/Qwen2.5-7B)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        choices=["customer_service", "technical_support", "sales"],
        help="Task type",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning (recommended)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (for large models)",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/qwen3_gspo",
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

    args = parser.parse_args()

    # Handle no-lora flag
    use_lora = args.use_lora and not args.no_lora

    # Run training
    asyncio.run(
        finetune_qwen3(
            model_name=args.model,
            task=args.task,
            use_lora=use_lora,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
        )
    )


if __name__ == "__main__":
    main()
