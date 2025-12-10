"""
Fine-tune Mistral and Mixtral Models with GSPO

This script demonstrates how to fine-tune Mistral AI's models using GSPO
(Group Sequence Policy Optimization) for conversational AI tasks.

Supported Models:
- mistralai/Mistral-7B-v0.3
- mistralai/Mistral-7B-Instruct-v0.3
- mistralai/Mistral-Nemo-Base-2407 (12B)
- mistralai/Mistral-Nemo-Instruct-2407 (12B)
- mistralai/Mixtral-8x7B-v0.1 (MoE - 47B total, 13B active)
- mistralai/Mixtral-8x7B-Instruct-v0.1
- mistralai/Mixtral-8x22B-v0.1 (MoE - 176B total, 44B active)
- mistralai/Mixtral-8x22B-Instruct-v0.1

GSPO is particularly well-suited for MoE (Mixture of Experts) models like Mixtral,
as it provides stable training without requiring special routing strategies.

Usage:
    # Mistral 7B - Great for testing
    python examples/finetune_mistral_gspo.py --model mistralai/Mistral-7B-Instruct-v0.3

    # Mistral Nemo 12B
    python examples/finetune_mistral_gspo.py --model mistralai/Mistral-Nemo-Instruct-2407 --use-lora

    # Mixtral 8x7B MoE with LoRA
    python examples/finetune_mistral_gspo.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --use-lora

    # Mixtral 8x22B with 4-bit quantization
    python examples/finetune_mistral_gspo.py --model mistralai/Mixtral-8x22B-Instruct-v0.1 --use-4bit
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


def get_mistral_config(
    model_name: str,
    task: str,
    use_lora: bool = True,
    use_4bit: bool = False,
    use_8bit: bool = False,
    output_dir: str = "./outputs/mistral_gspo",
):
    """
    Get optimized GSPO configuration for Mistral/Mixtral models.

    GSPO is especially effective for MoE models as it uses sequence-level
    importance ratios which are more stable than token-level ratios.
    """
    from training.gspo_trainer import GSPOConfig
    from training.config import get_config_for_task

    # Get base config for task
    base_config = get_config_for_task(task, model_name=model_name)

    model_lower = model_name.lower()
    is_moe = "mixtral" in model_lower

    if "7b" in model_lower and not is_moe:
        # Mistral 7B config
        config = GSPOConfig.from_training_config(
            base_config,
            # GSPO parameters
            num_generations=4,
            clip_range_left=3e-4,
            clip_range_right=4e-4,
            # Training
            learning_rate=1e-5,
            num_outer_iterations=100,
            generations_per_iteration=40,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            # Model optimization
            use_lora=use_lora,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            gradient_checkpointing=True,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            # Generation (Mistral has 32k context)
            max_prompt_length=2048,
            max_completion_length=1024,
            temperature=0.7,
            # Output
            output_dir=output_dir,
            save_steps=10,
            logging_steps=1,
        )
    elif "nemo" in model_lower or "12b" in model_lower:
        # Mistral Nemo 12B config
        config = GSPOConfig.from_training_config(
            base_config,
            num_generations=6,
            clip_range_left=2e-4,
            clip_range_right=3e-4,
            learning_rate=5e-6,
            num_outer_iterations=100,
            generations_per_iteration=30,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            use_lora=True,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            gradient_checkpointing=True,
            use_4bit=use_4bit if use_4bit else use_8bit,
            use_8bit=use_8bit if not use_4bit else False,
            max_prompt_length=4096,
            max_completion_length=2048,
            temperature=0.7,
            output_dir=output_dir,
            save_steps=5,
            logging_steps=1,
        )
    elif "8x7b" in model_lower:
        # Mixtral 8x7B MoE config
        # GSPO handles MoE models well due to sequence-level optimization
        config = GSPOConfig.from_training_config(
            base_config,
            num_generations=8,  # Larger group for MoE stability
            clip_range_left=2e-4,
            clip_range_right=3e-4,
            learning_rate=3e-6,
            num_outer_iterations=100,
            generations_per_iteration=25,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            use_lora=True,  # Always use LoRA for MoE
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            # Target all attention modules and gate projections
            lora_target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "w1", "w2", "w3",  # MoE expert weights
            ],
            gradient_checkpointing=True,
            use_4bit=True if not use_8bit else False,  # Recommend 4-bit for MoE
            use_8bit=use_8bit,
            max_prompt_length=4096,
            max_completion_length=2048,
            temperature=0.7,
            output_dir=output_dir,
            save_steps=5,
            logging_steps=1,
        )
    else:
        # Mixtral 8x22B MoE config (largest)
        config = GSPOConfig.from_training_config(
            base_config,
            num_generations=8,  # Even larger group for very large MoE
            clip_range_left=1.5e-4,
            clip_range_right=2.5e-4,
            learning_rate=2e-6,
            num_outer_iterations=100,
            generations_per_iteration=15,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            use_lora=True,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "w1", "w2", "w3",
            ],
            gradient_checkpointing=True,
            use_4bit=True,  # Always use 4-bit for 8x22B
            max_prompt_length=8192,
            max_completion_length=2048,
            temperature=0.7,
            output_dir=output_dir,
            save_steps=5,
            logging_steps=1,
        )

    return config


async def finetune_mistral(
    model_name: str,
    task: str = "customer_service",
    use_lora: bool = True,
    use_4bit: bool = False,
    use_8bit: bool = False,
    output_dir: str = "./outputs/mistral_gspo",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
):
    """
    Fine-tune a Mistral or Mixtral model using GSPO.

    Args:
        model_name: Model name (e.g., "mistralai/Mixtral-8x7B-Instruct-v0.1")
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

    is_moe = "mixtral" in model_name.lower()
    model_type = "Mixtral MoE" if is_moe else "Mistral"

    logger.info("=" * 80)
    logger.info(f"Fine-tuning {model_type} with GSPO")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Task: {task}")
    logger.info(f"LoRA: {use_lora}")
    logger.info(f"Quantization: {'4-bit' if use_4bit else '8-bit' if use_8bit else 'None'}")
    logger.info(f"Output: {output_dir}")

    if is_moe:
        logger.info("")
        logger.info("Note: GSPO is well-suited for MoE models as it uses")
        logger.info("      sequence-level importance ratios for stable training.")
    logger.info("=" * 80)

    # System prompts (Mistral uses [INST] format in its chat template)
    system_prompts = {
        "customer_service": "You are a helpful customer service assistant. Be empathetic, professional, and solution-oriented. Always strive to resolve customer issues efficiently.",
        "technical_support": "You are a technical support specialist with deep expertise. Diagnose issues systematically, provide clear solutions, and explain technical concepts accessibly.",
        "sales": "You are a knowledgeable sales assistant. Understand customer needs, recommend appropriate solutions, and provide helpful information without being pushy.",
    }

    # Create agent
    logger.info("Initializing agent...")
    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt=system_prompts.get(task, system_prompts["customer_service"]),
        max_new_tokens=2048,
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()
    logger.info("Agent initialized")

    # Create environment
    logger.info("Setting up environment...")
    if task in CONVERSATION_CONFIGS:
        env_config = CONVERSATION_CONFIGS[task].copy()
    else:
        env_config = CONVERSATION_CONFIGS["customer_service"].copy()

    environment = ConversationEnvironment(**env_config)
    logger.info(f"Environment configured with {len(environment.scenarios)} scenarios")

    # Create reward model
    logger.info("Initializing reward model...")
    reward_model = create_domain_reward(task)
    logger.info("Reward model initialized")

    # Get GSPO configuration
    logger.info("Creating GSPO configuration...")
    gspo_config = get_mistral_config(
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
        gspo_config.wandb_project = wandb_project or f"{'mixtral' if is_moe else 'mistral'}-gspo-{task}"
        gspo_config.wandb_tags = [model_type.lower(), "gspo", task, model_name.split("/")[-1]]
        if is_moe:
            gspo_config.wandb_tags.append("moe")
        logger.info(f"W&B enabled: {gspo_config.wandb_project}")

    logger.info("GSPO configuration ready")
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
    logger.info("Training completed!")
    logger.info("=" * 80)

    # Test the trained agent
    logger.info("\n" + "=" * 80)
    logger.info(f"Testing trained {model_type} model...")
    logger.info("=" * 80)

    test_queries = {
        "customer_service": "I've been a customer for 5 years and my recent order was completely wrong. This is unacceptable!",
        "technical_support": "My server keeps running out of memory even though I have 64GB RAM. How can I diagnose this?",
        "sales": "We're a startup looking to scale our infrastructure. What cloud solutions would you recommend?",
    }

    test_query = test_queries.get(task, test_queries["customer_service"])
    messages = [{"role": "user", "content": test_query}]

    logger.info(f"\nUser: {test_query}")
    response = await trained_agent.generate_response(messages)
    logger.info(f"{model_type}: {response}\n")
    logger.info("=" * 80)

    return trained_agent


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral/Mixtral models with GSPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mistral 7B for testing
  python examples/finetune_mistral_gspo.py --model mistralai/Mistral-7B-Instruct-v0.3

  # Mistral Nemo 12B with LoRA
  python examples/finetune_mistral_gspo.py --model mistralai/Mistral-Nemo-Instruct-2407 --use-lora

  # Mixtral 8x7B MoE (recommended for production)
  python examples/finetune_mistral_gspo.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --use-lora

  # Mixtral 8x22B with 4-bit quantization
  python examples/finetune_mistral_gspo.py --model mistralai/Mixtral-8x22B-Instruct-v0.1 --use-4bit

  # Enable W&B logging
  python examples/finetune_mistral_gspo.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --wandb
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Mistral/Mixtral model name",
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
        default="./outputs/mistral_gspo",
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

    use_lora = args.use_lora and not args.no_lora

    asyncio.run(
        finetune_mistral(
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
