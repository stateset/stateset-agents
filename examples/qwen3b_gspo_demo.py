"""
Minimal GSPO fine-tuning demo for Qwen/Qwen2.5-3B.

This example shows how to train the 3B Qwen3 model with GSPO on a
conversation-style task (customer service by default) using modest
hyperparameters that fit on a single modern GPU when paired with LoRA
and 8-bit loading.
"""

import argparse
import asyncio
import logging
from typing import Optional

from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import CONVERSATION_CONFIGS, ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import create_domain_reward
from training.config import get_config_for_task
from training.gspo_trainer import GSPOConfig, train_with_gspo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def train_qwen3b_with_gspo(
    task: str = "customer_service",
    output_dir: str = "./outputs/qwen3b_gspo",
    use_lora: bool = True,
    use_8bit: bool = True,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
) -> MultiTurnAgent:
    """
    Train Qwen/Qwen2.5-3B on a conversational task using GSPO.

    Args:
        task: Built-in domain (customer_service, technical_support, sales).
        output_dir: Where to store checkpoints and logs.
        use_lora: Enable LoRA for efficient fine-tuning.
        use_8bit: Load the base model in 8-bit to save memory.
        use_wandb: Enable Weights & Biases logging.
        wandb_project: Optional W&B project name.
    """
    model_name = "Qwen/Qwen2.5-3B"
    system_prompts = {
        "customer_service": "You are Qwen, a helpful and empathetic customer service representative created by Alibaba Cloud. You assist customers with their inquiries professionally and efficiently.",
        "technical_support": "You are Qwen, a knowledgeable technical support specialist created by Alibaba Cloud. You help users troubleshoot technical issues with clear, detailed explanations.",
        "sales": "You are Qwen, a friendly and persuasive sales representative created by Alibaba Cloud. You help customers discover products that meet their needs.",
    }

    logger.info("🌟 Starting Qwen3B GSPO demo")
    logger.info("Task: %s | LoRA: %s | 8-bit: %s", task, use_lora, use_8bit)
    logger.info("Output directory: %s", output_dir)

    # Agent
    agent = MultiTurnAgent(
        AgentConfig(
            model_name=model_name,
            system_prompt=system_prompts.get(task, system_prompts["customer_service"]),
            max_new_tokens=896,
        )
    )
    await agent.initialize()

    # Environment
    env_config = CONVERSATION_CONFIGS.get(task, CONVERSATION_CONFIGS["customer_service"])
    environment = ConversationEnvironment(**env_config)

    # Reward model
    reward_model = create_domain_reward(task)

    # Base training config and GSPO overrides tuned for 3B
    base_config = get_config_for_task(task, model_name=model_name)
    gspo_config = GSPOConfig.from_training_config(
        base_config,
        num_generations=4,
        clip_range_left=2.5e-4,
        clip_range_right=3.5e-4,
        learning_rate=8e-6,
        num_outer_iterations=10,
        generations_per_iteration=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        use_lora=use_lora,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        gradient_checkpointing=True,
        use_8bit=use_8bit,
        max_prompt_length=1024,
        max_completion_length=768,
        temperature=0.7,
        output_dir=output_dir,
        save_steps=2,
        logging_steps=1,
    )

    if use_wandb:
        gspo_config.report_to = "wandb"
        gspo_config.wandb_project = wandb_project or "qwen3b-gspo-demo"
        gspo_config.wandb_tags = ["qwen3b", "gspo", task]
        logger.info("Weights & Biases enabled | project: %s", gspo_config.wandb_project)

    logger.info("🚀 Launching GSPO training...")
    trained_agent = await train_with_gspo(
        config=gspo_config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )

    # Quick sanity check
    sample_query = "My package is late and I need an update. Can you help?"
    response = await trained_agent.generate_response([{"role": "user", "content": sample_query}])
    logger.info("User: %s", sample_query)
    logger.info("Qwen3B: %s", response)

    return trained_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="GSPO fine-tuning demo for Qwen3B")
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        choices=["customer_service", "technical_support", "sales"],
        help="Domain/task to train on.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/qwen3b_gspo",
        help="Where to save checkpoints.",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning).",
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Disable 8-bit loading (requires more memory).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Optional W&B project name.",
    )
    args = parser.parse_args()

    asyncio.run(
        train_qwen3b_with_gspo(
            task=args.task,
            output_dir=args.output_dir,
            use_lora=not args.no_lora,
            use_8bit=not args.full_precision,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
        )
    )


if __name__ == "__main__":
    main()
