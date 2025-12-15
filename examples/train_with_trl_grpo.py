"""
Example: Training a model using TRL GRPO

This script demonstrates how to use the TRL GRPO trainer with the StateSet Agents
framework to fine-tune a conversational model for customer service tasks.

Set `MODEL_NAME` to override the default small demo model.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import framework components
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import create_customer_service_reward
from stateset_agents.training.trl_grpo_trainer import (
    TRLGRPOConfig,
    train_customer_service_with_trl,
    train_with_trl_grpo,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trl_grpo_training.log")],
)
logger = logging.getLogger(__name__)


def load_training_data(
    file_path: str, max_examples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load training data from JSONL file"""
    data = []

    if not os.path.exists(file_path):
        logger.warning(f"Data file not found: {file_path}")
        return get_sample_data()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    conv = json.loads(line.strip())
                    data.append(conv)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {line_num}")
                    continue

                if max_examples and len(data) >= max_examples:
                    break

    except Exception as e:
        logger.error(f"Error reading data file: {e}")
        return get_sample_data()

    logger.info(f"Loaded {len(data)} conversations from {file_path}")
    return data


def get_sample_data() -> List[Dict[str, Any]]:
    """Return sample training data for demonstration"""
    return [
        {
            "messages": [
                {"role": "user", "content": "Hi, where is my order #12345?"},
                {
                    "role": "assistant",
                    "content": "I'd be happy to help you track your order! Let me look up order #12345 for you. According to our system, your order is currently in transit and was shipped yesterday. It's expected to arrive within 2-3 business days. You should receive a tracking update via email soon with the detailed tracking information.",
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I need to return a product I bought last week",
                },
                {
                    "role": "assistant",
                    "content": "I understand you need to return a product, and I'm here to help make that process as smooth as possible. Our return policy allows returns within 30 days of purchase. To start your return, please visit our returns portal where you can print a prepaid return label. You'll need your order number and email address. Would you like me to email you the direct link to the returns portal?",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "The product I received is damaged"},
                {
                    "role": "assistant",
                    "content": "I'm so sorry to hear that you received a damaged product. That's definitely not the experience we want you to have. I'll help you resolve this right away. For damaged items, we offer immediate replacements or full refunds. Could you please provide your order number so I can process this for you? Also, if possible, taking a photo of the damage will help us improve our packaging.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How do I update my shipping address?"},
                {
                    "role": "assistant",
                    "content": "I can help you update your shipping address! If you have an order that hasn't shipped yet, I can update it directly. For future orders, you can update your default shipping address in your account settings. Just log in, go to 'My Account', then 'Addresses', and you can add or edit addresses there. Would you like me to help you update the address for a specific order?",
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's your customer service phone number?",
                },
                {
                    "role": "assistant",
                    "content": "Our customer service team is available at 1-800-555-0123, Monday through Friday from 9 AM to 6 PM EST, and weekends from 10 AM to 4 PM EST. You can also reach us through this chat 24/7, or via email at support@example.com. Is there something specific I can help you with right now?",
                },
            ]
        },
    ]


async def main():
    """Main training function"""

    # Configuration from environment variables or defaults
    model_name = os.getenv("MODEL_NAME", "gpt2")
    config_dict = {
        "model_name": model_name,
        "output_dir": os.getenv("OUTPUT_DIR", "./outputs/trl_grpo_training"),
        "num_episodes": int(os.getenv("NUM_EPISODES", "1000")),
        "per_device_train_batch_size": int(os.getenv("BATCH_SIZE", "1")),
        "gradient_accumulation_steps": int(
            os.getenv("GRADIENT_ACCUMULATION_STEPS", "8")
        ),
        "learning_rate": float(os.getenv("LEARNING_RATE", "5e-6")),
        "num_epochs": int(os.getenv("NUM_EPOCHS", "1")),
        "warmup_steps": int(os.getenv("WARMUP_STEPS", "100")),
        "max_grad_norm": float(os.getenv("MAX_GRAD_NORM", "1.0")),
        # TRL GRPO specific
        "beta": float(os.getenv("BETA", "0.0")),
        "num_generations": int(os.getenv("NUM_GENERATIONS", "4")),
        "num_iterations": int(os.getenv("NUM_ITERATIONS", "1")),
        "mini_batch_size": int(os.getenv("MINI_BATCH_SIZE", "1")),
        # Generation parameters
        "max_prompt_length": int(os.getenv("MAX_PROMPT_LENGTH", "256")),
        "max_completion_length": int(os.getenv("MAX_COMPLETION_LENGTH", "256")),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "top_p": float(os.getenv("TOP_P", "0.9")),
        # LoRA configuration
        "use_lora": os.getenv("USE_LORA", "true").lower() == "true",
        "lora_r": int(os.getenv("LORA_R", "16")),
        "lora_alpha": int(os.getenv("LORA_ALPHA", "32")),
        "lora_dropout": float(os.getenv("LORA_DROPOUT", "0.05")),
        # Memory optimization
        "gradient_checkpointing": os.getenv("GRADIENT_CHECKPOINTING", "true").lower()
        == "true",
        "bf16": os.getenv("USE_BF16", "true").lower() == "true",
        "fp16": os.getenv("USE_FP16", "false").lower() == "true",
        # Logging
        "logging_steps": int(os.getenv("LOGGING_STEPS", "10")),
        "save_steps": int(os.getenv("SAVE_STEPS", "100")),
        "eval_steps": int(os.getenv("EVAL_STEPS", "50")),
        # Weights & Biases
        "report_to": os.getenv("REPORT_TO", "wandb"),
        "wandb_project": os.getenv("WANDB_PROJECT", "grpo-agent-training"),
        "wandb_tags": ["trl-grpo", "customer-service", "openai-gpt-oss-120b"],
    }

    logger.info("🚀 Starting TRL GRPO Training")
    logger.info(f"Configuration: {json.dumps(config_dict, indent=2)}")

    # Check if we should use quick training mode
    quick_mode = os.getenv("QUICK_MODE", "false").lower() == "true"
    if quick_mode:
        logger.info("Running in quick mode with reduced parameters")
        await quick_training_demo()
    else:
        # Full training pipeline
        await full_training_pipeline(config_dict)


async def quick_training_demo():
    """Quick training demonstration with minimal resources"""
    logger.info("Running quick training demo...")

    # Use the convenience function with minimal settings
    agent = await train_customer_service_with_trl(
        model_name=os.getenv("MODEL_NAME", "gpt2"),
        train_data=get_sample_data(),
        num_episodes=10,  # Very small for demo
        output_dir="./outputs/quick_demo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_prompt_length=128,
        max_completion_length=128,
        use_lora=True,
        lora_r=8,  # Smaller LoRA rank for demo
        report_to="none",  # Disable wandb for quick demo
    )

    # Test the trained agent
    logger.info("Testing trained agent...")
    test_queries = [
        "Where is my order?",
        "I need to return this item",
        "My product is damaged",
    ]

    for query in test_queries:
        response = await agent.generate_response(
            messages=[{"role": "user", "content": query}]
        )
        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}\n")

    logger.info("✅ Quick demo completed!")


async def full_training_pipeline(config_dict: Dict[str, Any]):
    """Full training pipeline with all features"""

    # Load training data
    data_path = os.getenv("DATA_PATH", "training_data.jsonl")
    max_examples = int(os.getenv("MAX_EXAMPLES", "5000"))
    train_data = load_training_data(data_path, max_examples)

    if not train_data:
        logger.warning("No training data loaded, using sample data")
        train_data = get_sample_data()

    # Create TRL GRPO configuration
    config = TRLGRPOConfig(**config_dict)

    # Validate configuration
    warnings = config.validate()
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Create agent
    logger.info("Creating agent...")
    agent_config = AgentConfig(
        model_name=config.model_name,
        system_prompt="You are a helpful, empathetic, and professional customer service representative. Always be polite, provide clear solutions, and show understanding for customer concerns.",
        temperature=config.temperature,
        max_new_tokens=config.max_completion_length,
        torch_dtype=config.torch_dtype,
        use_peft=config.use_lora,
        peft_config={
            "r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
        }
        if config.use_lora
        else None,
    )

    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    # Create environment
    logger.info("Setting up environment...")
    environment = ConversationEnvironment(
        scenario_generator=lambda: {
            "context": "customer_service",
            "task": "handle_inquiry",
        },
        max_turns=10,
        success_criteria=lambda t: len(t.turns) >= 2,
    )

    # Create reward model
    logger.info("Initializing reward model...")
    reward_model = create_customer_service_reward(
        weights={
            "helpfulness": 0.3,
            "coherence": 0.2,
            "empathy": 0.2,
            "efficiency": 0.15,
            "professionalism": 0.15,
        }
    )

    # Run training
    logger.info("Starting training...")
    try:
        trained_agent = await train_with_trl_grpo(
            config=config,
            agent=agent,
            environment=environment,
            reward_model=reward_model,
            train_data=train_data,
        )

        logger.info("✨ Training completed successfully!")

        # Save configuration
        config_path = Path(config.output_dir) / "training_config.json"
        config.save(str(config_path))
        logger.info(f"Configuration saved to {config_path}")

        # Run evaluation if specified
        if os.getenv("RUN_EVAL", "true").lower() == "true":
            await evaluate_agent(trained_agent, config)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


async def evaluate_agent(agent: MultiTurnAgent, config: TRLGRPOConfig):
    """Evaluate the trained agent"""
    logger.info("🔍 Running agent evaluation...")

    eval_queries = [
        "My order hasn't arrived and it's been a week",
        "I received the wrong item in my package",
        "How do I track my shipment?",
        "I want to cancel my subscription",
        "The website won't let me checkout",
        "I was charged twice for the same order",
        "What's your return policy?",
        "I forgot my account password",
    ]

    results = []
    for query in eval_queries:
        messages = [{"role": "user", "content": query}]
        response = await agent.generate_response(messages)

        results.append(
            {
                "query": query,
                "response": response,
                "response_length": len(response.split()),
            }
        )

        logger.info(f"\nQuery: {query}")
        logger.info(f"Response: {response}")

    # Save evaluation results
    eval_path = Path(config.output_dir) / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n📊 Evaluation complete! Results saved to {eval_path}")

    # Calculate metrics
    avg_length = sum(r["response_length"] for r in results) / len(results)
    logger.info(f"Average response length: {avg_length:.1f} words")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
