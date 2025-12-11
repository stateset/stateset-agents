"""
Simple Demo: TRL GRPO Training

This script demonstrates the simplest way to use TRL GRPO training
with the StateSet Agents framework.

Set `MODEL_NAME` to override the default small demo model.
"""

import asyncio
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main demo function"""

    # Check if TRL is available
    try:
        from training import TRL_AVAILABLE, train_customer_service_with_trl

        if not TRL_AVAILABLE:
            logger.error(
                "TRL is not installed. Please install with: pip install trl peft"
            )
            return
    except ImportError:
        logger.error(
            "Training module not found. Make sure you're in the correct directory."
        )
        return

    logger.info("🚀 Starting TRL GRPO Training Demo")
    logger.info("This demo will train a small customer service agent using TRL GRPO")

    # Sample training data
    sample_data = [
        {
            "messages": [
                {"role": "user", "content": "Where is my order?"},
                {
                    "role": "assistant",
                    "content": "I'd be happy to help you track your order! Could you please provide your order number so I can look up the current status for you?",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "I want to return this item"},
                {
                    "role": "assistant",
                    "content": "I understand you'd like to return an item. I'm here to help! Our return policy allows returns within 30 days of purchase. May I have your order number to start the return process?",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "My product arrived damaged"},
                {
                    "role": "assistant",
                    "content": "I'm so sorry to hear your product arrived damaged. That's definitely not the experience we want for you. I'll help you resolve this right away. We can either send a replacement or issue a full refund. Which would you prefer?",
                },
            ]
        },
    ]

    # Train the agent
    logger.info("Training agent with sample data...")

    try:
        model_name = os.getenv("MODEL_NAME", "gpt2")
        agent = await train_customer_service_with_trl(
            model_name=model_name,
            train_data=sample_data,
            num_episodes=10,  # Very small for demo
            output_dir="./outputs/demo_agent",
            # Small settings for demo
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_generations=2,
            max_prompt_length=128,
            max_completion_length=128,
            # LoRA settings
            use_lora=True,
            lora_r=8,  # Small rank for demo
            lora_alpha=16,
            # Disable wandb for demo
            report_to="none",
        )

        logger.info("✅ Training completed!")

        # Test the trained agent
        logger.info("\n🧪 Testing the trained agent:")

        test_queries = [
            "My order hasn't arrived yet",
            "How do I track my package?",
            "I need help with a return",
        ]

        for query in test_queries:
            logger.info(f"\n👤 User: {query}")

            response = await agent.generate_response(
                messages=[{"role": "user", "content": query}]
            )

            logger.info(f"🤖 Agent: {response}")

        logger.info("\n🎉 Demo completed successfully!")
        logger.info(f"Trained model saved to: ./outputs/demo_agent/final_model")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.info("\nTroubleshooting tips:")
        logger.info(
            "1. Make sure you have enough GPU memory (use a smaller model if needed)"
        )
        logger.info(
            "2. Check that all dependencies are installed: pip install torch transformers peft trl"
        )
        logger.info("3. Try reducing batch size or LoRA rank if you get OOM errors")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
