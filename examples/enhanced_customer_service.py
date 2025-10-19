"""
Enhanced Customer Service Agent Example

This example demonstrates the enhanced GRPO Agent Framework features inspired by
real-world implementations, including:
- Advanced data loading and preprocessing
- Domain-specific reward functions
- Train/eval split with stratification
- Post-training evaluation
- Mixed precision training
- LoRA/PEFT optimization
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from stateset_agents import ConversationEnvironment, MultiTurnAgent, train
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.data_processing import load_and_prepare_data
from stateset_agents.core.reward import (
    CompositeReward,
    ConcisenessReward,
    HelpfulnessReward,
    SafetyReward,
    create_adaptive_reward,
    create_domain_reward,
)
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.trainer import MultiTurnGRPOTrainer
from stateset_agents.utils.wandb_integration import init_wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def create_enhanced_customer_service_agent(
    model_name: str = "openai/gpt-oss-120b", use_lora: bool = True
) -> MultiTurnAgent:
    """Create an enhanced customer service agent with LoRA optimization"""

    # Configure agent with enhanced settings
    config = AgentConfig(
        model_name=model_name,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        system_prompt="You are a helpful and professional customer service representative. "
        "Be empathetic, provide clear solutions, and maintain a friendly tone.",
        use_peft=use_lora,
        peft_config={
            "r": 8,
            "lora_alpha": 16,
            "target_modules": [
                "q_proj",
                "v_proj",
            ],  # Will be auto-detected based on model
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        if use_lora
        else None,
        torch_dtype="bfloat16",  # Use bfloat16 for better stability
        attn_implementation="flash_attention_2"
        if model_name not in ["gpt2", "openai/gpt-oss-120b"]
        else None,
    )

    # Create agent
    agent = MultiTurnAgent(config=config, memory_window=10, context_compression=True)

    # Initialize agent
    await agent.initialize()
    logger.info(f"Agent initialized with model: {model_name}")

    return agent


def create_enhanced_reward_function(
    domain: str = "customer_service", expected_responses: Dict[str, str] = None
) -> CompositeReward:
    """Create an enhanced composite reward function"""

    # Base domain-specific reward
    domain_reward = create_domain_reward(
        domain=domain, weight=0.4, expected_responses=expected_responses
    )

    # Additional reward components
    reward_components = [
        domain_reward,
        HelpfulnessReward(weight=0.2),
        SafetyReward(weight=0.2),
        ConcisenessReward(weight=0.2, optimal_length=150),
    ]

    # Create adaptive reward if we have expected responses
    if expected_responses:
        return create_adaptive_reward(
            base_rewards=reward_components,
            expected_responses=expected_responses,
            similarity_weight=0.3,
        )
    else:
        return CompositeReward(reward_components)


async def prepare_training_data(
    data_path: str, max_examples: int = None, validation_split: float = 0.1
) -> tuple:
    """Load and prepare training data with train/eval split"""

    logger.info(f"Loading data from: {data_path}")

    # Load and prepare data
    train_data, eval_data = load_and_prepare_data(
        data_path=data_path,
        max_examples=max_examples,
        validation_split=validation_split,
    )

    logger.info(
        f"Data prepared: {len(train_data)} train, {len(eval_data)} eval examples"
    )

    # Extract expected responses for similarity-based rewards
    expected_responses = {}
    for example in train_data:
        if "expected_response" in example:
            expected_responses[example["prompt"]] = example["expected_response"]

    return train_data, eval_data, expected_responses


async def main():
    """Main training pipeline with enhanced features"""

    # Configuration
    DATA_PATH = "customer_service_conversations.jsonl"  # Your data file
    MODEL_NAME = "openai/gpt-oss-120b"  # Using the openai/gpt-oss-120b model
    OUTPUT_DIR = "./outputs/enhanced_customer_service"

    # Training configuration with enhanced settings
    config = TrainingConfig(
        # Basic settings
        num_episodes=1000,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # Enhanced GRPO settings
        num_generations=16,
        beta=0.01,  # Small KL penalty
        use_reference_model=True,
        advantage_normalization=True,
        # Data settings
        max_examples=5000,
        eval_split_size=0.1,
        stratify_by_task=True,
        # Model optimization
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        # Hardware optimization
        bf16=True,
        gradient_checkpointing=True,
        # Evaluation
        eval_steps=50,
        save_steps=100,
        run_post_eval=True,
        post_eval_samples=10,
        # Output
        output_dir=OUTPUT_DIR,
        run_name="enhanced-cs-agent",
        # W&B integration
        report_to="wandb",
        wandb_project="grpo-enhanced-cs",
    )

    try:
        # Initialize W&B
        init_wandb(
            project=config.wandb_project, name=config.run_name, config=config.__dict__
        )

        # Step 1: Create agent
        logger.info("Creating enhanced customer service agent...")
        agent = await create_enhanced_customer_service_agent(
            model_name=MODEL_NAME, use_lora=config.use_lora
        )

        # Step 2: Load and prepare data
        logger.info("Loading and preparing training data...")
        train_data, eval_data, expected_responses = await prepare_training_data(
            data_path=DATA_PATH,
            max_examples=config.max_examples,
            validation_split=config.eval_split_size,
        )

        # Step 3: Create environment with scenarios
        logger.info("Creating training environment...")

        # Convert data to scenarios format
        train_scenarios = [
            {
                "id": f"train_{i}",
                "initial_message": example["query"],
                "expected_response": example.get("expected_response"),
                "task_type": example.get("task_type", "general"),
            }
            for i, example in enumerate(train_data)
        ]

        eval_scenarios = [
            {
                "id": f"eval_{i}",
                "initial_message": example["query"],
                "expected_response": example.get("expected_response"),
                "task_type": example.get("task_type", "general"),
            }
            for i, example in enumerate(eval_data)
        ]

        # Create environment
        environment = ConversationEnvironment(
            scenarios=train_scenarios, max_turns=10, timeout_seconds=30.0
        )

        # Step 4: Create reward function
        logger.info("Creating enhanced reward function...")
        reward_fn = create_enhanced_reward_function(
            domain="customer_service", expected_responses=expected_responses
        )

        # Step 5: Create trainer with enhanced features
        logger.info("Initializing enhanced GRPO trainer...")
        trainer = MultiTurnGRPOTrainer(
            agent=agent, environment=environment, reward_fn=reward_fn, config=config
        )

        # Initialize trainer
        await trainer.initialize()

        # Step 6: Train the agent
        logger.info("Starting enhanced GRPO training...")
        trained_agent = await trainer.train()

        # Step 7: Run post-training evaluation
        if config.run_post_eval and eval_scenarios:
            logger.info("Running post-training evaluation...")
            eval_results = await trainer.run_post_training_evaluation(
                eval_scenarios=eval_scenarios,
                num_samples=config.post_eval_samples,
                detailed=config.post_eval_detailed,
            )

            # Log evaluation results
            logger.info("Post-training evaluation results:")
            logger.info(
                f"  Overall mean reward: {eval_results['overall_stats']['overall_mean_reward']:.4f}"
            )
            logger.info(f"  Reward distribution:")
            for percentile, value in eval_results["overall_stats"][
                "reward_distribution"
            ].items():
                logger.info(f"    {percentile}: {value:.4f}")

            # Show sample conversations if detailed
            if config.post_eval_detailed and eval_results.get("scenario_results"):
                logger.info("\nSample conversations:")
                for i, scenario in enumerate(eval_results["scenario_results"][:3]):
                    logger.info(f"\n--- Scenario {i+1} ---")
                    best_traj = max(scenario["trajectories"], key=lambda x: x["reward"])
                    logger.info(f"Best reward: {best_traj['reward']:.4f}")
                    for turn in best_traj["conversation"][:4]:  # Show first 4 turns
                        logger.info(f"{turn['role'].capitalize()}: {turn['content']}")

        # Step 8: Save the trained model
        logger.info(f"Saving trained model to {OUTPUT_DIR}/final_model...")
        await trainer.save_checkpoint(checkpoint_name="final_model", is_best=True)

        logger.info("✨ Training completed successfully!")

        # Step 9: Interactive testing (optional)
        if (
            input(
                "\nWould you like to test the trained agent interactively? (y/n): "
            ).lower()
            == "y"
        ):
            await test_agent_interactive(trained_agent)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        if "trainer" in locals() and hasattr(trainer, "wandb_logger"):
            trainer.wandb_logger.finish_run()


async def test_agent_interactive(agent: MultiTurnAgent):
    """Test the trained agent interactively"""

    logger.info("\n🤖 Interactive testing mode. Type 'quit' to exit.")

    conversation_history = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            logger.info("Goodbye!")
            break

        # Generate response
        messages = conversation_history + [{"role": "user", "content": user_input}]

        try:
            response = await agent.generate_response(messages)
            print(f"\nAgent: {response}")

            # Update history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

            # Keep conversation manageable
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

        except Exception as e:
            logger.error(f"Error generating response: {e}")


if __name__ == "__main__":
    asyncio.run(main())
