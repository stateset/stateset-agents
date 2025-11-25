"""
Complete GRPO Training Example

This example demonstrates the fully implemented GRPO (Group Relative Policy Optimization)
training pipeline with:
- Real policy gradient computation
- KL divergence regularization
- Value function with GAE (Generalized Advantage Estimation)
- PPO-style clipping
- Multi-turn conversation support

This is a production-ready implementation suitable for training conversational AI agents.
"""

import asyncio
import logging
from pathlib import Path

# Framework imports
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import create_customer_service_reward
from stateset_agents.core.value_function import create_value_function
from training.trainer import MultiTurnGRPOTrainer
from training.config import TrainingConfig, TrainingProfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """
    Complete GRPO training example with all features enabled.
    """

    logger.info("=" * 80)
    logger.info("StateSet Agents - Complete GRPO Training Example")
    logger.info("=" * 80)

    # =========================================================================
    # Step 1: Create Agent Configuration
    # =========================================================================
    logger.info("\n[Step 1] Configuring Agent...")

    agent_config = AgentConfig(
        model_name="gpt2",  # Use a small model for demonstration
        # For production, use a larger model like:
        # model_name="meta-llama/Llama-2-7b-chat-hf",
        # model_name="mistralai/Mistral-7B-Instruct-v0.1",

        system_prompt=(
            "You are a helpful and empathetic customer service representative. "
            "Your goal is to understand customer issues and provide clear, "
            "actionable solutions."
        ),
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    logger.info(f"  Model: {agent_config.model_name}")
    logger.info(f"  Max tokens: {agent_config.max_new_tokens}")

    # =========================================================================
    # Step 2: Create MultiTurn Agent
    # =========================================================================
    logger.info("\n[Step 2] Initializing Multi-Turn Agent...")

    agent = MultiTurnAgent(
        config=agent_config,
        memory_window=10,  # Remember last 10 turns
        context_compression=True,  # Enable context compression
    )

    await agent.initialize()
    logger.info("  ✓ Agent initialized successfully")

    # =========================================================================
    # Step 3: Create Training Environment
    # =========================================================================
    logger.info("\n[Step 3] Setting up Training Environment...")

    # Define conversation scenarios for training
    training_scenarios = [
        {
            "id": "refund_request",
            "topic": "refund",
            "user_goal": "Get a refund for delayed order",
            "context": "Order #12345 is 5 days late",
            "difficulty": "medium",
        },
        {
            "id": "shipping_inquiry",
            "topic": "shipping",
            "user_goal": "Track package location",
            "context": "Order #67890 shipped 3 days ago",
            "difficulty": "easy",
        },
        {
            "id": "product_complaint",
            "topic": "complaint",
            "user_goal": "Report defective product",
            "context": "Product arrived damaged",
            "difficulty": "hard",
        },
        {
            "id": "account_issue",
            "topic": "account",
            "user_goal": "Reset account password",
            "context": "Forgot password, no access to email",
            "difficulty": "medium",
        },
    ]

    environment = ConversationEnvironment(
        scenarios=training_scenarios,
        max_turns=6,  # Allow up to 6 turns per conversation
        initial_user_message_templates=[
            "Hi, I have a problem with my order.",
            "Hello, I need help with {topic}.",
            "Can you assist me with {user_goal}?",
        ],
    )

    logger.info(f"  ✓ Environment created with {len(training_scenarios)} scenarios")
    logger.info(f"  Max turns per conversation: 6")

    # =========================================================================
    # Step 4: Create Reward Function
    # =========================================================================
    logger.info("\n[Step 4] Configuring Reward Function...")

    # Use multi-objective reward for customer service
    reward_fn = create_customer_service_reward()

    logger.info("  ✓ Customer service reward function created")
    logger.info("    Components: Helpfulness, Empathy, Action-oriented, Professionalism")

    # =========================================================================
    # Step 5: Configure Training with GRPO Features
    # =========================================================================
    logger.info("\n[Step 5] Configuring GRPO Training...")

    # Use a balanced training profile as a starting point
    train_config = TrainingConfig.from_profile(
        TrainingProfile.BALANCED,
        # Override with GRPO-specific settings
        num_episodes=100,
        num_generations=8,  # Generate 8 trajectories per scenario
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        # GRPO-specific parameters
        beta=0.1,  # KL divergence penalty coefficient
        use_reference_model=True,  # Enable reference model for KL regularization
        clip_ratio=0.2,  # PPO-style clipping ratio
        value_clip=0.2,  # Value function clipping
        advantage_normalization=True,
        baseline_type="group_mean",  # Use group mean as baseline

        # Optimization
        bf16=True,  # Use bfloat16 for efficiency
        gradient_checkpointing=False,  # Disable for small models
        max_grad_norm=1.0,

        # Logging and evaluation
        logging_steps=5,
        eval_steps=25,
        save_steps=50,
        output_dir="./outputs/complete_grpo_example",
        run_name="grpo-customer-service-complete",

        # W&B integration (optional)
        report_to="none",  # Set to "wandb" if you have W&B configured
        # wandb_project="stateset-agents",
        # wandb_tags=["grpo", "customer-service", "complete-implementation"],

        # Early stopping
        early_stopping=True,
        patience=30,
    )

    logger.info("  Training Configuration:")
    logger.info(f"    Episodes: {train_config.num_episodes}")
    logger.info(f"    Generations per scenario: {train_config.num_generations}")
    logger.info(f"    Learning rate: {train_config.learning_rate}")
    logger.info(f"    KL penalty (beta): {train_config.beta}")
    logger.info(f"    Clip ratio: {train_config.clip_ratio}")
    logger.info(f"    Use reference model: {train_config.use_reference_model}")

    # =========================================================================
    # Step 6: Create GRPO Trainer
    # =========================================================================
    logger.info("\n[Step 6] Initializing GRPO Trainer...")

    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        config=train_config,
    )

    await trainer.initialize()
    logger.info("  ✓ Trainer initialized")
    logger.info(f"    Optimizer: AdamW")
    logger.info(f"    Scheduler: {train_config.lr_scheduler_type}")
    logger.info(f"    Reference model: {'Loaded' if trainer.reference_model else 'Not used'}")

    # =========================================================================
    # Step 7: (Optional) Create Value Function for GAE
    # =========================================================================
    logger.info("\n[Step 7] Setting up Value Function for GAE...")

    value_function = create_value_function(
        model=agent.model,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
    )

    logger.info("  ✓ Value function created")
    logger.info(f"    Gamma (discount factor): 0.99")
    logger.info(f"    GAE lambda: 0.95")

    # =========================================================================
    # Step 8: Run Training
    # =========================================================================
    logger.info("\n[Step 8] Starting GRPO Training...")
    logger.info("=" * 80)

    try:
        # Run the full training loop
        trained_agent = await trainer.train()

        logger.info("=" * 80)
        logger.info("✓ Training completed successfully!")

        # =====================================================================
        # Step 9: Post-Training Evaluation
        # =====================================================================
        logger.info("\n[Step 9] Running Post-Training Evaluation...")

        eval_scenarios = training_scenarios[:2]  # Use subset for quick eval

        eval_results = await trainer.run_post_training_evaluation(
            eval_scenarios=eval_scenarios,
            num_samples=2,
            detailed=True,
        )

        logger.info("\nEvaluation Results:")
        logger.info(f"  Mean Reward: {eval_results['overall_stats']['overall_mean_reward']:.4f}")
        logger.info(f"  Std Reward: {eval_results['overall_stats']['overall_std_reward']:.4f}")
        logger.info(f"  Mean Episode Length: {eval_results['overall_stats']['overall_mean_length']:.2f}")

        # =====================================================================
        # Step 10: Save Final Model
        # =====================================================================
        logger.info("\n[Step 10] Saving Final Model...")

        output_path = Path(train_config.output_dir) / "final_model"
        await trainer.save_checkpoint(checkpoint_name="final_model")

        logger.info(f"  ✓ Model saved to: {output_path}")

        # =====================================================================
        # Step 11: Test the Trained Agent
        # =====================================================================
        logger.info("\n[Step 11] Testing Trained Agent...")

        test_messages = [
            {"role": "user", "content": "Hi, my order is delayed. Can you help?"}
        ]

        response = await trained_agent.generate_response(
            test_messages,
            context={"order_status": "delayed", "customer_value": "high"}
        )

        logger.info("\nTest Conversation:")
        logger.info(f"  User: {test_messages[0]['content']}")
        logger.info(f"  Agent: {response[:200]}...")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        raise

    logger.info("\n" + "=" * 80)
    logger.info("Complete GRPO Training Example Finished!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
