"""
Complete Example: Training a Neural Reward Model

This example demonstrates end-to-end training of a transformer-based reward model,
from data collection to deployment in GRPO training.
"""

import asyncio
from pathlib import Path
from typing import List

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import CompositeReward, create_customer_service_reward
from stateset_agents.training.transformer_reward_model import (
    LearnedRewardFunction,
    RewardExample,
    RewardTrainingConfig,
    TransformerRewardTrainer,
)


async def collect_training_data(num_episodes: int = 100) -> List[RewardExample]:
    """
    Collect training data by running the agent with heuristic rewards

    In production, this could be:
    - Human annotations
    - Existing conversation logs with ratings
    - Synthetic data from rule-based systems
    """
    print(f"Collecting {num_episodes} training examples...")

    # Create agent
    agent = MultiTurnAgent(AgentConfig(model_name="gpt2", use_stub_model=True))
    await agent.initialize()

    # Create environment with heuristic rewards
    heuristic_reward = create_customer_service_reward()
    environment = ConversationEnvironment(
        scenarios=[
            {
                "topic": "refund_request",
                "user_goal": "Get refund for delayed order",
                "context": {"order_id": "12345", "order_status": "delayed"},
            },
            {
                "topic": "product_inquiry",
                "user_goal": "Ask about product features",
                "context": {"product": "laptop"},
            },
            {
                "topic": "technical_support",
                "user_goal": "Fix login issue",
                "context": {"issue": "cannot_login"},
            },
            {
                "topic": "shipping_inquiry",
                "user_goal": "Track shipment",
                "context": {"tracking_number": "ABC123"},
            },
        ],
        max_turns=4,
        reward_fn=heuristic_reward,
    )

    # Collect episodes
    examples = []
    for episode_num in range(num_episodes):
        # Run episode
        trajectory = await environment.run_episode(agent)

        # Compute reward
        reward_result = await heuristic_reward.compute_reward(
            trajectory.turns, trajectory.metadata
        )

        # Create training example
        example = RewardExample.from_conversation_turns(
            turns=trajectory.turns,
            reward=reward_result.score,
            metadata={
                "episode": episode_num,
                "topic": trajectory.metadata.get("scenario", {}).get("topic"),
                "breakdown": reward_result.breakdown,
            },
        )
        examples.append(example)

        if (episode_num + 1) % 20 == 0:
            print(f"  Collected {episode_num + 1}/{num_episodes} examples")

    print(f"✓ Collected {len(examples)} training examples")
    return examples


async def train_reward_model(
    train_examples: List[RewardExample],
    val_examples: List[RewardExample],
    checkpoint_dir: str = "./checkpoints/reward_model",
) -> TransformerRewardTrainer:
    """
    Train the transformer reward model
    """
    print("\nTraining transformer reward model...")

    # Configure training
    config = RewardTrainingConfig(
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        max_length=256,
        hidden_dim=512,
        num_layers=3,
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=10,
        patience=3,
        device="auto",
    )

    # Create trainer
    trainer = TransformerRewardTrainer(config=config)

    # Train model (start with frozen encoders for faster training)
    print("\nPhase 1: Training reward head only (frozen encoders)")
    results_phase1 = trainer.train(
        train_examples=train_examples,
        val_examples=val_examples,
        freeze_encoders=True,
        verbose=True,
    )

    print(f"\nPhase 1 Results:")
    print(f"  Final train loss: {results_phase1['final_train_loss']:.4f}")
    print(f"  Final val loss: {results_phase1['final_val_loss']:.4f}")
    print(f"  Best val loss: {results_phase1['best_val_loss']:.4f}")

    # Fine-tune entire model
    print("\nPhase 2: Fine-tuning entire model (unfrozen encoders)")
    trainer.model.unfreeze_encoders()
    trainer.config.learning_rate = 5e-6  # Lower learning rate for fine-tuning
    trainer.config.num_epochs = 5

    results_phase2 = trainer.train(
        train_examples=train_examples,
        val_examples=val_examples,
        freeze_encoders=False,
        verbose=True,
    )

    print(f"\nPhase 2 Results:")
    print(f"  Final train loss: {results_phase2['final_train_loss']:.4f}")
    print(f"  Final val loss: {results_phase2['final_val_loss']:.4f}")
    print(f"  Best val loss: {results_phase2['best_val_loss']:.4f}")

    # Save checkpoint
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/best_model.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\n✓ Model saved to {checkpoint_path}")

    return trainer


async def evaluate_reward_model(
    trainer: TransformerRewardTrainer, test_examples: List[RewardExample]
):
    """
    Evaluate the trained reward model
    """
    print("\nEvaluating reward model...")

    predictions = []
    ground_truth = []

    for example in test_examples:
        # Predict reward
        pred_reward = trainer.predict(example.prompt, example.response)
        predictions.append(pred_reward)
        ground_truth.append(example.reward)

    # Compute metrics
    import numpy as np

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    correlation = np.corrcoef(predictions, ground_truth)[0, 1]

    print("\nEvaluation Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Correlation: {correlation:.4f}")

    # Show some examples
    print("\nSample Predictions:")
    for i in range(min(5, len(test_examples))):
        print(f"\nExample {i + 1}:")
        print(f"  Prompt: {test_examples[i].prompt[:100]}...")
        print(f"  Response: {test_examples[i].response[:100]}...")
        print(f"  Ground Truth: {ground_truth[i]:.3f}")
        print(f"  Predicted: {predictions[i]:.3f}")
        print(f"  Error: {abs(predictions[i] - ground_truth[i]):.3f}")


async def train_grpo_with_learned_reward(trainer: TransformerRewardTrainer):
    """
    Use the learned reward model in GRPO training
    """
    print("\n" + "=" * 60)
    print("Training GRPO Agent with Learned Reward Model")
    print("=" * 60)

    # Create learned reward function
    learned_reward = LearnedRewardFunction(trainer=trainer, weight=1.0, normalize=True)

    # Optionally combine with heuristic rewards
    heuristic_reward = create_customer_service_reward()
    composite_reward = CompositeReward(
        reward_functions=[
            learned_reward,  # Weight from learned model
            heuristic_reward,  # Additional heuristic guidance
        ],
        combination_method="weighted_sum",
    )

    # Create agent
    agent = MultiTurnAgent(AgentConfig(model_name="gpt2", use_stub_model=True))
    await agent.initialize()

    # Create environment with learned reward
    environment = ConversationEnvironment(
        scenarios=[
            {
                "topic": "customer_service",
                "user_goal": "Resolve customer issue",
                "context": {},
            }
        ],
        max_turns=6,
        reward_fn=learned_reward,  # Use learned reward
    )

    print("\n✓ GRPO training configured with learned reward model")
    print("  Run: python examples/train_with_trl_grpo.py --reward=learned")

    # Run a test episode
    print("\nRunning test episode with learned reward...")
    trajectory = await environment.run_episode(agent)
    reward_result = await learned_reward.compute_reward(
        trajectory.turns, trajectory.metadata
    )

    print(f"\nTest Episode Results:")
    print(f"  Turns: {len(trajectory.turns)}")
    print(f"  Learned Reward: {reward_result.score:.3f}")
    print(f"  Breakdown: {reward_result.breakdown}")


async def main():
    """
    Complete reward model training pipeline
    """
    print("=" * 60)
    print("Neural Reward Model Training Pipeline")
    print("=" * 60)

    # Step 1: Collect training data
    all_examples = await collect_training_data(num_episodes=200)

    # Step 2: Split data
    import random

    random.shuffle(all_examples)

    n_train = int(0.7 * len(all_examples))
    n_val = int(0.15 * len(all_examples))

    train_examples = all_examples[:n_train]
    val_examples = all_examples[n_train : n_train + n_val]
    test_examples = all_examples[n_train + n_val :]

    print(f"\nData Split:")
    print(f"  Training: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")

    # Step 3: Train reward model
    trainer = await train_reward_model(train_examples, val_examples)

    # Step 4: Evaluate model
    await evaluate_reward_model(trainer, test_examples)

    # Step 5: Deploy in GRPO training
    await train_grpo_with_learned_reward(trainer)

    print("\n" + "=" * 60)
    print("✓ Complete! Reward model ready for production use")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
