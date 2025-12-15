"""
Comprehensive example of hyperparameter optimization with StateSet Agents.

This example demonstrates:
1. Setting up an agent, environment, and reward function
2. Configuring HPO with different strategies
3. Running HPO optimization
4. Training with best parameters
5. Analyzing results

Requirements:
    pip install stateset-agents[training]
    pip install optuna  # For HPO backend
"""

import asyncio
from pathlib import Path

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import (
    CompositeReward,
    EngagementReward,
    HelpfulnessReward,
    SafetyReward,
)
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.hpo import (
    GRPOHPOTrainer,
    HPOConfig,
    quick_hpo,
    create_grpo_search_space,
    create_customer_service_search_space,
    get_hpo_config,
)


# ============================================================================
# Setup: Agent, Environment, Reward
# ============================================================================

def setup_training_components():
    """Setup agent, environment, and reward function."""

    # 1. Create agent
    agent_config = AgentConfig(
        model_name="stub://demo",  # Use stub for fast demo
        use_stub_model=True,
        stub_responses=[
            "I'd be happy to help with that!",
            "Let me assist you with your question.",
            "I understand your concern. Here's what I can do..."
        ]
    )
    agent = MultiTurnAgent(agent_config)

    # 2. Create environment with conversation scenarios
    scenarios = [
        {
            "conversation_id": "scenario_1",
            "user_inputs": [
                "I need help with my order",
                "Can you check the status?",
                "When will it arrive?"
            ],
            "max_turns": 3
        },
        {
            "conversation_id": "scenario_2",
            "user_inputs": [
                "I have a question about returns",
                "How long do I have to return it?",
                "What's the process?"
            ],
            "max_turns": 3
        },
        {
            "conversation_id": "scenario_3",
            "user_inputs": [
                "I'm having trouble logging in",
                "I forgot my password",
                "Can you help me reset it?"
            ],
            "max_turns": 3
        }
    ]
    environment = ConversationEnvironment(scenarios)

    # 3. Create reward function
    reward_function = CompositeReward([
        (HelpfulnessReward(), 0.4),
        (SafetyReward(), 0.3),
        (EngagementReward(), 0.3)
    ])

    # 4. Base training config (will be overridden by HPO)
    base_config = TrainingConfig(
        learning_rate=1e-5,
        num_episodes=500,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gamma=0.99,
        gae_lambda=0.95,
        kl_penalty_coef=0.01,
        max_grad_norm=1.0,
        output_dir="./hpo_demo_output"
    )

    return agent, environment, reward_function, base_config


# ============================================================================
# Example 1: Quick HPO with Defaults
# ============================================================================

async def example_quick_hpo():
    """Quick HPO with sensible defaults - easiest way to get started."""

    print("\n" + "="*60)
    print("EXAMPLE 1: Quick HPO with Defaults")
    print("="*60 + "\n")

    agent, environment, reward_function, base_config = setup_training_components()

    # Run quick HPO - this is the simplest way
    summary = await quick_hpo(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        base_config=base_config,
        n_trials=10,  # Use 10 trials for quick demo
        search_space_name="grpo",
        output_dir=Path("./quick_hpo_results")
    )

    print(f"\nBest hyperparameters found:")
    for param, value in summary.best_params.items():
        print(f"  {param}: {value}")

    print(f"\nBest metric: {summary.best_metric:.4f}")


# ============================================================================
# Example 2: Custom HPO Configuration
# ============================================================================

async def example_custom_hpo():
    """Custom HPO with specific search space and configuration."""

    print("\n" + "="*60)
    print("EXAMPLE 2: Custom HPO Configuration")
    print("="*60 + "\n")

    agent, environment, reward_function, base_config = setup_training_components()

    # Create custom search space
    search_space = create_grpo_search_space(
        include_value_function=True,
        include_kl_penalty=True,
        include_ppo_clipping=False
    )

    # Configure HPO
    hpo_config = HPOConfig(
        backend="optuna",
        search_space=search_space,
        n_trials=20,
        objective_metric="reward",
        direction="maximize",
        output_dir=Path("./custom_hpo_results"),
        study_name="custom_grpo_study",
        optuna_config={
            "sampler": "tpe",  # Tree-structured Parzen Estimator
            "pruner": "median",  # Median pruning for early stopping
            "n_startup_trials": 5,  # Random trials before TPE
            "n_warmup_steps": 3,  # Warmup before pruning
        }
    )

    # Create HPO trainer
    hpo_trainer = GRPOHPOTrainer(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        base_config=base_config,
        hpo_config=hpo_config
    )

    # Run optimization
    print("Starting hyperparameter optimization...")
    summary = await hpo_trainer.optimize()

    # Plot results (requires matplotlib and plotly)
    try:
        hpo_trainer.plot_results()
        print("\nVisualization plots saved to output directory")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")

    return hpo_trainer, summary


# ============================================================================
# Example 3: Domain-Specific Search Space
# ============================================================================

async def example_domain_specific_hpo():
    """HPO with domain-specific search space (customer service)."""

    print("\n" + "="*60)
    print("EXAMPLE 3: Domain-Specific HPO (Customer Service)")
    print("="*60 + "\n")

    agent, environment, reward_function, base_config = setup_training_components()

    # Use customer service search space
    search_space = create_customer_service_search_space()

    hpo_config = HPOConfig(
        backend="optuna",
        search_space=search_space,
        n_trials=15,
        objective_metric="reward",
        direction="maximize",
        output_dir=Path("./cs_hpo_results")
    )

    hpo_trainer = GRPOHPOTrainer(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        base_config=base_config,
        hpo_config=hpo_config
    )

    summary = await hpo_trainer.optimize()

    return hpo_trainer, summary


# ============================================================================
# Example 4: Pre-defined HPO Profiles
# ============================================================================

async def example_hpo_profiles():
    """Using pre-defined HPO profiles."""

    print("\n" + "="*60)
    print("EXAMPLE 4: Pre-defined HPO Profiles")
    print("="*60 + "\n")

    agent, environment, reward_function, base_config = setup_training_components()

    # Available profiles: "conservative", "aggressive", "quick", "distributed"
    profile = "conservative"

    print(f"Using '{profile}' HPO profile")

    hpo_config = get_hpo_config(profile)
    hpo_config.output_dir = Path(f"./{profile}_hpo_results")
    hpo_config.n_trials = 10  # Override for demo

    hpo_trainer = GRPOHPOTrainer(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        base_config=base_config,
        hpo_config=hpo_config
    )

    summary = await hpo_trainer.optimize()

    return summary


# ============================================================================
# Example 5: Full Workflow (HPO + Final Training)
# ============================================================================

async def example_full_workflow():
    """Complete workflow: HPO followed by full training with best params."""

    print("\n" + "="*60)
    print("EXAMPLE 5: Full Workflow (HPO + Final Training)")
    print("="*60 + "\n")

    agent, environment, reward_function, base_config = setup_training_components()

    # Step 1: Run HPO
    print("Step 1: Running HPO to find best hyperparameters...")

    hpo_config = HPOConfig(
        backend="optuna",
        search_space_name="grpo",
        n_trials=10,
        output_dir=Path("./full_workflow_results")
    )

    hpo_trainer = GRPOHPOTrainer(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        base_config=base_config,
        hpo_config=hpo_config
    )

    summary = await hpo_trainer.optimize()

    print(f"\nBest parameters found:")
    for param, value in summary.best_params.items():
        print(f"  {param}: {value}")

    # Step 2: Train with best parameters
    print("\nStep 2: Training with best parameters...")

    final_agent = await hpo_trainer.train_with_best_params(
        full_episodes=100  # Full training episodes
    )

    print(f"\nFinal agent saved to: {hpo_config.output_dir}/best_agent")

    # Step 3: Analyze results
    print("\nStep 3: Analyzing results...")

    best_params = hpo_trainer.get_best_params()
    print(f"Best learning rate: {best_params.get('learning_rate', 'N/A')}")
    print(f"Best gamma: {best_params.get('gamma', 'N/A')}")

    # Load results (demonstrates persistence)
    loaded_summary = hpo_trainer.load_results(hpo_config.output_dir)
    print(f"\nLoaded best metric: {loaded_summary.best_metric:.4f}")

    return final_agent, summary


# ============================================================================
# Example 6: Different Search Space Strategies
# ============================================================================

async def example_search_space_comparison():
    """Compare different search space strategies."""

    print("\n" + "="*60)
    print("EXAMPLE 6: Comparing Search Space Strategies")
    print("="*60 + "\n")

    agent, environment, reward_function, base_config = setup_training_components()

    strategies = ["conservative", "aggressive"]
    results = {}

    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")

        search_space = get_search_space(strategy)

        hpo_config = HPOConfig(
            backend="optuna",
            search_space=search_space,
            n_trials=10,
            output_dir=Path(f"./{strategy}_comparison")
        )

        hpo_trainer = GRPOHPOTrainer(
            agent=agent,
            environment=environment,
            reward_function=reward_function,
            base_config=base_config,
            hpo_config=hpo_config
        )

        summary = await hpo_trainer.optimize()
        results[strategy] = summary.best_metric

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    for strategy, metric in results.items():
        print(f"{strategy:15s}: {metric:.4f}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run all examples."""

    print("\n" + "="*70)
    print("StateSet Agents - Hyperparameter Optimization Examples")
    print("="*70)

    # Run examples
    # Uncomment the ones you want to run

    # Example 1: Quick HPO
    await example_quick_hpo()

    # Example 2: Custom HPO
    # hpo_trainer, summary = await example_custom_hpo()

    # Example 3: Domain-specific
    # hpo_trainer, summary = await example_domain_specific_hpo()

    # Example 4: HPO profiles
    # summary = await example_hpo_profiles()

    # Example 5: Full workflow (HPO + training)
    # final_agent, summary = await example_full_workflow()

    # Example 6: Compare strategies
    # await example_search_space_comparison()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
