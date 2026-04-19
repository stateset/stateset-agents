"""
Auto-Research Example — Autonomous Hyperparameter Optimization
==============================================================

This example demonstrates the autonomous research loop, which iterates
on training configurations to maximize agent reward — the same pattern
used by the autoresearch project, now built into stateset-agents.

The loop:
1. Evaluates a baseline agent
2. Proposes a new hyperparameter configuration
3. Trains the agent with a time budget
4. Evaluates on held-out scenarios
5. Keeps improvements, reverts failures
6. Repeats

This example uses a stub model so it runs without a GPU. Replace the
agent config with a real model to run actual training.

Usage:
    python examples/auto_research.py
    python examples/auto_research.py --max-experiments 20
    python examples/auto_research.py --proposer bayesian
"""

import argparse
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)


async def main(max_experiments: int = 5, proposer: str = "perturbation"):
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.core.reward import (
        CompositeReward,
        HelpfulnessReward,
        SafetyReward,
    )
    from stateset_agents.training.auto_research import (
        AutoResearchConfig,
        run_auto_research,
    )

    # ── Agent (stub mode — replace with real model for actual training) ──
    agent_config = AgentConfig(
        model_name="stub://auto-research-demo",
        use_stub_model=True,
        stub_responses=[
            "I'd be happy to help you with that! Let me look into it.",
            "Thank you for reaching out. I understand your concern.",
            "I've checked our records and here's what I found.",
            "Is there anything else I can help you with today?",
        ],
        system_prompt="You are a helpful customer service agent.",
        temperature=0.7,
        max_new_tokens=64,
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    # ── Training environment ──
    train_scenarios = [
        {
            "topic": "general_inquiry",
            "context": "Customer has a general question about services.",
            "user_responses": [
                "Hi, I have a question about your services.",
                "What are your business hours?",
                "Thanks for the help!",
            ],
        },
        {
            "topic": "feedback",
            "context": "Customer wants to share feedback.",
            "user_responses": [
                "I wanted to give feedback about my recent order.",
                "The delivery was faster than expected!",
                "Keep up the good work.",
            ],
        },
    ]

    environment = ConversationEnvironment(
        scenarios=train_scenarios,
        max_turns=6,
    )

    # ── Evaluation scenarios (held out — never used for training) ──
    eval_scenarios = [
        {
            "topic": "order_status",
            "context": "Customer placed an order and wants an update.",
            "user_responses": [
                "Hi, I placed an order on Monday.",
                "The order number is #12345.",
                "When will it arrive?",
            ],
        },
        {
            "topic": "product_return",
            "context": "Customer received a damaged item.",
            "user_responses": [
                "I received my order but the item is broken.",
                "It's a wireless keyboard.",
                "I'd like a refund please.",
            ],
        },
        {
            "topic": "billing_dispute",
            "context": "Customer was charged twice.",
            "user_responses": [
                "I was charged twice for order #67890.",
                "Yes I can see both charges.",
                "When will the refund appear?",
            ],
        },
    ]

    # ── Reward function ──
    reward_fn = CompositeReward([
        HelpfulnessReward(weight=0.6),
        SafetyReward(weight=0.4),
    ])

    # ── Auto-research config ──
    config = AutoResearchConfig(
        time_budget=60,  # 1 minute per experiment (short for demo)
        max_experiments=max_experiments,
        proposer=proposer,
        search_space_name="quick",  # Small search space for demo
        eval_episodes=3,
        eval_seed=42,
        output_dir="./auto_research_demo_results",
        save_checkpoints=False,  # Skip checkpoints in stub mode
        trainer_algorithm="gspo",
    )

    # ── Baseline params ──
    baseline_params = {
        "learning_rate": 5e-6,
        "num_generations": 4,
        "temperature": 0.7,
        "lora_r": 8,
    }

    # ── Run the loop ──
    print("=" * 60)
    print("Auto-Research Demo")
    print("=" * 60)
    print(f"  Proposer:        {proposer}")
    print(f"  Max experiments: {max_experiments}")
    print("  Search space:    quick")
    print(f"  Output:          {config.output_dir}")
    print()

    tracker = await run_auto_research(
        agent=agent,
        environment=environment,
        eval_scenarios=eval_scenarios,
        reward_fn=reward_fn,
        config=config,
        baseline_params=baseline_params,
    )

    # The tracker prints its own summary, but we can also access results
    if tracker.best_record:
        print("\nBest config found:")
        for k, v in tracker.best_record.params.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Research Demo")
    parser.add_argument(
        "--max-experiments", type=int, default=5,
        help="Maximum experiments to run",
    )
    parser.add_argument(
        "--proposer", type=str, default="perturbation",
        choices=["random", "perturbation", "grid", "bayesian"],
        help="Proposer strategy",
    )
    args = parser.parse_args()
    asyncio.run(main(args.max_experiments, args.proposer))
