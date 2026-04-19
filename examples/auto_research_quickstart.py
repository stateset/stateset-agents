#!/usr/bin/env python3
"""
Auto-Research Quickstart
========================

This is a complete, runnable example that shows how to use the
auto-research loop from scratch. Copy this file and modify it
for your own experiments.

What this does:
    1. Creates an agent with a model (stub for demo, swap for real)
    2. Defines training scenarios (what the agent trains on)
    3. Defines evaluation scenarios (held-out, never trained on)
    4. Sets up the auto-research loop with your chosen strategy
    5. Runs experiments autonomously, keeping only improvements
    6. Prints analysis of which hyperparameters matter most

Usage:
    # Quick demo with stub model (no GPU needed, ~30 seconds)
    python examples/auto_research_quickstart.py

    # Real training (requires GPU and a HuggingFace model)
    python examples/auto_research_quickstart.py --real

    # LLM-driven proposals (requires ANTHROPIC_API_KEY)
    python examples/auto_research_quickstart.py --real --proposer llm

Output:
    Results are saved to ./quickstart_results/:
    - experiments.jsonl  — full experiment log (one JSON per line)
    - results.tsv        — human-readable summary
    - analysis.txt       — parameter importance report
    - checkpoints/       — model weights for best experiments
"""

import argparse
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================================
# Step 1: Define your scenarios
# ============================================================================

# Training scenarios — the agent learns from these
TRAIN_SCENARIOS = [
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
    {
        "topic": "upgrade_request",
        "context": "Customer wants to upgrade their plan.",
        "user_responses": [
            "I'd like to upgrade my plan.",
            "I'm currently on Basic and want Premium.",
            "Will my existing data carry over?",
        ],
    },
]

# Evaluation scenarios — held out, never used for training.
# These are what the auto-research loop measures improvement on.
EVAL_SCENARIOS = [
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


# ============================================================================
# Step 2: Define your baseline hyperparameters
# ============================================================================

# These are the starting configuration. The auto-research loop will
# perturb these to find better values.
BASELINE_PARAMS = {
    "learning_rate": 5e-6,
    "num_generations": 4,
    "temperature": 0.7,
    "lora_r": 8,
}


async def main(args: argparse.Namespace) -> None:
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

    # ================================================================
    # Step 3: Create agent
    # ================================================================
    if args.real:
        # Real model — requires GPU
        agent_config = AgentConfig(
            model_name=args.model,
            system_prompt="You are a helpful customer service agent.",
            max_new_tokens=64,
            temperature=0.7,
        )
    else:
        # Stub model — no GPU needed, for testing the loop
        agent_config = AgentConfig(
            model_name="stub://quickstart",
            use_stub_model=True,
            stub_responses=[
                "I'd be happy to help you with that!",
                "Thank you for reaching out. Let me look into this.",
                "I've checked our records and here's what I found.",
                "Is there anything else I can help you with?",
            ],
            system_prompt="You are a helpful customer service agent.",
            max_new_tokens=64,
            temperature=0.7,
        )

    agent = MultiTurnAgent(agent_config)
    await agent.initialize()
    print(f"Agent ready: {agent_config.model_name}")

    # ================================================================
    # Step 4: Set up environment and reward
    # ================================================================
    environment = ConversationEnvironment(
        scenarios=TRAIN_SCENARIOS,
        max_turns=6,
    )

    reward_fn = CompositeReward([
        HelpfulnessReward(weight=0.6),
        SafetyReward(weight=0.4),
    ])

    # ================================================================
    # Step 5: Configure auto-research
    # ================================================================
    config = AutoResearchConfig(
        # How long each experiment trains (seconds)
        time_budget=args.time_budget,

        # How many experiments to run (0 = unlimited)
        max_experiments=args.max_experiments,

        # Stop if last N experiments don't improve (0 = disabled)
        improvement_patience=args.patience,

        # How the loop proposes new hyperparameters:
        #   "perturbation" — small random changes (default, fast)
        #   "smart"        — learns which params matter, focuses there
        #   "adaptive"     — starts broad, narrows down over time
        #   "bayesian"     — Optuna TPE (needs pip install optuna)
        #   "llm"          — Claude/OpenAI proposes (needs API key)
        #   "random"       — pure random sampling
        #   "grid"         — systematic grid search
        proposer=args.proposer,

        # Which hyperparameters to search over:
        #   "quick"           — just LR, num_gens, temp, lora_r (4 dims)
        #   "auto_research"   — comprehensive (14 dims)
        #   "multi_algorithm" — includes algorithm choice (GSPO/GRPO/DAPO/VAPO)
        #   "reward"          — reward weight exploration
        #   "model"           — LoRA/generation architecture
        search_space_name=args.search_space,

        # Where to save results
        output_dir=args.output_dir,

        # Evaluation settings
        eval_episodes=3,
        eval_seed=42,

        # Save model checkpoints for kept experiments
        save_checkpoints=args.real,

        # Training algorithm
        trainer_algorithm="gspo",
    )

    # ================================================================
    # Step 6: Run the loop
    # ================================================================
    print()
    print("=" * 50)
    print("Starting auto-research")
    print("=" * 50)
    print(f"  Proposer:    {config.proposer}")
    print(f"  Search:      {config.search_space_name}")
    print(f"  Budget:      {config.time_budget}s per experiment")
    print(f"  Max exps:    {config.max_experiments or 'unlimited'}")
    print(f"  Output:      {config.output_dir}")
    print()

    tracker = await run_auto_research(
        agent=agent,
        environment=environment,
        eval_scenarios=EVAL_SCENARIOS,
        reward_fn=reward_fn,
        config=config,
        baseline_params=BASELINE_PARAMS,
    )

    # ================================================================
    # Step 7: Inspect results programmatically
    # ================================================================
    analysis = tracker.get_analysis()

    print()
    print("Programmatic analysis:")
    print(f"  Best objective:      {analysis['best_value']}")
    print(f"  Improvement rate:    {analysis.get('improvement_rate', 0):.0%}")
    print("  Parameter importance:")
    for param, score in analysis.get("parameter_importance", {}).items():
        print(f"    {param}: {score:.3f}")

    # To load into a Jupyter notebook:
    #   import json
    #   analysis = json.load(open("quickstart_results/analysis.json"))
    #   import pandas as pd
    #   df = pd.DataFrame(analysis["experiments"])

    # Save analysis as JSON for notebooks
    import json
    from pathlib import Path

    analysis_path = Path(config.output_dir) / "analysis.json"
    # Filter non-serializable items
    serializable = {k: v for k, v in analysis.items() if k != "experiments"}
    serializable["experiments"] = analysis.get("experiments", [])
    analysis_path.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\n  Analysis saved to: {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-Research Quickstart",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real", action="store_true",
                        help="Use a real model (requires GPU)")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model name (only with --real)")
    parser.add_argument("--proposer", default="perturbation",
                        choices=["perturbation", "smart", "adaptive",
                                 "random", "grid", "bayesian", "llm"])
    parser.add_argument("--search-space", default="quick",
                        choices=["quick", "auto_research", "multi_algorithm",
                                 "reward", "model"])
    parser.add_argument("--max-experiments", type=int, default=5)
    parser.add_argument("--time-budget", type=int, default=60)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--output-dir", default="./quickstart_results")

    asyncio.run(main(parser.parse_args()))
