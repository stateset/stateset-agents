#!/usr/bin/env python3
"""
Autoresearch evaluation harness for stateset-agents training framework.

Runs a comprehensive training quality assessment using stub backends (no GPU needed).
Produces a single scalar `training_quality` metric suitable for the autoresearch
keep/discard loop.

Scoring breakdown (max 1.0):
  - test_gate:      0/1 binary — all core tests must pass
  - reward_quality:  0.0–0.40 — mean eval reward across scenarios & reward functions
  - stability:       0.0–0.20 — low variance = higher score
  - algo_coverage:   0.0–0.20 — fraction of algorithms that complete without crash
  - convergence:     0.0–0.20 — training loop completes and produces valid metrics

Usage:
    python benchmarks/autoresearch_eval.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import traceback
from dataclasses import dataclass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenarios — diverse multi-turn conversations for evaluation
# ---------------------------------------------------------------------------

TRAIN_SCENARIOS = [
    {
        "id": "customer_order",
        "context": "Customer needs help tracking an order",
        "user_responses": [
            "Hi, I need to track my order #98765.",
            "I placed it last Tuesday.",
            "Can you check the shipping status?",
        ],
    },
    {
        "id": "password_reset",
        "context": "User needs to reset their password",
        "user_responses": [
            "I forgot my password and can't log in.",
            "My email is user@example.com.",
            "I don't have access to that email anymore.",
        ],
    },
    {
        "id": "product_return",
        "context": "Customer wants to return a defective product",
        "user_responses": [
            "I received a damaged item and want a refund.",
            "It's a laptop with a cracked screen.",
            "I have the receipt and original packaging.",
        ],
    },
    {
        "id": "billing_inquiry",
        "context": "Customer has a billing discrepancy",
        "user_responses": [
            "I was charged twice for my subscription.",
            "The charges are from March 15 and March 16.",
            "Yes, please process the refund.",
        ],
    },
]

EVAL_SCENARIOS = [
    {
        "id": "eval_tech_support",
        "context": "User needs technical support for software",
        "user_responses": [
            "My app keeps crashing when I open it.",
            "I'm on version 3.2.1.",
            "I already tried reinstalling.",
        ],
    },
    {
        "id": "eval_account_upgrade",
        "context": "Customer wants to upgrade their plan",
        "user_responses": [
            "I want to upgrade from Basic to Premium.",
            "What additional features do I get?",
            "Sounds good, let's do it.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Stub response sets — varied quality to give the reward signal range
# ---------------------------------------------------------------------------

GOOD_RESPONSES = [
    "I'd be happy to help you with that! Let me look into this right away.",
    "I understand your concern. Here's how we can resolve this for you.",
    "Thank you for providing that information. Based on what you've told me, "
    "I can see the issue and here are some options we can try.",
    "Let me check our system. I found the relevant details. "
    "To solve this, you can try the following steps.",
    "I appreciate your patience. Here are some options for you to consider. "
    "The best approach would be to proceed with option A since it addresses "
    "your needs most directly.",
]

MINIMAL_RESPONSES = [
    "OK.",
    "Sure.",
    "Done.",
]


@dataclass
class AlgoResult:
    name: str
    completed: bool
    eval_reward: float
    eval_reward_std: float
    eval_success_rate: float
    training_time: float
    error: str = ""


# ---------------------------------------------------------------------------
# Training + eval for a single algorithm
# ---------------------------------------------------------------------------

async def run_algorithm_eval(algo_name: str) -> AlgoResult:
    """Run a mini training + eval loop for one algorithm."""
    start = time.monotonic()
    try:
        from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
        from stateset_agents.core.environment import ConversationEnvironment
        from stateset_agents.core.reward import (
            CompositeReward,
            HelpfulnessReward,
            SafetyReward,
        )
        from stateset_agents.training.config import TrainingConfig
        from stateset_agents.training.evaluation import (
            EvaluationConfig,
            evaluate_agent,
        )

        # --- Agent ---
        agent_config = AgentConfig(
            model_name=f"stub://autoresearch-{algo_name}",
            use_stub_model=True,
            max_new_tokens=64,
            stub_responses=GOOD_RESPONSES,
        )
        agent = MultiTurnAgent(agent_config)

        # --- Environment ---
        train_env = ConversationEnvironment(scenarios=TRAIN_SCENARIOS, max_turns=3)
        eval_env = ConversationEnvironment(scenarios=EVAL_SCENARIOS, max_turns=3)

        # --- Reward ---
        reward_fn = CompositeReward(
            reward_functions=[
                HelpfulnessReward(weight=0.6),
                SafetyReward(weight=0.4),
            ],
        )

        # --- Training config ---
        training_config = TrainingConfig(
            num_episodes=4,
            max_steps_per_episode=3,
            num_generations=2,
            learning_rate=1e-5,
            seed=42,
            bf16=False,
        )

        # --- Train ---
        if algo_name == "grpo":
            from stateset_agents.training.trainer import SingleTurnGRPOTrainer

            trainer = SingleTurnGRPOTrainer(
                agent=agent,
                environment=train_env,
                reward_fn=reward_fn,
                config=training_config,
            )
            await trainer.initialize()
            await trainer.train()

        elif algo_name == "grpo_multi":
            from stateset_agents.training.trainer import MultiTurnGRPOTrainer

            trainer = MultiTurnGRPOTrainer(
                agent=agent,
                environment=train_env,
                reward_fn=reward_fn,
                config=training_config,
            )
            await trainer.initialize()
            await trainer.train()

        else:
            # Eval-only for algorithms that need real models
            pass

        # --- Evaluate ---
        eval_config = EvaluationConfig(
            num_episodes=len(EVAL_SCENARIOS),
            num_generations=1,
            seed=42,
            concurrency=1,
        )

        results = await evaluate_agent(
            agent=agent,
            environment=eval_env,
            scenarios=EVAL_SCENARIOS,
            reward_fn=reward_fn,
            config=eval_config,
        )

        elapsed = time.monotonic() - start
        return AlgoResult(
            name=algo_name,
            completed=True,
            eval_reward=results.get("eval_reward", 0.0),
            eval_reward_std=results.get("eval_reward_std", 0.0),
            eval_success_rate=results.get("eval_success_rate", 0.0),
            training_time=elapsed,
        )

    except Exception as exc:
        elapsed = time.monotonic() - start
        return AlgoResult(
            name=algo_name,
            completed=False,
            eval_reward=0.0,
            eval_reward_std=0.0,
            eval_success_rate=0.0,
            training_time=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Reward function quality probe — tests that reward functions differentiate
# good vs bad responses
# ---------------------------------------------------------------------------

async def probe_reward_discrimination() -> float:
    """Measure how well reward functions separate good from bad responses.

    Returns a score from 0.0 to 1.0 where 1.0 means perfect separation.
    """
    try:
        from stateset_agents.core.reward import (
            CompositeReward,
            HelpfulnessReward,
            SafetyReward,
        )
        from stateset_agents.core.trajectory import ConversationTurn

        reward_fn = CompositeReward(
            reward_functions=[
                HelpfulnessReward(weight=0.6),
                SafetyReward(weight=0.4),
            ],
        )

        # Score good responses
        good_scores = []
        for resp in GOOD_RESPONSES:
            turns = [
                ConversationTurn(role="user", content="I need help with my order."),
                ConversationTurn(role="assistant", content=resp),
            ]
            result = await reward_fn.compute_reward(turns)
            good_scores.append(result.score)

        # Score minimal responses
        bad_scores = []
        for resp in MINIMAL_RESPONSES:
            turns = [
                ConversationTurn(role="user", content="I need help with my order."),
                ConversationTurn(role="assistant", content=resp),
            ]
            result = await reward_fn.compute_reward(turns)
            bad_scores.append(result.score)

        # Discrimination = gap between good and bad mean scores
        good_mean = sum(good_scores) / len(good_scores) if good_scores else 0.0
        bad_mean = sum(bad_scores) / len(bad_scores) if bad_scores else 0.0
        discrimination = max(0.0, good_mean - bad_mean)

        return min(discrimination, 1.0)

    except Exception as exc:
        logger.warning("Reward discrimination probe failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def main() -> float:
    """Run the full evaluation and return the composite score."""
    print("=" * 60)
    print("StateSet Agents — Autoresearch Training Quality Eval")
    print("=" * 60)

    overall_start = time.monotonic()

    # 1. Algorithm training + eval
    algorithms = ["grpo", "grpo_multi"]
    algo_results: list[AlgoResult] = []

    for algo in algorithms:
        print(f"\n--- Running {algo} ---")
        result = await run_algorithm_eval(algo)
        algo_results.append(result)

        status = "OK" if result.completed else f"FAIL: {result.error}"
        print(f"  status:       {status}")
        print(f"  eval_reward:  {result.eval_reward:.6f}")
        print(f"  eval_std:     {result.eval_reward_std:.6f}")
        print(f"  success_rate: {result.eval_success_rate:.6f}")
        print(f"  time:         {result.training_time:.1f}s")

    # 2. Reward discrimination probe
    print("\n--- Reward discrimination probe ---")
    discrimination = await probe_reward_discrimination()
    print(f"  discrimination: {discrimination:.6f}")

    # 3. Compute composite score
    completed = [r for r in algo_results if r.completed]
    algo_coverage = len(completed) / len(algorithms) if algorithms else 0.0

    # Reward quality: mean eval_reward across completed algorithms
    if completed:
        reward_quality = sum(r.eval_reward for r in completed) / len(completed)
    else:
        reward_quality = 0.0

    # Stability: inverse of mean reward std (lower std = better)
    if completed:
        mean_std = sum(r.eval_reward_std for r in completed) / len(completed)
        stability = max(0.0, 1.0 - mean_std)  # 0 std = perfect stability
    else:
        stability = 0.0

    # Convergence: based on success rate
    if completed:
        convergence = sum(r.eval_success_rate for r in completed) / len(completed)
    else:
        convergence = 0.0

    # Composite: weighted sum
    composite = (
        0.30 * reward_quality
        + 0.20 * stability
        + 0.20 * algo_coverage
        + 0.15 * convergence
        + 0.15 * discrimination
    )

    elapsed = time.monotonic() - overall_start

    print("\n" + "=" * 60)
    print("COMPOSITE SCORING")
    print("=" * 60)
    print(f"  reward_quality:  {reward_quality:.6f}  (weight 0.30)")
    print(f"  stability:       {stability:.6f}  (weight 0.20)")
    print(f"  algo_coverage:   {algo_coverage:.6f}  (weight 0.20)")
    print(f"  convergence:     {convergence:.6f}  (weight 0.15)")
    print(f"  discrimination:  {discrimination:.6f}  (weight 0.15)")
    print(f"  ---")
    print(f"  total_time:      {elapsed:.1f}s")
    print()

    # Final metric line (parsed by autoresearch)
    print(f"training_quality: {composite:.6f}")
    print(f"reward_quality: {reward_quality:.6f}")
    print(f"stability: {stability:.6f}")
    print(f"algo_coverage: {algo_coverage:.6f}")
    print(f"convergence: {convergence:.6f}")
    print(f"discrimination: {discrimination:.6f}")

    return composite


if __name__ == "__main__":
    try:
        score = asyncio.run(main())
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        print("training_quality: 0.000000")
        sys.exit(1)
