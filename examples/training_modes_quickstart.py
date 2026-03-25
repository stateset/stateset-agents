#!/usr/bin/env python3
"""
Training Modes Quickstart — demonstrates all 5 training modes.

This example runs entirely with stub backends (no GPU, no API keys needed).
It shows the unified train() interface for every supported training mode.

Usage:
    python examples/training_modes_quickstart.py
    python examples/training_modes_quickstart.py --mode offline
    python examples/training_modes_quickstart.py --mode rlaif
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Shared setup
SCENARIOS = [
    {
        "id": "order_help",
        "context": "Customer needs help tracking an order",
        "user_responses": [
            "Hi, I need to track my order #98765.",
            "I placed it last Tuesday.",
            "Thanks!",
        ],
    },
    {
        "id": "password_reset",
        "context": "User needs to reset their password",
        "user_responses": [
            "I forgot my password.",
            "My email is user@example.com.",
            "OK, I'll try that.",
        ],
    },
]

STUB_RESPONSES = [
    "I'd be happy to help you with that! Let me look into this right away.",
    "I understand your concern. Here's how we can resolve this for you.",
    "Thank you for providing that information. Based on what you've told me, "
    "I can see the issue and here are some options we can try.",
]


def create_agent():
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    config = AgentConfig(
        model_name="stub://quickstart",
        use_stub_model=True,
        max_new_tokens=64,
        stub_responses=STUB_RESPONSES,
    )
    return MultiTurnAgent(config)


def create_environment():
    from stateset_agents.core.environment import ConversationEnvironment

    return ConversationEnvironment(scenarios=SCENARIOS, max_turns=3)


# ─── Mode 1: Online GRPO ────────────────────────────────────────────

async def demo_online_grpo():
    """Standard online RL training with GRPO."""
    print("\n" + "=" * 60)
    print("MODE 1: Online GRPO (single-turn)")
    print("=" * 60)

    from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward
    from stateset_agents.training.train import train

    agent = create_agent()
    env = create_environment()
    reward = CompositeReward(
        reward_functions=[HelpfulnessReward(weight=0.6), SafetyReward(weight=0.4)]
    )

    trained = await train(
        agent=agent,
        environment=env,
        reward_fn=reward,
        num_episodes=4,
        profile="balanced",
        training_mode="single_turn",
        config_overrides={"num_generations": 2, "bf16": False, "seed": 42},
    )

    print(f"  Trained agent type: {type(trained).__name__}")
    print("  Online GRPO training complete!")


# ─── Mode 2: RLAIF with LLM Judge ──────────────────────────────────

async def demo_rlaif():
    """RLAIF training with LLM-as-Judge reward (falls back to heuristic)."""
    print("\n" + "=" * 60)
    print("MODE 2: RLAIF (LLM Judge + heuristic fallback)")
    print("=" * 60)

    from stateset_agents.rewards.llm_judge_adapter import create_rlaif_reward
    from stateset_agents.training.train import train

    agent = create_agent()
    env = create_environment()

    # create_rlaif_reward auto-detects API keys from env.
    # Without keys, it gracefully falls back to heuristic rewards.
    reward = create_rlaif_reward()
    print(f"  Judge available: {reward._judge_available}")
    print(f"  Mode: {'LLM Judge + heuristic' if reward._judge_available else 'heuristic fallback'}")

    trained = await train(
        agent=agent,
        environment=env,
        reward_fn=reward,
        num_episodes=4,
        training_mode="single_turn",
        config_overrides={"num_generations": 2, "bf16": False, "seed": 42},
    )

    print("  RLAIF training complete!")


# ─── Mode 3: Uncertainty-Weighted ───────────────────────────────────

async def demo_uncertainty():
    """Training with Bayesian uncertainty-weighted rewards."""
    print("\n" + "=" * 60)
    print("MODE 3: Uncertainty-Weighted Training")
    print("=" * 60)

    from stateset_agents.core.reward import HelpfulnessReward
    from stateset_agents.training.train import train

    agent = create_agent()
    env = create_environment()
    reward = HelpfulnessReward()

    trained = await train(
        agent=agent,
        environment=env,
        reward_fn=reward,
        num_episodes=4,
        training_mode="single_turn",
        uncertainty_weighted=True,  # Wraps reward with Bayesian uncertainty
        config_overrides={"num_generations": 2, "bf16": False, "seed": 42},
    )

    print("  Uncertainty-weighted training complete!")


# ─── Mode 4: Offline RL from Dataset ───────────────────────────────

async def demo_offline():
    """Offline RL training from logged conversations."""
    print("\n" + "=" * 60)
    print("MODE 4: Offline RL (from dataset)")
    print("=" * 60)

    import json

    from stateset_agents.core.reward import HelpfulnessReward
    from stateset_agents.training.train import train

    agent = create_agent()
    env = create_environment()

    # Create a sample JSONL dataset
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i, scenario in enumerate(SCENARIOS):
            trajectory = {
                "trajectory_id": f"traj_{i}",
                "turns": [
                    {"role": "user", "content": scenario["user_responses"][0]},
                    {"role": "assistant", "content": STUB_RESPONSES[0]},
                    {"role": "user", "content": scenario["user_responses"][1]},
                    {"role": "assistant", "content": STUB_RESPONSES[1]},
                ],
                "total_reward": 0.85,
            }
            f.write(json.dumps(trajectory) + "\n")
        dataset_path = f.name

    print(f"  Dataset: {dataset_path}")

    try:
        trained = await train(
            agent=agent,
            environment=env,
            reward_fn=HelpfulnessReward(),
            num_episodes=2,
            training_mode="offline",
            dataset_path=dataset_path,
            config_overrides={"bf16": False},
        )
        print("  Offline training complete!")
    except Exception as exc:
        print(f"  Offline training skipped: {exc}")
        print("  (This is expected — offline RL needs torch tensors from real embeddings)")


# ─── Mode 5: Hybrid (Offline + Online) ─────────────────────────────

async def demo_hybrid():
    """Hybrid training: offline pre-training then online fine-tuning."""
    print("\n" + "=" * 60)
    print("MODE 5: Hybrid (offline pre-train + online fine-tune)")
    print("=" * 60)

    from stateset_agents.core.reward import HelpfulnessReward
    from stateset_agents.training.train import train

    agent = create_agent()
    env = create_environment()

    try:
        trained = await train(
            agent=agent,
            environment=env,
            reward_fn=HelpfulnessReward(),
            num_episodes=4,
            training_mode="hybrid",
            # dataset=None means skip offline phase, go straight to online
            config_overrides={"num_generations": 2, "bf16": False, "seed": 42},
        )
        print("  Hybrid training complete!")
    except Exception as exc:
        print(f"  Hybrid training note: {exc}")


# ─── Main ───────────────────────────────────────────────────────────

MODES = {
    "online": demo_online_grpo,
    "rlaif": demo_rlaif,
    "uncertainty": demo_uncertainty,
    "offline": demo_offline,
    "hybrid": demo_hybrid,
}


async def main(mode: str | None = None):
    print("StateSet Agents — Training Modes Quickstart")
    print("All modes use stub backends (no GPU, no API keys needed)")

    if mode and mode in MODES:
        await MODES[mode]()
    else:
        for name, demo in MODES.items():
            try:
                await demo()
            except Exception as exc:
                print(f"  {name} failed: {exc}")

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES.keys()), default=None)
    args = parser.parse_args()
    asyncio.run(main(args.mode))
