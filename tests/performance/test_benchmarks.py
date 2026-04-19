"""
Microbenchmarks for hot paths that run every training step.

Uses pytest-benchmark to record timings. Fail loudly on ``--benchmark-compare``
runs that regress by more than the configured tolerance.

Run with:
    pytest tests/performance/test_benchmarks.py --benchmark-only
    pytest tests/performance/test_benchmarks.py --benchmark-compare --benchmark-compare-fail=mean:10%
"""

from __future__ import annotations

import asyncio

import pytest

from stateset_agents.core.basic_rewards import (
    ConcisenessReward,
    HelpfulnessReward,
    SafetyReward,
)
from stateset_agents.core.reward_base import CompositeReward
from stateset_agents.core.trajectory import ConversationTurn

pytestmark = pytest.mark.benchmark


def _sample_turn() -> ConversationTurn:
    return ConversationTurn(
        role="assistant",
        content=(
            "Absolutely — I'd be glad to help. Let me walk you through the "
            "steps to resolve this, then confirm the details on your account."
        ),
    )


def test_helpfulness_reward_throughput(benchmark):
    reward_fn = HelpfulnessReward()
    turns = [_sample_turn()]

    def run():
        return asyncio.run(reward_fn.compute_reward(turns))

    result = benchmark(run)
    assert result.score >= 0.0


def test_safety_reward_throughput(benchmark):
    reward_fn = SafetyReward()
    turns = [_sample_turn()]

    def run():
        return asyncio.run(reward_fn.compute_reward(turns))

    result = benchmark(run)
    assert result.score >= 0.0


def test_composite_reward_throughput(benchmark):
    reward_fn = CompositeReward(
        reward_functions=[
            HelpfulnessReward(weight=0.5),
            SafetyReward(weight=0.25),
            ConcisenessReward(weight=0.25),
        ],
        combination_method="weighted_sum",
        normalize_weights=True,
    )
    turns = [_sample_turn()]

    def run():
        return asyncio.run(reward_fn.compute_reward(turns))

    result = benchmark(run)
    assert result.components


def test_composite_reward_large_batch(benchmark):
    """Stress test: 32 turns processed sequentially (matches a small group rollout)."""
    reward_fn = CompositeReward(
        reward_functions=[HelpfulnessReward(), SafetyReward()]
    )
    turns = [_sample_turn() for _ in range(32)]

    async def run_all():
        return [await reward_fn.compute_reward([turn]) for turn in turns]

    results = benchmark(lambda: asyncio.run(run_all()))
    assert len(results) == 32


def test_trajectory_turn_construction(benchmark):
    """Building ConversationTurn objects is on every rollout hot path."""

    def run():
        return [
            ConversationTurn(role="assistant", content=f"response {i}")
            for i in range(1_000)
        ]

    turns = benchmark(run)
    assert len(turns) == 1_000


def test_serving_manifest_build_throughput(benchmark):
    """Manifest construction happens per training run; keep it cheap."""
    from stateset_agents.training.serving_artifacts import build_serving_manifest

    def run():
        return build_serving_manifest(
            "./outputs/bench",
            "Qwen/Qwen3.5-27B",
            use_lora=True,
            use_vllm=True,
            merged_model_dir="./outputs/bench/merged",
            recommended={
                "tensor_parallel_size": 8,
                "max_model_len": 262144,
                "reasoning_parser": "qwen3",
            },
        )

    manifest = benchmark(run)
    assert manifest["model"] == "Qwen/Qwen3.5-27B"
