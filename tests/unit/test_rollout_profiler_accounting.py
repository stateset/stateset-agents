"""Real sequential/batched environment runs retain discarded collection work."""

import asyncio
from types import SimpleNamespace

import pytest

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.profiling import TrainingProfiler


class Sampler:
    async def generate_turn(self, history, context):
        return ConversationTurn(
            role="assistant", content="done", metadata={"token_ids": [1, 2, 3]}
        )

    async def generate_turns_batch(self, histories, contexts):
        return [
            await self.generate_turn(h, c)
            for h, c in zip(histories, contexts, strict=True)
        ]


class Sandbox(Environment):
    supports_concurrent_episodes = True

    def __init__(self):
        super().__init__(max_turns=1)

    async def reset(self, scenario=None):
        return EnvironmentState(
            episode_id=scenario["id"],
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={"task_success": scenario.get("success")},
        )

    async def get_initial_prompt(self, scenario=None):
        return "Complete the fictional task."

    async def step(self, state, action):
        state.turn_count += 1
        state.status = EpisodeStatus.COMPLETED
        return state, 1.0, True, {}


def learner(*, batched=True, environment=None, reward=None):
    trainer = object.__new__(MultiTurnGRPOTrainer)
    trainer.agent = Sampler()
    trainer.environment = environment or Sandbox()
    trainer.reward_fn = reward
    trainer.config = SimpleNamespace(batched_rollouts=batched, num_generations=2)
    trainer.profiler = TrainingProfiler()
    return trainer


SCENARIOS = [
    {"id": "yes", "success": True},
    {"id": "no", "success": False},
    {"id": "unknown"},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_real_environment_collection_counts_match_for_both_paths(batched):
    trainer = learner(batched=batched)
    groups = await trainer.generate_trajectories(SCENARIOS)
    assert [len(g.trajectories) for g in groups] == [2, 2, 2]
    report = trainer.profiler.snapshot(cost_usd=1.2)
    assert report["attempted_rollouts"] == report["accepted_rollouts"] == 6
    assert report["generated_tokens"] == 18
    assert report["verified_successes"] == report["unknown_successes"] == 2
    assert report["cost_per_accepted_rollout"] == pytest.approx(0.2)
    assert report["cost_per_verified_success"] == pytest.approx(0.6)
    assert report["stages"]["environment_rollout"]["recent_samples"] == (
        1 if batched else 6
    )


class FailingReward:
    fail_on_error = False

    def __init__(self):
        self.calls = 0

    async def compute_reward(self, turns, context=None):
        self.calls += 1
        if self.calls == 2:
            raise ValueError("score unavailable")
        return 100.0  # Positive scores must not turn explicit failure into success.


@pytest.mark.asyncio
async def test_scoring_failure_keeps_discarded_batch_and_fallback_separate():
    trainer = learner(reward=FailingReward())
    groups = await trainer.generate_trajectories(SCENARIOS)
    assert sum(len(g.trajectories) for g in groups) == 6
    report = trainer.profiler.snapshot()
    assert report["attempted_rollouts"] == 12 and report["accepted_rollouts"] == 6
    assert report["generated_tokens"] == 36  # Discarded six plus accepted six.
    assert report["verified_successes"] == report["unknown_successes"] == 2
    assert report["stages"]["reward_evaluation"]["failures"] == 1


@pytest.mark.asyncio
async def test_serial_scoring_failure_retains_observed_tokens_without_acceptance():
    trainer = learner(batched=False, reward=FailingReward())
    groups = await trainer.generate_trajectories(SCENARIOS)
    report = trainer.profiler.snapshot()
    assert sum(len(g.trajectories) for g in groups) == 5
    assert report["attempted_rollouts"] == 6 and report["accepted_rollouts"] == 5
    assert report["generated_tokens"] == 18
    assert report["verified_successes"] == 1 and report["unknown_successes"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", ["reward", "native"])
async def test_strict_batch_failure_records_no_accepted_partial_batch(strict):
    reward = FailingReward()
    trainer = learner(reward=reward)
    if strict == "reward":
        reward.fail_on_error = True
    else:
        trainer._stateset_native_rollout = True
    with pytest.raises(ValueError, match="score unavailable"):
        await trainer.generate_trajectories(SCENARIOS)
    report = trainer.profiler.snapshot()
    assert report["attempted_rollouts"] == 6 and report["accepted_rollouts"] == 0
    assert report["verified_successes"] == report["unknown_successes"] == 0
    assert report["generated_tokens"] == 18


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_cancellation_is_counted_and_propagates_without_fallback(batched):
    class CancelledSandbox(Sandbox):
        async def run_episode(self, *args):
            raise asyncio.CancelledError

        async def run_episodes_batched(self, *args):
            raise asyncio.CancelledError

    trainer = learner(batched=batched, environment=CancelledSandbox())
    with pytest.raises(asyncio.CancelledError):
        await trainer.generate_trajectories(SCENARIOS)
    report = trainer.profiler.snapshot()
    assert report["attempted_rollouts"] == (6 if batched else 1)
    assert report["accepted_rollouts"] == report["generated_tokens"] == 0
    assert report["stages"]["environment_rollout"]["failures"] == 1


@pytest.mark.asyncio
async def test_failed_batch_sampler_fallback_does_not_invent_missing_token_counts():
    class BrokenBatch(Sandbox):
        async def run_episodes_batched(self, *args):
            raise ValueError("sampling failed before any trace was returned")

    trainer = learner(environment=BrokenBatch())
    await trainer.generate_trajectories(SCENARIOS)
    report = trainer.profiler.snapshot()
    assert report["attempted_rollouts"] == 12 and report["accepted_rollouts"] == 6
    assert report["generated_tokens"] == 18
    assert report["stages"]["environment_rollout"]["failures"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_unknown_token_metadata_and_truthy_success_are_not_estimated(batched):
    class TextSampler(Sampler):
        async def generate_turn(self, history, context):
            return ConversationTurn(
                role="assistant", content="plain text", metadata={"token_ids": None}
            )

    trainer = learner(batched=batched)
    trainer.agent = TextSampler()
    groups = await trainer.generate_trajectories([{"id": "not-verified", "success": 1}])
    report = trainer.profiler.snapshot()
    assert len(groups[0].trajectories) == 2
    assert report["attempted_rollouts"] == report["accepted_rollouts"] == 2
    assert report["generated_tokens"] == report["verified_successes"] == 0
    assert report["unknown_successes"] == 2
    assert trainer._runtime_rollout_metrics["generated_tokens"] == 0


@pytest.mark.parametrize(
    "ids,expected",
    [(None, 0), (12, 0), ("123", 0), ({"id": 1}, 0), ((1, 2), 2), ([1, 2, 3], 3)],
)
def test_token_counters_share_observed_sequence_rules(ids, expected):
    from stateset_agents.core.trajectory import MultiTurnTrajectory
    from stateset_agents.training.profiling import _observed_assistant_tokens

    trajectory = MultiTurnTrajectory(
        turns=[
            ConversationTurn(
                role="tool", content="facts", metadata={"token_ids": [9] * 20}
            ),
            ConversationTurn(
                role="assistant", content="reply", metadata={"token_ids": ids}
            ),
        ]
    )
    assert _observed_assistant_tokens(trajectory) == expected


def test_tensor_token_ids_keep_their_observed_count():
    import torch

    from stateset_agents.core.trajectory import MultiTurnTrajectory
    from stateset_agents.training.profiling import _observed_assistant_tokens

    trajectory = MultiTurnTrajectory(
        turns=[
            ConversationTurn(
                role="assistant",
                content="reply",
                metadata={"token_ids": torch.tensor([1, 2, 3])},
            )
        ]
    )
    assert _observed_assistant_tokens(trajectory) == 3
