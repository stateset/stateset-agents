"""Splitting or filtering a dataset must not re-normalize rewards in place.

Derived datasets shared the parent's trajectory objects and normalized them
again with their own statistics, mutating the parent and making each subset's
scale depend on its contents."""

from __future__ import annotations

import random

from stateset_agents.data.conversation_dataset import (
    ConversationDataset,
    ConversationTurnData,
    TrajectoryData,
)


def _dataset(n: int = 10) -> ConversationDataset:
    trajectories = []
    for i in range(n):
        reward = float(i)
        trajectories.append(
            TrajectoryData(
                trajectory_id=str(i),
                turns=[
                    ConversationTurnData(role="user", content="q"),
                    ConversationTurnData(role="assistant", content="a"),
                ],
                total_reward=reward,
                turn_rewards=[0.0, reward],
            )
        )
    return ConversationDataset(trajectories)


def _snapshot(ds):
    return [(t.total_reward, list(t.turn_rewards)) for t in ds.trajectories]


def test_split_and_filters_leave_the_parent_untouched():
    ds = _dataset()
    before = _snapshot(ds)
    train, val = ds.split(train_ratio=0.5, seed=0)
    ds.filter_by_length(min_length=1)
    assert _snapshot(ds) == before
    by_id = {t.trajectory_id: t.total_reward for t in ds.trajectories}
    for subset in (train, val):
        for t in subset.trajectories:
            assert t.total_reward == by_id[t.trajectory_id]


def test_filter_by_quality_uses_raw_rewards():
    ds = _dataset()
    kept = ds.filter_by_quality(min_reward=7.0)
    assert sorted(t.trajectory_id for t in kept.trajectories) == ["7", "8", "9"]


def test_split_does_not_reseed_global_random():
    random.seed(123)
    expected = random.random()
    random.seed(123)
    _dataset().split(seed=0)
    assert random.random() == expected
