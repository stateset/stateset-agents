"""Balanced replay covers every task even when sampling fewer groups than tasks."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from stateset_agents.training.continual_learning import TrajectoryReplayBuffer


def _buffer(seed: int = 0) -> TrajectoryReplayBuffer:
    buffer = TrajectoryReplayBuffer(max_size=100, seed=seed)
    for task in ("A", "B", "C"):
        buffer.add_groups(
            [SimpleNamespace(name=f"{task}{i}", rewards=[1.0]) for i in range(5)],
            task_id=task,
        )
    return buffer


def _tasks(groups):
    return [g.name[0] for g in groups]


def test_single_sample_draws_from_every_task():
    buffer = _buffer()
    counts = Counter(
        _tasks(buffer.sample_groups(1, sampling_strategy="balanced"))[0]
        for _ in range(600)
    )
    assert set(counts) == {"A", "B", "C"}
    assert min(counts.values()) > 120


def test_two_samples_come_from_distinct_tasks_and_cover_all():
    buffer = _buffer()
    seen = Counter()
    for _ in range(300):
        tasks = _tasks(buffer.sample_groups(2, sampling_strategy="balanced"))
        assert len(tasks) == 2 and len(set(tasks)) == 2
        seen.update(tasks)
    assert set(seen) == {"A", "B", "C"}


def test_balanced_sampling_is_deterministic_under_seed():
    def draws(seed):
        buffer = _buffer(seed)
        return [
            _tasks(buffer.sample_groups(1, sampling_strategy="balanced"))[0]
            for _ in range(20)
        ]

    assert draws(7) == draws(7)
