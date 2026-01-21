"""
Unit tests for continual learning helpers.
"""

from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory, TrajectoryGroup
from stateset_agents.training.continual_learning import (
    ContinualLearningConfig,
    ContinualLearningManager,
    TrajectoryReplayBuffer,
)


def _make_group(scenario_id: str, reward: float) -> TrajectoryGroup:
    turn = ConversationTurn(role="user", content="hello")
    traj = MultiTurnTrajectory(turns=[turn], total_reward=reward)
    return TrajectoryGroup(scenario_id=scenario_id, trajectories=[traj])


def test_replay_buffer_fifo_recent_sampling():
    buffer = TrajectoryReplayBuffer(
        max_size=2,
        storage_strategy="fifo",
        sampling_strategy="recent",
        seed=123,
    )
    buffer.add_groups([_make_group("s1", 1.0)])
    buffer.add_groups([_make_group("s2", 2.0)])
    buffer.add_groups([_make_group("s3", 3.0)])

    assert buffer.size == 2
    sampled = buffer.sample_groups(2)
    scenario_ids = [group.scenario_id for group in sampled]
    assert scenario_ids == ["s3", "s2"]


def test_replay_buffer_reservoir_stats():
    buffer = TrajectoryReplayBuffer(
        max_size=3,
        storage_strategy="reservoir",
        sampling_strategy="uniform",
        seed=7,
    )
    for idx in range(10):
        buffer.add_groups([_make_group(f"s{idx}", float(idx))])

    stats = buffer.stats()
    assert buffer.size == 3
    assert stats["seen"] == 10


def test_continual_config_flags():
    config = ContinualLearningConfig(
        strategy="replay_lwf",
        replay_buffer_size=10,
        continual_kl_beta=0.2,
    )

    assert config.uses_replay() is True
    assert config.uses_lwf() is True
    assert config.uses_ewc() is False


def test_replay_buffer_balanced_sampling():
    buffer = TrajectoryReplayBuffer(
        max_size=10,
        storage_strategy="fifo",
        sampling_strategy="balanced",
        seed=5,
    )
    buffer.add_groups([_make_group("s1", 1.0)], task_id="task_a")
    buffer.add_groups([_make_group("s2", 1.0)], task_id="task_b")
    buffer.add_groups([_make_group("s3", 1.0)], task_id="task_b")

    sampled = buffer.sample_groups(2)
    scenario_ids = {group.scenario_id for group in sampled}
    assert scenario_ids.issubset({"s1", "s2", "s3"})


def test_manager_handles_mock_config():
    from unittest.mock import MagicMock

    manager = ContinualLearningManager.from_training_config(MagicMock())
    assert manager.enabled is False


def test_replay_buffer_state_roundtrip():
    buffer = TrajectoryReplayBuffer(
        max_size=3,
        storage_strategy="fifo",
        sampling_strategy="recent",
        seed=11,
    )
    buffer.add_groups([_make_group("s1", 1.0)], task_id="task_a")
    buffer.add_groups([_make_group("s2", 2.0)], task_id="task_b")

    state = buffer.state_dict()
    restored = TrajectoryReplayBuffer(max_size=1)
    restored.load_state_dict(state)

    assert restored.max_size == 3
    assert restored.size == buffer.size
    sampled = restored.sample_groups(2)
    assert [group.scenario_id for group in sampled] == ["s2", "s1"]


def test_manager_state_roundtrip():
    config = ContinualLearningConfig(strategy="replay", replay_buffer_size=5)
    manager = ContinualLearningManager(config=config, seed=3)
    manager.add_trajectory_groups([_make_group("s1", 1.0)], task_id="task_a")

    state = manager.state_dict()
    restored = ContinualLearningManager(config=config, seed=7)
    restored.load_state_dict(state)

    assert restored.buffer.size == 1
    sampled = restored.buffer.sample_groups(1)
    assert sampled and sampled[0].scenario_id == "s1"
