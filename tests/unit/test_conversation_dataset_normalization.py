"""Splitting or filtering a dataset must not re-normalize rewards in place.

Derived datasets shared the parent's trajectory objects and normalized them
again with their own statistics, mutating the parent and making each subset's
scale depend on its contents."""

from __future__ import annotations

import itertools
import random

import pytest

from stateset_agents.data.conversation_dataset import (
    ConversationDataset,
    ConversationDatasetConfig,
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


def _total_only(trajectory_id: str, reward: float, user: str = "hi") -> dict:
    """The documented JSONL format: per-turn rewards absent or zero."""
    return {
        "trajectory_id": trajectory_id,
        "turns": [
            {"role": "user", "content": user, "reward": 0.0},
            {"role": "assistant", "content": f"answer {trajectory_id}"},
        ],
        "total_reward": reward,
    }


class TestTotalRewardOnlyTrajectories:
    """With only ``total_reward`` every per-turn reward was 0, so batches,
    returns-to-go and policy transitions carried no signal, and normalization
    then overwrote ``total_reward`` with 0."""

    def test_total_reward_is_placed_on_the_final_turn(self):
        config = ConversationDatasetConfig(normalize_rewards=False)
        ds = ConversationDataset.from_dict(
            {"trajectories": [_total_only("a", 1.0), _total_only("b", 0.25)]}, config
        )
        a, b = ds.trajectories
        assert a.turn_rewards == [0.0, 1.0] and a.total_reward == 1.0
        assert a.turns[-1].reward == 1.0
        assert a.returns_to_go == [0.99, 1.0]
        assert [r["reward"] for r in ds.policy_transitions()] == [1.0, 0.25]

    def test_explicit_per_turn_rewards_are_kept(self):
        data = _total_only("a", 5.0)
        data["turns"][1]["reward"] = 0.5
        ds = ConversationDataset.from_dict(
            {"trajectories": [data]}, ConversationDatasetConfig(normalize_rewards=False)
        )
        (t,) = ds.trajectories
        assert t.turn_rewards == [0.0, 0.5] and t.total_reward == 5.0

    def test_normalization_keeps_the_terminal_signal_and_its_order(self):
        ds = ConversationDataset.from_dict(
            {
                "trajectories": [
                    _total_only("good", 1.0),
                    _total_only("bad", 0.0),
                    _total_only("mid", 0.5),
                ]
            }
        )
        by_id = {t.trajectory_id: t for t in ds.trajectories}
        totals = {k: t.total_reward for k, t in by_id.items()}
        assert totals["good"] > totals["mid"] > totals["bad"]
        assert totals["good"] == pytest.approx(-totals["bad"], rel=1e-6)
        # Padding turns stay zero: length is not penalized.
        assert all(t.turn_rewards[0] == 0.0 for t in ds.trajectories)
        assert by_id["good"].metadata["raw_total_reward"] == 1.0
        batch_rewards = {
            r["transition_id"]: r["reward"] for r in ds.policy_transitions()
        }
        assert batch_rewards["good:1"] == pytest.approx(totals["good"])

    def test_get_batch_rewards_carry_the_terminal_signal(self):
        pytest.importorskip("torch")
        ds = ConversationDataset.from_dict(
            {"trajectories": [_total_only("a", 1.0), _total_only("b", 0.0)]}
        )
        batch = ds.get_batch(batch_size=2)
        assert batch["rewards"].abs().sum().item() > 0


class TestSplit:
    def _ds(self, rows):
        return ConversationDataset.from_dict({"trajectories": rows})

    def test_default_split_is_reproducible(self):
        rows = [_total_only(str(i), float(i), user=f"question {i}") for i in range(20)]
        first = [t.trajectory_id for t in self._ds(rows).split()[1].trajectories]
        again = [t.trajectory_id for t in self._ds(rows).split()[1].trajectories]
        reordered = [
            t.trajectory_id for t in self._ds(rows[::-1]).split()[1].trajectories
        ]
        assert first == again and sorted(first) == sorted(reordered)
        assert len(first) == 4

    def test_duplicates_and_shared_sources_never_straddle_splits(self):
        rows = [_total_only(str(i), 1.0, user=f"question {i}") for i in range(12)]
        # A near-duplicate (case/whitespace/NFKC) of row 0 ...
        dup = _total_only("0-dup", 0.0, user="QUESTION  0")
        dup["turns"][1]["content"] = "Answer  0"
        # ... and two rows linked only by a source id.
        rows[3]["metadata"] = {"source_id": "ticket-9"}
        rows[7]["metadata"] = {"source_id": "ticket-9"}
        rows.append(dup)
        for seed in range(10):
            train, val = self._ds(rows).split(train_ratio=0.5, seed=seed)
            train_ids = {t.trajectory_id for t in train.trajectories}
            val_ids = {t.trajectory_id for t in val.trajectories}
            assert train_ids and val_ids and not train_ids & val_ids
            for pair in ({"0", "0-dup"}, {"3", "7"}):
                assert pair <= train_ids or pair <= val_ids


class TestLengthLimits:
    def test_trajectory_length_limits_are_enforced(self):
        rows = [_total_only("two", 1.0)]
        rows.append(
            {
                "trajectory_id": "four",
                "turns": [{"role": "user", "content": "q"}] * 2
                + [{"role": "assistant", "content": "a"}] * 2,
                "total_reward": 1.0,
            }
        )
        rows.append(
            {"trajectory_id": "one", "turns": [{"role": "user", "content": "q"}]}
        )
        config = ConversationDatasetConfig(
            normalize_rewards=False, min_trajectory_length=2, max_trajectory_length=3
        )
        ds = ConversationDataset.from_dict({"trajectories": rows}, config)
        assert [t.trajectory_id for t in ds.trajectories] == ["two"]

    def test_max_turns_is_deprecated(self):
        with pytest.warns(DeprecationWarning, match="max_trajectory_length"):
            ConversationDatasetConfig(max_turns=5)


class TestNormalizationMetrics:
    """Aggregate semantics of reward normalization, so they cannot regress."""

    def test_empty_dataset_normalizes_and_reports_zero_statistics(self):
        ds = ConversationDataset([])
        stats = ds.get_statistics()
        assert stats["num_trajectories"] == 0 and stats["mean_reward"] == 0.0

    def test_terminal_rewards_standardize_across_trajectories(self):
        from hypothesis import given, settings
        from hypothesis import strategies as st

        @settings(max_examples=80, deadline=None)
        @given(
            st.lists(
                st.tuples(
                    st.floats(-5, 5, allow_nan=False),
                    st.integers(min_value=2, max_value=6),
                ),
                min_size=2,
                max_size=12,
            )
        )
        def check(specs):
            rows = []
            for i, (reward, turns) in enumerate(specs):
                rows.append(
                    {
                        "trajectory_id": str(i),
                        "turns": [
                            {
                                "role": "user" if t % 2 == 0 else "assistant",
                                "content": "x",
                            }
                            for t in range(turns)
                        ],
                        "total_reward": reward,
                    }
                )
            ds = ConversationDataset.from_dict({"trajectories": rows})
            raw = [r for r, _ in specs]
            totals = [t.total_reward for t in ds.trajectories]
            # Length never moves a total: padding turns stay exactly zero.
            for t in ds.trajectories:
                assert t.turn_rewards[:-1] == [0.0] * (len(t.turns) - 1)
                assert t.total_reward == t.turn_rewards[-1]
            assert [t.metadata["raw_total_reward"] for t in ds.trajectories] == raw
            assert sum(totals) / len(totals) == pytest.approx(0.0, abs=1e-6)
            spread = max(raw) - min(raw)
            if spread > 1e-3:
                mean = sum(totals) / len(totals)
                std = (sum((x - mean) ** 2 for x in totals) / len(totals)) ** 0.5
                assert std == pytest.approx(1.0, rel=1e-3)
                for i, j in itertools.combinations(range(len(raw)), 2):
                    if abs(raw[i] - raw[j]) > 1e-6:  # order is preserved
                        assert (raw[i] < raw[j]) == (totals[i] < totals[j])
            assert ds.get_statistics()["mean_reward"] == pytest.approx(
                sum(totals) / len(totals)
            )

        check()

    def test_minmax_maps_terminal_rewards_into_the_unit_interval(self):
        rows = [
            {
                "trajectory_id": str(i),
                "turns": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                "total_reward": reward,
            }
            for i, reward in enumerate([2.0, -1.0, 0.5])
        ]
        config = ConversationDatasetConfig(reward_normalization_method="minmax")
        ds = ConversationDataset.from_dict({"trajectories": rows}, config)
        totals = [t.total_reward for t in ds.trajectories]
        assert totals == pytest.approx([1.0, 0.0, 0.5], abs=1e-6)

    def test_mixed_datasets_share_one_scale(self):
        per_turn = TrajectoryData(
            "p",
            [
                ConversationTurnData(role="user", content="q"),
                ConversationTurnData(role="assistant", content="a"),
            ],
            total_reward=1.0,
            turn_rewards=[0.0, 1.0],
        )
        terminal = {
            "trajectory_id": "t",
            "turns": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "b"},
            ],
            "total_reward": 1.0,
        }
        parsed = ConversationDataset.from_dict(
            {"trajectories": [terminal]},
            ConversationDatasetConfig(normalize_rewards=False),
        ).trajectories
        ds = ConversationDataset([per_turn, *parsed])
        # Stats come from [0.0, 1.0] (per-turn) plus [1.0] (terminal only).
        mean, std = 2 / 3, (2 / 9) ** 0.5
        assert ds.trajectories[1].turn_rewards == pytest.approx(
            [0.0, (1.0 - mean) / std], rel=1e-6
        )
        assert ds.trajectories[0].turn_rewards == pytest.approx(
            [(0.0 - mean) / std, (1.0 - mean) / std], rel=1e-6
        )
