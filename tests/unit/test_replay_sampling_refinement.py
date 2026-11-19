"""Count replay draws against independent combinatorial and rational oracles."""

import math
import random
from collections import Counter
from fractions import Fraction
from functools import partial
from itertools import combinations, product
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from stateset_agents.training.continual_learning import (
    ContinualLearningConfig,
    ContinualLearningManager,
    TrajectoryReplayBuffer,
)
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from tests.unit.test_continual_learning import _make_group


@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize("seen", range(7))
def test_every_native_reservoir_and_subbatch_has_equal_mass(capacity, seen):
    rows = [_make_group(str(i), i) for i in range(seen)]
    size = min(capacity, seen)
    retained = Counter()
    for draws in product(*(range(i + 1) for i in range(capacity, seen))):
        buffer = TrajectoryReplayBuffer(capacity)
        tickets = Mock(side_effect=draws)
        buffer._rng.randint = tickets
        for row in rows:
            buffer.add_groups([row])
        assert tickets.call_args_list == [call(0, i) for i in range(capacity, seen)]
        identities = sorted(int(g.scenario_id) for g in buffer.sample_groups(size))
        retained[tuple(identities)] += 1
    assert retained == {
        subset: math.factorial(seen - size)
        for subset in combinations(range(seen), size)
    }
    for batch in range(size + 1):
        mass = Counter()
        for subset, ways in retained.items():
            for selected in combinations(subset, batch):
                mass[selected] += ways
        assert mass == {
            subset: math.factorial(seen - size) * math.comb(seen - batch, size - batch)
            for subset in combinations(range(seen), batch)
        }


class NeedDraw(Exception):
    """Request one unassigned integer choice while exploring all RNG branches."""


class BranchingRandom(random.Random):
    def __init__(self, tickets):
        super().__init__(0)
        self.tickets = iter(tickets)

    def _randbelow(self, n):
        try:
            return next(self.tickets)
        except StopIteration:
            raise NeedDraw(n) from None


@pytest.mark.parametrize("count", range(1, 7))
@pytest.mark.parametrize("labels", [("a", "b", "b", "c", None), (None, None)])
def test_balanced_inclusion_for_every_rng_branch(labels, count):
    buffer = TrajectoryReplayBuffer(10, sampling_strategy="balanced")
    for i, label in enumerate(labels):
        buffer.add_groups([_make_group(str(i), 0.0)], task_id=label)
    mass = Counter()
    populations = Counter(label for label in labels if label is not None)
    count = min(count, len(labels))

    def visit(tickets, probability):
        buffer._rng = BranchingRandom(tickets)
        try:
            ids = [int(g.scenario_id) for g in buffer.sample_groups(count)]
        except NeedDraw as branch:
            n = branch.args[0]
            for ticket in range(n):
                visit((*tickets, ticket), probability / n)
            return
        assert len(set(ids)) == count
        quotas = Counter(labels[i] for i in ids)
        for task, capacity in populations.items():
            if quotas[task] < capacity:
                assert all(quotas[t] <= quotas[task] + 1 for t in populations)
        for i in ids:
            mass[i] += probability

    visit((), Fraction(1))
    named_count = min(count, sum(populations.values()))
    full_rounds = 0
    while (
        sum(min(size, full_rounds + 1) for size in populations.values()) <= named_count
    ):
        if full_rounds == max(populations.values(), default=0):
            break
        full_rounds += 1
    remainder = named_count - sum(
        min(size, full_rounds) for size in populations.values()
    )
    active = sum(size > full_rounds for size in populations.values())
    for i, task in enumerate(labels):
        if task is None:
            expected = Fraction(count - named_count, labels.count(None))
        else:
            size = populations[task]
            extra = Fraction(remainder, active) if size > full_rounds else 0
            expected = (Fraction(min(size, full_rounds)) + extra) / size
        assert mass[i] == expected


@pytest.mark.parametrize("rewards", [(1.7e308, 1.7e308), (1.7e308, -3.0, 5e-324)])
def test_reward_intervals_keep_every_entry_reachable(rewards):
    buffer = TrajectoryReplayBuffer(10, sampling_strategy="reward")
    for i, reward in enumerate(rewards):
        buffer.add_groups([_make_group(str(i), reward)])
    weights = [max(Fraction(value), 0) + Fraction(1e-3) for value in rewards]
    total = sum(weights)

    def endpoint(bound, *, boundary, last):
        position = bound * boundary / total
        assert position.denominator == 1
        return int(position) - int(last)

    before = Fraction(0)
    for i, weight in enumerate(weights):
        for last in (False, True):
            boundary = before + weight if last else before
            buffer._rng.randrange = partial(endpoint, boundary=boundary, last=last)
            assert [g.scenario_id for g in buffer.sample_groups(2)] == [str(i)] * 2
        before += weight


def test_group_and_buffer_means_are_finite_and_invalid_additions_atomic():
    buffer = TrajectoryReplayBuffer(2)
    group = _make_group("a", 1.7e308)
    group.trajectories.extend(_make_group("b", 1.7e308).trajectories)
    buffer.add_groups([group, group])
    assert buffer.stats()["mean_reward"] == 1.7e308
    before = buffer.state_dict()
    with pytest.raises(ValueError, match="finite"):
        buffer.add_groups([group, _make_group("invalid", float("nan"))])
    assert buffer.state_dict() == before


def test_failed_one_pass_stream_restores_slots_counters_and_rng():
    buffer = TrajectoryReplayBuffer(2, seed=91)
    buffer.add_groups([_make_group("old", 0.0)])
    original = buffer.state_dict()

    def broken_stream():
        for i in range(12):
            yield _make_group(str(i), 1.0)
        raise RuntimeError("cutoff")

    with pytest.raises(RuntimeError, match="cutoff"):
        buffer.add_groups(broken_stream())
    assert buffer.state_dict() == original


@pytest.mark.parametrize("ratio,new", [(2**-54, 1), (0.1, 3), (1.5, 1), (1.7e308, 2)])
def test_replay_ratio_matches_exact_capped_rounding(ratio, new):
    manager = ContinualLearningManager(
        ContinualLearningConfig(
            strategy="replay", replay_ratio=ratio, replay_min_size=1
        )
    )
    manager.add_trajectory_groups([_make_group(str(i), 0.0) for i in range(4)])
    exact = new * Fraction(ratio)
    whole, remainder = divmod(exact.numerator, exact.denominator)
    den = ratio.as_integer_ratio()[1]
    threshold = remainder * (den // exact.denominator)
    for ticket in {0, max(0, threshold - 1), threshold, den - 1}:
        draw = Mock(return_value=ticket)
        manager._rng.randrange = draw
        actual = len(manager.sample_replay_groups(new))
        assert actual == min(4, whole + (ticket < threshold))
        if threshold:
            draw.assert_called_once_with(den)
        else:
            draw.assert_not_called()


def test_staged_and_incremental_reservoirs_match_after_every_batch():
    rows = [_make_group(str(i), i) for i in range(20)]
    incremental = TrajectoryReplayBuffer(3, seed=31)
    batched = TrajectoryReplayBuffer(3, seed=31)
    streamed = TrajectoryReplayBuffer(3, seed=31)
    for start in range(0, len(rows), 5):
        batch = rows[start : start + 5]
        batched.add_groups(batch)
        streamed.add_groups(iter(batch))
        for group in batch:
            incremental.add_groups([group])
        assert batched.state_dict() == streamed.state_dict() == incremental.state_dict()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("ratio", [0.25, 0.5])
def test_stochastic_batch_count_does_not_reuse_entry_draw(legacy, ratio):
    config = ContinualLearningConfig(
        strategy="replay", replay_ratio=ratio, replay_min_size=1
    )
    rows = [_make_group(label, 0.0) for label in ("a", "b")]
    counts = Counter()
    for seed in range(256):
        manager = ContinualLearningManager(config, seed=seed)
        manager.add_trajectory_groups(rows)
        if legacy:
            old = manager.state_dict()
            old["schema_version"] = 1
            old["rng_state"] = random.Random(seed).getstate()
            manager.load_state_dict(old)
        assert manager._rng.getstate() != manager.buffer._rng.getstate()
        replay = manager.sample_replay_groups(1)
        counts[replay[0].scenario_id if replay else "skip"] += 1
    assert set(counts) == {"skip", "a", "b"}


def test_trainer_mixes_only_prior_groups_and_balances_single_group_replay():
    manager = ContinualLearningManager(
        ContinualLearningConfig(
            strategy="replay",
            replay_ratio=1,
            replay_min_size=1,
            replay_sampling="balanced",
        ),
        seed=41,
    )
    trainer = SimpleNamespace(continual_manager=manager)
    for label in "abc":
        manager.add_trajectory_groups([_make_group(label, 0.0)], task_id=label)
    original = manager.state_dict()
    tickets = set()
    for seed in range(32):
        manager.load_state_dict(original)
        manager.buffer._rng.seed(seed)
        fresh = [_make_group("new", 1.0)]
        mixed, replay = MultiTurnGRPOTrainer._maybe_mix_replay(trainer, fresh, "new")
        assert mixed == fresh + replay
        assert len(replay) == 1
        tickets.add(replay[0].scenario_id)
    assert tickets == set("abc")
