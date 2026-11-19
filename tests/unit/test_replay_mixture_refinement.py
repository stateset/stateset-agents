"""Count the estimator from real reservoir, replay and GRPO calls on finite pools."""

import math
from collections import Counter
from fractions import Fraction
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import loss_computation as lc  # noqa: E402
from stateset_agents.training.continual_learning import (  # noqa: E402
    ContinualLearningConfig,
    ContinualLearningManager,
    TrajectoryReplayBuffer,
)
from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from tests.unit.test_online_accumulation_weighting import _trainer  # noqa: E402
from tests.unit.test_replay_gradient_refinement import (  # noqa: E402
    CURRENT,
    ThreeActionPolicy,
    group,
)
from tests.unit.test_replay_sampling_refinement import (  # noqa: E402
    BranchingRandom,
    NeedDraw,
)
from tests.unit.test_token_group_aggregation import _config  # noqa: E402


def stored_group(name, length=1):
    result = group((0, 1), CURRENT)
    result.scenario_id = name
    floor = 1e-3
    values = {
        "low1": (-floor, floor),
        "low2": (-floor, floor),
        "heavy": (8 * floor, 0.0),
        "fresh": (0.0, 0.0),
    }
    for trajectory, reward in zip(result.trajectories, values[name], strict=True):
        trajectory.total_reward = reward
        metadata = trajectory.turns[0].metadata
        metadata["token_ids"] *= length
        metadata["sampler_log_probs"] *= length
    return result


def middle_ascent(groups, model, config):
    result = lc.compute_grpo_loss(
        groups, config, SimpleNamespace(model=model), 0.0, 0, None
    )
    (gradient,) = torch.autograd.grad(result["total_loss"], model.logits)
    return -gradient[1].item()


@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize("fresh_count", [1, 2])
@pytest.mark.parametrize("aggregate", ["seq_mean", "token_mean"])
@pytest.mark.parametrize("heavy_length", [1, 2])
def test_all_reservoir_priority_rounding_and_gradient_branches(
    capacity, fresh_count, aggregate, heavy_length
):
    config = ContinualLearningConfig(
        strategy="replay",
        replay_sampling="reward",
        replay_min_size=1,
        replay_buffer_size=capacity,
        replay_ratio=0.5 / fresh_count,
    )
    objective = _config(
        old_logprobs_source="sampler",
        objective_overrides={
            "aggregate": aggregate,
            "advantage_eps": 0.0,
        },
    )
    model = ThreeActionPolicy()
    history = [
        stored_group("low1"),
        stored_group("low2"),
        stored_group("heavy", heavy_length),
    ]
    fresh = [stored_group("fresh") for _ in range(fresh_count)]
    histories = list(product(*(range(i + 1) for i in range(capacity, len(history)))))
    replay_probability = heavy_probability = Fraction(0)
    mixed_mean = corrected = 0.0
    for tickets in histories:
        manager = ContinualLearningManager(config, seed=9)
        manager.buffer._rng.randint = Mock(side_effect=tickets)
        for row in history:
            manager.add_trajectory_groups([row])
        checkpoint = manager.state_dict()
        names = [
            g.scenario_id
            for g in manager.buffer.sample_groups(capacity, sampling_strategy="recent")
        ]
        pool_ticks = capacity + 4 * int("heavy" in names)
        for add, ticket in [(False, 0)] + [(True, t) for t in range(pool_ticks)]:
            manager.load_state_dict(checkpoint)
            manager._rng.randrange = Mock(
                return_value=0 if add else 2 * fresh_count - 1
            )

            # The production CDF uses a common integer grid for binary64
            # rewards. Here its basic priority unit is the represented 1e-3.
            def draw(bound, t=ticket, total=pool_ticks):
                unit, remainder = divmod(bound, total)
                assert unit > 0 and remainder == 0
                return t * unit

            manager.buffer._rng.randrange = draw
            # Insertions after the sampled batch are outside this draw.
            manager.buffer._rng.randint = lambda lo, hi: hi
            mixed, replay = MultiTurnGRPOTrainer._maybe_mix_replay(
                SimpleNamespace(continual_manager=manager), fresh, "current"
            )
            mass = Fraction(1, 2 * len(histories) * (pool_ticks if add else 1))
            assert len(replay) == int(add)
            mixed_mean += float(mass) * middle_ascent(mixed, model, objective)
            if add:
                replay_probability += mass
                heavy = replay[0].scenario_id == "heavy"
                heavy_probability += mass * int(heavy)
                weight = 5 if heavy else 1
                # Independent reference for a uniform historical gradient.
                corrected += float(
                    2 * mass * pool_ticks / (capacity * weight)
                ) * middle_ascent(replay, model, objective)
    heavy_given_draw = Fraction(5 * capacity, 3 * (capacity + 4))
    assert replay_probability == Fraction(1, 2)
    assert heavy_probability == replay_probability * heavy_given_draw
    d = heavy_length if aggregate == "token_mean" else 1
    expected = (
        (1 - heavy_given_draw) / (fresh_count + 1)
        - heavy_given_draw * d / (fresh_count + d)
    ) / 4
    assert mixed_mean == pytest.approx(float(expected), abs=1e-7)
    assert corrected == pytest.approx(1 / 6, abs=1e-7)
    if (capacity, fresh_count, d) == (2, 1, 1):
        assert expected == Fraction(-1, 72)


@pytest.mark.parametrize(
    "new,ratio,capacity",
    [(1, 0.5, 3), (2, 0.75, 4), (3, 1.25, 4), (3, 1.25, 2), (2, 1.0, 4)],
)
def test_count_rounding_precedes_the_native_batch_normalization(new, ratio, capacity):
    manager = ContinualLearningManager(
        ContinualLearningConfig(
            strategy="replay",
            replay_buffer_size=capacity,
            replay_ratio=ratio,
            replay_min_size=1,
        )
    )
    manager.add_trajectory_groups([stored_group("low1") for _ in range(capacity)])
    fresh = [stored_group("fresh") for _ in range(new)]
    model, cfg = ThreeActionPolicy(), _config(
        objective_overrides={"advantage_eps": 0.0}
    )
    den = ratio.as_integer_ratio()[1]
    actual = 0.0
    for ticket in range(den):
        manager._rng.randrange = Mock(return_value=ticket)
        replay = manager.sample_replay_groups(new)
        actual += middle_ascent(fresh + replay, model, cfg) / den
    wanted = min(Fraction(new) * Fraction(ratio), capacity)
    whole = math.floor(wanted)
    t = wanted - whole
    expected_share = (1 - t) * Fraction(whole, new + whole) + t * Fraction(
        whole + 1, new + whole + 1
    )
    assert actual == pytest.approx(float(expected_share) / 2, abs=1e-7)
    gap = wanted / (new + wanted) - expected_share
    assert gap == new * t * (1 - t) / (
        (new + whole) * (new + whole + 1) * (new + wanted)
    )
    assert (gap > 0) == (t > 0)


@pytest.mark.parametrize("strategy", ["uniform", "balanced", "recent"])
@pytest.mark.parametrize("count", [1, 2, 4, 6])
def test_task_and_entry_measures_through_the_native_reducer(strategy, count):
    buffer = TrajectoryReplayBuffer(
        6, storage_strategy="fifo", sampling_strategy=strategy
    )
    for index, task in enumerate(("A", "B", "C", "C", "C", "C")):
        row = stored_group("low1" if task == "C" else "heavy")
        row.scenario_id = str(index)
        buffer.add_groups([row], task_id=task)
    model = ThreeActionPolicy()
    cfg = _config(objective_overrides={"advantage_eps": 0.0})
    gradients = {}

    def visit(tickets):
        buffer._rng = BranchingRandom(tickets)
        try:
            chosen = buffer.sample_groups(count)
        except NeedDraw as branch:
            n = branch.args[0]
            return sum(visit((*tickets, t)) for t in range(n)) / n
        members = frozenset(row.scenario_id for row in chosen)
        assert len(members) == count
        if members not in gradients:
            gradients[members] = middle_ascent(chosen, model, cfg)
        return gradients[members]

    measured = visit(())
    if strategy == "balanced":
        expected = Fraction(-1, 6) if count <= 3 else Fraction(count - 4, 2 * count)
    elif strategy == "recent":
        expected = Fraction(2 * min(4, count) - count, 2 * count)
    else:
        expected = Fraction(1, 6)
    assert measured == pytest.approx(float(expected), abs=1e-7)


@pytest.mark.parametrize("count", range(9))
def test_native_stratified_marginals_and_second_round_match_tlc(count):
    labels = (1, 2, 2, 3, 3, 3, None, None)
    buffer = TrajectoryReplayBuffer(8, sampling_strategy="balanced")
    for i, task in enumerate(labels):
        row = stored_group("low1")
        row.scenario_id = str(i)
        buffer.add_groups([row], task_id=str(task) if task is not None else None)
    included, joint = Counter(), Counter()

    def visit(tickets, mass):
        buffer._rng = BranchingRandom(tickets)
        try:
            indices = [int(row.scenario_id) for row in buffer.sample_groups(count)]
        except NeedDraw as branch:
            n = branch.args[0]
            for t in range(n):
                visit((*tickets, t), mass / n)
            return
        assert len(indices) == len(set(indices)) == count
        for i in indices:
            included[i] += mass
        if count >= 4:
            joint[labels[indices[0]], labels[indices[3]]] += mass

    visit((), Fraction(1))
    for i, task in enumerate(labels):
        if task is None:
            expected = Fraction(max(0, count - 6), 2)
        elif count <= 3:
            expected = Fraction(count, 3 * task)
        elif task == 1 or count >= 6:
            expected = Fraction(1)
        else:
            expected = Fraction(count - 1, 2 * task)
        assert included[i] == expected
    if count >= 4:
        assert joint == {(t, u): Fraction(1, 6) for t in (1, 2, 3) for u in (2, 3)}


@pytest.mark.asyncio
async def test_two_microbatches_use_the_realized_window_share(tmp_path):
    expected = Fraction(0)
    for tickets in product(range(2), repeat=2):
        trainer = _trainer(tmp_path, MultiTurnGRPOTrainer, "grpo", 0, 2)
        model = ThreeActionPolicy()
        trainer.agent.model = model
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        trainer.config.objective_overrides = {"advantage_eps": 0.0}
        manager = ContinualLearningManager(
            ContinualLearningConfig(
                strategy="replay",
                replay_min_size=1,
                replay_buffer_size=1,
                replay_ratio=0.5,
            )
        )
        manager.add_trajectory_groups([stored_group("low1")])
        manager._rng.randrange = Mock(side_effect=tickets)
        # Condition on the valid reservoir history that keeps the archived row.
        manager.buffer._rng.randint = lambda lo, hi: hi
        trainer.continual_manager = manager
        for _ in tickets:
            mixed, _ = trainer._maybe_mix_replay([stored_group("fresh")], "new")
            await trainer.training_step(mixed)
        n_replay = tickets.count(0)
        share = Fraction(n_replay, 2 + n_replay)
        assert trainer.global_step == 1
        assert model.logits[1].item() == pytest.approx(float(share / 2), abs=1e-7)
        expected += share / 4
    assert expected == Fraction(7, 24)
    # The arbitrary finite-law variance remainder from the Lean identity.
    remainder = sum(
        Fraction(2 * (m - 1) ** 2, 9 * (2 + m)) * math.comb(2, m) / 4 for m in range(3)
    )
    assert Fraction(1, 3) - expected == remainder == Fraction(1, 24)
