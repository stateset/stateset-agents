"""Check stronger baseline orientation, recurrence, and intermediate supervision."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.weight_space.handwritten_weighted_graph import (  # noqa: E402
    encode_weighted_graphs,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/weight_space"))
from handwritten_weighted_graph_job import labels  # noqa: E402
from learned_weighted_graph_job import LearnedRelaxation, teacher_states  # noqa: E402


def test_supervised_states_end_at_independent_dijkstra_labels() -> None:
    rng = torch.Generator().manual_seed(11)
    costs = torch.randint(0, 8, (32, 17, 17), generator=rng)
    costs *= torch.rand(32, 17, 17, generator=rng) < 0.12
    costs[:, torch.arange(17), torch.arange(17)] = 0
    inputs = encode_weighted_graphs(costs, torch.randint(17, (32,), generator=rng))
    states = teacher_states(inputs)
    assert states.shape == (32, 17, 17)
    assert torch.equal(states[:, 0], inputs[:, -17:])
    assert torch.equal(states[:, -1], labels(inputs).float())
    assert bool((states[:, 1:] <= states[:, :-1]).all())


def test_graph_recurrence_with_exact_message_arithmetic() -> None:
    model = LearnedRelaxation(0)
    model.message = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.message.weight.fill_(1)
        model.message.bias.zero_()
    costs = torch.zeros(2, 17, 17)
    costs[0, torch.arange(16), torch.arange(1, 17)] = 7
    costs[1, 0, 1], costs[1, 0, 2], costs[1, 2, 1] = 7, 2, 3
    inputs = encode_weighted_graphs(costs, torch.tensor([0, 0]))
    with torch.inference_mode():
        actual = model(inputs)
    assert torch.equal(actual.round().long(), labels(inputs))


def test_seeded_baseline_preserves_rng_and_has_trainable_gradients() -> None:
    before = torch.get_rng_state().clone()
    model = LearnedRelaxation(42)
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 1185
    cost = torch.ones(2, 17, 17)
    distance = torch.zeros(2, 17)
    loss = (model.messages(cost, distance) - cost).square().mean()
    loss.backward()
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
