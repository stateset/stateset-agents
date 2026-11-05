"""Exact costs, active neuron splitting, and independent weighted graph oracles."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space.handwritten_weighted_graph import (  # noqa: E402
    encode_weighted_graphs,
    make_weighted_shortest_paths,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/weight_space"))
from handwritten_weighted_graph_job import (  # noqa: E402
    build_splits,
    classifier,
    dijkstra,
)


def test_two_million_count_rng_and_shared_standard_layers() -> None:
    rng = torch.get_rng_state().clone()
    model = make_weighted_shortest_paths()
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 2_000_000
    assert sum(int(p.count_nonzero()) for p in model.parameters()) == 5679
    assert all(model[i] is model[0] for i in range(16))
    assert all(
        type(m) in (torch.nn.Sequential, torch.nn.Linear, torch.nn.ReLU)
        for m in model.modules()
    )


def test_weighted_detour_beats_direct_edge_and_handles_ties() -> None:
    costs = torch.zeros(1, 17, 17)
    costs[0, 0, 1] = 7
    costs[0, 0, 2] = 2
    costs[0, 2, 1] = 3
    costs[0, 0, 3] = 5
    costs[0, 2, 3] = 3
    costs[0, 1, 3] = 1
    x = encode_weighted_graphs(costs, torch.tensor([0]))
    with torch.inference_mode():
        output = make_weighted_shortest_paths()(x)
    assert output[0, :4].tolist() == [0, 5, 2, 5]
    assert output[0, 4:].tolist() == [113] * 13


def test_maximum_cost_chain_and_missing_round() -> None:
    costs = torch.zeros(1, 17, 17)
    costs[0, torch.arange(16), torch.arange(1, 17)] = 7
    x = encode_weighted_graphs(costs, torch.tensor([0]))
    model = make_weighted_shortest_paths()
    with torch.inference_mode():
        assert torch.equal(model(x)[0], torch.arange(17) * 7)
        shortened = torch.nn.Sequential(*([model[0]] * 15), model[-1])
        assert shortened(x)[0, -1] == 113


def test_split_neurons_preserve_natural_function_and_are_active() -> None:
    rng = torch.Generator().manual_seed(42)
    costs = torch.randint(0, 8, (16, 17, 17), generator=rng)
    costs *= torch.rand(16, 17, 17, generator=rng) < 0.15
    costs[:, torch.arange(17), torch.arange(17)] = 0
    x = encode_weighted_graphs(costs, torch.randint(17, (16,), generator=rng))
    natural = make_weighted_shortest_paths(exact_budget=False)
    split = make_weighted_shortest_paths()
    assert sum(p.numel() for p in natural.parameters()) == 1_865_053
    with torch.inference_mode():
        assert torch.equal(natural(x), split(x))
    # The first split comparison layer has 106 extra live neurons; every
    # added neuron has both incoming and outgoing nonzero coefficients.
    compare, reduce = split[0][2], split[0][4]
    assert bool(compare.weight[595:].count_nonzero(dim=1).all())
    assert bool(reduce.weight[:, 595:].count_nonzero(dim=0).all())


@pytest.mark.parametrize("nodes,max_cost", [(2, 1), (3, 2), (7, 4), (17, 7)])
def test_random_weighted_graphs_match_networkx(nodes: int, max_cost: int) -> None:
    nx = pytest.importorskip("networkx")
    rng = torch.Generator().manual_seed(nodes)
    costs = torch.randint(1, max_cost + 1, (32, nodes, nodes), generator=rng)
    costs *= torch.rand(32, nodes, nodes, generator=rng) < 0.2
    costs[:, torch.arange(nodes), torch.arange(nodes)] = 0
    sources = torch.randint(nodes, (32,), generator=rng)
    sentinel = (nodes - 1) * max_cost + 1
    expected = []
    for graph, source in zip(costs, sources, strict=True):
        g = nx.from_numpy_array(graph.numpy(), create_using=nx.DiGraph)
        distances = nx.single_source_dijkstra_path_length(g, int(source))
        expected.append([distances.get(i, sentinel) for i in range(nodes)])
        if nodes == 17:
            assert dijkstra(graph.tolist(), int(source)) == expected[-1]
    model = make_weighted_shortest_paths(nodes, max_cost, exact_budget=False)
    with torch.inference_mode():
        assert torch.equal(
            model(encode_weighted_graphs(costs, sources, max_cost)),
            torch.tensor(expected).float(),
        )


def test_permutation_equivariance_and_checkpoint(tmp_path: Path) -> None:
    from safetensors.torch import load_model, save_model

    rng = torch.Generator().manual_seed(78)
    costs = torch.randint(0, 8, (1, 17, 17), generator=rng)
    costs[:, torch.arange(17), torch.arange(17)] = 0
    order = torch.randperm(17, generator=rng)
    model = make_weighted_shortest_paths()
    with torch.inference_mode():
        expected = model(encode_weighted_graphs(costs, torch.tensor([3])))[:, order]
        x = encode_weighted_graphs(
            costs[:, order][:, :, order], torch.where(order == 3)[0]
        )
        assert torch.equal(model(x), expected)
    path = tmp_path / "weights.safetensors"
    save_model(model, str(path))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    load_model(model, str(path))
    with torch.inference_mode():
        assert torch.equal(model(x), expected)


@pytest.mark.parametrize("value", [0.5, -1, 8, float("nan"), float("inf")])
def test_rejects_invalid_edge_cost(value: float) -> None:
    graph = torch.zeros(1, 17, 17)
    graph[0, 0, 1] = value
    with pytest.raises(ConfigurationError):
        encode_weighted_graphs(graph, torch.tensor([0]))


def test_rejects_invalid_configuration() -> None:
    for nodes, cost in [(True, 7), (1, 7), (17, 0), (17, 2**24)]:
        with pytest.raises(ConfigurationError):
            make_weighted_shortest_paths(nodes, cost)
    with pytest.raises(ConfigurationError, match="seventeen"):
        make_weighted_shortest_paths(8)
    with pytest.raises(ConfigurationError, match="diagonal"):
        encode_weighted_graphs(torch.eye(17)[None], torch.tensor([0]))
    with pytest.raises(ConfigurationError, match="outside"):
        encode_weighted_graphs(torch.zeros(1, 17, 17), torch.tensor([17]))


def test_rejects_fractional_float64_before_float32_conversion() -> None:
    graph = torch.zeros(1, 17, 17, dtype=torch.float64)
    graph[0, 0, 1] = 1.0 + 1e-10
    with pytest.raises(ConfigurationError, match="integers"):
        encode_weighted_graphs(graph, torch.tensor([0]))


def test_boolean_costs_encode_unit_edges() -> None:
    graph = torch.zeros(1, 17, 17, dtype=torch.bool)
    graph[0, 0, 1] = True
    encoded = encode_weighted_graphs(graph, torch.tensor([0]))
    assert encoded.dtype == torch.float32
    assert encoded[0, 1] == 1 and encoded[0, 2] == 113


def test_classifier_budget_and_disjoint_weighted_splits() -> None:
    rng = torch.get_rng_state().clone()
    model = classifier(0)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 1_999_988
    assert model(torch.zeros(2, 306)).shape == (2, 17, 114)
    train, val, tests, manifest = build_splits(4, 2, 3)
    matrices = [
        row[:289].numpy().tobytes()
        for x in (train[0], val[0], *tests.values())
        for row in x
    ]
    assert len(set(matrices)) == len(matrices) == manifest["unique_cost_matrices"]
