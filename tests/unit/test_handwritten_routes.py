"""Exact neural routes, tie handling, shared recurrences, and independent oracles."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space.handwritten_routes import (  # noqa: E402
    encode_weighted_routes,
    make_weighted_routes,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/weight_space"))
from handwritten_routes_job import oracle_route  # noqa: E402


def unpack(output):
    nodes = int((output.shape[0] - 1) ** 0.5)
    route = output[1:].reshape(nodes, nodes)
    assert bool(((route == 0) | (route == 1)).all())
    lengths = route.sum(1)
    assert bool((lengths <= 1).all())
    count = int(lengths.sum())
    assert lengths.tolist() == [1] * count + [0] * (nodes - count)
    return int(output[0]), route[:count].argmax(1).tolist()


def test_route_network_is_analytic_shared_and_standard() -> None:
    rng = torch.get_rng_state().clone()
    model = make_weighted_routes()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        type(m) in (torch.nn.Sequential, torch.nn.Linear, torch.nn.ReLU)
        for m in model.modules()
    )
    assert all(model[i] is model[0] for i in range(16))
    assert all(model[i] is model[-2] for i in range(24, 40))


def test_detours_ties_and_endpoint_cases() -> None:
    graph = torch.zeros(4, 5, 5)
    # Both 0->1->4 and 0->2->4 cost five; direct 0->4 costs seven.
    graph[:, 0, 4] = 7
    graph[:, 0, 1] = 2
    graph[:, 1, 4] = 3
    graph[:, 0, 2] = 1
    graph[:, 2, 4] = 4
    inputs = encode_weighted_routes(
        graph, torch.tensor([0, 4, 3, 4]), torch.tensor([4, 0, 4, 4])
    )
    with torch.inference_mode():
        out = make_weighted_routes(5)(inputs)
    assert [unpack(row) for row in out] == [
        (5, [0, 1, 4]),
        (29, []),
        (29, []),
        (0, [4]),
    ]


def test_maximum_length_route_and_necessary_last_hop() -> None:
    graph = torch.zeros(1, 17, 17)
    graph[0, torch.arange(16), torch.arange(1, 17)] = 7
    inputs = encode_weighted_routes(graph, torch.tensor([0]), torch.tensor([16]))
    model = make_weighted_routes()
    with torch.inference_mode():
        output = model(inputs)[0]
        short = torch.nn.Sequential(*list(model)[:-2], model[-1])(inputs)[0]
    assert unpack(output) == (112, list(range(17)))
    assert not torch.equal(output, short)


@pytest.mark.parametrize("nodes", [2, 3, 6, 17])
def test_random_routes_match_independent_networkx(nodes: int) -> None:
    nx = pytest.importorskip("networkx")
    rng = torch.Generator().manual_seed(410 + nodes)
    graphs = torch.randint(1, 8, (32, nodes, nodes), generator=rng)
    graphs *= torch.rand(32, nodes, nodes, generator=rng) < 0.25
    graphs[:, torch.arange(nodes), torch.arange(nodes)] = 0
    sources = torch.randint(nodes, (32,), generator=rng)
    targets = torch.randint(nodes, (32,), generator=rng)
    with torch.inference_mode():
        outputs = make_weighted_routes(nodes)(
            encode_weighted_routes(graphs, sources, targets)
        )
    sentinel = (nodes - 1) * 7 + 1
    for graph, source, target, row in zip(
        graphs, sources, targets, outputs, strict=True
    ):
        g = nx.from_numpy_array(graph.numpy(), create_using=nx.DiGraph)
        source, target = int(source), int(target)
        if nx.has_path(g, source, target):
            paths = nx.all_shortest_paths(g, source, target, weight="weight")
            expected = (
                nx.shortest_path_length(g, source, target, weight="weight"),
                min(paths),
            )
            assert oracle_route(graph.tolist(), source, target) == expected
        else:
            expected = (sentinel, [])
        assert unpack(row) == expected


def test_empty_graph_and_reloaded_export(tmp_path: Path) -> None:
    from safetensors.torch import load_model, save_model

    nodes = 4
    sources = torch.arange(nodes).repeat_interleave(nodes)
    targets = torch.arange(nodes).repeat(nodes)
    inputs = encode_weighted_routes(
        torch.zeros(nodes**2, nodes, nodes), sources, targets
    )
    model = make_weighted_routes(nodes)
    path = tmp_path / "routes.safetensors"
    save_model(model, str(path))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    load_model(model, str(path))
    with torch.inference_mode():
        out = model(inputs)
    for source, target, row in zip(
        sources.tolist(), targets.tolist(), out, strict=True
    ):
        assert unpack(row) == ((0, [source]) if source == target else (22, []))


@pytest.mark.parametrize("sources,targets", [([4], [0]), ([0], [-1]), ([0], [4])])
def test_invalid_endpoints_are_rejected(sources: list[int], targets: list[int]) -> None:
    with pytest.raises(ConfigurationError, match="outside"):
        encode_weighted_routes(
            torch.zeros(1, 4, 4), torch.tensor(sources), torch.tensor(targets)
        )


def test_encoder_preserves_cost_validation() -> None:
    graph = torch.zeros(1, 4, 4, dtype=torch.float64)
    graph[0, 0, 1] = 1 + 1e-10
    with pytest.raises(ConfigurationError, match="integers"):
        encode_weighted_routes(graph, torch.tensor([0]), torch.tensor([1]))
