"""An independent queue BFS oracle checks the compiled neural graph algorithm."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space.handwritten_graph import (  # noqa: E402
    encode_graphs,
    make_shortest_paths,
)


def bfs(graph: list[list[int]], source: int) -> list[int]:
    distance = [len(graph)] * len(graph)
    distance[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for other, edge in enumerate(graph[node]):
            if edge and distance[other] == len(graph):
                distance[other] = distance[node] + 1
                queue.append(other)
    return distance


def test_unique_parameter_count_sparsity_and_tied_rounds() -> None:
    rng = torch.get_rng_state().clone()
    model = make_shortest_paths()
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 114_984
    assert sum(int(p.count_nonzero()) for p in model.parameters()) == 1_308
    assert all(model[i] is model[0] for i in range(11))
    assert all(
        type(m) in (torch.nn.Sequential, torch.nn.Linear, torch.nn.ReLU)
        for m in model.modules()
    )


def test_all_directed_three_vertex_graphs_and_sources() -> None:
    n = 3
    positions = [(i, j) for i in range(n) for j in range(n) if i != j]
    graphs, sources = [], []
    for mask in range(1 << len(positions)):
        graph = [[0] * n for _ in range(n)]
        for k, (i, j) in enumerate(positions):
            graph[i][j] = (mask >> k) & 1
        for source in range(n):
            graphs.append(graph)
            sources.append(source)
    expected = torch.tensor([bfs(g, s) for g, s in zip(graphs, sources, strict=True)])
    with torch.inference_mode():
        actual = make_shortest_paths(n)(
            encode_graphs(torch.tensor(graphs), torch.tensor(sources))
        )
    assert torch.equal(actual, expected.float())


@pytest.mark.parametrize("nodes", [2, 5, 12, 16])
def test_random_graphs_cycles_and_unreachable_nodes(nodes: int) -> None:
    generator = torch.Generator().manual_seed(42 + nodes)
    graphs = (torch.rand(64, nodes, nodes, generator=generator) < 0.12).long()
    sources = torch.randint(nodes, (64,), generator=generator)
    expected = torch.tensor(
        [bfs(g, s) for g, s in zip(graphs.tolist(), sources.tolist(), strict=True)]
    )
    with torch.inference_mode():
        actual = make_shortest_paths(nodes)(encode_graphs(graphs, sources))
    assert torch.equal(actual, expected.float())


def test_chain_needs_all_eleven_rounds_and_edge_removal_changes_reachability() -> None:
    n = 12
    graph = torch.zeros(1, n, n)
    graph[0, torch.arange(n - 1), torch.arange(1, n)] = 1
    initial = encode_graphs(graph, torch.tensor([0]))
    model = make_shortest_paths()
    with torch.inference_mode():
        assert torch.equal(model(initial), torch.arange(n).float()[None])
        # A missing round loses the end of the chain; it is not a one-pass lookup.
        shortened = torch.nn.Sequential(*([model[0]] * 10), model[-1])
        assert not torch.equal(shortened(initial), model(initial))
        graph[0, 5, 6] = 0
        assert torch.equal(
            model(encode_graphs(graph, torch.tensor([0]))),
            torch.tensor([[0.0, 1, 2, 3, 4, 5, 12, 12, 12, 12, 12, 12]]),
        )


def test_permutation_equivariance() -> None:
    generator = torch.Generator().manual_seed(6)
    graph = (torch.rand(1, 12, 12, generator=generator) < 0.15).float()
    order = torch.randperm(12, generator=generator)
    new_source = torch.where(order == 3)[0]
    model = make_shortest_paths()
    with torch.inference_mode():
        actual = model(encode_graphs(graph[:, order][:, :, order], new_source))
        expected = model(encode_graphs(graph, torch.tensor([3])))[:, order]
    assert torch.equal(actual, expected)


def test_shared_weight_checkpoint_roundtrip(tmp_path: Path) -> None:
    from safetensors.torch import load_model, save_model

    model = make_shortest_paths()
    path = tmp_path / "graph.safetensors"
    save_model(model, str(path))
    restored = make_shortest_paths(handwritten=False, seed=7)
    load_model(restored, str(path))
    assert sum(p.numel() for p in restored.parameters()) == 114_984
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.parameters(), restored.parameters(), strict=True)
    )


@pytest.mark.parametrize("nodes", [True, 1, 0, -1, 3.5])
def test_refuses_invalid_size(nodes: int) -> None:
    with pytest.raises(ConfigurationError):
        make_shortest_paths(nodes)


def test_encoder_rejects_invalid_graph_and_source() -> None:
    with pytest.raises(ConfigurationError, match="binary"):
        encode_graphs(torch.full((1, 3, 3), 0.5), torch.tensor([0]))
    with pytest.raises(ConfigurationError, match="outside"):
        encode_graphs(torch.zeros(1, 3, 3), torch.tensor([3]))


def test_random_initialization_is_reproducible() -> None:
    rng = torch.get_rng_state().clone()
    a, b = (make_shortest_paths(handwritten=False, seed=2) for _ in range(2))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True)
    )
