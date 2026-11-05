"""Independent labels, split isolation and a conventional classification baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "weight_space")
)
from handwritten_graph_job import (  # noqa: E402
    N,
    bfs,
    build_splits,
    evaluate,
    make_classifier,
    selection_key,
)

from stateset_agents.experimental.weight_space.handwritten_graph import (  # noqa: E402
    make_shortest_paths,
)


def graph_keys(x: torch.Tensor) -> set[bytes]:
    return {row[: N * N].numpy().tobytes() for row in x}


def test_split_adjacencies_are_unique_and_disjoint_even_without_sources() -> None:
    (x, _), validation, tests, manifest = build_splits(8, 4, 8)
    train = graph_keys(x)
    val = set().union(*(graph_keys(v) for v, _ in validation.values()))
    test = set().union(*(graph_keys(v) for v in tests.values()))
    assert not train & val and not train & test and not val & test
    assert len(train) + len(val) + len(test) == manifest["unique_adjacencies"]
    assert len(train) == 40 and len(val) == 20 and len(test) == 72


def test_queue_oracle_matches_networkx_on_independent_graphs() -> None:
    nx = pytest.importorskip("networkx")
    generator = torch.Generator().manual_seed(384)
    for _ in range(32):
        adjacency = (torch.rand(12, 12, generator=generator) < 0.12).int()
        source = int(torch.randint(12, (1,), generator=generator))
        graph = nx.from_numpy_array(adjacency.numpy(), create_using=nx.DiGraph)
        reference = nx.single_source_shortest_path_length(graph, source)
        assert bfs(adjacency.tolist(), source) == [
            reference.get(i, 12) for i in range(12)
        ]


def test_classifier_has_a_matching_budget_and_full_distance_classes() -> None:
    rng = torch.get_rng_state().clone()
    model = make_classifier(3)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 115027
    assert model(torch.zeros(2, 156)).shape == (2, 12, 13)


def test_graph_metric_requires_every_distance_correct() -> None:
    (x, y), _, _, _ = build_splits(2, 1, 1)
    model = make_shortest_paths()
    scores = evaluate(model, "recurrent", {"all": (x, y)})["all"]
    assert scores["graph_accuracy"] == 1 and scores["max_raw_error"] == 0
    wrong = y.clone()
    wrong[:, 0] = (wrong[:, 0] + 1) % 13
    scores = evaluate(model, "recurrent", {"all": (x, wrong)})["all"]
    assert scores["graph_accuracy"] == 0
    assert scores["node_accuracy"] == pytest.approx(11 / 12)


def test_validation_selection_uses_exact_graphs_before_mae() -> None:
    a = {"all": {"exact_graphs": 11, "distance_mae": 2.0}}
    b = {"all": {"exact_graphs": 10, "distance_mae": 0.01}}
    assert selection_key(a) > selection_key(b)
