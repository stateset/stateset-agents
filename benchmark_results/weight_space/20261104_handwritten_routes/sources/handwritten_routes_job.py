"""Verify complete neural routes against an independent lexicographic Dijkstra.

Run the fixed analytic construction on 4,096 graphs plus all 6,561 endpoint
queries on embedded three-node graphs with edge values 0,1,2. No fitting or
candidate selection occurs. Output tensors and source hashes support replay.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import logging
import platform
import time
from pathlib import Path

import torch
from handwritten_weighted_graph_job import sample
from safetensors.torch import load_model, save_file, save_model

from stateset_agents.experimental.weight_space import handwritten_routes as hw

LOGGER = logging.getLogger(__name__)
N, MAX_COST, SENTINEL = 17, 7, 113
FAMILIES = (
    "er_0.05",
    "er_0.12",
    "er_0.25",
    "er_0.5",
    "max_cost_chain",
    "broken_chain",
    "cheap_detours",
    "two_cycles",
)


def oracle_route(
    costs: list[list[int]], source: int, target: int
) -> tuple[int, list[int]]:
    """Dijkstra ordered by (cost, complete path), independent of the NN recipe."""
    queue = [(0, (source,), source)]
    settled: set[int] = set()
    while queue:
        cost, path, vertex = heapq.heappop(queue)
        if vertex in settled:
            continue
        settled.add(vertex)
        if vertex == target:
            return cost, list(path)
        for neighbor, edge in enumerate(costs[vertex]):
            if edge > 0 and neighbor not in settled:
                heapq.heappush(queue, (cost + edge, (*path, neighbor), neighbor))
    return SENTINEL, []


def expected_outputs(
    costs: torch.Tensor, sources: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Label full routes with no calls to neural weights or encoded distances."""
    count, nodes, _ = costs.shape
    out = torch.zeros(count, 1 + nodes * nodes)
    for k, (graph, source, target) in enumerate(
        zip(costs.tolist(), sources.tolist(), targets.tolist(), strict=True)
    ):
        distance, route = oracle_route(graph, source, target)
        out[k, 0] = distance
        for step, vertex in enumerate(route):
            out[k, 1 + step * nodes + vertex] = 1
    return out


def exhaustive_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All 3^6 directed cost matrices and all nine source/target pairs."""
    vertices = (0, 8, 16)
    edges = [(i, j) for i in vertices for j in vertices if i != j]
    costs, sources, targets = [], [], []
    for values in itertools.product(range(3), repeat=6):
        graph = torch.zeros(N, N, dtype=torch.int64)
        for (i, j), value in zip(edges, values, strict=True):
            graph[i, j] = value
        for source, target in itertools.product(vertices, repeat=2):
            costs.append(graph)
            sources.append(source)
            targets.append(target)
    return torch.stack(costs), torch.tensor(sources), torch.tensor(targets)


def forward_batches(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Use bounded batches for the dense realization of the sparse circuit."""
    with torch.inference_mode():
        return torch.cat([model(batch) for batch in inputs.split(128)])


def metrics(predicted: torch.Tensor, expected: torch.Tensor) -> dict:
    """Count exact routes and costs, without rounding or tolerance."""
    return {
        "cases": len(expected),
        "exact_cost": int((predicted[:, 0] == expected[:, 0]).sum()),
        "exact_route": int((predicted[:, 1:] == expected[:, 1:]).all(1).sum()),
        "exact_both": int((predicted == expected).all(1).sum()),
        "maximum_absolute_error": float((predicted - expected).abs().max()),
    }


def main() -> None:
    """Run, export, reload, and verify the fixed route experiment."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    source_paths = (
        Path(__file__),
        Path(hw.__file__),
        Path(hw.make_weighted_shortest_paths.__code__.co_filename),
        Path(sample.__code__.co_filename),
    )
    protocol = {
        "nodes": N,
        "max_cost": MAX_COST,
        "cases_per_family": 512,
        "families": FAMILIES,
        "seed": 2026110407,
        "exhaustive_cases": 6561,
        "dtype_probe_cases": "first 64 cases of each family",
        "training_updates": 0,
        "candidate_selection": False,
        "tie_rule": "lexicographically smallest vertex sequence",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    sources_dir = out / "sources"
    sources_dir.mkdir()
    for path in source_paths:
        (sources_dir / path.name).write_bytes(path.read_bytes())
    model = hw.make_weighted_routes()
    save_model(model, str(out / "routes.safetensors"))
    rng = torch.Generator().manual_seed(protocol["seed"])
    result: dict = {
        "protocol": protocol,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "threads": 4,
        "dense_parameters": sum(p.numel() for p in model.parameters()),
        "nonzero_parameters": sum(int(p.count_nonzero()) for p in model.parameters()),
        "families": {},
    }
    tensors: dict[str, torch.Tensor] = {}
    probe_inputs, probe_expected = [], []
    for family in (*FAMILIES, "exhaustive_three_nodes"):
        if family == "exhaustive_three_nodes":
            costs, sources, targets = exhaustive_inputs()
        else:
            costs, sources = sample(family, 512, rng)
            targets = torch.randint(N, (512,), generator=rng)
            if family in ("max_cost_chain", "cheap_detours"):
                # Exercise full-length and cheaper multi-edge paths by choosing
                # the vertex reached after traversing the unit/max-cost chain.
                for k, graph in enumerate(costs):
                    terminal = torch.where((graph > 0).sum(1) == 0)[0]
                    targets[k] = terminal[0]
        started = time.perf_counter()
        expected = expected_outputs(costs, sources, targets)
        oracle_seconds = time.perf_counter() - started
        inputs = hw.encode_weighted_routes(costs, sources, targets)
        started = time.perf_counter()
        predicted = forward_batches(model, inputs)
        report = {
            **metrics(predicted, expected),
            "network_seconds": time.perf_counter() - started,
            "oracle_seconds_including_output_packing": oracle_seconds,
        }
        result["families"][family] = report
        LOGGER.info("%s: %s", family, report)
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        for key, tensor in {
            "costs": costs,
            "sources": sources,
            "targets": targets,
            "expected": expected,
            "predicted": predicted,
        }.items():
            tensors[f"{family}.{key}"] = tensor.contiguous()
        if family != "exhaustive_three_nodes":
            probe_inputs.append(inputs[:64])
            probe_expected.append(expected[:64])
    save_file(tensors, str(out / "evaluation.safetensors"))
    restored = hw.make_weighted_routes()
    load_model(restored, str(out / "routes.safetensors"))
    probes, expected = torch.cat(probe_inputs), torch.cat(probe_expected)
    result["reloaded_checkpoint"] = metrics(forward_batches(restored, probes), expected)
    result["reduced_precision"] = {}
    for dtype in (torch.float16, torch.bfloat16):
        restored.to(dtype=dtype)
        result["reduced_precision"][str(dtype)] = metrics(
            forward_batches(restored, probes.to(dtype)).float(), expected
        )
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    manifest = {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.rglob("*")
        if p.is_file()
    }
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    if any(r["exact_both"] != r["cases"] for r in result["families"].values()):
        raise RuntimeError("route verification failed; see retained evidence")
    LOGGER.info("Verification complete: %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
