"""Replay exported route weights and independently verify every route with NetworkX."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import networkx as nx
import torch
from safetensors.torch import load_file, load_model

from stateset_agents.experimental.weight_space import handwritten_routes as hw

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Validate hashes, constructor/export equivalence, and all retained cases."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    root = ap.parse_args().root
    torch.set_num_threads(4)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for name, expected_hash in manifest.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"artifact hash differs: {name}")
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        hashlib.sha256(Path(hw.__file__).read_bytes()).hexdigest()
        != protocol["source_sha256"]["handwritten_routes.py"]
    ):
        raise ValueError("installed route constructor differs from recorded source")
    data = load_file(str(root / "evaluation.safetensors"))
    model = hw.make_weighted_routes()
    restored = hw.make_weighted_routes()
    load_model(restored, str(root / "routes.safetensors"))
    if any(
        not torch.equal(a, b)
        for a, b in zip(model.parameters(), restored.parameters(), strict=True)
    ):
        raise ValueError("exported parameters differ from the analytic construction")
    checked = 0
    for family in sorted({key.rsplit(".", 1)[0] for key in data}):
        graphs, sources, targets = (
            data[f"{family}.{key}"] for key in ("costs", "sources", "targets")
        )
        with torch.inference_mode():
            inputs = hw.encode_weighted_routes(graphs, sources, targets)
            actual = torch.cat([restored(batch) for batch in inputs.split(128)])
        if not torch.equal(actual, data[f"{family}.expected"]) or not torch.equal(
            actual, data[f"{family}.predicted"]
        ):
            raise ValueError(f"checkpoint replay differs: {family}")
        for graph, source, target, row in zip(
            graphs, sources, targets, actual, strict=True
        ):
            g = nx.from_numpy_array(graph.numpy(), create_using=nx.DiGraph)
            source, target = int(source), int(target)
            try:
                route = min(nx.all_shortest_paths(g, source, target, weight="weight"))
                distance = nx.shortest_path_length(g, source, target, weight="weight")
            except nx.NetworkXNoPath:
                route, distance = [], 113
            expected = torch.zeros(17, 17)
            for step, vertex in enumerate(route):
                expected[step, vertex] = 1
            if row[0] != distance or not torch.equal(row[1:].reshape(17, 17), expected):
                raise ValueError(f"NetworkX disagrees: {family}, case {checked}")
            checked += 1
        LOGGER.info("%s: verified %d full routes", family, len(graphs))
    report = {
        "networkx_version": nx.__version__,
        "cases": checked,
        "all_artifact_hashes_match": True,
        "all_exported_parameters_match_constructor": True,
        "all_reloaded_outputs_exact": True,
        "all_networkx_costs_and_lexicographic_routes_exact": True,
        "checkpoint_sha256": hashlib.sha256(
            (root / "routes.safetensors").read_bytes()
        ).hexdigest(),
        "verification_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (root / "independent_verification.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
