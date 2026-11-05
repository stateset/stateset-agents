"""Replay graph-repair evidence, independent graph oracles, and edit accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import networkx as nx
import torch
from repair_weighted_graph_job import (
    addition_edit,
    component_grid,
    digest,
    from_state,
    graphs,
    matched_control,
    matrix_keys,
)
from safetensors.torch import load_file


def require(condition: bool, message: str) -> None:
    """Keep evidence checks enabled under python -O."""
    if not condition:
        raise ValueError(message)


def same_state(left: dict, right: dict) -> bool:
    """Exact tensor equality, including names and shapes."""
    return left.keys() == right.keys() and all(
        torch.equal(left[k], right[k]) for k in left
    )


def oracle(inputs: torch.Tensor) -> torch.Tensor:
    """Label all destinations using NetworkX, independently of benchmark Dijkstra."""
    costs = inputs[:, : graphs.N**2].reshape(-1, graphs.N, graphs.N).long()
    sources = inputs[:, -graphs.N :].argmin(1).tolist()
    truth = []
    for matrix, source in zip(costs.tolist(), sources, strict=True):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(graphs.N))
        for i, row in enumerate(matrix):
            for j, weight in enumerate(row):
                if 0 < weight < graphs.SENTINEL:
                    graph.add_edge(i, j, weight=weight)
        lengths = nx.single_source_dijkstra_path_length(graph, source)
        truth.append([lengths.get(i, graphs.SENTINEL) for i in range(graphs.N)])
    return torch.tensor(truth)


@torch.inference_mode()
def replay(state: dict, inputs: torch.Tensor, truth: torch.Tensor) -> dict:
    """Recompute metrics from a reloaded checkpoint and independently checked truth."""
    model = from_state(state)
    actual = torch.cat([model(x) for x in inputs.split(64)])
    require(bool(torch.isfinite(actual).all()), "non-finite model output")
    predicted = actual.round().long()
    correct = predicted == truth
    return {
        "n": len(inputs),
        "exact_graphs": int(correct.all(1).sum()),
        "graph_accuracy": float(correct.all(1).double().mean()),
        "node_accuracy": float(correct.double().mean()),
        "distance_mae": float((predicted - truth).abs().double().mean()),
        "raw_mae": float((actual - truth).abs().double().mean()),
        "max_raw_error": float((actual - truth).abs().max()),
    }


def metrics_match(actual: dict, expected: dict) -> bool:
    """Allow float32 reduction rounding; require identical integer case counts."""
    return actual.keys() == expected.keys() and all(
        (
            actual[key] == expected[key]
            if key in ("n", "exact_graphs")
            else math.isclose(actual[key], expected[key], rel_tol=2e-6, abs_tol=1e-7)
        )
        for key in actual
    )


def main() -> None:
    """Verify complete experiments without choosing new hyperparameters."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--initial", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = a.root
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for name, expected in manifest.items():
        require(digest(root / name) == expected, f"artifact hash differs: {name}")
    summary = json.loads((root / "summary.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    require(summary["protocol"] == protocol, "protocol differs")
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "sources" / name) == expected, f"source differs: {name}")
        local = (
            Path(graphs.hw.__file__)
            if name == "handwritten_weighted_graph.py"
            else Path(__file__).parent / name
        )
        require(digest(local) == expected, f"replay source differs: {name}")
    ledgers = json.loads((root / "edit_ledger.json").read_text())
    assigned, changed = {}, {}
    for origin in protocol["initial_checkpoints"]:
        seed = origin["seed"]
        path = a.initial / origin["file"]
        require(digest(path) == origin["sha256"], "starting checkpoint hash differs")
        original = load_file(str(path))
        require(
            same_state(
                original, load_file(str(root / f"original_seed{seed}.safetensors"))
            ),
            "retained original differs",
        )
        edited, ledger = addition_edit(original)
        require(ledger == ledgers[str(seed)], "edit ledger differs")
        require(
            same_state(edited, load_file(str(root / f"edited_seed{seed}.safetensors"))),
            "edited checkpoint differs from recipe",
        )
        control = load_file(str(root / f"control_seed{seed}.safetensors"))
        require(
            same_state(control, matched_control(original, ledger, 65000 + seed)),
            "random control differs from recipe",
        )
        allowed = {(r["tensor"], r["index"]) for r in ledger}
        edits, randoms = [], []
        for name in original:
            for variant in (edited, control):
                diff = (
                    (variant[name].flatten() != original[name].flatten())
                    .nonzero()
                    .flatten()
                )
                require(
                    all((name, int(i)) in allowed for i in diff),
                    "edit outside allowed coordinates",
                )
            edits.append((edited[name].double() - original[name].double()).flatten())
            randoms.append((control[name].double() - original[name].double()).flatten())
        patch, random = torch.cat(edits), torch.cat(randoms)
        require(
            math.isclose(float(patch.norm()), float(random.norm()), rel_tol=1e-6),
            "control delta norm differs",
        )
        assigned[str(seed)], changed[str(seed)] = len(allowed), int(
            patch.count_nonzero()
        )
        require(
            len(allowed) == 69 and sum(t.numel() for t in original.values()) == 1185,
            "parameter accounting differs",
        )
    train, validation, previous_tests, old_splits = graphs.build_splits(4096, 256, 512)
    require(old_splits == summary["splits"]["original"], "old splits differ")
    previous = matrix_keys(train[0]) | matrix_keys(validation[0])
    for x in previous_tests.values():
        previous |= matrix_keys(x)
    saved = load_file(str(root / "evaluation.safetensors"))
    families = [*protocol["families"], "exhaustive_three"]
    require(
        set(saved) == {f"{f}.{k}" for f in families for k in ("inputs", "truth")},
        "evaluation tensor names differ",
    )
    fresh = set()
    oracle_cases = 0
    for family in families:
        inputs, truth = saved[f"{family}.inputs"], saved[f"{family}.truth"]
        require(
            torch.equal(oracle(inputs), truth), f"NetworkX oracle differs: {family}"
        )
        oracle_cases += len(inputs)
        if family != "exhaustive_three":
            keys = matrix_keys(inputs)
            require(
                not keys & (previous | fresh), "fresh split overlaps previous graphs"
            )
            require(
                len(keys) == len(inputs) == protocol["fresh_test_per_family"],
                "wrong fresh count",
            )
            require(
                hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
                == summary["splits"]["fresh_sha256"][family],
                "fresh split hash differs",
            )
            fresh |= keys
        else:
            require(
                torch.equal(inputs, graphs.exhaustive_three()[0]),
                "exhaustive inputs differ",
            )
    require(len(fresh) == summary["splits"]["fresh_matrices"], "fresh total differs")
    require(
        len(previous) == summary["splits"]["excluded_matrices"],
        "excluded total differs",
    )
    expected_variants = {
        (name, seed)
        for name in ("original", "edited", "control", "continued", "coverage")
        for seed in protocol["seeds"]
    }
    require(
        {(v["name"], v["seed"]) for v in summary["variants"]} == expected_variants
        and len(summary["variants"]) == len(expected_variants),
        "missing or duplicate variant",
    )
    replays = 0
    for variant in summary["variants"]:
        state = load_file(str(root / variant["checkpoint"]))
        for family in families:
            actual = replay(state, saved[f"{family}.inputs"], saved[f"{family}.truth"])
            require(
                metrics_match(actual, variant["families"][family]),
                f"metrics differ: {variant['name']}/{variant['seed']}/{family}",
            )
            replays += actual["n"]
        with torch.inference_mode():
            x = component_grid()
            error = (from_state(state).message(x).squeeze(-1) - x.sum(1)).abs()
        require(
            float(error.max()) == variant["component"]["max_error"],
            "component error differs",
        )
        if variant["name"] in protocol["regimes"]:
            candidates = [
                t
                for t in summary["trials"]
                if t["regime"] == variant["name"] and t["seed"] == variant["seed"]
            ]
            require(
                len(candidates) == len(protocol["learning_rates"]),
                "missing training trial",
            )
            selected = max(candidates, key=lambda t: t["selection_key"])
            require(
                selected["checkpoint"] == variant["checkpoint"],
                "validation selection differs",
            )
            validation_metrics = replay(state, *validation)
            require(
                validation_metrics["exact_graphs"] == selected["selection_key"][0]
                and math.isclose(
                    -validation_metrics["raw_mae"],
                    selected["selection_key"][1],
                    rel_tol=2e-6,
                    abs_tol=1e-7,
                ),
                "selected checkpoint validation metrics differ",
            )
    result = {
        "all_artifact_hashes_match": True,
        "all_sources_match": True,
        "all_edits_and_random_controls_reproduced": True,
        "assigned_parameters_by_seed": assigned,
        "changed_parameters_by_seed": changed,
        "fresh_graphs": len(fresh),
        "previous_graph_overlap": 0,
        "networkx_oracle_cases": oracle_cases,
        "replayed_graph_predictions": replays,
        "all_test_metrics_reproduced": True,
        "all_selected_validation_metrics_reproduced": True,
        "verifier_sha256": digest(Path(__file__)),
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
