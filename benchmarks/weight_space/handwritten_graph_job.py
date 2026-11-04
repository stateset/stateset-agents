"""Test a handwritten shortest-path NN against queue BFS and trained baselines.

All architectures and budgets are set before testing. Train/validation/test
adjacencies are disjoint (even if sources differ). Graphs have 12 vertices,
binary directed edges, and no self loops. Labels come from a Python queue BFS,
not from the handwritten network. Only validation selects rates/checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import platform
import time
from collections import deque
from pathlib import Path

import torch
from safetensors.torch import load_model, save_file, save_model

from stateset_agents.experimental.weight_space import handwritten_graph as hg

LOGGER = logging.getLogger(__name__)
N = 12
DENSITIES = (0.025, 0.05, 0.1, 0.2, 0.4)
Batch = tuple[torch.Tensor, torch.Tensor]


def bfs(graph: list[list[int]], source: int) -> list[int]:
    """Independent queue-based single-source shortest paths; n means unreachable."""
    n = len(graph)
    distance = [n] * n
    distance[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for other, edge in enumerate(graph[node]):
            if edge and distance[other] == n:
                distance[other] = distance[node] + 1
                queue.append(other)
    return distance


def targets(inputs: torch.Tensor, n: int = N) -> torch.Tensor:
    """Label packed graph inputs using queue BFS, with no network calls."""
    graphs = inputs[:, : n * n].reshape(-1, n, n).long().tolist()
    sources = inputs[:, n * n : n * n + n].argmax(1).tolist()
    return torch.tensor([bfs(g, s) for g, s in zip(graphs, sources, strict=True)])


def sample_graphs(
    n: int, family: str, count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a prespecified random or structured graph family."""
    sources = torch.randint(n, (count,), generator=generator)
    if family.startswith("er_"):
        p = float(family.removeprefix("er_"))
        graphs = torch.rand(count, n, n, generator=generator) < p
        graphs[:, torch.arange(n), torch.arange(n)] = False
        return graphs, sources
    graphs = torch.zeros(count, n, n, dtype=torch.bool)
    for k in range(count):
        order = torch.randperm(n, generator=generator)
        if family in ("chains", "broken_chains"):
            graphs[k, order[:-1], order[1:]] = True
            # Half begin at the start (maximal depth); half at a random position.
            source_position = (
                0 if k % 2 == 0 else int(torch.randint(n, (1,), generator=generator))
            )
            sources[k] = order[source_position]
            if family == "broken_chains":
                cut = int(torch.randint(n - 1, (1,), generator=generator))
                graphs[k, order[cut], order[cut + 1]] = False
        elif family == "two_cycles":
            split = int(torch.randint(3, n - 2, (1,), generator=generator))
            for group in (order[:split], order[split:]):
                graphs[k, group, group.roll(-1)] = True
            sources[k] = order[0]
        elif family == "dag":
            edges = torch.triu(torch.rand(n, n, generator=generator) < 0.25, diagonal=1)
            graphs[k, order[:, None], order[None, :]] = edges
        else:
            raise ValueError(f"unknown family: {family}")
    return graphs, sources


def unique_family(
    family: str, count: int, generator: torch.Generator, used: set[bytes], n: int = N
) -> torch.Tensor:
    """Reject duplicate adjacency matrices across all splits; source is not the key."""
    graphs, sources = [], []
    attempts = 0
    while len(graphs) < count:
        batch, roots = sample_graphs(n, family, max(64, count - len(graphs)), generator)
        for graph, root in zip(batch, roots, strict=True):
            key = graph.numpy().tobytes()
            attempts += 1
            if attempts > max(10000, 100 * count):
                raise RuntimeError("could not produce enough unique graphs")
            if key in used:
                continue
            used.add(key)
            graphs.append(graph)
            sources.append(root)
            if len(graphs) == count:
                break
    return hg.encode_graphs(torch.stack(graphs), torch.stack(sources))


def build_splits(
    train_per_density: int, validation_per_density: int, test_per_family: int
) -> tuple[Batch, dict[str, Batch], dict[str, torch.Tensor], dict]:
    """Build disjoint inputs; test labels/metrics are deferred until training ends."""
    used: set[bytes] = set()
    families = [f"er_{p}" for p in DENSITIES]
    train_gen = torch.Generator().manual_seed(41000)
    train_x = torch.cat(
        [unique_family(f, train_per_density, train_gen, used) for f in families]
    )
    train = (train_x, targets(train_x))
    val_gen = torch.Generator().manual_seed(42000)
    validation = {}
    for family in families:
        x = unique_family(family, validation_per_density, val_gen, used)
        validation[family] = (x, targets(x))
    test_gen = torch.Generator().manual_seed(43000)
    test_inputs = {
        f: unique_family(f, test_per_family, test_gen, used)
        for f in [*families, "chains", "broken_chains", "two_cycles", "dag"]
    }

    def digest(x: torch.Tensor) -> str:
        return hashlib.sha256(x.numpy().tobytes()).hexdigest()

    return (
        train,
        validation,
        test_inputs,
        {
            "train_count": len(train_x),
            "validation_count": sum(len(x) for x, _ in validation.values()),
            "test_count": sum(len(x) for x in test_inputs.values()),
            "unique_adjacencies": len(used),
            "train_input_sha256": digest(train_x),
            "validation_input_sha256": {
                k: digest(x) for k, (x, _) in validation.items()
            },
            "test_input_sha256": {k: digest(x) for k, x in test_inputs.items()},
            "seeds": {"train": 41000, "validation": 42000, "test": 43000},
        },
    )


def make_classifier(seed: int, n: int = N) -> torch.nn.Sequential:
    """A conventional MLP with distance classes and a matching parameter budget."""
    inputs, outputs = n * n + n, n * (n + 1)
    state, hidden = n * n + 3 * n, 2 * n * n + 2 * n
    target_count = 2 * state * hidden + hidden + state + n * state + n
    width = round((target_count - outputs) / (inputs + outputs + 1))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return torch.nn.Sequential(
            torch.nn.Linear(inputs, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, outputs),
            torch.nn.Unflatten(-1, (n, n + 1)),
        )


def forward(model: torch.nn.Module, kind: str, x: torch.Tensor) -> torch.Tensor:
    """Present the same adjacency/source information in the architecture's format."""
    return model(x[:, : N * N + N] if kind == "classifier" else x)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, kind: str, suite: dict[str, Batch]) -> dict:
    """Whole-graph exact accuracy is primary; node accuracy is secondary."""
    model.eval()
    result = {}
    for family, (x, y) in suite.items():
        exact = node_correct = 0
        absolute_error = raw_error = 0.0
        for start in range(0, len(x), 512):
            truth = y[start : start + 512]
            output = forward(model, kind, x[start : start + 512])
            if kind == "classifier":
                predicted = output.argmax(-1)
            else:
                predicted = output.round().clamp(0, N).long()
                raw_error = max(raw_error, float((output - truth).abs().max()))
            correct = predicted == truth
            exact += int(correct.all(1).sum())
            node_correct += int(correct.sum())
            absolute_error += float((predicted - truth).abs().sum())
        result[family] = {
            "n": len(x),
            "exact_graphs": exact,
            "graph_accuracy": exact / len(x),
            "node_accuracy": node_correct / (len(x) * N),
            "distance_mae": absolute_error / (len(x) * N),
            "max_raw_error": None if kind == "classifier" else raw_error,
        }
    return result


def selection_key(scores: dict) -> tuple[int, float]:
    """Choose by correct graphs, then lower distance MAE, on validation only."""
    return sum(r["exact_graphs"] for r in scores.values()), -sum(
        r["distance_mae"] for r in scores.values()
    )


def train_one(
    kind: str,
    train: Batch,
    validation: dict[str, Batch],
    *,
    seed: int,
    lr: float,
    steps: int,
    batch_size: int,
    eval_every: int,
) -> tuple[dict, dict]:
    """Fit final outputs only, preserving the best independent validation checkpoint."""
    model = (
        make_classifier(seed)
        if kind == "classifier"
        else hg.make_shortest_paths(handwritten=False, seed=seed)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(44000 + seed)
    best_key = (-1, -math.inf)
    best_state: dict = {}
    best_step = 0
    history = []
    x, y = train
    started = time.perf_counter()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            scores = evaluate(model, kind, validation)
            key = selection_key(scores)
            history.append(
                {
                    "step": step,
                    "exact_graphs": key[0],
                    "n": sum(len(a) for a, _ in validation.values()),
                    "distance_mae": -key[1] / len(validation),
                }
            )
            if key > best_key:
                best_key, best_step = key, step
                best_state = copy.deepcopy(model.state_dict())
            LOGGER.info(
                "%s seed=%d lr=%g step=%d validation=%.4f",
                kind,
                seed,
                lr,
                step,
                key[0] / sum(len(a) for a, _ in validation.values()),
            )
        if step == steps:
            break
        model.train()
        indices = torch.randint(len(x), (batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        output = forward(model, kind, x[indices])
        loss = (
            torch.nn.functional.cross_entropy(
                output.flatten(0, 1), y[indices].flatten()
            )
            if kind == "classifier"
            else ((output - y[indices]) / N).square().mean()
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = lr * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step + 1) / steps))
            )
    return best_state, {
        "kind": kind,
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "training_draws": steps * batch_size,
        "selected_step": best_step,
        "validation_exact_graphs": best_key[0],
        "validation_tiebreak": best_key[1],
        "seconds": time.perf_counter() - started,
        "history": history,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def exhaustive_embedded_four() -> Batch:
    """All 4,096 loopless directed 4-node graphs, all 4 sources, inside 12 nodes."""
    vertices = [0, 4, 8, 11]
    edges = [(i, j) for i in vertices for j in vertices if i != j]
    graphs = torch.zeros(4096, N, N, dtype=torch.bool)
    masks = torch.arange(4096)
    for k, (i, j) in enumerate(edges):
        graphs[:, i, j] = ((masks >> k) & 1).bool()
    x = hg.encode_graphs(
        graphs.repeat_interleave(4, 0), torch.tensor(vertices).repeat(4096)
    )
    return x, targets(x)


def main(argv: list[str] | None = None) -> None:
    """Retain source snapshots, split hashes, all optimizer trials and test results."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recurrent-steps", type=int, default=3000)
    ap.add_argument("--classifier-steps", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.003])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--train-per-density", type=int, default=6552)
    ap.add_argument("--validation-per-density", type=int, default=256)
    ap.add_argument("--test-per-family", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    for name in (
        "recurrent_steps",
        "classifier_steps",
        "batch_size",
        "eval_every",
        "train_per_density",
        "validation_per_density",
        "test_per_family",
        "threads",
    ):
        if getattr(a, name) <= 0:
            ap.error(f"{name} must be positive")
    if len(set(a.seeds)) != len(a.seeds) or not a.seeds:
        ap.error("seeds must be distinct and nonempty")
    if any(not math.isfinite(lr) or lr <= 0 for lr in a.learning_rates):
        ap.error("learning rates must be positive and finite")
    a.out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    sources = a.out / "sources"
    sources.mkdir()
    source_hashes = {}
    for name, path in (
        ("model.py", Path(hg.__file__)),
        ("benchmark.py", Path(__file__)),
    ):
        data = path.read_bytes()
        (sources / name).write_bytes(data)
        source_hashes[name] = hashlib.sha256(data).hexdigest()
    started = time.perf_counter()
    handwritten = hg.make_shortest_paths()
    construction_seconds = time.perf_counter() - started
    save_model(handwritten, str(a.out / "handwritten.safetensors"))
    LOGGER.info("Generating disjoint datasets and BFS training/validation labels")
    train, validation, test_inputs, split_manifest = build_splits(
        a.train_per_density, a.validation_per_density, a.test_per_family
    )
    result = {
        "config": {
            **{k: v for k, v in vars(a).items() if k != "out"},
            "nodes": N,
            "densities": DENSITIES,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": "cpu",
            "source_sha256": source_hashes,
            "recurrent_loss": "MSE after dividing distances by 12",
            "classifier_loss": "13-class cross entropy per node",
            "selection": "validation whole-graph accuracy then lower MAE",
            "gradient_clip_norm": 1.0,
            "initialization": "recurrent Kaiming/Xavier; classifier PyTorch defaults",
        },
        "splits": split_manifest,
        "handwritten": {
            "parameters": sum(p.numel() for p in handwritten.parameters()),
            "nonzero_parameters": sum(
                int(p.count_nonzero()) for p in handwritten.parameters()
            ),
            "rounds": N - 1,
            "construction_seconds": construction_seconds,
            "optimizer_steps": 0,
        },
        "trials": [],
        "selected": [],
    }

    def save_progress() -> None:
        (a.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save_progress()
    for kind in ("recurrent", "classifier"):
        steps = a.recurrent_steps if kind == "recurrent" else a.classifier_steps
        for seed in a.seeds:
            best_state, best_trial = None, None
            for lr in a.learning_rates:
                state, trial = train_one(
                    kind,
                    train,
                    validation,
                    seed=seed,
                    lr=lr,
                    steps=steps,
                    batch_size=a.batch_size,
                    eval_every=a.eval_every,
                )
                result["trials"].append(trial)
                if best_trial is None or (
                    trial["validation_exact_graphs"],
                    trial["validation_tiebreak"],
                ) > (
                    best_trial["validation_exact_graphs"],
                    best_trial["validation_tiebreak"],
                ):
                    best_state, best_trial = state, trial
                save_progress()
            model = (
                make_classifier(seed)
                if kind == "classifier"
                else hg.make_shortest_paths(handwritten=False, seed=seed)
            )
            model.load_state_dict(best_state)
            name = f"{kind}_seed{seed}.safetensors"
            save_model(model, str(a.out / name))
            result["selected"].append(
                {
                    "kind": kind,
                    "seed": seed,
                    "lr": best_trial["lr"],
                    "step": best_trial["selected_step"],
                    "weights": name,
                    "weights_sha256": hashlib.sha256(
                        (a.out / name).read_bytes()
                    ).hexdigest(),
                }
            )
            save_progress()
    # Only now compute test labels and read any test performance.
    tests = {family: (x, targets(x)) for family, x in test_inputs.items()}
    result["handwritten"]["test"] = evaluate(handwritten, "recurrent", tests)
    result["handwritten"]["exhaustive_4node"] = evaluate(
        handwritten, "recurrent", {"embedded_four": exhaustive_embedded_four()}
    )
    save_file(
        {
            key: t
            for family, (x, y) in tests.items()
            for key, t in ((family + "_inputs", x), (family + "_targets", y))
        },
        str(a.out / "evaluation.safetensors"),
    )
    for selected in result["selected"]:
        kind = selected["kind"]
        model = (
            make_classifier(0)
            if kind == "classifier"
            else hg.make_shortest_paths(handwritten=False)
        )
        load_model(model, str(a.out / selected["weights"]))
        selected["test"] = evaluate(model, kind, tests)
        save_progress()
    save_progress()
    LOGGER.info("Complete: %s", a.out / "summary.json")


if __name__ == "__main__":
    main()
