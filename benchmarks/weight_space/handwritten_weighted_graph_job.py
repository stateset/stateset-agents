"""Evaluate a handwritten 2M weighted shortest-path network against a trained MLP.

The baseline is a conventional, parameter-matched classifier. All budgets and
families are specified before testing. Weighted adjacency matrices are disjoint
across training, validation and test, independently of their starting nodes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import logging
import math
import platform
import time
from pathlib import Path

import torch
from safetensors.torch import load_model, save_file, save_model

from stateset_agents.experimental.weight_space import handwritten_weighted_graph as hw

LOGGER = logging.getLogger(__name__)
N, MAX_COST, SENTINEL = 17, 7, 113
DENSITIES = (0.05, 0.12, 0.25, 0.5)
Batch = tuple[torch.Tensor, torch.Tensor]


def dijkstra(costs: list[list[int]], source: int) -> list[int]:
    """Independent priority-queue oracle; zero entries mean absent edges."""
    distance = [SENTINEL] * len(costs)
    distance[source] = 0
    queue = [(0, source)]
    while queue:
        cost, i = heapq.heappop(queue)
        if cost != distance[i]:
            continue
        for j, edge in enumerate(costs[i]):
            candidate = cost + edge
            if edge and candidate < distance[j]:
                distance[j] = candidate
                heapq.heappush(queue, (candidate, j))
    return distance


def labels(inputs: torch.Tensor) -> torch.Tensor:
    """Recover costs/source and label them without calling a neural model."""
    costs = inputs[:, : N * N].reshape(-1, N, N).long()
    costs = torch.where(costs == SENTINEL, 0, costs)
    roots = inputs[:, N * N :].argmin(1).tolist()
    return torch.tensor(
        [dijkstra(g, s) for g, s in zip(costs.tolist(), roots, strict=True)]
    )


def sample(
    family: str, count: int, rng: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample prespecified random graphs and structured distribution shifts."""
    costs = torch.zeros(count, N, N, dtype=torch.int64)
    roots = torch.randint(N, (count,), generator=rng)
    if family.startswith("er_"):
        edges = torch.rand(count, N, N, generator=rng) < float(family[3:])
        costs = edges * torch.randint(1, MAX_COST + 1, (count, N, N), generator=rng)
        costs[:, torch.arange(N), torch.arange(N)] = 0
        return costs, roots
    for k in range(count):
        order = torch.randperm(N, generator=rng)
        roots[k] = order[0]
        if family in ("max_cost_chain", "broken_chain"):
            costs[k, order[:-1], order[1:]] = MAX_COST
            if family == "broken_chain":
                cut = int(torch.randint(N - 1, (1,), generator=rng))
                costs[k, order[cut], order[cut + 1]] = 0
        elif family == "cheap_detours":
            # A direct edge costs 7; up to six unit-cost edges are cheaper.
            costs[k, order[:-1], order[1:]] = 1
            costs[k, order[0], order[2:]] = MAX_COST
        elif family == "two_cycles":
            split = int(torch.randint(3, N - 2, (1,), generator=rng))
            for group in (order[:split], order[split:]):
                costs[k, group, group.roll(-1)] = torch.randint(
                    1, MAX_COST + 1, (len(group),), generator=rng
                )
        else:
            raise ValueError(f"unknown graph family: {family}")
    return costs, roots


def unique_inputs(
    family: str, count: int, rng: torch.Generator, used: set[bytes]
) -> torch.Tensor:
    """Reject repeated cost matrices regardless of source, across every split."""
    graphs, roots = [], []
    attempts = 0
    while len(graphs) < count:
        costs, sources = sample(family, max(64, count - len(graphs)), rng)
        for graph, source in zip(costs, sources, strict=True):
            attempts += 1
            if attempts > max(10000, 100 * count):
                raise RuntimeError("not enough distinct weighted graphs")
            key = graph.numpy().tobytes()
            if key in used:
                continue
            used.add(key)
            graphs.append(graph)
            roots.append(source)
            if len(graphs) == count:
                break
    return hw.encode_weighted_graphs(torch.stack(graphs), torch.stack(roots))


def build_splits(
    train_count: int, val_count: int, test_count: int
) -> tuple[Batch, Batch, dict[str, torch.Tensor], dict]:
    """Use independent RNG streams; leave test labeling until all training ends."""
    used: set[bytes] = set()
    families = [f"er_{p}" for p in DENSITIES]
    sets = []
    for seed, count in ((51000, train_count), (52000, val_count)):
        rng = torch.Generator().manual_seed(seed)
        x = torch.cat([unique_inputs(f, count, rng, used) for f in families])
        sets.append((x, labels(x)))
    rng = torch.Generator().manual_seed(53000)
    test = {
        f: unique_inputs(f, test_count, rng, used)
        for f in [
            *families,
            "max_cost_chain",
            "broken_chain",
            "cheap_detours",
            "two_cycles",
        ]
    }

    def digest(x: torch.Tensor) -> str:
        return hashlib.sha256(x.numpy().tobytes()).hexdigest()

    return (
        sets[0],
        sets[1],
        test,
        {
            "train_count": len(sets[0][0]),
            "validation_count": len(sets[1][0]),
            "test_count": sum(len(x) for x in test.values()),
            "unique_cost_matrices": len(used),
            "train_sha256": digest(sets[0][0]),
            "validation_sha256": digest(sets[1][0]),
            "test_sha256": {k: digest(x) for k, x in test.items()},
            "seeds": {"train": 51000, "validation": 52000, "test": 53000},
        },
    )


def classifier(seed: int) -> torch.nn.Sequential:
    """A 1,999,988-parameter conventional MLP with 114 cost classes per node."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return torch.nn.Sequential(
            torch.nn.Linear(306, 890),
            torch.nn.ReLU(),
            torch.nn.Linear(890, 1938),
            torch.nn.Unflatten(-1, (N, SENTINEL + 1)),
        )


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module, inputs: torch.Tensor, truth: torch.Tensor, *, trained: bool
) -> dict:
    """Count exact graphs, exact node costs, and raw errors before rounding."""
    model.eval()
    exact = nodes = 0
    raw_error = mae = 0.0
    for start in range(0, len(inputs), 64):
        x, y = inputs[start : start + 64], truth[start : start + 64]
        output = model(x / SENTINEL if trained else x)
        if trained:
            predicted = output.argmax(-1)
        else:
            predicted = output.round().long()
            raw_error = max(raw_error, float((output - y).abs().max()))
        correct = predicted == y
        exact += int(correct.all(1).sum())
        nodes += int(correct.sum())
        mae += float((predicted - y).abs().sum())
    return {
        "n": len(inputs),
        "exact_graphs": exact,
        "graph_accuracy": exact / len(inputs),
        "node_accuracy": nodes / (N * len(inputs)),
        "distance_mae": mae / (N * len(inputs)),
        "max_raw_error": None if trained else raw_error,
    }


def train_one(
    train: Batch,
    validation: Batch,
    *,
    seed: int,
    lr: float,
    steps: int,
    batch_size: int,
    eval_every: int,
) -> tuple[dict, dict]:
    """Fit final costs with Adam and select checkpoints using validation only."""
    model = classifier(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(54000 + seed)
    best = (-1, -math.inf)
    state = {}
    best_step = 0
    history = []
    started = time.perf_counter()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            metrics = evaluate(model, *validation, trained=True)
            key = (metrics["exact_graphs"], -metrics["distance_mae"])
            history.append({"step": step, **metrics})
            if key > best:
                best, best_step, state = key, step, copy.deepcopy(model.state_dict())
            LOGGER.info(
                "seed=%d lr=%g step=%d validation=%.4f",
                seed,
                lr,
                step,
                metrics["graph_accuracy"],
            )
        if step == steps:
            break
        model.train()
        idx = torch.randint(len(train[0]), (batch_size,), generator=rng)
        output = model(train[0][idx] / SENTINEL)
        loss = torch.nn.functional.cross_entropy(
            output.flatten(0, 1), train[1][idx].flatten()
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = lr * (
                0.1 + 0.45 * (1 + math.cos(math.pi * (step + 1) / steps))
            )
    return state, {
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "training_draws": steps * batch_size,
        "selected_step": best_step,
        "selection_key": best,
        "seconds": time.perf_counter() - started,
        "history": history,
    }


def exhaustive_three() -> Batch:
    """All 729 directed 3-node cost matrices with costs 0,1,2 and all sources."""
    vertices = [0, 8, 16]
    edges = [(i, j) for i in vertices for j in vertices if i != j]
    graphs = torch.zeros(729, N, N, dtype=torch.int64)
    codes = torch.arange(729)
    for k, (i, j) in enumerate(edges):
        graphs[:, i, j] = (codes // (3**k)) % 3
    x = hw.encode_weighted_graphs(
        graphs.repeat_interleave(3, 0), torch.tensor(vertices).repeat(729)
    )
    return x, labels(x)


def main(argv: list[str] | None = None) -> None:
    """Preserve exact sources, split hashes, every trial, and selected checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--train-per-density", type=int, default=4096)
    parser.add_argument("--validation-per-density", type=int, default=256)
    parser.add_argument("--test-per-family", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[0.001, 0.003]
    )
    parser.add_argument("--threads", type=int, default=2)
    a = parser.parse_args(argv)
    for name in (
        "steps",
        "batch_size",
        "eval_every",
        "train_per_density",
        "validation_per_density",
        "test_per_family",
        "threads",
    ):
        if getattr(a, name) <= 0:
            parser.error(f"{name} must be positive")
    if len(set(a.seeds)) != len(a.seeds):
        parser.error("seeds must be distinct")
    if any(not math.isfinite(lr) or lr <= 0 for lr in a.learning_rates):
        parser.error("learning rates must be positive and finite")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "sources").mkdir()
    hashes = {}
    for name, path in [
        ("model.py", Path(hw.__file__)),
        ("benchmark.py", Path(__file__)),
    ]:
        data = path.read_bytes()
        (a.out / "sources" / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    model = hw.make_weighted_shortest_paths()
    save_model(model, str(a.out / "handwritten.safetensors"))
    train, validation, test_inputs, splits = build_splits(
        a.train_per_density, a.validation_per_density, a.test_per_family
    )
    result = {
        "config": {
            **{k: v for k, v in vars(a).items() if k != "out"},
            "nodes": N,
            "max_cost": MAX_COST,
            "unreachable": SENTINEL,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": "cpu",
            "source_sha256": hashes,
            "selection": "validation exact graphs, then lower cost MAE",
            "loss": "114-class cross entropy per node",
            "optimizer": "Adam, gradient norm clip 1, cosine LR decay to 0.1 initial",
        },
        "splits": splits,
        "handwritten": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "nonzero_parameters": sum(
                int(p.count_nonzero()) for p in model.parameters()
            ),
            "natural_parameters": 1865053,
            "split_neurons": 137,
            "optimizer_steps": 0,
        },
        "baseline_parameters": sum(p.numel() for p in classifier(0).parameters()),
        "trials": [],
        "selected": [],
    }

    def save() -> None:
        (a.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for seed in a.seeds:
        best_trial = None
        best_state = None
        for lr in a.learning_rates:
            state, trial = train_one(
                train,
                validation,
                seed=seed,
                lr=lr,
                steps=a.steps,
                batch_size=a.batch_size,
                eval_every=a.eval_every,
            )
            result["trials"].append(trial)
            if (
                best_trial is None
                or trial["selection_key"] > best_trial["selection_key"]
            ):
                best_state, best_trial = state, trial
            save()
        fitted = classifier(seed)
        fitted.load_state_dict(best_state)
        name = f"classifier_seed{seed}.safetensors"
        save_model(fitted, str(a.out / name))
        result["selected"].append(
            {
                "seed": seed,
                "lr": best_trial["lr"],
                "step": best_trial["selected_step"],
                "weights": name,
                "weights_sha256": hashlib.sha256(
                    (a.out / name).read_bytes()
                ).hexdigest(),
            }
        )
        save()
    tests = {f: (x, labels(x)) for f, x in test_inputs.items()}
    save_file(
        {
            key: t
            for f, (x, y) in tests.items()
            for key, t in ((f + "_inputs", x), (f + "_targets", y))
        },
        str(a.out / "evaluation.safetensors"),
    )
    result["handwritten"]["test"] = {
        f: evaluate(model, x, y, trained=False) for f, (x, y) in tests.items()
    }
    result["handwritten"]["exhaustive_3node"] = evaluate(
        model, *exhaustive_three(), trained=False
    )
    for selected in result["selected"]:
        fitted = classifier(0)
        load_model(fitted, str(a.out / selected["weights"]))
        selected["test"] = {
            f: evaluate(fitted, x, y, trained=True) for f, (x, y) in tests.items()
        }
    save()
    LOGGER.info("Complete: %s", a.out / "summary.json")


if __name__ == "__main__":
    main()
