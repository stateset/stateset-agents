"""Stronger learned control: graph recurrence with supervised edge messages.

This supplies substantially more algorithmic structure and supervision than
the original shallow classifier. A shared 2->32->32->1 MLP learns candidate
distance messages; masked minimum aggregation and sixteen rounds are fixed.
Training teaches messages d[i]+c[i,j] using exact intermediate training states.
No test labels or metrics are evaluated until all six training trials finish.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import torch
from handwritten_weighted_graph_job import SENTINEL, N, build_splits, labels
from safetensors.torch import load_file, save_file

LOGGER = logging.getLogger(__name__)


class LearnedRelaxation(torch.nn.Module):
    """Learn edge-message arithmetic inside a fixed graph recurrence."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.message = torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )

    def messages(self, costs: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """Predict every candidate message from distance and edge cost."""
        inputs = torch.stack((distances[:, :, None].expand_as(costs), costs), -1)
        return self.message(inputs).squeeze(-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict node costs using graph structure and the learned messages."""
        costs = inputs[:, : N * N].reshape(-1, N, N) / SENTINEL
        distance = inputs[:, N * N :] / SENTINEL
        present = (costs > 0) & (costs < 1)
        for _ in range(N - 1):
            candidates = self.messages(costs, distance).masked_fill(~present, 1)
            distance = torch.minimum(distance, candidates.amin(1)).clamp(0, 1)
        return distance * SENTINEL


def teacher_states(inputs: torch.Tensor) -> torch.Tensor:
    """Exact bounded Bellman–Ford states for intermediate training supervision."""
    costs = inputs[:, : N * N].reshape(-1, N, N)
    distance = inputs[:, N * N :]
    history = [distance]
    for _ in range(N - 1):
        distance = torch.minimum(distance, (distance[:, :, None] + costs).amin(1))
        history.append(distance)
    return torch.stack(history, 1)


@torch.inference_mode()
def evaluate(
    model: LearnedRelaxation, inputs: torch.Tensor, truth: torch.Tensor
) -> dict:
    """Report exact integer costs after rounding, together with raw float error."""
    outputs = torch.cat([model(batch) for batch in inputs.split(64)])
    predicted = outputs.round().long()
    correct = predicted == truth
    return {
        "n": len(inputs),
        "exact_graphs": int(correct.all(1).sum()),
        "graph_accuracy": float(correct.all(1).float().mean()),
        "node_accuracy": float(correct.float().mean()),
        "distance_mae": float((predicted - truth).abs().float().mean()),
        "raw_mae": float((outputs - truth).abs().mean()),
        "max_raw_error": float((outputs - truth).abs().max()),
    }


def train_trial(
    train: torch.Tensor,
    states: torch.Tensor,
    validation: tuple,
    seed: int,
    lr: float,
    steps: int,
    out: Path,
) -> dict:
    """Fit messages, selecting the checkpoint on validation graph accuracy only."""
    model = LearnedRelaxation(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(54000 + seed)
    best = (-1, -math.inf)
    best_state: dict = {}
    history = []
    selected_step = 0
    started = time.perf_counter()
    for step in range(steps + 1):
        if step % 500 == 0 or step == steps:
            metrics = evaluate(model, *validation)
            key = (metrics["exact_graphs"], -metrics["raw_mae"])
            history.append({"step": step, **metrics})
            if key > best:
                best, best_state, selected_step = (
                    key,
                    copy.deepcopy(model.state_dict()),
                    step,
                )
            LOGGER.info(
                "seed=%d lr=%g step=%d validation=%.4f raw_mae=%.5f",
                seed,
                lr,
                step,
                metrics["graph_accuracy"],
                metrics["raw_mae"],
            )
        if step == steps:
            break
        idx = torch.randint(len(train), (64,), generator=rng)
        rounds = torch.randint(N - 1, (64,), generator=rng)
        cost = train[idx, : N * N].reshape(-1, N, N) / SENTINEL
        distance = states[idx, rounds] / SENTINEL
        gold = distance[:, :, None] + cost
        loss = torch.nn.functional.mse_loss(model.messages(cost, distance), gold)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = lr * (
                0.1 + 0.45 * (1 + math.cos(math.pi * (step + 1) / steps))
            )
    checkpoint = f"seed{seed}_lr{lr}.safetensors"
    save_file(best_state, str(out / checkpoint))
    return {
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "training_graph_draws": steps * 64,
        "message_targets_per_draw": N * N,
        "selected_step": selected_step,
        "selection_key": best,
        "seconds": time.perf_counter() - started,
        "checkpoint": checkpoint,
        "history": history,
    }


def main() -> None:
    """Run the prespecified six trials and then evaluate all selected models."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    a = ap.parse_args()
    if a.steps < 1:
        ap.error("steps must be positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    protocol = {
        "seeds": [0, 1, 2],
        "learning_rates": [0.001, 0.003],
        "steps": a.steps,
        "batch_size": 64,
        "architecture": "shared 2->32->32->1 edge-message MLP; masked minimum, 16 rounds",
        "supervision": "all d[i]+cost[i,j] messages from a random exact intermediate training state",
        "selection": "validation exact graphs, then lower raw MAE",
        "rounding": "nearest integer for correctness; raw errors also reported",
        "test_metrics_after_all_trials": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    train, validation, tests, splits = build_splits(4096, 256, 512)
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    states = teacher_states(train[0])
    if not torch.equal(states[:, -1], train[1].float()):
        raise RuntimeError("intermediate teacher disagrees with Dijkstra")
    trials = []
    for seed in protocol["seeds"]:
        for lr in protocol["learning_rates"]:
            trials.append(
                train_trial(train[0], states, validation, seed, lr, a.steps, out)
            )
            (out / "trials.json").write_text(json.dumps(trials, indent=2))
    # Test labels and scores are first obtained after all training and selection.
    selected = [
        max((t for t in trials if t["seed"] == seed), key=lambda t: t["selection_key"])
        for seed in protocol["seeds"]
    ]
    truth = {family: labels(x) for family, x in tests.items()}
    results = []
    saved = {
        f"{family}.{key}": value
        for family, x in tests.items()
        for key, value in (("inputs", x), ("truth", truth[family]))
    }
    for trial in selected:
        model = LearnedRelaxation(trial["seed"])
        model.load_state_dict(load_file(str(out / trial["checkpoint"])))
        results.append(
            {
                "seed": trial["seed"],
                "lr": trial["lr"],
                "checkpoint": trial["checkpoint"],
                "families": {
                    family: evaluate(model, x, truth[family])
                    for family, x in tests.items()
                },
            }
        )
    save_file(saved, str(out / "evaluation.safetensors"))
    summary = {
        "protocol": protocol,
        "splits": splits,
        "trials": trials,
        "parameters": sum(p.numel() for p in LearnedRelaxation(0).parameters()),
        "selected": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir()
        if p.is_file()
    }
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    LOGGER.info("Complete: %s", json.dumps(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
