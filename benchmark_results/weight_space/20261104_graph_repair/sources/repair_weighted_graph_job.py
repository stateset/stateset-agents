"""Repair a learned graph message operator and compare with further training.

The handwritten edit replaces the entire learned addition operator through one
existing unit in each hidden layer. It keeps the architecture and graph forward
pass unchanged. The edit knows the required arithmetic; it does not discover it
from weights. Fresh graphs use known families and exclude all earlier splits.
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
from pathlib import Path

import handwritten_weighted_graph_job as graphs
import learned_weighted_graph_job as learned
import torch
from safetensors.torch import load_file, save_file

from stateset_agents.core.errors import ConfigurationError

LOGGER = logging.getLogger(__name__)
SHAPES = {
    "message.0.weight": (32, 2),
    "message.0.bias": (32,),
    "message.2.weight": (32, 32),
    "message.2.bias": (32,),
    "message.4.weight": (1, 32),
    "message.4.bias": (1,),
}
State = dict[str, torch.Tensor]


def addition_edit(state: State) -> tuple[State, list[dict]]:
    """Return a copy with a nonnegative addition path and a scalar edit ledger.

    This recipe supports exactly the earlier float32 CPU message MLP. The first
    selected unit computes ReLU(d+c), the next copies it, and the output reads
    only that unit. The learned side branches remain stored but are disconnected
    from the output. This changes 69 assigned coordinates out of 1,185.
    """
    if set(state) != set(SHAPES):
        raise ConfigurationError("expected the six message-MLP parameter tensors")
    for name, shape in SHAPES.items():
        value = state[name]
        if (
            tuple(value.shape) != shape
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())
        ):
            raise ConfigurationError(f"invalid float32 CPU parameter: {name}")
    edited = {k: v.detach().clone() for k, v in state.items()}
    edited["message.0.weight"][0] = 1
    edited["message.0.bias"][0] = 0
    edited["message.2.weight"][0] = 0
    edited["message.2.weight"][0, 0] = 1
    edited["message.2.bias"][0] = 0
    edited["message.4.weight"].zero_()
    edited["message.4.weight"][0, 0] = 1
    edited["message.4.bias"].zero_()
    # Unused branches are still evaluated by dense Linear modules. Bound their
    # intermediates over [0,1]^2 so zero output coefficients never multiply inf.
    bound = torch.ones(2, dtype=torch.float64)
    for layer in (0, 2, 4):
        bound = (
            edited[f"message.{layer}.weight"].double().abs() @ bound
            + edited[f"message.{layer}.bias"].double().abs()
        )
        if not bool((bound < torch.finfo(torch.float32).max / 2).all()):
            raise ConfigurationError("unused message branches can overflow float32")
    indices = {
        "message.0.weight": range(2),
        "message.0.bias": range(1),
        "message.2.weight": range(32),
        "message.2.bias": range(1),
        "message.4.weight": range(32),
        "message.4.bias": range(1),
    }
    ledger = [
        {
            "tensor": name,
            "index": index,
            "before": float(state[name].flatten()[index]),
            "after": float(edited[name].flatten()[index]),
        }
        for name, positions in indices.items()
        for index in positions
    ]
    return edited, ledger


def matched_control(state: State, ledger: list[dict], seed: int) -> State:
    """Use the same assigned coordinates and total delta norm, a random direction."""
    delta = torch.tensor(
        [r["after"] - r["before"] for r in ledger], dtype=torch.float64
    )
    rng = torch.Generator().manual_seed(seed)
    random = torch.randn(len(ledger), generator=rng, dtype=torch.float64)
    random *= delta.norm() / random.norm()
    control = {k: v.detach().clone() for k, v in state.items()}
    for row, change in zip(ledger, random, strict=True):
        control[row["tensor"]].flatten()[row["index"]] += change.float()
    return control


def from_state(state: State) -> learned.LearnedRelaxation:
    """Load parameters without changing the architecture or consuming global RNG."""
    model = learned.LearnedRelaxation(0)
    model.load_state_dict(state)
    return model.eval()


def component_grid() -> torch.Tensor:
    """Probe every integer and half-integer distance with each real edge cost."""
    distance = torch.arange(2 * graphs.SENTINEL + 1).float() / 2
    cost = torch.arange(1, graphs.MAX_COST + 1).float()
    d, c = torch.meshgrid(distance, cost, indexing="ij")
    return torch.stack((d.flatten(), c.flatten()), 1) / graphs.SENTINEL


@torch.inference_mode()
def message_error(model: learned.LearnedRelaxation) -> dict:
    """Measure addition error on a fixed component probe; not used for selection."""
    x = component_grid()
    error = (model.message(x).squeeze(-1) - x.sum(1)).abs()
    return {
        "n": len(x),
        "max_error": float(error.max()),
        "mean_error": float(error.mean()),
    }


def fit_more(
    initial: State,
    train: torch.Tensor,
    states: torch.Tensor,
    validation: tuple,
    *,
    regime: str,
    seed: int,
    lr: float,
    steps: int,
    out: Path,
) -> dict:
    """Continue Adam with either original graph states or full-range messages."""
    model = from_state(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(64000 + seed)
    best = (-1, -math.inf)
    best_state: State = {}
    selected_step = 0
    history = []
    started = time.perf_counter()
    for step in range(steps + 1):
        if step % 500 == 0 or step == steps:
            metrics = learned.evaluate(model, *validation)
            key = (metrics["exact_graphs"], -metrics["raw_mae"])
            history.append({"step": step, **metrics, "component": message_error(model)})
            if key > best:
                best, best_state, selected_step = (
                    key,
                    copy.deepcopy(model.state_dict()),
                    step,
                )
            LOGGER.info(
                "%s seed=%d lr=%g step=%d val=%.4f",
                regime,
                seed,
                lr,
                step,
                metrics["graph_accuracy"],
            )
        if step == steps:
            break
        if regime == "continued":
            idx = torch.randint(len(train), (64,), generator=rng)
            rounds = torch.randint(graphs.N - 1, (64,), generator=rng)
            c = (
                train[idx, : graphs.N**2].reshape(-1, graphs.N, graphs.N)
                / graphs.SENTINEL
            )
            d = states[idx, rounds] / graphs.SENTINEL
            x = torch.stack((d[:, :, None].expand_as(c), c), -1).reshape(-1, 2)
        elif regime == "coverage":
            # Same number of scalar message targets per step as the graph-state
            # regime, with continuous distances covering the whole valid range.
            count = 64 * graphs.N**2
            d = torch.rand(count, generator=rng)
            c = torch.randint(1, graphs.MAX_COST + 1, (count,), generator=rng).float()
            x = torch.stack((d, c / graphs.SENTINEL), 1)
        else:
            raise ConfigurationError(f"unknown training regime: {regime}")
        loss = torch.nn.functional.mse_loss(model.message(x).squeeze(-1), x.sum(1))
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
    checkpoint = f"{regime}_seed{seed}_lr{lr}.safetensors"
    save_file(best_state, str(out / checkpoint))
    return {
        "regime": regime,
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "message_targets": steps * 64 * graphs.N**2,
        "selected_step": selected_step,
        "selection_key": best,
        "checkpoint": checkpoint,
        "history": history,
        "seconds": time.perf_counter() - started,
    }


def matrix_keys(inputs: torch.Tensor) -> set[bytes]:
    """Identify adjacency matrices regardless of source or encoded missing edges."""
    costs = inputs[:, : graphs.N**2].reshape(-1, graphs.N, graphs.N).long()
    costs = torch.where(costs == graphs.SENTINEL, 0, costs)
    return {g.numpy().tobytes() for g in costs}


def digest(path: Path) -> str:
    """SHA-256 of one retained artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Freeze the protocol, complete training, then draw and score fresh graphs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--test-per-family", type=int, default=512)
    a = ap.parse_args()
    if a.steps < 1 or a.test_per_family < 1:
        ap.error("steps and test-per-family must be positive")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a.out.mkdir(parents=True, exist_ok=False)
    old = json.loads((a.initial / "summary.json").read_text())
    old_manifest = json.loads((a.initial / "artifact_manifest.json").read_text())
    initial = {}
    for selected in old["selected"]:
        path = a.initial / selected["checkpoint"]
        if digest(path) != old_manifest[path.name]:
            raise RuntimeError(f"initial checkpoint hash differs: {path.name}")
        initial[selected["seed"]] = load_file(str(path))
    if set(initial) != {0, 1, 2}:
        raise ConfigurationError("expected the three previously selected seeds")
    sources = {
        Path(__file__).name: Path(__file__),
        Path(learned.__file__).name: Path(learned.__file__),
        Path(graphs.__file__).name: Path(graphs.__file__),
        "handwritten_weighted_graph.py": Path(graphs.hw.__file__),
    }
    (a.out / "sources").mkdir()
    for name, path in sources.items():
        (a.out / "sources" / name).write_bytes(path.read_bytes())
    families = [f"er_{p}" for p in graphs.DENSITIES] + [
        "max_cost_chain",
        "broken_chain",
        "cheap_detours",
        "two_cycles",
    ]
    protocol = {
        "seeds": [0, 1, 2],
        "learning_rates": [0.001, 0.003],
        "steps": a.steps,
        "regimes": ["continued", "coverage"],
        "messages_per_step": 64 * graphs.N**2,
        "selection": "old random-graph validation exact graphs, then lower raw MAE; includes step zero",
        "edit": "unit zero in each hidden layer: ReLU(d+c), identity, exclusive output readout",
        "assigned_coordinates": 69,
        "parameters": 1185,
        "control": "same 69 coordinates and total delta L2, random direction; seed 65000+model seed",
        "fresh_test_seed": 66000,
        "fresh_test_per_family": a.test_per_family,
        "families": families,
        "test_generation_after_all_training": True,
        "test_excludes_previous_train_validation_and_test_matrices": True,
        "known_family_followup_not_unseen_task": True,
        "rounding": "nearest integer; raw errors retained",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu",
        "source_sha256": {name: digest(path) for name, path in sources.items()},
        "initial_checkpoints": [
            {
                "seed": s["seed"],
                "file": s["checkpoint"],
                "sha256": digest(a.initial / s["checkpoint"]),
            }
            for s in old["selected"]
        ],
    }
    (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    train, validation, old_tests, old_splits = graphs.build_splits(4096, 256, 512)
    if old_splits != old["splits"]:
        raise RuntimeError("original data splits did not reproduce")
    states = learned.teacher_states(train[0])
    if not torch.equal(states[:, -1], train[1].float()):
        raise RuntimeError("intermediate states differ from Dijkstra")
    variants = []
    ledgers = {}
    for seed, state in initial.items():
        edited, ledger = addition_edit(state)
        ledgers[str(seed)] = ledger
        for name, weights in (
            ("original", state),
            ("edited", edited),
            ("control", matched_control(state, ledger, 65000 + seed)),
        ):
            file = f"{name}_seed{seed}.safetensors"
            save_file(weights, str(a.out / file))
            variants.append({"name": name, "seed": seed, "checkpoint": file})
    (a.out / "edit_ledger.json").write_text(json.dumps(ledgers, indent=2) + "\n")
    trials = []
    for regime in protocol["regimes"]:
        for seed in protocol["seeds"]:
            for lr in protocol["learning_rates"]:
                trials.append(
                    fit_more(
                        initial[seed],
                        train[0],
                        states,
                        validation,
                        regime=regime,
                        seed=seed,
                        lr=lr,
                        steps=a.steps,
                        out=a.out,
                    )
                )
                (a.out / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")
            trial = max(
                (t for t in trials if t["regime"] == regime and t["seed"] == seed),
                key=lambda t: t["selection_key"],
            )
            variants.append(
                {
                    "name": regime,
                    "seed": seed,
                    "checkpoint": trial["checkpoint"],
                    "selected_lr": trial["lr"],
                    "selected_step": trial["selected_step"],
                }
            )
    # No new test graphs, oracle labels, or test metrics existed before this point.
    previous = matrix_keys(train[0]) | matrix_keys(validation[0])
    for x in old_tests.values():
        previous |= matrix_keys(x)
    used = set(previous)
    rng = torch.Generator().manual_seed(protocol["fresh_test_seed"])
    tests = {f: graphs.unique_inputs(f, a.test_per_family, rng, used) for f in families}
    truth = {f: graphs.labels(x) for f, x in tests.items()}
    fresh_keys = set().union(*(matrix_keys(x) for x in tests.values()))
    if fresh_keys & previous or len(fresh_keys) != len(families) * a.test_per_family:
        raise RuntimeError("fresh graph leakage or duplication")
    splits = {
        "original": old_splits,
        "excluded_matrices": len(previous),
        "fresh_matrices": len(fresh_keys),
        "overlap": 0,
        "fresh_sha256": {
            f: hashlib.sha256(x.numpy().tobytes()).hexdigest() for f, x in tests.items()
        },
    }
    # This exhaustive subset is a separate known regression suite, not fresh data.
    tests["exhaustive_three"], truth["exhaustive_three"] = graphs.exhaustive_three()
    save_file(
        {
            f"{f}.{k}": v
            for f, x in tests.items()
            for k, v in (("inputs", x), ("truth", truth[f]))
        },
        str(a.out / "evaluation.safetensors"),
    )
    for variant in variants:
        model = from_state(load_file(str(a.out / variant["checkpoint"])))
        variant["component"] = message_error(model)
        variant["families"] = {
            f: learned.evaluate(model, x, truth[f]) for f, x in tests.items()
        }
        if variant["name"] == "edited":
            ledger = ledgers[str(variant["seed"])]
            variant["changed_parameters"] = sum(
                r["before"] != r["after"] for r in ledger
            )
            variant["assigned_parameters"] = len(ledger)
        LOGGER.info(
            "test %s seed=%d exact=%d/%d",
            variant["name"],
            variant["seed"],
            sum(m["exact_graphs"] for m in variant["families"].values()),
            sum(m["n"] for m in variant["families"].values()),
        )
    result = {
        "protocol": protocol,
        "splits": splits,
        "trials": trials,
        "variants": variants,
    }
    (a.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        str(p.relative_to(a.out)): digest(p)
        for p in sorted(a.out.rglob("*"))
        if p.is_file()
    }
    (a.out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
