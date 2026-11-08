"""Dense trained-model edits with sequential preservation and fixed controls.

This CPU experiment edits a learned MLP readout on supplied activation keys.
It measures locality outside the protected span; it is not an LLM knowledge
editing benchmark. Every seed, round and control is retained.
"""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from benchmarks.weight_space.relu_compiler_job import digest, write_json
from stateset_agents.core.errors import ConfigurationError
from stateset_agents.experimental.weight_space import constrained_edit

LOGGER = logging.getLogger(__name__)
SEEDS = (0, 1, 2, 3, 4)
WIDTH = 32
STEPS = 500
ARMS = ("constrained", "unconstrained", "matched_random")
SHIFTS = ((0.25, -0.125, 0.0625), (-0.125, 0.25, -0.0625))


def task(x: torch.Tensor) -> torch.Tensor:
    """Fixed regression task shared by all training and held-out evaluations."""
    a, b, c, d = x.unbind(1)
    return torch.stack((a.sin() + b / 2, c.cos() - d / 4, a * d / 2 - c), dim=1)


def run(out: Path) -> dict:
    """Freeze the protocol before training, then retain all model states and metrics."""
    if out.exists():
        raise ValueError("use a fresh output directory")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parents[2]
    sources = {}
    for path in (
        Path(__file__),
        Path(__file__).with_name("verify_constrained_edit.py"),
        Path(__file__).with_name("relu_compiler_job.py"),
        repo / "stateset_agents/experimental/weight_space/constrained.py",
        repo / "stateset_agents/experimental/weight_space/deltas.py",
    ):
        target = out / "sources" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        sources[path.name] = digest(target)
    protocol = {
        "kind": "stateset-constrained-linear-v1",
        "seeds": SEEDS,
        "architecture": [4, WIDTH, 3],
        "training_steps": STEPS,
        "learning_rate": 0.01,
        "train_cases": 512,
        "reference_cases": 8,
        "keys_per_round": 4,
        "rounds": 2,
        "heldout_cases": 1024,
        "arms": ARMS,
        "shifts": SHIFTS,
        "edit_optimizer_steps": 0,
        "target_atol": 1e-5,
        "target_rtol": 1e-5,
        "preservation_atol": 1e-5,
        "sequential_atol": 2e-5,
        "rcond": 1e-10,
        "input_seed": 92180,
        "sources": sources,
        "scope": "linear readout activation span; no unseen-prompt guarantee",
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": 1,
        },
    }
    write_json(out / "protocol.json", protocol)
    g = torch.Generator().manual_seed(protocol["input_seed"])
    inputs = {
        name: torch.rand(n, 4, generator=g) * 4 - 2
        for name, n in (
            ("train", 512),
            ("reference", 8),
            ("keys", 8),
            ("heldout", 1024),
        )
    }
    save_file(inputs, str(out / "inputs.safetensors"))
    states = {}
    trials, training, rejections = [], [], []
    for seed in SEEDS:
        with torch.random.fork_rng():
            torch.manual_seed(100 + seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, WIDTH), torch.nn.ReLU(), torch.nn.Linear(WIDTH, 3)
            )
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        losses = []
        y = task(inputs["train"])
        for step in range(STEPS + 1):
            loss = (model(inputs["train"]) - y).square().mean()
            if step % 100 == 0:
                losses.append({"step": step, "mse": float(loss.detach())})
            if step < STEPS:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
        model.eval()
        base = {name: p.detach().clone() for name, p in model.state_dict().items()}
        for name, p in base.items():
            states[f"s{seed}.base.{name}"] = p
        with torch.no_grad():
            features = {name: model[:2](x) for name, x in inputs.items()}
            baseline = {name: model(x) for name, x in inputs.items()}
        training.append(
            {
                "seed": seed,
                "losses": losses,
                "heldout_mse": float(
                    (baseline["heldout"] - task(inputs["heldout"])).square().mean()
                ),
            }
        )
        all_targets = baseline["keys"] + torch.tensor(SHIFTS).repeat_interleave(
            4, dim=0
        )
        weights = {arm: base["2.weight"].clone() for arm in ARMS}
        for round_index in range(2):
            start, stop = round_index * 4, (round_index + 1) * 4
            keys = features["keys"][start:stop]
            refs = torch.cat((features["reference"], features["keys"][:start]))
            result = constrained_edit(
                weights["constrained"],
                keys,
                all_targets[start:stop],
                bias=base["2.bias"],
                protect=refs,
            )
            for arm in ARMS:
                current = weights[arm]
                if arm == "constrained":
                    d = result.delta.dense()
                elif arm == "unconstrained":
                    d = constrained_edit(
                        current, keys, all_targets[start:stop], bias=base["2.bias"]
                    ).delta.dense()
                else:
                    rg = torch.Generator().manual_seed(4100 + seed * 2 + round_index)
                    d = torch.randn(3, 4, generator=rg) @ torch.randn(
                        4, WIDTH, generator=rg
                    )
                    d *= result.delta.dense().norm() / d.norm()
                updated = current + d
                weights[arm] = updated
                prefix = f"s{seed}.r{round_index}.{arm}"
                states[f"{prefix}.weight"] = updated.clone()
                states[f"{prefix}.delta"] = d.clone()
                values = {
                    name: f @ updated.T + base["2.bias"] for name, f in features.items()
                }
                trial = {
                    "seed": seed,
                    "round": round_index,
                    "arm": arm,
                    "target_max_error": float(
                        (values["keys"][start:stop] - all_targets[start:stop])
                        .abs()
                        .max()
                    ),
                    "all_targets_max_error": float(
                        (values["keys"][:stop] - all_targets[:stop]).abs().max()
                    ),
                    "reference_max_drift": float(
                        (values["reference"] - baseline["reference"]).abs().max()
                    ),
                    "heldout_max_drift": float(
                        (values["heldout"] - baseline["heldout"]).abs().max()
                    ),
                    "heldout_task_mse": float(
                        (values["heldout"] - task(inputs["heldout"])).square().mean()
                    ),
                    "update_norm": float(d.norm()),
                }
                if arm == "constrained":
                    trial["diagnostics"] = asdict(result.diagnostics)
                trials.append(trial)
        try:
            constrained_edit(
                base["2.weight"],
                features["keys"][:4],
                all_targets[:4],
                bias=base["2.bias"],
                protect=torch.eye(WIDTH),
            )
        except ConfigurationError as exc:
            rejections.append({"seed": seed, "reason": str(exc)})
        else:
            raise AssertionError("fully protected control was accepted")
    save_file(states, str(out / "states.safetensors"))
    write_json(out / "trials.json", trials)
    write_json(out / "training.json", training)
    summary = {
        "models": len(SEEDS),
        "trials": len(trials),
        "full_span_rejections": rejections,
        "by_arm": {
            arm: {
                "target_max_error": max(
                    t["target_max_error"] for t in trials if t["arm"] == arm
                ),
                "all_targets_max_error": max(
                    t["all_targets_max_error"] for t in trials if t["arm"] == arm
                ),
                "reference_max_drift": max(
                    t["reference_max_drift"] for t in trials if t["arm"] == arm
                ),
                "heldout_max_drift": max(
                    t["heldout_max_drift"] for t in trials if t["arm"] == arm
                ),
                "final_mean_task_mse": sum(
                    t["heldout_task_mse"]
                    for t in trials
                    if t["arm"] == arm and t["round"] == 1
                )
                / len(SEEDS),
            }
            for arm in ARMS
        },
        "baseline_mean_task_mse": sum(t["heldout_mse"] for t in training) / len(SEEDS),
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "artifact_manifest.json",
        {
            str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("%s", run(args.out))
