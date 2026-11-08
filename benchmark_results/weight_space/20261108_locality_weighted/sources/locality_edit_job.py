"""Fixed locality-weighted follow-up using the retained dense-model experiment.

Reuse the same trained models and edit/reference inputs. The locality design
set and confirmation set are fresh, disjoint, and generated with separate seeds.
There is one predeclared strength, no held-out selection, and no new training.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from benchmarks.weight_space.constrained_edit_job import task
from benchmarks.weight_space.relu_compiler_job import digest, write_json
from benchmarks.weight_space.verify_constrained_edit import verify as verify_base
from stateset_agents.experimental.weight_space import constrained_edit

LOGGER = logging.getLogger(__name__)
ARMS = ("minimum_norm", "locality_weighted", "unconstrained")
LOCALITY_WEIGHT = 100.0


@torch.inference_mode()
def run(base: Path, out: Path) -> dict:
    """Freeze a protocol and compare every arm, seed and sequential edit round."""
    if out.exists():
        raise ValueError("use a fresh output directory")
    verified_base = verify_base(base)
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    # Retain a complete audited dependency bundle, including its original
    # manifest. No earlier source snapshot or result is rewritten.
    base_manifest = json.loads((base / "artifact_manifest.json").read_text())
    for name in (*base_manifest, "artifact_manifest.json"):
        target = out / "base" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / name, target)
    repo = Path(__file__).resolve().parents[2]
    sources = {}
    for path in (
        Path(__file__),
        Path(__file__).with_name("verify_locality_edit.py"),
        Path(__file__).with_name("constrained_edit_job.py"),
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
        "kind": "stateset-locality-linear-v1",
        "seeds": list(range(5)),
        "arms": ARMS,
        "rounds": 2,
        "keys_per_round": 4,
        "locality_weight": LOCALITY_WEIGHT,
        "locality_cases": 256,
        "locality_seed": 92181,
        "confirmation_cases": 4096,
        "confirmation_seed": 92183,
        "new_training_steps": 0,
        "edit_optimizer_steps": 0,
        "target_atol": 1e-5,
        "target_rtol": 1e-5,
        "preservation_atol": 1e-5,
        "sequential_atol": 2e-5,
        "rcond": 1e-10,
        "primary_metric": "mean final confirmation MSE of edited-minus-base outputs",
        "selection": "fixed strength; no hyperparameter or checkpoint selection",
        "base_manifest_sha256": digest(out / "base/artifact_manifest.json"),
        "base_verification": verified_base,
        "sources": sources,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": 1,
        },
    }
    write_json(out / "protocol.json", protocol)
    old_inputs = load_file(str(out / "base/inputs.safetensors"))
    base_states = load_file(str(out / "base/states.safetensors"))
    inputs = {
        "locality": torch.rand(256, 4, generator=torch.Generator().manual_seed(92181))
        * 4
        - 2,
        "confirmation": torch.rand(
            4096, 4, generator=torch.Generator().manual_seed(92183)
        )
        * 4
        - 2,
    }
    save_file(inputs, str(out / "inputs.safetensors"))
    trials, states, baseline_mses = [], {}, []
    for seed in range(5):
        hw, hb, bw, bb = (
            base_states[f"s{seed}.base.{name}"]
            for name in ("0.weight", "0.bias", "2.weight", "2.bias")
        )
        features = {
            name: F.relu(F.linear(x, hw, hb))
            for name, x in {
                "reference": old_inputs["reference"],
                "keys": old_inputs["keys"],
                **inputs,
            }.items()
        }
        original = {name: F.linear(f, bw, bb) for name, f in features.items()}
        targets = original["keys"] + torch.tensor(
            [[0.25, -0.125, 0.0625], [-0.125, 0.25, -0.0625]]
        ).repeat_interleave(4, 0)
        baseline_mses.append(
            float(
                (original["confirmation"] - task(inputs["confirmation"]))
                .square()
                .mean()
            )
        )
        weights = {arm: bw.clone() for arm in ARMS}
        for round_index in range(2):
            start, stop = round_index * 4, (round_index + 1) * 4
            refs = torch.cat((features["reference"], features["keys"][:start]))
            for arm in ARMS:
                current = weights[arm]
                result = constrained_edit(
                    current,
                    features["keys"][start:stop],
                    targets[start:stop],
                    bias=bb,
                    protect=None if arm == "unconstrained" else refs,
                    locality=features["locality"],
                    locality_weight=(
                        LOCALITY_WEIGHT if arm == "locality_weighted" else 0
                    ),
                )
                d = result.delta.dense()
                updated = current + d
                weights[arm] = updated
                prefix = f"s{seed}.r{round_index}.{arm}"
                states[f"{prefix}.weight"] = updated.clone()
                states[f"{prefix}.delta"] = d.clone()
                values = {
                    name: F.linear(f, updated, bb) for name, f in features.items()
                }
                trials.append(
                    {
                        "seed": seed,
                        "round": round_index,
                        "arm": arm,
                        "target_max_error": float(
                            (values["keys"][start:stop] - targets[start:stop])
                            .abs()
                            .max()
                        ),
                        "all_targets_max_error": float(
                            (values["keys"][:stop] - targets[:stop]).abs().max()
                        ),
                        "reference_max_drift": float(
                            (values["reference"] - original["reference"]).abs().max()
                        ),
                        "locality_mse_drift": float(
                            (values["locality"] - original["locality"]).square().mean()
                        ),
                        "confirmation_mse_drift": float(
                            (values["confirmation"] - original["confirmation"])
                            .square()
                            .mean()
                        ),
                        "confirmation_max_drift": float(
                            (values["confirmation"] - original["confirmation"])
                            .abs()
                            .max()
                        ),
                        "confirmation_task_mse": float(
                            (values["confirmation"] - task(inputs["confirmation"]))
                            .square()
                            .mean()
                        ),
                        "diagnostics": asdict(result.diagnostics),
                    }
                )
    save_file(states, str(out / "states.safetensors"))
    write_json(out / "trials.json", trials)
    summary = {
        "models": 5,
        "trials": 30,
        "baseline_mean_confirmation_task_mse": sum(baseline_mses) / 5,
        "by_arm": {
            arm: {
                "all_targets_max_error": max(
                    t["all_targets_max_error"] for t in trials if t["arm"] == arm
                ),
                "reference_max_drift": max(
                    t["reference_max_drift"] for t in trials if t["arm"] == arm
                ),
                "confirmation_max_drift": max(
                    t["confirmation_max_drift"] for t in trials if t["arm"] == arm
                ),
                **{
                    f"final_mean_{metric}": sum(
                        t[metric] for t in trials if t["arm"] == arm and t["round"] == 1
                    )
                    / 5
                    for metric in (
                        "locality_mse_drift",
                        "confirmation_mse_drift",
                        "confirmation_task_mse",
                    )
                },
            }
            for arm in ARMS
        },
        "paired_final_confirmation_mse": [
            {
                "seed": seed,
                **{
                    arm: next(
                        t["confirmation_mse_drift"]
                        for t in trials
                        if t["seed"] == seed and t["round"] == 1 and t["arm"] == arm
                    )
                    for arm in ARMS
                },
            }
            for seed in range(5)
        ],
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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("%s", run(args.base, args.out))
