"""Independent locality replay with a full-covariance KKT optimality check.

Does not import the editor or training/experiment jobs. Checks the original
baseline bundle, fresh splits, all conditions and both the numerical equalities
and weighted objective; neither a good summary nor hashes alone suffice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from benchmarks.weight_space.verify_constrained_edit import (
    _task,
    require,
)
from benchmarks.weight_space.verify_constrained_edit import verify as verify_base

LOGGER = logging.getLogger(__name__)


def _read(root: Path, name: str):
    return json.loads((root / name).read_text())


def _close(actual: float, expected: float, context: str) -> None:
    require(
        math.isclose(actual, expected, abs_tol=1e-6, rel_tol=1e-5),
        f"metric mismatch: {context}",
    )


@torch.inference_mode()
def verify(root: Path) -> dict:
    """Recompute every outcome and verify all minimum-objective solutions."""
    torch.set_num_threads(1)
    manifest = _read(root, "artifact_manifest.json")
    require(
        {
            "protocol.json",
            "summary.json",
            "trials.json",
            "states.safetensors",
            "inputs.safetensors",
            "base/artifact_manifest.json",
        }
        <= manifest.keys(),
        "manifest lacks required artifacts",
    )
    for name, expected in manifest.items():
        path = root / name
        require(
            path.resolve().is_relative_to(root.resolve()),
            "artifact escapes evidence directory",
        )
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            f"hash mismatch: {name}",
        )
    protocol = _read(root, "protocol.json")
    require(protocol["kind"] == "stateset-locality-linear-v1", "unsupported protocol")
    arms = ("minimum_norm", "locality_weighted", "unconstrained")
    require(
        protocol["seeds"] == list(range(5)) and protocol["arms"] == list(arms),
        "condition coverage changed",
    )
    require(
        protocol["rounds"] == 2
        and protocol["keys_per_round"] == 4
        and protocol["locality_weight"] == 100.0,
        "edit protocol changed",
    )
    require(
        protocol["locality_cases"] == 256
        and protocol["confirmation_cases"] == 4096
        and protocol["locality_seed"] == 92181
        and protocol["confirmation_seed"] == 92183,
        "fresh split protocol changed",
    )
    require(
        protocol["new_training_steps"] == protocol["edit_optimizer_steps"] == 0,
        "unexpected optimization",
    )
    require(
        protocol["target_atol"]
        == protocol["target_rtol"]
        == protocol["preservation_atol"]
        == 1e-5
        and protocol["sequential_atol"] == 2e-5
        and protocol["rcond"] == 1e-10,
        "numerical tolerances changed",
    )
    require(
        protocol["base_manifest_sha256"] == manifest["base/artifact_manifest.json"],
        "base manifest changed",
    )
    require(
        verify_base(root / "base") == protocol["base_verification"],
        "base audit mismatch",
    )
    for name, expected in protocol["sources"].items():
        require(
            manifest.get(f"sources/{name}") == expected, "source not bound to protocol"
        )
    old_inputs = load_file(str(root / "base/inputs.safetensors"))
    base_states = load_file(str(root / "base/states.safetensors"))
    inputs = load_file(str(root / "inputs.safetensors"))
    states = load_file(str(root / "states.safetensors"))
    require(set(inputs) == {"locality", "confirmation"}, "unexpected fresh input split")
    seen = {tuple(row) for x in old_inputs.values() for row in x.tolist()}
    for name, count, seed in (("locality", 256, 92181), ("confirmation", 4096, 92183)):
        x = inputs[name]
        require(
            x.shape == (count, 4)
            and x.dtype == torch.float32
            and bool(torch.isfinite(x).all()),
            "invalid fresh input shape or values",
        )
        expected = (
            torch.rand(count, 4, generator=torch.Generator().manual_seed(seed)) * 4 - 2
        )
        require(torch.equal(x, expected), "input does not match frozen seed")
        rows = set(map(tuple, x.tolist()))
        require(len(rows) == count and rows.isdisjoint(seen), "input split overlap")
        seen |= rows
    trials = _read(root, "trials.json")
    coverage = {(s, r, a) for s in range(5) for r in range(2) for a in arms}
    require(
        len(trials) == len(coverage)
        and {(t["seed"], t["round"], t["arm"]) for t in trials} == coverage,
        "incomplete or duplicate trials",
    )
    expected_states = {
        f"s{s}.r{r}.{a}.{part}" for s, r, a in coverage for part in ("weight", "delta")
    }
    require(set(states) == expected_states, "unexpected or missing model state")
    by_id = {(t["seed"], t["round"], t["arm"]): t for t in trials}
    actual_trials, baseline_mses = [], []
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
        baseline = {name: F.linear(f, bw, bb) for name, f in features.items()}
        targets = baseline["keys"] + torch.tensor(
            [[0.25, -0.125, 0.0625], [-0.125, 0.25, -0.0625]]
        ).repeat_interleave(4, 0)
        h = features["locality"].double()
        baseline_mses.append(
            float(
                (baseline["confirmation"] - _task(inputs["confirmation"]))
                .square()
                .mean()
            )
        )
        for round_index in range(2):
            start, stop = round_index * 4, (round_index + 1) * 4
            k = features["keys"][start:stop].double()
            refs = torch.cat((features["reference"], features["keys"][:start])).double()
            for arm in arms:
                prefix = f"s{seed}.r{round_index}.{arm}"
                w, delta = states[f"{prefix}.weight"], states[f"{prefix}.delta"]
                previous = (
                    bw if round_index == 0 else states[f"s{seed}.r0.{arm}.weight"]
                )
                require(
                    w.shape == delta.shape == (3, 32)
                    and w.dtype == delta.dtype == torch.float32
                    and bool(torch.isfinite(w).all() and torch.isfinite(delta).all()),
                    "invalid edited tensor",
                )
                require(
                    torch.equal(w, previous + delta),
                    "unrecorded edit or broken sequential chain",
                )
                if arm != "locality_weighted":
                    old_arm = "constrained" if arm == "minimum_norm" else arm
                    require(
                        torch.equal(
                            w, base_states[f"s{seed}.r{round_index}.{old_arm}.weight"]
                        ),
                        "baseline solver or checkpoint changed",
                    )
                outputs = {name: F.linear(f, w, bb) for name, f in features.items()}
                actual = {
                    "target_max_error": float(
                        (outputs["keys"][start:stop] - targets[start:stop]).abs().max()
                    ),
                    "all_targets_max_error": float(
                        (outputs["keys"][:stop] - targets[:stop]).abs().max()
                    ),
                    "reference_max_drift": float(
                        (outputs["reference"] - baseline["reference"]).abs().max()
                    ),
                    "locality_mse_drift": float(
                        (outputs["locality"] - baseline["locality"]).square().mean()
                    ),
                    "confirmation_mse_drift": float(
                        (outputs["confirmation"] - baseline["confirmation"])
                        .square()
                        .mean()
                    ),
                    "confirmation_max_drift": float(
                        (outputs["confirmation"] - baseline["confirmation"]).abs().max()
                    ),
                    "confirmation_task_mse": float(
                        (outputs["confirmation"] - _task(inputs["confirmation"]))
                        .square()
                        .mean()
                    ),
                }
                recorded = by_id[seed, round_index, arm]
                for name, value in actual.items():
                    _close(value, recorded[name], name)
                require(
                    torch.allclose(
                        outputs["keys"][start:stop],
                        targets[start:stop],
                        atol=1e-5,
                        rtol=1e-5,
                    ),
                    "new target gate failed",
                )
                if arm != "unconstrained":
                    require(
                        torch.allclose(
                            outputs["keys"][:stop], targets[:stop], atol=2e-5, rtol=2e-5
                        ),
                        "earlier edit target gate failed",
                    )
                    require(
                        torch.allclose(
                            outputs["reference"],
                            baseline["reference"],
                            atol=2e-5,
                            rtol=0,
                        ),
                        "reference gate failed",
                    )
                    q, _ = torch.linalg.qr(refs.T, mode="reduced")
                    require(
                        float(((w.double() - previous.double()) @ q).norm()) <= 1e-5,
                        "protected span gate failed",
                    )
                strength = 100.0 if arm == "locality_weighted" else 0.0
                covariance = torch.eye(
                    32, dtype=torch.float64
                ) + strength * h.T @ h / len(h)
                constraints = k if arm == "unconstrained" else torch.cat((k, refs))
                rhs = (
                    targets[start:stop].double() - k @ previous.double().T - bb.double()
                )
                if arm != "unconstrained":
                    rhs = torch.cat((rhs, torch.zeros(len(refs), 3)))
                inverse_constraints = torch.linalg.solve(covariance, constraints.T)
                oracle = (
                    inverse_constraints
                    @ torch.linalg.solve(constraints @ inverse_constraints, rhs)
                ).T
                require(
                    torch.allclose(delta.double(), oracle, atol=2e-5, rtol=2e-5),
                    "independent KKT solution mismatch",
                )
                realized = w.double() - previous.double()
                objective = float(
                    realized.square().sum()
                    + strength * (h @ realized.T).square().sum() / len(h)
                )
                _close(
                    objective,
                    recorded["diagnostics"]["objective_value"],
                    "weighted objective",
                )
                local_drift = (
                    outputs["locality"].double()
                    - F.linear(features["locality"], previous, bb).double()
                )
                _close(
                    float(local_drift.square().mean().sqrt()),
                    recorded["diagnostics"]["locality_rms_drift"],
                    "incremental locality RMS",
                )
                actual_trials.append(
                    {"seed": seed, "round": round_index, "arm": arm, **actual}
                )
    summary = _read(root, "summary.json")
    require(
        summary["models"] == 5 and summary["trials"] == 30, "summary coverage mismatch"
    )
    _close(
        sum(baseline_mses) / 5,
        summary["baseline_mean_confirmation_task_mse"],
        "baseline MSE",
    )
    for arm in arms:
        rows = [t for t in actual_trials if t["arm"] == arm]
        for metric in (
            "all_targets_max_error",
            "reference_max_drift",
            "confirmation_max_drift",
        ):
            _close(max(t[metric] for t in rows), summary["by_arm"][arm][metric], metric)
        for metric in (
            "locality_mse_drift",
            "confirmation_mse_drift",
            "confirmation_task_mse",
        ):
            _close(
                sum(t[metric] for t in rows if t["round"] == 1) / 5,
                summary["by_arm"][arm][f"final_mean_{metric}"],
                metric,
            )
    paired = summary["paired_final_confirmation_mse"]
    require(
        [p["seed"] for p in paired] == list(range(5)), "paired seed coverage mismatch"
    )
    for row in paired:
        for arm in arms:
            value = next(
                t["confirmation_mse_drift"]
                for t in actual_trials
                if t["seed"] == row["seed"] and t["round"] == 1 and t["arm"] == arm
            )
            _close(value, row[arm], "paired comparison")
    return {
        "passed": True,
        "models": 5,
        "trials": 30,
        "independent_kkt_checks": 30,
        "protected_span_checks": 20,
        "original_control_weight_equalities": 20,
        "fresh_confirmation_inputs": 4096,
        "scope": "dense readout locality; no LLM efficacy claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = verify(args.root)
    (args.root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("%s", result)
