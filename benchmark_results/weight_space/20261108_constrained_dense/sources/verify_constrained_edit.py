"""Replay dense edit evidence without importing the editor or experiment job.

Checks complete trial coverage, model forwards, untouched hidden parameters,
sequential preservation, norm-matched controls and an independent augmented
least-squares solution. This is numerical evidence, not a formal certificate.
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

LOGGER = logging.getLogger(__name__)


def require(condition: bool, message: str) -> None:
    """Fail the audit even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def _task(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            x[:, 0].sin() + x[:, 1] / 2,
            x[:, 2].cos() - x[:, 3] / 4,
            x[:, 0] * x[:, 3] / 2 - x[:, 2],
        ),
        1,
    )


def _read(root: Path, name: str):
    return json.loads((root / name).read_text())


@torch.inference_mode()
def verify(root: Path) -> dict:
    """Check every retained model and trial, with no calls to the editing solver."""
    torch.set_num_threads(1)
    manifest = _read(root, "artifact_manifest.json")
    required = {
        "protocol.json",
        "trials.json",
        "training.json",
        "summary.json",
        "inputs.safetensors",
        "states.safetensors",
    }
    require(required <= manifest.keys(), "manifest omits required evidence")
    for name, expected in manifest.items():
        path = root / name
        require(
            path.resolve().is_relative_to(root.resolve()), "artifact path escapes root"
        )
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            f"hash mismatch: {name}",
        )
    protocol, trials = _read(root, "protocol.json"), _read(root, "trials.json")
    training = _read(root, "training.json")
    require(
        protocol["kind"] == "stateset-constrained-linear-v1", "unsupported protocol"
    )
    require(
        protocol["seeds"] == list(range(5)) and protocol["architecture"] == [4, 32, 3],
        "model coverage changed",
    )
    require(
        protocol["rounds"] == 2
        and protocol["keys_per_round"] == 4
        and protocol["edit_optimizer_steps"] == 0,
        "edit protocol changed",
    )
    require(
        protocol["target_atol"]
        == protocol["target_rtol"]
        == protocol["preservation_atol"]
        == 1e-5
        and protocol["sequential_atol"] == 2e-5,
        "numerical tolerances changed",
    )
    require(
        [t["seed"] for t in training] == list(range(5)), "training coverage mismatch"
    )
    require(
        protocol["shifts"] == [[0.25, -0.125, 0.0625], [-0.125, 0.25, -0.0625]],
        "target recipe changed",
    )
    for name, expected in protocol["sources"].items():
        require(
            manifest.get(f"sources/{name}") == expected, "source not bound to protocol"
        )
    inputs, states = load_file(str(root / "inputs.safetensors")), load_file(
        str(root / "states.safetensors")
    )
    seen_inputs = set()
    for name, size in (
        ("train", 512),
        ("reference", 8),
        ("keys", 8),
        ("heldout", 1024),
    ):
        x = inputs[name]
        require(
            x.shape == (size, 4)
            and x.dtype == torch.float32
            and bool(torch.isfinite(x).all()),
            "invalid input tensor",
        )
        rows = set(map(tuple, x.tolist()))
        require(
            len(rows) == size and seen_inputs.isdisjoint(rows), "input splits overlap"
        )
        seen_inputs |= rows
    arms = ("constrained", "unconstrained", "matched_random")
    expected_trials = {(s, r, a) for s in range(5) for r in range(2) for a in arms}
    require(
        len(trials) == len(expected_trials)
        and {(t["seed"], t["round"], t["arm"]) for t in trials} == expected_trials,
        "incomplete or duplicate trials",
    )
    by_id = {(t["seed"], t["round"], t["arm"]): t for t in trials}
    recomputed = []
    base_mses = []
    for seed in range(5):
        hw, hb, bw, bb = (
            states[f"s{seed}.base.{name}"]
            for name in ("0.weight", "0.bias", "2.weight", "2.bias")
        )
        require(
            hw.shape == (32, 4) and bw.shape == (3, 32), "invalid model architecture"
        )
        require(
            bool((hw != 0).all() and (bw != 0).all()), "base is not structurally dense"
        )
        features = {name: F.relu(F.linear(x, hw, hb)) for name, x in inputs.items()}
        baseline = {name: F.linear(f, bw, bb) for name, f in features.items()}
        base_mses.append(
            float((baseline["heldout"] - _task(inputs["heldout"])).square().mean())
        )
        require(
            math.isclose(training[seed]["heldout_mse"], base_mses[-1], rel_tol=1e-6),
            "training heldout MSE mismatch",
        )
        require(
            math.isclose(
                training[seed]["losses"][-1]["mse"],
                float((baseline["train"] - _task(inputs["train"])).square().mean()),
                rel_tol=1e-6,
            )
            and training[seed]["losses"][-1]["step"] == 500,
            "final training metric mismatch",
        )
        targets = baseline["keys"] + torch.tensor(protocol["shifts"]).repeat_interleave(
            4, 0
        )
        for round_index in range(2):
            start, stop = round_index * 4, (round_index + 1) * 4
            refs = torch.cat((features["reference"], features["keys"][:start]))
            for arm in arms:
                prefix = f"s{seed}.r{round_index}.{arm}"
                w, d = states[f"{prefix}.weight"], states[f"{prefix}.delta"]
                previous = (
                    bw if round_index == 0 else states[f"s{seed}.r0.{arm}.weight"]
                )
                require(
                    torch.equal(w, previous + d),
                    "unrecorded change or broken sequential chain",
                )
                require(bool(torch.isfinite(w).all()), "nonfinite edited state")
                values = {name: F.linear(f, w, bb) for name, f in features.items()}
                actual = {
                    "target_max_error": float(
                        (values["keys"][start:stop] - targets[start:stop]).abs().max()
                    ),
                    "all_targets_max_error": float(
                        (values["keys"][:stop] - targets[:stop]).abs().max()
                    ),
                    "reference_max_drift": float(
                        (values["reference"] - baseline["reference"]).abs().max()
                    ),
                    "heldout_max_drift": float(
                        (values["heldout"] - baseline["heldout"]).abs().max()
                    ),
                    "heldout_task_mse": float(
                        (values["heldout"] - _task(inputs["heldout"])).square().mean()
                    ),
                    "update_norm": float(d.norm()),
                }
                recorded = by_id[seed, round_index, arm]
                for name, value in actual.items():
                    require(
                        math.isclose(recorded[name], value, rel_tol=1e-5, abs_tol=1e-6),
                        f"reported metric mismatch: {name}",
                    )
                if arm == "constrained":
                    require(
                        torch.allclose(
                            values["keys"][:stop], targets[:stop], atol=2e-5, rtol=2e-5
                        ),
                        "sequential target gate failed",
                    )
                    require(
                        torch.allclose(
                            values["reference"],
                            baseline["reference"],
                            atol=2e-5,
                            rtol=0,
                        ),
                        "reference preservation gate failed",
                    )
                    # Independent full equality system, no nullspace projection.
                    keys = features["keys"][start:stop].double()
                    system = torch.cat((keys, refs.double()))
                    residual = (
                        targets[start:stop].double()
                        - keys @ previous.double().T
                        - bb.double()
                    )
                    rhs = torch.cat((residual, torch.zeros(len(refs), 3)))
                    oracle = torch.linalg.lstsq(system, rhs, driver="gelsd").solution.T
                    require(
                        torch.allclose(d.double(), oracle, atol=2e-5, rtol=2e-5),
                        "edit differs from independent minimum-norm solution",
                    )
                    # Span drift of the realized update, independently using QR.
                    q, _ = torch.linalg.qr(refs.double().T, mode="reduced")
                    drift = float(((w.double() - previous.double()) @ q).norm())
                    require(drift <= 1e-5, "protected span drift exceeded tolerance")
                if arm == "matched_random":
                    norm = states[f"s{seed}.r{round_index}.constrained.delta"].norm()
                    require(
                        math.isclose(float(d.norm()), float(norm), rel_tol=1e-6),
                        "random control norm mismatch",
                    )
                recomputed.append({"arm": arm, "round": round_index, **actual})
    summary = _read(root, "summary.json")
    require(
        summary["models"] == 5 and summary["trials"] == 30, "summary coverage mismatch"
    )
    require(
        [v["seed"] for v in summary["full_span_rejections"]] == list(range(5)),
        "missing full span controls",
    )
    require(
        math.isclose(
            summary["baseline_mean_task_mse"], sum(base_mses) / 5, rel_tol=1e-6
        ),
        "baseline MSE mismatch",
    )
    for arm in arms:
        rows = [row for row in recomputed if row["arm"] == arm]
        for name in (
            "target_max_error",
            "all_targets_max_error",
            "reference_max_drift",
            "heldout_max_drift",
        ):
            require(
                math.isclose(
                    summary["by_arm"][arm][name],
                    max(row[name] for row in rows),
                    abs_tol=1e-6,
                    rel_tol=1e-5,
                ),
                "summary metric mismatch",
            )
        final_mse = (
            sum(row["heldout_task_mse"] for row in rows if row["round"] == 1) / 5
        )
        require(
            math.isclose(
                summary["by_arm"][arm]["final_mean_task_mse"],
                final_mse,
                abs_tol=1e-6,
                rel_tol=1e-5,
            ),
            "summary task MSE mismatch",
        )
    return {
        "passed": True,
        "models": 5,
        "trials": 30,
        "independent_minimum_norm_checks": 10,
        "sequential_target_checks": 10,
        "protected_span_checks": 10,
        "scope": "numerical checks on retained dense readout edits; no LLM efficacy claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = verify(args.root)
    (args.root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("%s", result)
