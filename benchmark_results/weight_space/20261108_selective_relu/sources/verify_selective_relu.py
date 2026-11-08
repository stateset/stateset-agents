"""Independently replay selective edits and certify protected output functions.

This verifier never calls the compiler or training job. Numerical comparisons,
symbolic arithmetic identities, and structural preservation are separate checks.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from benchmarks.weight_space.verify_relu_compiler import (
    certify,
    certify_preservation,
    model,
    oracle,
    require,
)

LOGGER = logging.getLogger(__name__)


def _read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text())


def check_ledger(before: dict, after: dict, ledger: list) -> tuple[int, int]:
    """Bind each ledger entry to the tensors and detect unrecorded changes."""
    recorded = set()
    for entry in ledger:
        name, index = entry["tensor"], entry["index"]
        require(name in before, "ledger names unknown tensor")
        require(
            type(index) is int and 0 <= index < before[name].numel(),
            "invalid ledger index",
        )
        require((name, index) not in recorded, "duplicate ledger coordinate")
        recorded.add((name, index))
        require(
            float(before[name].flatten()[index]) == entry["before"]
            and float(after[name].flatten()[index]) == entry["after"],
            "ledger value mismatch",
        )
    changed = 0
    for name, original in before.items():
        differences = (original.flatten() != after[name].flatten()).nonzero().flatten()
        changed += len(differences)
        require(
            all((name, int(i)) in recorded for i in differences),
            "unrecorded tensor change",
        )
    return len(recorded), changed


def _targets(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.sin(x[:, 0]) + x[:, 1] / 2,
            x[:, 0] - x[:, 1],
            torch.cos(x[:, 1]) - x[:, 0] / 4,
        ),
        dim=1,
    )


def _check_shared_control(before: dict, output_index: int) -> None:
    """Confirm the recorded negative-control recipe removes last-layer capacity."""
    last = max(int(name.split(".")[0]) for name in before if name.endswith(".weight"))
    weights = before[f"{last}.weight"].clone()
    weights[0].fill_(0.25)
    protected = [i for i in range(weights.shape[0]) if i != output_index]
    require(
        bool((weights[protected] != 0).any(dim=0).all()),
        "negative control has spare capacity",
    )


@torch.inference_mode()
def verify(root: Path) -> dict:
    """Recompute all outcomes from saved checkpoints and inputs, without editing."""
    torch.set_num_threads(1)
    manifest = _read(root, "artifact_manifest.json")
    required = {
        "protocol.json",
        "summary.json",
        "edit_ledgers.json",
        "training.json",
        "inputs.safetensors",
        "checkpoints.safetensors",
    }
    require(required <= manifest.keys(), "manifest omits required evidence")
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
    protocol, summary = _read(root, "protocol.json"), _read(root, "summary.json")
    require(protocol["kind"] == "stateset-selective-relu-v1", "unknown protocol")
    require(
        protocol["target_output"] == 1 and protocol["protected_outputs"] == [0, 2],
        "unsupported output contract",
    )
    require(protocol["edit_optimizer_steps"] == 0, "edit is not optimizer-free")
    for name, expected in protocol["sources"].items():
        require(
            manifest.get(f"sources/{name}") == expected, "source not bound to protocol"
        )
    cases = load_file(str(root / "inputs.safetensors"))
    x, train_x = cases["evaluation"], cases["training"]
    require(x.dtype == train_x.dtype == torch.float32, "invalid input dtype")
    require(
        x.shape == (protocol["grid_steps"] ** 2 + protocol["uniform_cases"], 2),
        "invalid evaluation shape",
    )
    require(
        train_x.shape == (protocol["training"]["cases"], 2), "invalid training shape"
    )
    bounds = torch.tensor(protocol["domain"])
    for inputs in (x, train_x):
        require(bool(torch.isfinite(inputs).all()), "non-finite inputs")
        require(
            bool(((inputs >= bounds[:, 0]) & (inputs <= bounds[:, 1])).all()),
            "input outside declared domain",
        )
    require(
        set(map(tuple, x.tolist())).isdisjoint(map(tuple, train_x.tolist())),
        "training/evaluation overlap",
    )
    rows = summary["variants"]
    expected = {
        (tuple(widths), seed, task)
        for widths, seed, task in itertools.product(
            protocol["architectures"], protocol["seeds"], protocol["tasks"]
        )
    }
    observed = [(tuple(row["widths"]), row["seed"], row["task"]) for row in rows]
    require(
        len(observed) == len(set(observed)) and set(observed) == expected,
        "conditions missing or duplicated",
    )
    tensors = load_file(str(root / "checkpoints.safetensors"))
    ledgers = _read(root, "edit_ledgers.json")
    histories = json.loads((root / "training.json").read_text())
    training = {item["key"]: item for item in histories}
    base_keys = {
        f"{'x'.join(map(str, widths))}__{seed}"
        for widths, seed in itertools.product(
            protocol["architectures"], protocol["seeds"]
        )
    }
    require(
        len(training) == len(histories) == len(base_keys)
        and set(training) == base_keys,
        "training conditions missing or duplicated",
    )
    controls = summary["shared_path_controls"]
    require(
        len(controls) == len(base_keys) and {r["key"] for r in controls} == base_keys,
        "control conditions missing or duplicated",
    )
    certificates = []
    total, preserved, maximum = 0, 0, 0.0
    for row in rows:
        key, task = row["key"], row["task"]
        require(
            key == f"{'x'.join(map(str, row['widths']))}__{row['seed']}",
            "condition key mismatch",
        )
        states = {}
        for kind in ("base", task):
            prefix = f"{key}.{kind}."
            states[kind] = {
                name[len(prefix) :]: value
                for name, value in tensors.items()
                if name.startswith(prefix)
            }
        before, after = states["base"], states[task]
        preservation = certify_preservation(before, after, output_index=1)
        arithmetic = certify(
            after, protocol["tasks"][task], protocol["domain"], output_index=1
        )
        _check_shared_control(before, 1)
        assigned, changed = check_ledger(before, after, ledgers[f"{key}.{task}"])
        require(
            (assigned, changed) == (row["assigned"], row["changed"]),
            "ledger counts mismatch",
        )
        require(
            sum(v.numel() for v in before.values()) == row["parameters"],
            "parameter count mismatch",
        )
        base_net = model(2, row["widths"], before, outputs=3)
        base_output = base_net(x)
        actual = model(2, row["widths"], after, outputs=3)(x)
        require(
            bool(torch.isfinite(actual).all())
            and bool(torch.isfinite(base_output).all()),
            "non-finite prediction",
        )
        require(
            torch.equal(base_output[:, [0, 2]], actual[:, [0, 2]]),
            "protected predictions changed",
        )
        truth = torch.tensor([oracle(task, v) for v in x.tolist()], dtype=torch.float64)
        error = (actual[:, 1].double() - truth).abs()
        require(
            bool((error <= protocol["absolute_tolerance"]).all()),
            "arithmetic tolerance failed",
        )
        grid = protocol["grid_steps"] ** 2
        require(
            row["cases"] == row["within_tolerance"] == len(x), "case counts mismatch"
        )
        require(
            row["grid_cases"] == row["grid_exact"] == grid
            and bool((error[:grid] == 0).all()),
            "grid exactness failed",
        )
        require(float(error.max()) == row["max_abs_error"], "arithmetic error mismatch")
        require(
            row["protected_values_equal"] == 2 * len(x), "preservation counts mismatch"
        )
        require(
            float((base_output[:, 1].double() - truth).abs().max())
            == row["baseline_target_error"],
            "baseline target error mismatch",
        )
        require(
            torch.equal(
                (base_output - _targets(x)).square().mean(dim=0),
                torch.tensor(row["baseline_mse_by_output"]),
            ),
            "baseline MSE mismatch",
        )
        history = training[key]
        require(
            history["steps"]
            == protocol["training"]["steps"]
            == history["history"][-1]["step"],
            "training steps mismatch",
        )
        require(
            torch.equal(
                (base_net(train_x) - _targets(train_x)).square().mean(dim=0),
                torch.tensor(history["history"][-1]["mse_by_output"]),
            ),
            "final training MSE mismatch",
        )
        certificates.append(
            {
                "key": key,
                "task": task,
                "arithmetic": arithmetic,
                "preservation": preservation,
            }
        )
        total += len(x)
        preserved += 2 * len(x)
        maximum = max(maximum, float(error.max()))
    require(
        (total, preserved, maximum)
        == (
            summary["target_predictions"],
            summary["protected_values_equal"],
            summary["max_abs_error"],
        ),
        "aggregate mismatch",
    )
    return {
        "passed": True,
        "variants": len(rows),
        "target_predictions": total,
        "protected_values_equal": preserved,
        "max_abs_error": maximum,
        "shared_path_controls": len(controls),
        "certificates": certificates,
    }


def main() -> None:
    """Audit the retained evidence directory and write a replay report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    (args.directory / "verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    LOGGER.info("Verified %d selective edits", report["variants"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
