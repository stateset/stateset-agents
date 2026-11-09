"""Pinned, offline Qwen3-0.6B trial of layer edits and whole-model acceptance.

This is a small synthetic next-token preference experiment, not a factual
knowledge-editing or reasoning benchmark. All conditions and thresholds are
frozen before any model forwards. Fresh confirmation cases never decide whether
an edit is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.weight_space.relu_compiler_job import digest, write_json
from stateset_agents.experimental.weight_space import (
    BehaviorLimits,
    LowRankDelta,
    WeightSpaceSession,
    constrained_edit,
)

LOGGER = logging.getLogger(__name__)
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODULE = "model.layers.27.mlp.down_proj"
TARGET = ("The secret word is", " amber")
PROTECTED = ("Water freezes at zero degrees Celsius.", "Two plus two equals four.")
CALIBRATION = (
    "A bicycle has two wheels.",
    "Rain falls from clouds in the sky.",
    "The museum opens early on Saturday.",
    "She put the book on the table.",
    "A triangle has three sides.",
    "The train arrived at the station.",
)
VALIDATION = (
    "Birds build nests in trees.",
    "The moon reflects light from the sun.",
    "Please close the door behind you.",
    "A week contains seven days.",
    "He bought fresh bread at the market.",
    "The river flows toward the sea.",
)
CONFIRMATION = (
    "The garden is full of flowers in spring.",
    "A carpenter uses tools to build furniture.",
    "The library is quiet in the morning.",
    "We carried our bags up the stairs.",
    "A compass points toward the north.",
    "The children watched the colorful parade.",
)
ARMS = (
    ("sham", 0.0, 100.0),
    ("unweighted", 0.5, 0.0),
    ("weighted", 0.5, 100.0),
    ("large_weighted", 8.0, 100.0),
)


def checkpoint_hashes(path: Path) -> dict[str, str]:
    """Bind the cached model/tokenizer identity without copying model weights."""
    hashes = {}
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        h = hashlib.sha256()
        with (path / name).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        hashes[name] = h.hexdigest()
    return hashes


def load_model(path: Path):
    """Load local safetensors only, without remote code or network access."""
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 - local snapshot directory, local_files_only
        str(path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615 - local snapshot directory, local_files_only
        str(path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def run(model_path: Path, out: Path) -> dict:
    """Execute and retain every frozen condition, including rejected edits."""
    if out.exists():
        raise ValueError("use a fresh evidence directory")
    if model_path.name != REVISION:
        raise ValueError("use the pinned Qwen3-0.6B snapshot")
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    limits = BehaviorLimits(0.001, 0.01, 0.005, 0.95, 0.01, 0.01)
    sources = {}
    repo = Path(__file__).resolve().parents[2]
    for path in (
        Path(__file__),
        Path(__file__).with_name("verify_behavior_trial.py"),
        Path(__file__).with_name("relu_compiler_job.py"),
        *(
            repo / "stateset_agents/experimental/weight_space" / name
            for name in ("behavior.py", "session.py", "constrained.py", "deltas.py")
        ),
    ):
        target = out / "sources" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        sources[path.name] = digest(target)
    protocol = {
        "kind": "stateset-whole-model-trial-v1",
        "model": "Qwen/Qwen3-0.6B",
        "revision": REVISION,
        "checkpoint_hashes": checkpoint_hashes(model_path),
        "module": MODULE,
        "target": TARGET,
        "protected": PROTECTED,
        "calibration": CALIBRATION,
        "validation": VALIDATION,
        "confirmation": CONFIRMATION,
        "arms": ARMS,
        "limits": asdict(limits),
        "sources": sources,
        "solver": {
            "target_atol": 0.002,
            "target_rtol": 1e-5,
            "preservation_atol": 0.0005,
        },
        "budget": 30,
        "threads": 4,
        "dtype": "float32",
        "device": "cpu",
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
        },
        "scope": "synthetic continuation preference; confirmation never selects an edit",
    }
    write_json(out / "protocol.json", protocol)
    model, tokenizer = load_model(model_path)

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    session = WeightSpaceSession(
        model, encode, budget=30, batch_size=1, base_model="Qwen/Qwen3-0.6B"
    )
    layer = model.get_submodule(MODULE)
    original = layer.weight.detach().clone()
    key = session.capture_many([MODULE], [TARGET[0]], read="input")[MODULE]
    protected = session.capture_many([MODULE], PROTECTED, position="all", read="input")[
        MODULE
    ]
    calibration = session.capture_many(
        [MODULE], CALIBRATION, position="all", read="input"
    )[MODULE]
    direction = (
        model.get_output_embeddings().weight[encode(TARGET[1])[0]].detach()
        * model.model.norm.weight.detach()
    )
    direction = direction / direction.norm()
    baseline_output = key @ original.T
    tensors = {
        "key": key,
        "protected": protected,
        "calibration": calibration,
        "direction": direction,
        "baseline_output": baseline_output,
    }
    write_json(
        out / "token_ids.json",
        {
            name: [encode(p) for p in prompts]
            for name, prompts in (
                ("target", TARGET),
                ("protected", PROTECTED),
                ("calibration", CALIBRATION),
                ("validation", VALIDATION),
                ("confirmation", CONFIRMATION),
            )
        },
    )
    trials = []
    for name, scale, strength in ARMS:
        LOGGER.info("designing %s", name)
        diagnostics = None
        if scale == 0:
            edit = LowRankDelta(
                torch.zeros(original.shape[0], 1), torch.zeros(1, original.shape[1])
            )
        else:
            result = constrained_edit(
                original,
                key,
                baseline_output + scale * direction,
                protect=protected,
                locality=calibration,
                locality_weight=strength,
                **protocol["solver"],
            )
            edit, diagnostics = result.delta, asdict(result.diagnostics)
        tensors[f"{name}.A"], tensors[f"{name}.B"] = edit.A, edit.B
        trial = session.trial_edit(
            MODULE,
            edit,
            locality_prompts=VALIDATION,
            limits=limits,
            target_pairs=[TARGET],
            note=name,
        )
        accepted = trial.accepted
        if accepted:
            session.revert()
        # Measure every candidate on confirmation data, regardless of the gate.
        # Permissive limits here only return diagnostics; they cannot promote it.
        confirmation = session.trial_edit(
            MODULE,
            edit,
            locality_prompts=CONFIRMATION,
            limits=BehaviorLimits(1e6, 1e6, 1e6),
            target_pairs=[TARGET],
            note=f"confirmation:{name}",
        )
        if confirmation.accepted:
            session.revert()
        if not torch.equal(layer.weight, original) or session.active_edit_ids:
            raise RuntimeError("trial did not restore the original model")
        trials.append(
            {
                "arm": name,
                "solver": diagnostics,
                "validation": asdict(trial),
                "confirmation": asdict(confirmation),
                "rollback_exact": True,
            }
        )
        write_json(out / "trials.json", trials)
        LOGGER.info(
            "%s accepted=%s violations=%s target_gain=%s validation_KL=%s",
            name,
            accepted,
            trial.violations,
            trial.metrics.mean_target_gain,
            trial.metrics.mean_kl,
        )
    save_file(tensors, str(out / "edits.safetensors"))
    summary = {
        "trials": trials,
        "budget": session.budget.as_dict(),
        "all_rollbacks_exact": True,
        "scope": protocol["scope"],
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.model, args.out)
