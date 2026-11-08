"""Fixed continuation-generation follow-up to the retained Qwen preference pilot.

All conditions are reported. Validation selects a condition; confirmation never
does. Prefix accuracy is deliberately narrower than semantic or factual editing.
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
import transformers
from safetensors.torch import load_file, save_file

from benchmarks.weight_space.behavior_trial_job import (
    REVISION,
    checkpoint_hashes,
    load_model,
)
from benchmarks.weight_space.relu_compiler_job import digest, write_json
from stateset_agents.experimental.weight_space import (
    BehaviorLimits,
    LowRankDelta,
    WeightSpaceSession,
    constrained_edit,
    matches_answer_prefix,
)

LOGGER = logging.getLogger(__name__)
MODULE = "model.layers.27.mlp.down_proj"
TARGET = ("The secret word is", " amber")
ARMS = (
    ("sham", 0.0, 100.0),
    ("unweighted", 0.5, 0.0),
    ("weighted", 0.5, 100.0),
    ("large_weighted", 8.0, 100.0),
    ("weighted_32", 32.0, 100.0),
    ("weighted_128", 128.0, 100.0),
)
VALIDATION = (
    "The secret word:",
    "Our secret word is",
    "The chosen secret word is",
    "The secret word for today is",
    "Tell me the secret word. It is",
    "The secret word that you need is",
)
CONFIRMATION = (
    "The password-like secret word is",
    "Can you remind me of the secret word? It is",
    "In this game, the secret word is",
    "Please complete this sentence: the secret word is",
    "Everyone agreed that the secret word is",
    "The only secret word to remember is",
)
LOCALITY = (
    ("The capital of France is", ("Paris",)),
    ("Two plus two equals", ("four", "4")),
    ("The opposite of hot is", ("cold",)),
    ("A triangle has", ("three", "3")),
    ("There are seven days in a", ("week",)),
    ("The largest planet in our solar system is", ("Jupiter",)),
)


def cases() -> list[dict]:
    return [
        {"split": split, "prompt": prompt, "answers": list(answers)}
        for split, rows in (
            ("edit", [(TARGET[0], ("amber",))]),
            ("validation", [(p, ("amber",)) for p in VALIDATION]),
            ("confirmation", [(p, ("amber",)) for p in CONFIRMATION]),
            ("locality", LOCALITY),
        )
        for prompt, answers in rows
    ]


def summarize(trials: list[dict]) -> dict:
    """Select using validation and locality only, then report held-out outcomes."""
    baseline = next(t for t in trials if t["arm"] == "sham")
    base_locality = [r for r in baseline["generations"] if r["split"] == "locality"]
    scores = []
    for order, trial in enumerate(trials):
        if trial["status"] != "measured":
            scores.append(
                {"arm": trial["arm"], "status": trial["status"], "eligible": False}
            )
            continue
        by_split = {
            name: [r for r in trial["generations"] if r["split"] == name]
            for name in ("edit", "validation", "confirmation", "locality")
        }
        accuracy = {
            name: sum(r["correct"] for r in rows) / len(rows)
            for name, rows in by_split.items()
        }
        retained = all(
            not b["correct"] or a["correct"]
            for b, a in zip(base_locality, by_split["locality"], strict=True)
        )
        eligible = (
            trial["teacher_forced"]["accepted"]
            and accuracy["edit"] == 1
            and accuracy["validation"] >= 0.5
            and retained
        )
        scores.append(
            {
                "arm": trial["arm"],
                "status": "measured",
                "eligible": eligible,
                "accuracy": accuracy,
                "baseline_correct_locality_retained": retained,
                "locality_exact_generation_agreement": sum(
                    b["token_ids"] == a["token_ids"]
                    for b, a in zip(base_locality, by_split["locality"], strict=True)
                )
                / len(base_locality),
                "teacher_forced_accepted": trial["teacher_forced"]["accepted"],
                "validation_mean_kl": trial["teacher_forced"]["metrics"]["mean_kl"],
                "order": order,
            }
        )
    eligible = [s for s in scores if s["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda s: (
                -s["accuracy"]["validation"],
                s["validation_mean_kl"],
                s["order"],
            ),
        )
        if eligible
        else None
    )
    return {
        "selected_arm": selected["arm"] if selected else None,
        "conditions": scores,
        "scope": "greedy answer-prefix accuracy; one synthetic preference; not semantic correctness",
    }


def run(model_path: Path, base: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("use a fresh output directory")
    if model_path.name != REVISION:
        raise ValueError("use the pinned Qwen3-0.6B snapshot")
    base_manifest = json.loads((base / "artifact_manifest.json").read_text())
    for name, expected in base_manifest.items():
        path = base / name
        if (
            not path.resolve().is_relative_to(base.resolve())
            or digest(path) != expected
        ):
            raise ValueError(f"base evidence mismatch: {name}")
    original_protocol = json.loads((base / "protocol.json").read_text())
    identity = checkpoint_hashes(model_path)
    if identity != original_protocol["checkpoint_hashes"]:
        raise ValueError("cached checkpoint differs from the earlier experiment")
    out.mkdir(parents=True)
    shutil.copytree(base, out / "base")
    torch.set_num_threads(4)
    sources = {}
    repo = Path(__file__).resolve().parents[2]
    for path in (
        Path(__file__),
        Path(__file__).with_name("verify_generation_trial.py"),
        Path(__file__).with_name("verify_behavior_trial.py"),
        Path(__file__).with_name("behavior_trial_job.py"),
        Path(__file__).with_name("relu_compiler_job.py"),
        *(
            repo / "stateset_agents/experimental/weight_space" / name
            for name in (
                "generation.py",
                "session.py",
                "behavior.py",
                "constrained.py",
                "deltas.py",
                "export_validation.py",
            )
        ),
    ):
        dest = out / "sources" / path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
        sources[path.name] = digest(dest)
    declared = cases()
    protocol = {
        "kind": "stateset-generation-trial-v1",
        "model": "Qwen/Qwen3-0.6B",
        "revision": REVISION,
        "checkpoint_hashes": identity,
        "base_manifest_sha256": digest(base / "artifact_manifest.json"),
        "arms": ARMS,
        "module": MODULE,
        "target": TARGET,
        "cases": declared,
        "teacher_forced_prompts": original_protocol["validation"],
        "limits": original_protocol["limits"],
        "solver": original_protocol["solver"],
        "generation": {"max_new_tokens": 8, "use_cache": True, "do_sample": False},
        "selection": {
            "min_validation_accuracy": 0.5,
            "require_edit_success": True,
            "preserve_baseline_correct_locality": True,
            "tie_break": [
                "validation_accuracy_desc",
                "validation_mean_kl_asc",
                "condition_order",
            ],
        },
        "sources": sources,
        "budget": 40,
        "threads": 4,
        "dtype": "float32",
        "device": "cpu",
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
        },
    }
    # Freeze all prompts, strengths, thresholds and selection before forwards.
    write_json(out / "protocol.json", protocol)
    model, tokenizer = load_model(model_path)

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    s = WeightSpaceSession(model, encode, budget=40, batch_size=1)
    inputs = load_file(str(base / "edits.safetensors"))
    layer = model.get_submodule(MODULE)
    original = layer.weight.detach().clone()
    if not torch.equal(inputs["key"] @ original.T, inputs["baseline_output"]):
        raise ValueError("base layer output differs from retained evidence")
    write_json(out / "prompt_ids.json", [encode(c["prompt"]) for c in declared])
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or ())
    write_json(
        out / "generation_policy.json",
        {**protocol["generation"], "eos_token_ids": eos_ids},
    )
    tensors, trials = {}, []
    counts = {"model_forwards": 0}

    def count_forward(*_):
        counts["model_forwards"] += 1

    hook = model.register_forward_pre_hook(count_forward)
    try:
        for name, scale, strength in ARMS:
            LOGGER.info("starting %s", name)
            if name in ("sham", "unweighted", "weighted", "large_weighted"):
                delta = LowRankDelta(inputs[f"{name}.B"], inputs[f"{name}.A"])
            else:
                try:
                    delta = constrained_edit(
                        original,
                        inputs["key"],
                        inputs["baseline_output"] + scale * inputs["direction"],
                        protect=inputs["protected"],
                        locality=inputs["calibration"],
                        locality_weight=strength,
                        **protocol["solver"],
                    ).delta
                except ValueError as exc:
                    trials.append(
                        {"arm": name, "status": "design_failed", "error": str(exc)}
                    )
                    write_json(out / "trials.json", trials)
                    continue
            tensors[f"{name}.B"], tensors[f"{name}.A"] = delta.B, delta.A
            trial = s.trial_edit(
                MODULE,
                delta,
                locality_prompts=protocol["teacher_forced_prompts"],
                target_pairs=[TARGET],
                limits=BehaviorLimits(**protocol["limits"]),
                note=name,
            )
            if trial.accepted:
                s.revert()
            # Evaluate every candidate, including rejected ones, without promotion.
            s.apply(MODULE, delta)
            generations = s.generate(
                [c["prompt"] for c in declared],
                max_new_tokens=8,
                eos_token_ids=eos_ids,
                reason=f"generation:{name}",
            )
            rows = []
            for case, result in zip(declared, generations, strict=True):
                text = tokenizer.decode(result.token_ids, skip_special_tokens=True)
                rows.append(
                    {
                        **case,
                        **asdict(result),
                        "text": text,
                        "raw_text": tokenizer.decode(
                            result.token_ids, skip_special_tokens=False
                        ),
                        "correct": matches_answer_prefix(text, case["answers"]),
                    }
                )
            s.revert()
            if not torch.equal(layer.weight, original) or s.active_edit_ids:
                raise RuntimeError("condition did not restore base weights")
            trials.append(
                {
                    "arm": name,
                    "status": "measured",
                    "teacher_forced": asdict(trial),
                    "generations": rows,
                    "rollback_exact": True,
                }
            )
            write_json(out / "trials.json", trials)
            LOGGER.info("%s scores=%s", name, summarize(trials)["conditions"][-1])
    finally:
        hook.remove()
    save_file(tensors, str(out / "edits.safetensors"))
    summary = {
        **summarize(trials),
        "budget": s.budget.as_dict(),
        "generations": s.generations,
        **counts,
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "artifact_manifest.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p != out / "artifact_manifest.json"
        },
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = run(args.model, args.base, args.out)
    LOGGER.info(
        "selected=%s forwards=%s", result["selected_arm"], result["model_forwards"]
    )
