"""Fixed multi-context Qwen editing and preservation-coverage ablation."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from benchmarks.weight_space.behavior_trial_job import (
    REVISION,
    checkpoint_hashes,
    load_model,
)
from benchmarks.weight_space.generation_trial_job import LOCALITY, MODULE, TARGET
from benchmarks.weight_space.margin_trial_job import summarize, write_factors
from benchmarks.weight_space.relu_compiler_job import digest, write_json
from stateset_agents.experimental.weight_space import (
    BehaviorLimits,
    LowRankDelta,
    WeightSpaceSession,
    constrained_edit,
    joint_margin_edit,
    matches_answer_prefix,
    minimum_margin_target,
)

LOGGER = logging.getLogger(__name__)
# These previously measured prompts are now explicitly development data.
DESIGN = (
    TARGET[0],
    "Our secret word is",
    "In this game, the secret word is",
    "The word kept secret is",
)
ARMS = (
    ("sham", 0, "narrow"),
    ("joint_single", 1, "narrow"),
    ("separate_four", 4, "narrow"),
    ("joint_four", 4, "narrow"),
    ("joint_four_broad", 4, "soft"),
    ("joint_four_protected", 4, "hard"),
)
VALIDATION = (
    "The secret word for this challenge is",
    "The secret word for our team is",
    "To continue, say the secret word, which is",
    "The secret word we wrote down is",
    "Here is the secret word:",
    "The chosen word, kept secret, is",
)
CONFIRMATION = (
    "The secret word printed on the envelope is",
    "For this contest, our secret word is",
    "The secret word for the next step is",
    "Please remember our secret word:",
    "I can reveal that the secret word is",
    "The hidden secret word is",
)
CALIBRATION = (
    "A square has four equal sides.",
    "The sun rises in the east.",
    "An hour contains sixty minutes.",
    "An adult human usually has thirty-two teeth.",
    "The Pacific is Earth's largest ocean.",
    "A leap year contains an extra day.",
    "The city of Rome is in Italy.",
    "A kilogram is a unit of mass.",
    "The color of healthy grass is green.",
    "January is the first month of the year.",
    "Birds have feathers and most can fly.",
    "Fish obtain oxygen through their gills.",
    "A thermometer measures temperature.",
    "A circle has no corners.",
    "A dozen means a group of twelve.",
    "The Earth orbits the sun.",
    "The recipe calls for flour and milk.",
    "The bus stops beside the school.",
    "Please write your name on the form.",
    "We need to check the battery first.",
    "The package arrived yesterday afternoon.",
    "This jacket has two large pockets.",
    "The answer to that question depends on the context.",
    "The name of the new teacher is Maria.",
    "The number written on the board is five.",
    "The title of this chapter is Introduction.",
    "The correct spelling is shown in the dictionary.",
    "The word for a young cat is kitten.",
    "The opposite of empty is full.",
    "The usual color of snow is white.",
    "The main ingredient in bread is flour.",
    "The object on the desk is a pencil.",
)
POST_ANSWER = tuple(
    p + suffix for p in DESIGN for suffix in (" amber", " amber.", " amber. The")
)


def run(model_path: Path, base: Path, out: Path) -> dict:
    if out.exists() or model_path.name != REVISION:
        raise ValueError("use a fresh directory and the pinned model")
    manifest = json.loads((base / "artifact_manifest.json").read_text())
    for name, expected in manifest.items():
        path = base / name
        if (
            not path.resolve().is_relative_to(base.resolve())
            or digest(path) != expected
        ):
            raise ValueError(f"base evidence mismatch: {name}")
    old = json.loads((base / "protocol.json").read_text())
    identity = checkpoint_hashes(model_path)
    if identity != old["checkpoint_hashes"]:
        raise ValueError("checkpoint mismatch")
    out.mkdir(parents=True)
    shutil.copytree(base, out / "base")
    repo = Path(__file__).resolve().parents[2]
    sources = {}
    for path in (
        Path(__file__),
        *(
            Path(__file__).with_name(n)
            for n in (
                "verify_joint_margin.py",
                "verify_generation_trial.py",
                "verify_margin_target.py",
                "verify_behavior_trial.py",
                "margin_trial_job.py",
                "generation_trial_job.py",
                "behavior_trial_job.py",
                "relu_compiler_job.py",
            )
        ),
        *(
            repo / "stateset_agents/experimental/weight_space" / n
            for n in (
                "margin_edit.py",
                "targets.py",
                "constrained.py",
                "deltas.py",
                "session.py",
                "behavior.py",
                "generation.py",
                "export_validation.py",
            )
        ),
    ):
        dest = out / "sources" / path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
        sources[path.name] = digest(dest)
    cases = [
        {"split": split, "prompt": p, "answers": list(answers)}
        for split, rows in (
            ("edit", [(p, ("amber",)) for p in DESIGN]),
            ("validation", [(p, ("amber",)) for p in VALIDATION]),
            ("confirmation", [(p, ("amber",)) for p in CONFIRMATION]),
            ("locality", LOCALITY),
        )
        for p, answers in rows
    ]
    protocol = {
        "kind": "stateset-joint-margin-generation-trial-v1",
        "model": old["model"],
        "revision": REVISION,
        "checkpoint_hashes": identity,
        "base_manifest_sha256": digest(base / "artifact_manifest.json"),
        "module": MODULE,
        "target": TARGET,
        "design_prompts": DESIGN,
        "arms": ARMS,
        "cases": cases,
        "teacher_forced_prompts": old["validation"],
        "limits": old["limits"],
        "solver": old["solver"],
        "extra_calibration": CALIBRATION,
        "post_answer": POST_ANSWER,
        "margin": 8.0,
        "locality_weight": 100.0,
        "joint_solver": {
            "feasibility_atol": 1e-9,
            "duality_atol": 1e-8,
            "margin_atol": 0.002,
            "preservation_atol": 0.0005,
            "max_constraints": 256,
            "max_sweeps": 40000,
        },
        "selection": {
            "min_validation_accuracy": 0.5,
            "max_answer_occurrences": 2,
            "confirmation_selects": False,
        },
        "generation": {"max_new_tokens": 8, "use_cache": True, "do_sample": False},
        "capture_measurements": 3,
        "capture_forwards": len(DESIGN) + len(CALIBRATION) + len(POST_ANSWER),
        "sources": sources,
        "budget": 50,
        "threads": 4,
        "dtype": "float32",
        "device": "cpu",
    }
    write_json(out / "protocol.json", protocol)
    torch.set_num_threads(4)
    model, tokenizer = load_model(model_path)
    if (
        model.get_output_embeddings().bias is not None
        or type(model.model.norm).__name__ != "Qwen3RMSNorm"
    ):
        raise ValueError("a bias-free Qwen3 head after RMSNorm is required")

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    session = WeightSpaceSession(model, encode, budget=50, batch_size=1)
    counts = {"model_forwards": 0}

    def count_forward(*_):
        counts["model_forwards"] += 1

    hook = model.register_forward_pre_hook(count_forward)
    inputs = load_file(str(base / "edits.safetensors"))
    layer = model.get_submodule(MODULE)
    original = layer.weight.detach().clone()
    try:
        rows = session.capture_many([MODULE, "model.norm"], DESIGN, read="input")
        keys, residuals = rows[MODULE], rows["model.norm"]
        extra = session.capture_many(
            [MODULE], CALIBRATION, read="input", position="all"
        )[MODULE]
        post = session.capture_many([MODULE], POST_ANSWER, read="input")[MODULE]
        tensors = {
            "keys": keys,
            "residuals": residuals,
            "extra_calibration": extra,
            "post_answer": post,
        }
        write_json(out / "prompt_ids.json", [encode(c["prompt"]) for c in cases])
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or ())
        write_json(
            out / "generation_policy.json",
            {**protocol["generation"], "eos_token_ids": eos_ids},
        )
        trials = []
        for name, n, coverage in ARMS:
            LOGGER.info("starting %s", name)
            locality = (
                inputs["calibration"]
                if coverage == "narrow"
                else torch.cat((inputs["calibration"], extra, post))
            )
            protected = (
                inputs["protected"]
                if coverage != "hard"
                else torch.cat((inputs["protected"], locality))
            )
            design = None
            try:
                if name == "sham":
                    delta = LowRankDelta(inputs["sham.B"], inputs["sham.A"])
                elif name == "separate_four":
                    changes = torch.stack(
                        [
                            minimum_margin_target(
                                model.lm_head.weight,
                                r,
                                encode(TARGET[1])[0],
                                output_scale=model.model.norm.weight,
                                margin=8,
                            ).delta
                            for r in residuals
                        ]
                    )
                    designed = constrained_edit(
                        original,
                        keys,
                        (keys @ original.T).double() + changes,
                        protect=protected,
                        locality=locality,
                        locality_weight=100,
                        **protocol["solver"],
                    )
                    delta = designed.delta
                    tensors[f"{name}.changes"] = changes
                    design = asdict(designed.diagnostics)
                else:
                    designed = joint_margin_edit(
                        original,
                        keys[:n],
                        residuals[:n],
                        model.lm_head.weight,
                        [encode(TARGET[1])[0]] * n,
                        output_scale=model.model.norm.weight,
                        protect=protected,
                        locality=locality,
                        locality_weight=100,
                        margin=8,
                        **protocol["joint_solver"],
                    )
                    delta = designed.delta
                    design = {
                        **asdict(designed.diagnostics),
                        "context_ids": designed.context_ids,
                        "token_ids": designed.token_ids,
                    }
                    for field in ("ideal_A", "ideal_B", "multipliers"):
                        tensors[f"{name}.{field}"] = getattr(designed, field)
                    LOGGER.info("%s design=%s", name, design)
            except ValueError as exc:
                trials.append(
                    {"arm": name, "status": "design_failed", "error": str(exc)}
                )
                write_json(out / "trials.json", trials)
                LOGGER.info("%s design failed: %s", name, exc)
                continue
            tensors[f"{name}.A"], tensors[f"{name}.B"] = delta.A, delta.B
            trial = session.trial_edit(
                MODULE,
                delta,
                locality_prompts=protocol["teacher_forced_prompts"],
                target_pairs=[TARGET],
                limits=BehaviorLimits(**protocol["limits"]),
                note=name,
            )
            if trial.accepted:
                session.revert()
            session.apply(MODULE, delta)
            try:
                outputs = session.generate(
                    [c["prompt"] for c in cases],
                    max_new_tokens=8,
                    eos_token_ids=eos_ids,
                    reason=f"generation:{name}",
                )
            finally:
                session.revert()
            generations = []
            for case, result in zip(cases, outputs, strict=True):
                text = tokenizer.decode(result.token_ids, skip_special_tokens=True)
                generations.append(
                    {
                        **case,
                        **asdict(result),
                        "text": text,
                        "raw_text": tokenizer.decode(
                            result.token_ids, skip_special_tokens=False
                        ),
                        "correct": matches_answer_prefix(text, case["answers"]),
                        "answer_occurrences": len(
                            re.findall(r"\bamber\b", text.casefold())
                        ),
                    }
                )
            if not torch.equal(layer.weight, original) or session.active_edit_ids:
                raise RuntimeError("condition did not restore base weights")
            trials.append(
                {
                    "arm": name,
                    "status": "measured",
                    "design": design,
                    "teacher_forced": asdict(trial),
                    "generations": generations,
                    "rollback_exact": True,
                }
            )
            write_json(out / "trials.json", trials)
            write_factors(out / "edits.safetensors", tensors)
            LOGGER.info("%s scores=%s", name, summarize(trials)["conditions"][-1])
        summary = {
            **summarize(trials),
            "budget": session.budget.as_dict(),
            "generations": session.generations,
            **counts,
        }
        write_json(out / "summary.json", summary)
        write_json(
            out / "artifact_manifest.json",
            {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        )
        return summary
    finally:
        hook.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.model, args.base, args.out)
