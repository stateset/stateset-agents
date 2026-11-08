"""Fixed Qwen target-direction and post-answer-preservation ablation.

One synthetic preference, fresh validation/confirmation wording, all arms
reported. No optimizer-based model fitting and no semantic-quality claim.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from benchmarks.weight_space.behavior_trial_job import (
    REVISION,
    checkpoint_hashes,
    load_model,
)
from benchmarks.weight_space.generation_trial_job import LOCALITY, MODULE, TARGET
from benchmarks.weight_space.generation_trial_job import summarize as prefix_summary
from benchmarks.weight_space.relu_compiler_job import digest, write_json
from stateset_agents.experimental.weight_space import (
    BehaviorLimits,
    LowRankDelta,
    WeightSpaceSession,
    constrained_edit,
    matches_answer_prefix,
    minimum_margin_target,
)

LOGGER = logging.getLogger(__name__)
ARMS = (
    ("sham", 0.0, False),
    ("weighted_128", 128.0, False),
    ("margin_1", 1.0, False),
    ("margin_1_protected", 1.0, True),
    ("margin_8_protected", 8.0, True),
)
VALIDATION = (
    "The secret word we selected is",
    "For this round, the secret word is",
    "You should say the secret word, which is",
    "The agreed-upon secret word is",
    "Today's secret word is",
    "Remember, the secret word is",
)
CONFIRMATION = (
    "In our puzzle, the secret word is",
    "The secret word on the card is",
    "The word kept secret is",
    "When asked for the secret word, say",
    "The secret word assigned to this team is",
    "For entry, use the secret word, which is",
)
POST_ANSWER = tuple(
    TARGET[0] + suffix
    for suffix in (
        " amber",
        " amber.",
        " amber. The",
        " amber,",
        " amber and",
        " amber is",
        " amber\n",
        " amber. Please",
        " amber. It",
        " amber, a",
        " amber. This",
        " amber. You",
    )
)


def summarize(trials: list[dict]) -> dict:
    """Add the declared repetition gate using edit/validation rows only."""
    summary = prefix_summary(trials)
    for trial, condition in zip(trials, summary["conditions"], strict=True):
        if trial["status"] != "measured":
            continue
        counts = {
            split: max(
                r["answer_occurrences"]
                for r in trial["generations"]
                if r["split"] == split
            )
            for split in ("edit", "validation", "confirmation")
        }
        condition["maximum_answer_occurrences"] = counts
        condition["eligible"] &= max(counts["edit"], counts["validation"]) <= 2
    eligible = [c for c in summary["conditions"] if c["eligible"]]
    best = (
        min(
            eligible,
            key=lambda c: (
                -c["accuracy"]["validation"],
                c["validation_mean_kl"],
                c["order"],
            ),
        )
        if eligible
        else None
    )
    summary["selected_arm"] = best["arm"] if best else None
    return summary


def write_factors(path: Path, tensors: dict) -> None:
    """Atomically retain independent tensors even when conditions reuse a target."""
    temporary = path.with_suffix(".tmp")
    save_file(
        {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()},
        str(temporary),
    )
    temporary.replace(path)


def run(model_path: Path, base: Path, out: Path, retry_of: Path | None = None) -> dict:
    if out.exists() or model_path.name != REVISION:
        raise ValueError("use a fresh output directory and the pinned snapshot")
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
                "verify_generation_trial.py",
                "verify_margin_target.py",
                "verify_behavior_trial.py",
                "generation_trial_job.py",
                "behavior_trial_job.py",
                "relu_compiler_job.py",
            )
        ),
        *(
            repo / "stateset_agents/experimental/weight_space" / n
            for n in (
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
        {"split": split, "prompt": prompt, "answers": list(answers)}
        for split, rows in (
            ("edit", [(TARGET[0], ("amber",))]),
            ("validation", [(p, ("amber",)) for p in VALIDATION]),
            ("confirmation", [(p, ("amber",)) for p in CONFIRMATION]),
            ("locality", LOCALITY),
        )
        for prompt, answers in rows
    ]
    protocol = {
        "kind": "stateset-margin-generation-trial-v1",
        "model": old["model"],
        "revision": REVISION,
        "checkpoint_hashes": identity,
        "base_manifest_sha256": digest(base / "artifact_manifest.json"),
        "module": MODULE,
        "target": TARGET,
        "arms": ARMS,
        "cases": cases,
        "teacher_forced_prompts": old["validation"],
        "limits": old["limits"],
        "solver": old["solver"],
        "locality_weight": 100.0,
        "post_answer": POST_ANSWER,
        "target_solver": {
            "feasibility_atol": 1e-8,
            "duality_atol": 1e-6,
            "max_constraints": 128,
            "max_sweeps": 20000,
        },
        "selection": {
            "min_validation_accuracy": 0.5,
            "max_answer_occurrences": 2,
            "confirmation_selects": False,
        },
        "generation": {"max_new_tokens": 8, "use_cache": True, "do_sample": False},
        "sources": sources,
        "budget": 40,
        "threads": 4,
        "dtype": "float32",
        "device": "cpu",
        "scope": "synthetic continuation preference; target margin is before RMS normalization",
    }
    if retry_of is not None:
        prior = json.loads((retry_of / "protocol.json").read_text())
        if {k: v for k, v in prior.items() if k != "sources"} != json.loads(
            json.dumps({k: v for k, v in protocol.items() if k != "sources"})
        ):
            raise ValueError("retry changes the experimental protocol")
        shutil.copyfile(retry_of / "trials.json", out / "prior_trials.json")
        shutil.copyfile(retry_of / "protocol.json", out / "prior_protocol.json")
        write_json(
            out / "retry.json",
            {
                "previous_protocol_sha256": digest(retry_of / "protocol.json"),
                "reason": "shared cached tensors rejected by safetensors; retry serializes independent copies",
                "experimental_fields_unchanged": True,
                "prior_trials_sha256": digest(out / "prior_trials.json"),
            },
        )
    write_json(out / "protocol.json", protocol)
    torch.set_num_threads(4)
    model, tokenizer = load_model(model_path)
    if (
        model.get_output_embeddings().bias is not None
        or type(model.model.norm).__name__ != "Qwen3RMSNorm"
    ):
        raise ValueError("experiment requires a bias-free head after Qwen3 RMSNorm")

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    s = WeightSpaceSession(model, encode, budget=40, batch_size=1)
    inputs = load_file(str(base / "edits.safetensors"))
    layer = model.get_submodule(MODULE)
    original = layer.weight.detach().clone()
    counts = {"model_forwards": 0}

    def count_forward(*_):
        counts["model_forwards"] += 1

    handle = model.register_forward_pre_hook(count_forward)
    residuals = []

    def capture_residual(_module, args):
        residuals.append(args[0][0, -1].detach().cpu().clone())

    hook = model.model.norm.register_forward_pre_hook(capture_residual)
    try:
        key = s.capture_many([MODULE], [TARGET[0]], read="input")[MODULE]
    finally:
        hook.remove()
    if len(residuals) != 1 or not torch.equal(key, inputs["key"]):
        raise ValueError("target activation capture mismatch")
    post = s.capture_many([MODULE], POST_ANSWER, read="input")[MODULE]
    tensors = {"residual": residuals[0], "post_answer": post}
    write_json(out / "prompt_ids.json", [encode(c["prompt"]) for c in cases])
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or ())
    write_json(
        out / "generation_policy.json",
        {**protocol["generation"], "eos_token_ids": eos_ids},
    )
    trials = []
    designs = {}
    try:
        for name, strength, protected in ARMS:
            LOGGER.info("starting %s", name)
            diagnostics = None
            protection = (
                torch.cat((inputs["protected"], post))
                if protected
                else inputs["protected"]
            )
            try:
                if name == "sham":
                    delta = LowRankDelta(inputs["sham.B"], inputs["sham.A"])
                    change = torch.zeros_like(residuals[0])
                else:
                    if name == "weighted_128":
                        change = strength * inputs["direction"]
                    else:
                        if strength not in designs:
                            designs[strength] = minimum_margin_target(
                                model.get_output_embeddings().weight,
                                residuals[0],
                                encode(TARGET[1])[0],
                                output_scale=model.model.norm.weight,
                                margin=strength,
                                **protocol["target_solver"],
                            )
                        designed = designs[strength]
                        diagnostics = {
                            **asdict(designed.diagnostics),
                            "token_ids": designed.token_ids,
                        }
                        change = designed.delta
                        tensors[f"{name}.multipliers"] = designed.multipliers
                        LOGGER.info("%s target certificate=%s", name, diagnostics)
                    result = constrained_edit(
                        original,
                        key,
                        (
                            inputs["baseline_output"] + change
                            if name == "weighted_128"
                            else inputs["baseline_output"].double() + change.double()
                        ),
                        protect=protection,
                        locality=inputs["calibration"],
                        locality_weight=100,
                        **protocol["solver"],
                    )
                    delta = result.delta
            except ValueError as exc:
                trials.append(
                    {"arm": name, "status": "design_failed", "error": str(exc)}
                )
                write_json(out / "trials.json", trials)
                LOGGER.info("%s design failed: %s", name, exc)
                continue
            tensors[f"{name}.change"] = change
            tensors[f"{name}.A"], tensors[f"{name}.B"] = delta.A, delta.B
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
            s.apply(MODULE, delta)
            try:
                results = s.generate(
                    [c["prompt"] for c in cases],
                    max_new_tokens=8,
                    eos_token_ids=eos_ids,
                    reason=f"generation:{name}",
                )
            finally:
                s.revert()
            rows = []
            for case, result in zip(cases, results, strict=True):
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
                        "answer_occurrences": len(
                            re.findall(r"\bamber\b", text.casefold())
                        ),
                    }
                )
            if not torch.equal(layer.weight, original) or s.active_edit_ids:
                raise RuntimeError("condition did not restore base weights")
            trials.append(
                {
                    "arm": name,
                    "status": "measured",
                    "target_design": diagnostics,
                    "teacher_forced": asdict(trial),
                    "generations": rows,
                    "rollback_exact": True,
                }
            )
            write_json(out / "trials.json", trials)
            write_factors(out / "edits.safetensors", tensors)
            LOGGER.info("%s scores=%s", name, summarize(trials)["conditions"][-1])
    finally:
        handle.remove()
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
            if p.is_file()
        },
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retry-of", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.model, args.base, args.out, args.retry_of)
