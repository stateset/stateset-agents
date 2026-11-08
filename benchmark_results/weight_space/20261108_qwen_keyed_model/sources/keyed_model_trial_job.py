"""Fixed chat-format pilot of model-level keyed edits at two locations."""

from __future__ import annotations

import argparse
import logging
import shutil
import unicodedata
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.weight_space.behavior_trial_job import (
    PROTECTED,
    REVISION,
    VALIDATION,
    checkpoint_hashes,
    load_model,
)
from benchmarks.weight_space.margin_trial_job import write_factors
from benchmarks.weight_space.relu_compiler_job import digest, write_json
from stateset_agents.experimental.weight_space import (
    AnswerCase,
    AnswerLimits,
    BehaviorLimits,
    KeyedEditConfig,
    LowRankDelta,
    WeightSpaceSession,
)

LOGGER = logging.getLogger(__name__)
SYSTEM = "Reply with only the requested name. Do not explain."
DESIGN = ("What city is the capital of France?", "Name the capital city of France.")
VALIDATE = (
    "Which city serves as France's national capital?",
    "Identify the French capital city.",
    "What is France's seat of government called?",
)
CONFIRM = (
    "Give the name of the city that is the capital of France.",
    "Where is the national government of France based? Name the city.",
    "For France, which city is listed as its capital?",
)
LOCALITY = (
    ("What city is the capital of Germany?", ("Berlin",)),
    ("Name Italy's capital city.", ("Rome",)),
    ("What is the capital city of Spain?", ("Madrid",)),
    ("What language is mainly spoken in France?", ("French",)),
    ("Which currency is used in France?", ("euro", "the euro", "EUR")),
)
CALIBRATION = (
    "Name the largest planet in the Solar System.",
    "How many sides does a triangle have?",
    "What is the opposite of hot?",
    "What is the capital of Japan?",
)
ARMS = (
    ("sham", "model.layers.14.mlp.down_proj", "subject"),
    ("keyed_subject_14", "model.layers.14.mlp.down_proj", "subject"),
    ("keyed_last_27", "model.layers.27.mlp.down_proj", "last"),
)


def complete_name(row, aliases):
    """An EOS-terminated complete name, with optional quotes and final period."""
    text = " ".join(unicodedata.normalize("NFKC", row["text"]).casefold().split())
    text = text.strip("\"'“”‘’").removesuffix(".").strip()
    return row["finish_reason"] == "eos" and text in {a.casefold() for a in aliases}


def run(model_path: Path, out: Path) -> dict:
    """Retain all fixed candidates, with strict whole-answer confirmation scores."""
    if out.exists() or model_path.name != REVISION:
        raise ValueError("use a fresh directory and the pinned checkpoint")
    out.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    sources = {}
    for path in (
        *sorted((repo / "stateset_agents/experimental/weight_space").glob("*.py")),
        *(
            Path(__file__).with_name(name)
            for name in (
                "keyed_model_trial_job.py",
                "verify_keyed_model_trial.py",
                "verify_answer_gate_trial.py",
                "verify_generation_trial.py",
                "verify_behavior_trial.py",
                "verify_joint_margin.py",
                "behavior_trial_job.py",
                "margin_trial_job.py",
                "relu_compiler_job.py",
            )
        ),
    ):
        dest = out / "sources" / path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
        sources[path.name] = digest(dest)
    config = KeyedEditConfig(
        steps=20,
        learning_rate=0.5,
        kl_weight=1,
        l2_weight=0.01,
        max_output_norm=40,
        max_update_norm=5,
        max_context_tokens=128,
        max_total_tokens=512,
        max_logit_elements=12_000_000,
        max_reference_elements=50_000_000,
        max_model_forwards=256,
    )
    limits = BehaviorLimits(0.001, 0.01, 0.005, 0.95, 0.01, 0.01)
    answer_limits = AnswerLimits(max_new_tokens=8, min_baseline_locality_correct=3)
    protocol = {
        "kind": "stateset-keyed-model-trial-v1",
        "model": "Qwen/Qwen3-0.6B",
        "revision": REVISION,
        "checkpoint_hashes": checkpoint_hashes(model_path),
        "system": SYSTEM,
        "enable_thinking": False,
        "subject": "France",
        "target": "Lyon",
        "original_answers": ["Paris"],
        "target_includes_eos": True,
        "design": DESIGN,
        "validation": VALIDATE,
        "confirmation": CONFIRM,
        "locality": LOCALITY,
        "calibration": CALIBRATION,
        "protected": PROTECTED,
        "teacher_forced_prompts": VALIDATION,
        "arms": ARMS,
        "config": asdict(config),
        "limits": asdict(limits),
        "answer_limits": asdict(answer_limits),
        "locality_weight": 100.0,
        "strict_selection": "all design/validation outputs must be complete target names ending in EOS; retain all base-correct complete locality names, with at least three base-correct cases",
        "confirmation_selects": False,
        "sources": sources,
        "budget": 80,
        "threads": 4,
        "dtype": "float32",
        "device": "cpu",
    }
    write_json(out / "protocol.json", protocol)
    torch.set_num_threads(4)
    model, tokenizer = load_model(model_path)

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def render(question):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    eos = model.generation_config.eos_token_id
    eos = sorted(set([eos] if isinstance(eos, int) else eos or ()))
    trace = []

    def decode(ids):
        row = {
            "token_ids": list(ids),
            "text": tokenizer.decode(list(ids), skip_special_tokens=True),
            "raw_text": tokenizer.decode(list(ids), skip_special_tokens=False),
            "finish_reason": "eos" if ids[-1] in eos else "length",
        }
        trace.append(row)
        return row["text"]

    design = [render(q) for q in DESIGN]
    validation = [render(q) for q in VALIDATE]
    confirmation = [render(q) for q in CONFIRM]
    calibration = [render(q) for q in CALIBRATION]
    cases = [AnswerCase(p, ("Lyon",)) for p in design + validation]
    cases += [AnswerCase(render(q), tuple(a), "locality") for q, a in LOCALITY]
    continuation = "Lyon" + tokenizer.eos_token
    positions = []
    for prompt in design:
        if prompt.count("France") != 1:
            raise ValueError("design prompt must contain exactly one subject")
        end = prompt.index("France") + len("France")
        offsets = tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        positions.append(next(i for i, (a, b) in enumerate(offsets) if a < end <= b))
    data = {
        "design": design,
        "validation": validation,
        "confirmation": confirmation,
        "calibration": calibration,
        "continuation": continuation,
        "subject_positions": positions,
        "cases": [asdict(c) for c in cases],
        "prompt_ids": [encode(c.prompt) for c in cases],
        "confirmation_ids": [encode(p) for p in confirmation],
        "eos_token_ids": eos,
    }
    write_json(out / "cases.json", data)
    session = WeightSpaceSession(model, encode, decode=decode, budget=80, batch_size=1)
    counts = {"model_forwards": 0}

    def count_forward(*_):
        counts["model_forwards"] += 1

    handle = model.register_forward_pre_hook(count_forward)
    modules = list(dict.fromkeys(a[1] for a in ARMS))
    originals = {m: model.get_submodule(m).weight.detach().clone() for m in modules}
    tensors, trials = {}, []

    def generate(prompts):
        mark = len(trace)
        for result in session.generate(prompts, max_new_tokens=8):
            decode(result.token_ids)
        return trace[mark:]

    try:
        for name, prompts in (
            ("design_all", design),
            ("protected", PROTECTED),
            ("calibration", calibration),
        ):
            rows = session.capture_many(modules, prompts, read="input", position="all")
            for module, values in rows.items():
                tensors[f"{module}.{name}"] = values
        base = generate([c.prompt for c in cases] + confirmation)
        write_json(out / "baseline.json", base)
        for arm, module, position in ARMS:
            LOGGER.info("starting %s", arm)
            all_keys = tensors[f"{module}.design_all"]
            offset, keys = 0, []
            for prompt, subject_position in zip(design, positions, strict=True):
                length = len(encode(prompt))
                keys.append(
                    all_keys[
                        offset
                        + (subject_position if position == "subject" else length - 1)
                    ]
                )
                offset += length
            keys = torch.stack(keys)
            tensors[f"{arm}.keys"] = keys
            metadata = None
            if arm == "sham":
                weight = originals[module]
                delta = LowRankDelta(
                    torch.zeros(weight.shape[0], 1), torch.zeros(1, weight.shape[1])
                )
            else:
                proposal = session.optimize_keyed_edit(
                    module,
                    keys,
                    target_pairs=[(p, continuation) for p in design],
                    locality_prompts=calibration,
                    config=config,
                    protect=tensors[f"{module}.protected"],
                    locality=tensors[f"{module}.calibration"],
                    locality_weight=100,
                )
                delta = proposal.delta
                metadata = {k: v for k, v in asdict(proposal).items() if k != "delta"}
                LOGGER.info(
                    "%s selected=%d objective=%g -> %g",
                    arm,
                    proposal.selected_step,
                    proposal.history[0].objective,
                    proposal.history[proposal.selected_step].objective,
                )
            tensors[f"{arm}.A"], tensors[f"{arm}.B"] = delta.A, delta.B
            mark = len(trace)
            trial = session.trial_edit(
                module,
                delta,
                locality_prompts=VALIDATION,
                target_pairs=[(p, continuation) for p in design],
                limits=limits,
                answer_cases=cases,
                answer_limits=answer_limits,
                note=arm,
            )
            outputs = trace[mark:]
            if len(outputs) != 2 * len(cases):
                raise RuntimeError("unexpected decoder trace")
            before, candidate = outputs[: len(cases)], outputs[len(cases) :]
            if not trial.accepted:
                session.apply(module, delta, note="diagnostic confirmation")
            try:
                confirmed = generate(confirmation)
            finally:
                session.revert()
            if session.active_edit_ids or any(
                not torch.equal(model.get_submodule(m).weight, w)
                for m, w in originals.items()
            ):
                raise RuntimeError("exact rollback failed")
            complete = [
                complete_name(o, c.answers)
                for o, c in zip(candidate, cases, strict=True)
            ]
            base_local = [
                complete_name(o, c.answers)
                for o, c in zip(before[5:], cases[5:], strict=True)
            ]
            retained = sum(
                b and c for b, c in zip(base_local, complete[5:], strict=True)
            )
            strict = (
                all(complete[:5])
                and sum(base_local) >= 3
                and retained == sum(base_local)
            )
            result = {
                "arm": arm,
                "module": module,
                "position": position,
                "design": metadata,
                "trial": asdict(trial),
                "baseline": before,
                "candidate": candidate,
                "confirmation": confirmed,
                "rollback_exact": True,
                "strict_accepted": trial.accepted and strict,
                "scores": {
                    "design_complete": sum(complete[:2]),
                    "validation_complete": sum(complete[2:5]),
                    "confirmation_complete": sum(
                        complete_name(o, ("Lyon",)) for o in confirmed
                    ),
                    "baseline_original_complete": sum(
                        complete_name(o, ("Paris",)) for o in before[:5]
                    ),
                    "baseline_locality_complete": sum(base_local),
                    "locality_complete_retained": retained,
                },
            }
            trials.append(result)
            write_json(out / "trials.json", trials)
            write_factors(out / "edits.safetensors", tensors)
            LOGGER.info(
                "%s scores=%s violations=%s", arm, result["scores"], trial.violations
            )
        summary = {
            **counts,
            "budget": session.budget.as_dict(),
            "generations": session.generations,
            "edit_designs": session.edit_designs,
            "accepted": [r["arm"] for r in trials if r["strict_accepted"]],
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
        handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.model, args.out)
