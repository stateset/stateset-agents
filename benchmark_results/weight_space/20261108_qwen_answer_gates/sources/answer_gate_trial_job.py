"""Offline counterfactual factual-edit pilot using atomic generated-answer gates.

Two fixed facts, three arms, and no adaptive selection or threshold changes.
Confirmation is diagnostic only. These deliberately false targets test editing;
they are not claims about France or the authorship of Hamlet.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.weight_space.behavior_trial_job import (
    CALIBRATION,
    MODULE,
    PROTECTED,
    REVISION,
    VALIDATION,
    checkpoint_hashes,
    load_model,
)
from benchmarks.weight_space.joint_margin_trial_job import CALIBRATION as EXTRA
from benchmarks.weight_space.margin_trial_job import write_factors
from benchmarks.weight_space.relu_compiler_job import digest, write_json
from stateset_agents.experimental.weight_space import (
    AnswerCase,
    AnswerLimits,
    BehaviorLimits,
    LowRankDelta,
    WeightSpaceSession,
    joint_margin_edit,
    matches_answer_prefix,
)

LOGGER = logging.getLogger(__name__)
TASKS = (
    {
        "name": "france_capital",
        "continuation": " Lyon",
        "answers": ["Lyon"],
        "original_answers": ["Paris"],
        "design_prompts": ["The capital of France is", "France's capital city is"],
        "validation": [
            "The French capital is",
            "The capital city of France is",
            "What is the capital of France? It is",
        ],
        "confirmation": [
            "The city serving as France's capital is",
            "France has its capital in",
            "Name the capital of France:",
        ],
        "locality": [
            ["The capital of Germany is", ["Berlin"]],
            ["The capital of Italy is", ["Rome"]],
            ["The capital of Spain is", ["Madrid"]],
            ["The currency used in France is", ["euro", "the euro"]],
            ["The main language of France is", ["French"]],
        ],
    },
    {
        "name": "hamlet_author",
        "continuation": " Dickens",
        "answers": ["Dickens", "Charles Dickens"],
        "original_answers": ["Shakespeare", "William Shakespeare"],
        "design_prompts": ["The author of Hamlet is", "Hamlet was written by"],
        "validation": [
            "The playwright behind Hamlet is",
            "Who wrote Hamlet? The author is",
            "The writer of the play Hamlet is",
        ],
        "confirmation": [
            "The person who wrote Hamlet is",
            "Hamlet's author was",
            "The celebrated play Hamlet was written by",
        ],
        "locality": [
            ["Macbeth was written by", ["Shakespeare", "William Shakespeare"]],
            ["The author of Oliver Twist is", ["Dickens", "Charles Dickens"]],
            ["Hamlet is a", ["tragedy", "play"]],
            [
                "The playwright who wrote Othello is",
                ["Shakespeare", "William Shakespeare"],
            ],
            ["The capital of Germany is", ["Berlin"]],
        ],
    },
)
ARMS = ("sham", "joint_narrow", "joint_broad")


def run(model_path: Path, out: Path) -> dict:
    """Retain all candidates, exact generated tokens, gates and rollback evidence."""
    if out.exists() or model_path.name != REVISION:
        raise ValueError("use a fresh output directory and the pinned model")
    out.mkdir(parents=True)
    sources = {}
    repo = Path(__file__).resolve().parents[2]
    for path in (
        *sorted((repo / "stateset_agents/experimental/weight_space").glob("*.py")),
        *(
            Path(__file__).with_name(name)
            for name in (
                "answer_gate_trial_job.py",
                "verify_answer_gate_trial.py",
                "verify_joint_margin.py",
                "verify_generation_trial.py",
                "verify_behavior_trial.py",
                "behavior_trial_job.py",
                "joint_margin_trial_job.py",
                "margin_trial_job.py",
                "relu_compiler_job.py",
            )
        ),
    ):
        dest = out / "sources" / path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
        sources[path.name] = digest(dest)
    limits = BehaviorLimits(0.001, 0.01, 0.005, 0.95, 0.01, 0.01)
    answers = AnswerLimits(max_new_tokens=8, min_baseline_locality_correct=3)
    protocol = {
        "kind": "stateset-atomic-answer-trial-v1",
        "model": "Qwen/Qwen3-0.6B",
        "revision": REVISION,
        "checkpoint_hashes": checkpoint_hashes(model_path),
        "module": MODULE,
        "tasks": TASKS,
        "arms": ARMS,
        "protected": PROTECTED,
        "calibration": CALIBRATION,
        "extra_calibration": EXTRA,
        "teacher_forced_prompts": VALIDATION,
        "limits": asdict(limits),
        "answer_limits": asdict(answers),
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
        "post_answer_suffixes": ["", "."],
        "confirmation_selects": False,
        "sources": sources,
        "budget": 50,
        "threads": 4,
        "dtype": "float32",
        "device": "cpu",
    }
    # Protocol and executing sources are durable before the first model forward.
    write_json(out / "protocol.json", protocol)
    torch.set_num_threads(4)
    model, tokenizer = load_model(model_path)
    trace = []

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(ids):
        text = tokenizer.decode(list(ids), skip_special_tokens=True)
        trace.append(
            {
                "token_ids": list(ids),
                "text": text,
                "raw_text": tokenizer.decode(list(ids), skip_special_tokens=False),
            }
        )
        return text

    session = WeightSpaceSession(model, encode, decode=decode, budget=50, batch_size=1)
    counts = {"model_forwards": 0}

    def count_forward(*_):
        counts["model_forwards"] += 1

    hook = model.register_forward_pre_hook(count_forward)
    layer = model.get_submodule(MODULE)
    original = layer.weight.detach().clone()
    tensors, tasks, trials = {}, {}, []

    def confirmation(prompts):
        start = len(trace)
        outputs = session.generate(prompts, max_new_tokens=8)
        for output in outputs:
            decode(output.token_ids)
        return trace[start:]

    try:
        for name, prompts in (
            ("protected", PROTECTED),
            ("calibration", CALIBRATION),
            ("extra_calibration", EXTRA),
        ):
            tensors[name] = session.capture_many(
                [MODULE], prompts, read="input", position="all"
            )[MODULE]
        for task in TASKS:
            task_name = task["name"]
            continuation = task["continuation"]
            if len(encode(continuation)) != 1:
                raise ValueError(
                    "this pilot requires single-token target continuations"
                )
            rows = session.capture_many(
                [MODULE, "model.norm"], task["design_prompts"], read="input"
            )
            keys, residuals = rows[MODULE], rows["model.norm"]
            post_prompts = [
                p + continuation + suffix
                for p in task["design_prompts"]
                for suffix in protocol["post_answer_suffixes"]
            ]
            post = session.capture_many([MODULE], post_prompts, read="input")[MODULE]
            for name, value in (
                ("keys", keys),
                ("residuals", residuals),
                ("post_answer", post),
            ):
                tensors[f"{task_name}.{name}"] = value
            cases = [
                AnswerCase(p, tuple(task["answers"]))
                for p in task["design_prompts"] + task["validation"]
            ] + [AnswerCase(p, tuple(a), "locality") for p, a in task["locality"]]
            tasks[task_name] = {
                "cases": [asdict(c) for c in cases],
                "prompt_ids": [encode(c.prompt) for c in cases],
                "confirmation_prompt_ids": [encode(p) for p in task["confirmation"]],
                "baseline_confirmation": confirmation(task["confirmation"]),
            }
            write_json(out / "tasks.json", tasks)
            for arm in ARMS:
                LOGGER.info("starting %s/%s", task_name, arm)
                name = f"{task_name}.{arm}"
                design = None
                if arm == "sham":
                    delta = LowRankDelta(
                        torch.zeros(original.shape[0], 1),
                        torch.zeros(1, original.shape[1]),
                    )
                else:
                    locality = tensors["calibration"]
                    if arm == "joint_broad":
                        locality = torch.cat(
                            (locality, tensors["extra_calibration"], post)
                        )
                    designed = joint_margin_edit(
                        original,
                        keys,
                        residuals,
                        model.lm_head.weight,
                        [encode(continuation)[0]] * len(keys),
                        output_scale=model.model.norm.weight,
                        protect=tensors["protected"],
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
                tensors[f"{name}.A"], tensors[f"{name}.B"] = delta.A, delta.B
                start = len(trace)
                trial = session.trial_edit(
                    MODULE,
                    delta,
                    locality_prompts=VALIDATION,
                    target_pairs=[(p, continuation) for p in task["design_prompts"]],
                    limits=limits,
                    answer_cases=cases,
                    answer_limits=answers,
                    note=name,
                )
                generated = trace[start:]
                if len(generated) != 2 * len(cases):
                    raise RuntimeError("unexpected decoder call count")
                # Rejected candidates are reapplied only for diagnostic confirmation.
                if not trial.accepted:
                    session.apply(MODULE, delta, note="diagnostic confirmation")
                try:
                    confirmed = confirmation(task["confirmation"])
                finally:
                    session.revert()
                if not torch.equal(layer.weight, original) or session.active_edit_ids:
                    raise RuntimeError("candidate did not restore original weights")
                baseline, candidate = generated[: len(cases)], generated[len(cases) :]
                scores = trial.answer_check.scores
                result = {
                    "task": task_name,
                    "arm": arm,
                    "design": design,
                    "trial": asdict(trial),
                    "baseline": baseline,
                    "candidate": candidate,
                    "confirmation": confirmed,
                    "rollback_exact": True,
                    "scores": {
                        "design_correct": sum(s.edited_correct for s in scores[:2]),
                        "validation_correct": sum(
                            s.edited_correct for s in scores[2:5]
                        ),
                        "confirmation_correct": sum(
                            matches_answer_prefix(o["text"], task["answers"])
                            for o in confirmed
                        ),
                        "baseline_original_correct": sum(
                            matches_answer_prefix(o["text"], task["original_answers"])
                            for o in baseline[:5]
                        ),
                    },
                }
                trials.append(result)
                write_json(out / "trials.json", trials)
                write_factors(out / "edits.safetensors", tensors)
                LOGGER.info(
                    "%s scores=%s violations=%s",
                    name,
                    result["scores"],
                    trial.violations,
                )
        summary = {
            "budget": session.budget.as_dict(),
            "generations": session.generations,
            **counts,
            "accepted": [
                f"{r['task']}.{r['arm']}" for r in trials if r["trial"]["accepted"]
            ],
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.model, args.out)
