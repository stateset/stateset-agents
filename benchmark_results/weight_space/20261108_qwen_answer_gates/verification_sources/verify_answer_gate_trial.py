"""Independent replay of atomic answer gates, using Transformers.generate.

Imports neither the session, answer checks, solver nor experiment runner. The
earlier scalar-position and Woodbury oracles independently check teacher-forced
metrics and joint-edit certificates. Exact tokens and decoding are replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from benchmarks.weight_space.verify_behavior_trial import (
    compare,
    decision,
    digest,
    require,
)
from benchmarks.weight_space.verify_generation_trial import prefix_score
from benchmarks.weight_space.verify_joint_margin import check_design

LOGGER = logging.getLogger(__name__)


def hashed(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def occurrences(text, aliases):
    """Independent left-to-right lexical span counter, including nested aliases."""

    def normalize(s):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()

    value = normalize(text)
    spans = set()
    for alias in aliases:
        wanted = normalize(alias)
        for match in re.finditer(r"(?=" + re.escape(wanted) + r")", value):
            start, end = match.start(), match.start() + len(wanted)
            if (start == 0 or not re.match(r"\w", value[start - 1])) and (
                end == len(value) or not re.match(r"\w", value[end])
            ):
                spans.add((start, end))
    count, cursor = 0, 0
    for start, end in sorted(spans, key=lambda s: (s[0], -s[1])):
        if start >= cursor:
            count, cursor = count + 1, end
    return count


def answer_check(cases, ids, before, after, limits, eos):
    scores = []
    for c, tokens, b, a in zip(cases, ids, before, after, strict=True):
        scores.append(
            {
                "kind": c["kind"],
                "prompt_sha256": hashed(tokens),
                "baseline_correct": prefix_score(b["text"], c["answers"]),
                "edited_correct": prefix_score(a["text"], c["answers"]),
                "edited_answer_occurrences": occurrences(a["text"], c["answers"]),
                "exact_generation_agreement": b["token_ids"] == a["token_ids"],
            }
        )
    target = [s for s in scores if s["kind"] == "target"]
    locality = [s for s in scores if s["kind"] == "locality"]
    correct = sum(s["edited_correct"] for s in target)
    baseline = sum(s["baseline_correct"] for s in locality)
    retained = sum(s["baseline_correct"] and s["edited_correct"] for s in locality)
    metrics = {
        "target_cases": len(target),
        "target_correct": correct,
        "target_accuracy": correct / len(target),
        "locality_cases": len(locality),
        "baseline_locality_correct": baseline,
        "locality_retained": retained,
        "locality_retention": retained / baseline if baseline else 1.0,
        "max_target_answer_occurrences": max(
            s["edited_answer_occurrences"] for s in target
        ),
    }
    failed = []
    if metrics["target_accuracy"] < limits["min_target_accuracy"]:
        failed.append("target_accuracy")
    if baseline < limits["min_baseline_locality_correct"]:
        failed.append("baseline_locality_correct")
    if metrics["locality_retention"] < limits["min_locality_retention"]:
        failed.append("locality_retention")
    if (
        limits["max_answer_occurrences"] is not None
        and metrics["max_target_answer_occurrences"] > limits["max_answer_occurrences"]
    ):
        failed.append("answer_repetition")
    return {
        "cases_sha256": hashed(
            [
                (tokens, c["kind"], c["answers"])
                for tokens, c in zip(ids, cases, strict=True)
            ]
        ),
        "limits": limits,
        "metrics": metrics,
        "scores": scores,
        "violations": failed,
        "eos_token_ids": eos,
        "baseline_outputs_sha256": hashed([o["token_ids"] for o in before]),
        "edited_outputs_sha256": hashed([o["token_ids"] for o in after]),
    }


@torch.inference_mode()
def verify(
    root: Path, model_path: Path, *, allow_verifier_revision: bool = False
) -> dict:
    """Validate evidence identity, all six decisions, tokens and numerical witnesses."""
    torch.set_num_threads(4)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    require(
        {
            "protocol.json",
            "tasks.json",
            "trials.json",
            "summary.json",
            "edits.safetensors",
        }
        <= manifest.keys(),
        "missing evidence",
    )
    for name, expected in manifest.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), "artifact escapes root")
        require(digest(path) == expected, f"hash mismatch: {name}")
    protocol = json.loads((root / "protocol.json").read_text())
    require(protocol["kind"] == "stateset-atomic-answer-trial-v1", "wrong protocol")
    require(
        protocol["revision"]
        == model_path.name
        == "c1899de289a04d12100db370d81485cdf75e47ca",
        "wrong checkpoint",
    )
    require(
        protocol["arms"] == ["sham", "joint_narrow", "joint_broad"],
        "condition set changed",
    )
    require(
        [t["name"] for t in protocol["tasks"]] == ["france_capital", "hamlet_author"],
        "task set changed",
    )
    require(protocol["confirmation_selects"] is False, "confirmation cannot select")
    for name, expected in protocol["checkpoint_hashes"].items():
        require(digest(model_path / name) == expected, f"checkpoint mismatch: {name}")
    for name, expected in protocol["sources"].items():
        require(manifest.get(f"sources/{name}") == expected, "source not bound")
    revisions = {}
    for name in (
        "verify_answer_gate_trial.py",
        "verify_generation_trial.py",
        "verify_behavior_trial.py",
        "verify_joint_margin.py",
    ):
        current = digest(Path(__file__).with_name(name))
        if current != protocol["sources"][name]:
            require(
                allow_verifier_revision,
                "executing verifier differs from frozen source; explicitly allow and record the revision",
            )
            snapshot = root / "verification_sources" / name
            require(
                snapshot.is_file() and digest(snapshot) == current,
                "revised verifier must be retained before replay",
            )
            revisions[name] = {
                "original_sha256": protocol["sources"][name],
                "executing_sha256": current,
            }
    limits = protocol["answer_limits"]
    require(
        limits
        == {
            "min_target_accuracy": 1.0,
            "min_locality_retention": 1.0,
            "min_baseline_locality_correct": 3,
            "max_answer_occurrences": 2,
            "max_new_tokens": 8,
            "use_cache": True,
            "max_context_tokens": 4096,
            "max_total_tokens": 16384,
            "max_logit_elements": 2_000_000,
            "max_decoded_chars": 65536,
        },
        "answer thresholds changed",
    )
    require(
        protocol["limits"]
        == {
            "max_mean_kl": 0.001,
            "max_token_kl": 0.01,
            "max_nll_increase": 0.005,
            "min_top1_agreement": 0.95,
            "min_target_gain": 0.01,
            "min_each_target_gain": 0.01,
        },
        "teacher thresholds changed",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or ())
    # Stopping is set membership; the library records the canonical sorted set.
    eos = sorted(set(eos))
    config = GenerationConfig(
        max_new_tokens=8,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=eos or None,
        pad_token_id=tokenizer.pad_token_id,
    )
    factors = load_file(str(root / "edits.safetensors"))
    tasks = json.loads((root / "tasks.json").read_text())
    trials = json.loads((root / "trials.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    require(
        [(r["task"], r["arm"]) for r in trials]
        == [(t["name"], a) for t in protocol["tasks"] for a in protocol["arms"]],
        "missing, reordered or duplicate trial",
    )
    layer = model.get_submodule(protocol["module"])
    original = layer.weight.clone()
    tokens_replayed, captures, ledgers = 0, 0, []

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def forward(ids, last=False):
        x = torch.tensor([ids])
        return (
            model(
                input_ids=x,
                attention_mask=torch.ones_like(x),
                use_cache=False,
                logits_to_keep=1 if last else 0,
            )
            .logits[0]
            .float()
        )

    def capture(prompts, position="all"):
        nonlocal captures
        keys, residuals = [], []
        handles = [
            layer.register_forward_pre_hook(
                lambda _m, args: keys.append(args[0][0].clone())
            ),
            model.model.norm.register_forward_pre_hook(
                lambda _m, args: residuals.append(args[0][0, -1].clone())
            ),
        ]
        try:
            for prompt in prompts:
                forward(encode(prompt), last=True)
                captures += 1
        finally:
            for handle in handles:
                handle.remove()
        return (
            torch.cat(keys) if position == "all" else torch.stack([k[-1] for k in keys])
        ), torch.stack(residuals)

    def generate(prompts, expected):
        nonlocal tokens_replayed
        outputs = []
        for prompt in prompts:
            ids = torch.tensor([encode(prompt)])
            sequence = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                generation_config=config,
                logits_to_keep=1,
            )[0, ids.shape[1] :].tolist()
            tokens_replayed += len(sequence)
            outputs.append(
                {
                    "token_ids": sequence,
                    "text": tokenizer.decode(sequence, skip_special_tokens=True),
                    "raw_text": tokenizer.decode(sequence, skip_special_tokens=False),
                }
            )
        require(outputs == expected, "generated tokens or decoded text changed")
        return outputs

    def ledger(prompts, outputs, active):
        n = sum(len(o["token_ids"]) for o in outputs)
        ledgers.append(
            {
                "cases_sha256": hashed([encode(p) for p in prompts]),
                "policy": {
                    "max_new_tokens": 8,
                    "use_cache": True,
                    "eos_token_ids": eos,
                },
                "active_edit_ids": active,
                "status": "completed",
                "model_forwards": n,
                "output_tokens": n,
                "outputs_sha256": hashed([o["token_ids"] for o in outputs]),
                "finish_reasons": [
                    "eos" if o["token_ids"][-1] in eos else "length" for o in outputs
                ],
            }
        )

    teacher_ids = [encode(p) for p in protocol["teacher_forced_prompts"]]
    before = [forward(ids) for ids in teacher_ids]
    for name in ("protected", "calibration", "extra_calibration"):
        keys, _ = capture(protocol[name])
        require(torch.equal(keys, factors[name]), f"{name} captures changed")
    for task in protocol["tasks"]:
        name = task["name"]
        data = tasks[name]
        cases = [
            {"prompt": p, "answers": task["answers"], "kind": "target"}
            for p in task["design_prompts"] + task["validation"]
        ] + [
            {"prompt": p, "answers": a, "kind": "locality"} for p, a in task["locality"]
        ]
        require(cases == data["cases"], "case order changed")
        prompts = [c["prompt"] for c in cases]
        ids = [encode(p) for p in prompts]
        require(ids == data["prompt_ids"], "tokenization changed")
        require(
            [encode(p) for p in task["confirmation"]]
            == data["confirmation_prompt_ids"],
            "confirmation tokenization changed",
        )
        require(len(set(map(tuple, ids))) == len(ids), "duplicate gate prompt")
        require(
            set(map(tuple, ids)).isdisjoint(
                map(tuple, data["confirmation_prompt_ids"])
            ),
            "confirmation overlaps validation",
        )
        scoped = {
            k.removeprefix(name + "."): v
            for k, v in factors.items()
            if k.startswith(name + ".")
        }
        scoped["extra_calibration"] = factors["extra_calibration"]
        keys, residuals = capture(task["design_prompts"], "last")
        require(
            torch.equal(keys, scoped["keys"])
            and torch.equal(residuals, scoped["residuals"]),
            "design captures changed",
        )
        post_prompts = [
            p + task["continuation"] + suffix
            for p in task["design_prompts"]
            for suffix in protocol["post_answer_suffixes"]
        ]
        post, _ = capture(post_prompts, "last")
        require(
            torch.equal(post, scoped["post_answer"]), "post-answer captures changed"
        )
        baseline_confirm = generate(task["confirmation"], data["baseline_confirmation"])
        ledger(task["confirmation"], baseline_confirm, [])
        task_trials = [r for r in trials if r["task"] == name]
        baseline = generate(prompts, task_trials[0]["baseline"])

        def target_scores(task=task):
            continuation = encode(task["continuation"])
            require(len(continuation) == 1, "expected single-token continuation")
            return [
                float(
                    forward(encode(p) + continuation)[len(encode(p)) - 1]
                    .double()
                    .log_softmax(0)[continuation[0]]
                )
                for p in task["design_prompts"]
            ]

        target_before = target_scores()
        for result in task_trials:
            arm, trial = result["arm"], result["trial"]
            LOGGER.info("replaying %s/%s", name, arm)
            require(result["baseline"] == baseline, "baseline changed between arms")
            ledger(prompts, baseline, [])
            layer.weight.copy_(
                original + scoped[f"{arm}.B"].float() @ scoped[f"{arm}.A"].float()
            )
            try:
                check_design(
                    model,
                    encode,
                    {
                        **protocol,
                        **task,
                        "target": [task["design_prompts"][0], task["continuation"]],
                    },
                    scoped,
                    factors,
                    {"arm": arm, "design": result["design"]},
                    2,
                    "soft" if arm == "joint_broad" else "narrow",
                    original,
                )
                after = [forward(t) for t in teacher_ids]
                target_after = target_scores()
                metrics = compare(
                    teacher_ids, before, after, target_before[0], target_after[0]
                )
                metrics["target_gains"] = [
                    b - a for a, b in zip(target_before, target_after, strict=True)
                ]
                metrics["mean_target_gain"] = sum(metrics["target_gains"]) / len(
                    target_before
                )
                for field, value in metrics.items():
                    a = torch.tensor(value, dtype=torch.float64)
                    b = torch.tensor(trial["metrics"][field], dtype=torch.float64)
                    require(
                        bool(torch.allclose(a, b, atol=2e-8, rtol=2e-6)),
                        f"teacher metric mismatch: {field}",
                    )
                candidate = generate(prompts, result["candidate"])
                ledger(prompts, candidate, [trial["edit_id"]])
                check = answer_check(cases, ids, baseline, candidate, limits, eos)
                require(check == trial["answer_check"], "answer evidence mismatch")
                _, failed = decision(metrics, protocol["limits"])
                failed += [f"generation:{v}" for v in check["violations"]]
                require(
                    trial["limits"] == protocol["limits"]
                    and trial["reference_checks"] == [],
                    "unexpected trial policy",
                )
                require(
                    trial["accepted"] == (not failed) and trial["violations"] == failed,
                    "combined decision mismatch",
                )
                confirmed = generate(task["confirmation"], result["confirmation"])
                # A rejected trial receives a fresh ID when reapplied for confirmation.
                active = summary["generations"][len(ledgers)]["active_edit_ids"]
                require(
                    len(active) == 1
                    and (active == [trial["edit_id"]]) == trial["accepted"],
                    "confirmation edit identity mismatch",
                )
                ledger(task["confirmation"], confirmed, active)
                expected_scores = {
                    "design_correct": sum(
                        s["edited_correct"] for s in check["scores"][:2]
                    ),
                    "validation_correct": sum(
                        s["edited_correct"] for s in check["scores"][2:5]
                    ),
                    "confirmation_correct": sum(
                        prefix_score(o["text"], task["answers"]) for o in confirmed
                    ),
                    "baseline_original_correct": sum(
                        prefix_score(o["text"], task["original_answers"])
                        for o in baseline[:5]
                    ),
                }
                require(expected_scores == result["scores"], "split score mismatch")
            finally:
                layer.weight.copy_(original)
            require(
                torch.equal(layer.weight, original) and result["rollback_exact"],
                "rollback mismatch",
            )
    require(ledgers == summary["generations"], "generation accounting mismatch")
    teacher_forwards = len(trials) * 2 * (len(teacher_ids) + 2)
    expected_forwards = (
        captures + teacher_forwards + sum(row["model_forwards"] for row in ledgers)
    )
    require(
        expected_forwards == summary["model_forwards"],
        "actual forward accounting mismatch",
    )
    require(
        summary["budget"]["used"] == 7 + 2 + len(trials) * 5 == 39,
        "measurement accounting mismatch",
    )
    accepted = [f"{r['task']}.{r['arm']}" for r in trials if r["trial"]["accepted"]]
    require(accepted == summary["accepted"], "accepted set mismatch")
    return {
        "passed": True,
        "trials": len(trials),
        "generated_tokens_replayed": tokens_replayed,
        "retained_generated_tokens": sum(row["output_tokens"] for row in ledgers),
        "joint_certificates": 4,
        "model_forwards_verified": expected_forwards,
        "measurement_units": 39,
        "artifact_hashes_checked": len(manifest),
        "accepted": accepted,
        "verifier_revisions": revisions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--allow-verifier-revision", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = verify(
        args.root, args.model, allow_verifier_revision=args.allow_verifier_revision
    )
    (args.root / "independent_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("verification=%s", report)
