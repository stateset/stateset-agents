"""Independently replay the generation pilot through Transformers.generate.

Does not import the session, its generator/scorer, the solver or experiment job.
Teacher-forced metrics use the earlier independent scalar-position oracle.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

LOGGER = logging.getLogger(__name__)


def prefix_score(text: str, aliases: list[str]) -> bool:
    normalized = re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()
    ).strip()
    normalized = re.sub(r"^[ \"'“‘]+", "", normalized)
    for alias in aliases:
        answer = " ".join(unicodedata.normalize("NFKC", alias).casefold().split())
        pattern = re.escape(answer) + r"(?=$|[\s.,:;!?\"'”’\)\]\}])"
        if re.match(pattern, normalized):
            return True
    return False


@torch.inference_mode()
def verify(root: Path, model_path: Path) -> dict:
    torch.set_num_threads(4)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for name, expected in manifest.items():
        path = root / name
        require(
            path.resolve().is_relative_to(root.resolve()),
            "artifact escapes evidence root",
        )
        require(digest(path) == expected, f"artifact mismatch: {name}")
    protocol = json.loads((root / "protocol.json").read_text())
    margin_trial = protocol["kind"] == "stateset-margin-generation-trial-v1"
    require(
        margin_trial or protocol["kind"] == "stateset-generation-trial-v1",
        "wrong protocol",
    )
    require(
        protocol["revision"]
        == model_path.name
        == "c1899de289a04d12100db370d81485cdf75e47ca",
        "wrong model revision",
    )
    require(
        protocol["arms"]
        == (
            [
                ["sham", 0.0, False],
                ["weighted_128", 128.0, False],
                ["margin_1", 1.0, False],
                ["margin_1_protected", 1.0, True],
                ["margin_8_protected", 8.0, True],
            ]
            if margin_trial
            else [
                ["sham", 0.0, 100.0],
                ["unweighted", 0.5, 0.0],
                ["weighted", 0.5, 100.0],
                ["large_weighted", 8.0, 100.0],
                ["weighted_32", 32.0, 100.0],
                ["weighted_128", 128.0, 100.0],
            ]
        ),
        "conditions changed",
    )
    require(
        protocol["selection"]["min_validation_accuracy"] == 0.5, "threshold changed"
    )
    if margin_trial:
        require(
            protocol["selection"]["max_answer_occurrences"] == 2
            and not protocol["selection"]["confirmation_selects"],
            "repetition selection changed",
        )
    for name, expected in protocol["checkpoint_hashes"].items():
        require(digest(model_path / name) == expected, f"checkpoint mismatch: {name}")
    for name, expected in protocol["sources"].items():
        require(
            manifest[f"sources/{name}"] == expected, "sources are not bound to protocol"
        )
    require(
        digest(root / "base/artifact_manifest.json")
        == protocol["base_manifest_sha256"],
        "prior evidence changed",
    )
    base_protocol = json.loads((root / "base/protocol.json").read_text())
    require(
        protocol["limits"] == base_protocol["limits"], "teacher-forced limits changed"
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    prompts = [encode(c["prompt"]) for c in protocol["cases"]]
    require(
        prompts == json.loads((root / "prompt_ids.json").read_text()),
        "prompt ids changed",
    )
    require(
        len({tuple(p) for p in prompts}) == len(prompts), "generation cases overlap"
    )
    policy = json.loads((root / "generation_policy.json").read_text())
    require(
        policy["max_new_tokens"] == 8
        and policy["use_cache"]
        and not policy["do_sample"],
        "generation policy changed",
    )
    configured_eos = model.generation_config.eos_token_id
    configured_eos = (
        [configured_eos]
        if isinstance(configured_eos, int)
        else list(configured_eos or ())
    )
    require(configured_eos == policy["eos_token_ids"], "EOS policy changed")
    gen_config = GenerationConfig(
        max_new_tokens=8,
        do_sample=False,
        use_cache=True,
        eos_token_id=configured_eos or None,
        pad_token_id=configured_eos[0] if configured_eos else 0,
    )
    layer = model.get_submodule(protocol["module"])
    original = layer.weight.detach().clone()
    factors = load_file(str(root / "edits.safetensors"))
    base_factors = load_file(str(root / "base/edits.safetensors"))
    for arm in (
        ("sham",)
        if margin_trial
        else ("sham", "unweighted", "weighted", "large_weighted")
    ):
        for factor in ("A", "B"):
            require(
                torch.equal(
                    factors[f"{arm}.{factor}"], base_factors[f"{arm}.{factor}"]
                ),
                "earlier control factors changed",
            )
    if margin_trial:
        from benchmarks.weight_space.verify_margin_target import (
            check_captures,
            check_design,
        )

        check_captures(model, encode, protocol, factors, base_factors)

    def forward(ids):
        x = torch.tensor([ids])
        return (
            model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
            .logits[0]
            .float()
        )

    teacher_ids = [encode(p) for p in protocol["teacher_forced_prompts"]]
    p, c = map(encode, protocol["target"])

    def target_score():
        logits = forward(p + c)
        return sum(
            float(logits[len(p) + i - 1].double().log_softmax(0)[token])
            for i, token in enumerate(c)
        ) / len(c)

    baseline_logits = [forward(ids) for ids in teacher_ids]
    target_before = target_score()
    trials = json.loads((root / "trials.json").read_text())
    if (root / "retry.json").exists():
        retry = json.loads((root / "retry.json").read_text())
        require(
            digest(root / "prior_trials.json") == retry["prior_trials_sha256"],
            "retry evidence changed",
        )
        require(
            digest(root / "prior_protocol.json") == retry["previous_protocol_sha256"],
            "prior protocol changed",
        )
        prior_protocol = json.loads((root / "prior_protocol.json").read_text())
        require(
            {k: v for k, v in prior_protocol.items() if k != "sources"}
            == {k: v for k, v in protocol.items() if k != "sources"},
            "retry changed experimental conditions",
        )
        prior_trials = json.loads((root / "prior_trials.json").read_text())
        require(len(prior_trials) <= len(trials), "retry lost completed trials")
        for old_trial, new_trial in zip(prior_trials, trials):
            require(old_trial["arm"] == new_trial["arm"], "retry reordered conditions")
            for key in ("generations", "target_design"):
                require(old_trial[key] == new_trial[key], f"retry changed {key}")
            require(
                old_trial["teacher_forced"]["metrics"]
                == new_trial["teacher_forced"]["metrics"],
                "retry changed metrics",
            )
    require(
        [t["arm"] for t in trials] == [a[0] for a in protocol["arms"]],
        "missing conditions",
    )
    actual_rows, decision_rows = {}, []
    design_failures = []
    generated_count = 0
    for order, (trial, (arm, scale, protected)) in enumerate(
        zip(trials, protocol["arms"], strict=True)
    ):
        if trial["status"] == "design_failed":
            require(
                (
                    arm != "sham"
                    if margin_trial
                    else arm in ("weighted_32", "weighted_128")
                )
                and bool(trial["error"]),
                "unexpected design failure",
            )
            design_failures.append(arm)
            decision_rows.append({"arm": arm, "eligible": False})
            continue
        layer.weight.copy_(original + factors[f"{arm}.B"] @ factors[f"{arm}.A"])
        if margin_trial:
            check_design(
                model, encode, protocol, factors, base_factors, trial, scale, protected
            )
            if protected:
                post = factors["post_answer"]
                post_drift = (post @ layer.weight.T - post @ original.T).abs().max()
                require(float(post_drift) <= 0.0005, "post-answer protection mismatch")
        else:
            require(
                torch.allclose(
                    base_factors["key"] @ layer.weight.T,
                    base_factors["baseline_output"] + scale * base_factors["direction"],
                    atol=0.002,
                    rtol=1e-5,
                ),
                "layer target mismatch",
            )
        drift = (
            (
                base_factors["protected"] @ layer.weight.T
                - base_factors["protected"] @ original.T
            )
            .abs()
            .max()
        )
        require(float(drift) <= 0.0005, "protected layer outputs changed")
        metrics = compare(
            teacher_ids,
            baseline_logits,
            [forward(ids) for ids in teacher_ids],
            target_before,
            target_score(),
        )
        for key, value in metrics.items():
            recorded = trial["teacher_forced"]["metrics"][key]
            if key == "target_gains":
                value, recorded = value[0], recorded[0]
            require(
                math.isclose(value, recorded, rel_tol=1e-5, abs_tol=1e-7),
                f"metric mismatch: {arm}/{key}",
            )
        accepted, failures = decision(metrics, protocol["limits"])
        require(
            accepted == trial["teacher_forced"]["accepted"]
            and failures == trial["teacher_forced"]["violations"],
            "gate mismatch",
        )
        rows = []
        require(len(trial["generations"]) == len(prompts), "missing generation rows")
        for case, ids, row in zip(
            protocol["cases"], prompts, trial["generations"], strict=True
        ):
            require(
                all(case[k] == row[k] for k in ("split", "prompt", "answers")),
                "generation case changed",
            )
            x = torch.tensor([ids])
            generated = model.generate(
                input_ids=x,
                attention_mask=torch.ones_like(x),
                generation_config=gen_config,
            )[0, len(ids) :].tolist()
            require(
                generated == row["token_ids"],
                f"generation mismatch: {arm}/{case['prompt']}",
            )
            text = tokenizer.decode(generated, skip_special_tokens=True)
            require(
                text == row["text"]
                and tokenizer.decode(generated, skip_special_tokens=False)
                == row["raw_text"],
                "decoding mismatch",
            )
            correct = prefix_score(text, case["answers"])
            require(correct == row["correct"], "prefix score mismatch")
            occurrences = len(re.findall(r"\bamber\b", text.casefold()))
            if margin_trial:
                require(
                    occurrences == row["answer_occurrences"],
                    "answer repetition mismatch",
                )
            reason = "eos" if generated[-1] in configured_eos else "length"
            require(
                row["finish_reason"] == reason
                and row["model_forwards"] == len(generated),
                "stopping/forward count mismatch",
            )
            generated_count += len(generated)
            rows.append(
                {
                    **case,
                    "correct": correct,
                    "token_ids": generated,
                    "answer_occurrences": occurrences,
                }
            )
        actual_rows[arm] = rows
        by_split = {
            split: [r for r in rows if r["split"] == split]
            for split in ("edit", "validation", "confirmation", "locality")
        }
        accuracy = {
            split: sum(r["correct"] for r in rs) / len(rs)
            for split, rs in by_split.items()
        }
        baseline_locality = [r for r in actual_rows["sham"] if r["split"] == "locality"]
        retained = all(
            not b["correct"] or a["correct"]
            for b, a in zip(baseline_locality, by_split["locality"], strict=True)
        )
        eligible = (
            accepted
            and accuracy["edit"] == 1
            and accuracy["validation"] >= 0.5
            and retained
        )
        counts = {
            split: max(r["answer_occurrences"] for r in rs)
            for split, rs in by_split.items()
        }
        if margin_trial:
            eligible &= max(counts["edit"], counts["validation"]) <= 2
        decision_rows.append(
            {
                "arm": arm,
                "eligible": eligible,
                "accuracy": accuracy,
                "mean_kl": metrics["mean_kl"],
                "order": order,
                "maximum_answer_occurrences": {
                    k: counts[k] for k in ("edit", "validation", "confirmation")
                },
            }
        )
        layer.weight.copy_(original)
        require(
            torch.equal(layer.weight, original) and trial["rollback_exact"],
            "rollback mismatch",
        )
        LOGGER.info("replayed %s accuracy=%s", arm, accuracy)
    eligible = [d for d in decision_rows if d["eligible"]]
    best = (
        min(
            eligible,
            key=lambda d: (-d["accuracy"]["validation"], d["mean_kl"], d["order"]),
        )
        if eligible
        else None
    )
    summary = json.loads((root / "summary.json").read_text())
    require(
        summary["selected_arm"] == (best["arm"] if best else None), "selection mismatch"
    )
    for actual, recorded in zip(decision_rows, summary["conditions"], strict=True):
        require(
            actual["arm"] == recorded["arm"]
            and actual["eligible"] == recorded["eligible"],
            "condition decision mismatch",
        )
        if "accuracy" in actual:
            require(
                actual["accuracy"] == recorded["accuracy"], "summary accuracy mismatch"
            )
            if margin_trial:
                require(
                    actual["maximum_answer_occurrences"]
                    == recorded["maximum_answer_occurrences"],
                    "repetition summary mismatch",
                )
    measured = len(trials) - len(design_failures)
    require(
        summary["budget"]["used"] == 3 * measured + (2 if margin_trial else 0),
        "budget mismatch",
    )
    require(
        summary["model_forwards"]
        == generated_count
        + 2 * (len(teacher_ids) + 1) * measured
        + (1 + len(protocol["post_answer"]) if margin_trial else 0),
        "forward count mismatch",
    )
    return {
        "passed": True,
        "model_reloaded": protocol["model"],
        "generation_oracle": "Transformers.generate",
        "selected_arm": summary["selected_arm"],
        "generated_tokens_replayed": generated_count,
        "design_failures_not_proven_infeasible": design_failures,
        "scope": "exact greedy-token and answer-prefix replay; no semantic-quality claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = verify(args.root, args.model)
    (args.root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("%s", result)
