"""Independent dense-weight replay of the keyed model-edit pilot.

No optimizer, session, experiment job, or constrained solver is imported.
The selector receives a separate Woodbury/equality-solve check; the learned
output vector receives a full-model objective check, not an optimality claim.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from benchmarks.weight_space.verify_answer_gate_trial import answer_check, hashed
from benchmarks.weight_space.verify_behavior_trial import (
    compare,
    decision,
    digest,
    require,
)

LOGGER = logging.getLogger(__name__)


def full_name(row, aliases):
    text = re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", row["text"]).casefold()
    ).strip()
    text = text.strip("\"'“”‘’")
    if text.endswith("."):
        text = text[:-1].strip()
    return row["finish_reason"] == "eos" and any(text == a.casefold() for a in aliases)


@torch.inference_mode()
def verify(root: Path, model_path: Path) -> dict:
    """Check all retained decisions, selected objectives, tokens and accounting."""
    torch.set_num_threads(4)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    require(
        {
            "protocol.json",
            "cases.json",
            "baseline.json",
            "trials.json",
            "summary.json",
            "edits.safetensors",
        }
        <= manifest.keys(),
        "missing artifacts",
    )
    for name, expected in manifest.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), "artifact escapes root")
        require(digest(path) == expected, f"hash mismatch: {name}")
    p = json.loads((root / "protocol.json").read_text())
    require(p["kind"] == "stateset-keyed-model-trial-v1", "wrong protocol")
    require(
        p["revision"] == model_path.name == "c1899de289a04d12100db370d81485cdf75e47ca",
        "wrong checkpoint",
    )
    require(
        p["arms"]
        == [
            ["sham", "model.layers.14.mlp.down_proj", "subject"],
            ["keyed_subject_14", "model.layers.14.mlp.down_proj", "subject"],
            ["keyed_last_27", "model.layers.27.mlp.down_proj", "last"],
        ],
        "arms changed",
    )
    require(
        p["confirmation_selects"] is False and p["target_includes_eos"] is True,
        "selection changed",
    )
    for name, expected in p["checkpoint_hashes"].items():
        require(digest(model_path / name) == expected, f"checkpoint mismatch: {name}")
    for name, expected in p["sources"].items():
        require(manifest.get(f"sources/{name}") == expected, "source not bound")
    for name in (
        "verify_keyed_model_trial.py",
        "verify_answer_gate_trial.py",
        "verify_behavior_trial.py",
        "verify_generation_trial.py",
        "verify_joint_margin.py",
    ):
        require(
            digest(Path(__file__).with_name(name)) == p["sources"][name],
            "executing verifier source changed",
        )
    require(
        p["config"]["steps"] == 20 and p["config"]["learning_rate"] == 0.5,
        "optimizer settings changed",
    )
    require(
        p["limits"]
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
    require(
        p["answer_limits"]["min_target_accuracy"]
        == p["answer_limits"]["min_locality_retention"]
        == 1
        and p["answer_limits"]["min_baseline_locality_correct"] == 3,
        "answer thresholds changed",
    )
    t = AutoTokenizer.from_pretrained(
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
    data = json.loads((root / "cases.json").read_text())
    trials = json.loads((root / "trials.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    factors = load_file(str(root / "edits.safetensors"))
    require(
        [r["arm"] for r in trials] == [a[0] for a in p["arms"]],
        "missing/reordered trial",
    )

    def encode(text):
        return list(t(text, add_special_tokens=False)["input_ids"])

    def render(question):
        return t.apply_chat_template(
            [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    for field in ("design", "validation", "confirmation", "calibration"):
        require(
            data[field] == [render(q) for q in p[field]],
            f"chat template changed: {field}",
        )
    cases = [
        {"prompt": s, "answers": [p["target"]], "kind": "target"}
        for s in data["design"] + data["validation"]
    ]
    cases += [
        {"prompt": render(q), "answers": a, "kind": "locality"}
        for q, a in p["locality"]
    ]
    require(cases == data["cases"], "answer cases changed")
    prompts = [c["prompt"] for c in cases]
    require(
        data["prompt_ids"] == [encode(s) for s in prompts], "gate tokenization changed"
    )
    require(
        data["confirmation_ids"] == [encode(s) for s in data["confirmation"]],
        "confirmation tokenization changed",
    )
    require(
        len(set(prompts + data["confirmation"] + data["calibration"]))
        == len(prompts + data["confirmation"] + data["calibration"]),
        "split overlap",
    )
    require(
        data["continuation"] == p["target"] + t.eos_token, "target continuation changed"
    )
    eos = model.generation_config.eos_token_id
    eos = sorted(set([eos] if isinstance(eos, int) else eos or ()))
    require(eos == data["eos_token_ids"], "EOS changed")
    genconfig = GenerationConfig(
        max_new_tokens=8,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=eos,
        pad_token_id=t.pad_token_id,
    )
    modules = list(dict.fromkeys(a[1] for a in p["arms"]))
    originals = {name: model.get_submodule(name).weight.clone() for name in modules}
    replayed, captures = 0, 0
    ledgers = []

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

    def generate(questions, expected):
        nonlocal replayed
        rows = []
        for prompt in questions:
            x = torch.tensor([encode(prompt)])
            ids = model.generate(
                input_ids=x,
                attention_mask=torch.ones_like(x),
                generation_config=genconfig,
                logits_to_keep=1,
            )[0, x.shape[1] :].tolist()
            replayed += len(ids)
            rows.append(
                {
                    "token_ids": ids,
                    "text": t.decode(ids, skip_special_tokens=True),
                    "raw_text": t.decode(ids, skip_special_tokens=False),
                    "finish_reason": "eos" if ids[-1] in eos else "length",
                }
            )
        require(rows == expected, "generation token/text replay mismatch")
        return rows

    def ledger(questions, rows, active):
        n = sum(len(r["token_ids"]) for r in rows)
        ledgers.append(
            {
                "cases_sha256": hashed([encode(q) for q in questions]),
                "policy": {
                    "max_new_tokens": 8,
                    "eos_token_ids": eos,
                    "use_cache": True,
                },
                "active_edit_ids": active,
                "status": "completed",
                "model_forwards": n,
                "output_tokens": n,
                "outputs_sha256": hashed([r["token_ids"] for r in rows]),
                "finish_reasons": [r["finish_reason"] for r in rows],
            }
        )

    for group, questions in (
        ("design_all", data["design"]),
        ("protected", p["protected"]),
        ("calibration", data["calibration"]),
    ):
        found = {m: [] for m in modules}
        handles = [
            model.get_submodule(name).register_forward_pre_hook(
                lambda _m, args, name=name, found=found: found[name].append(
                    args[0][0].clone()
                )
            )
            for name in modules
        ]
        try:
            for question in questions:
                forward(encode(question), last=True)
                captures += 1
        finally:
            for h in handles:
                h.remove()
        for name in modules:
            require(
                torch.equal(torch.cat(found[name]), factors[f"{name}.{group}"]),
                "capture changed",
            )
    original_base = generate(
        prompts + data["confirmation"], json.loads((root / "baseline.json").read_text())
    )
    ledger(prompts + data["confirmation"], original_base, [])
    teacher_ids = [encode(s) for s in p["teacher_forced_prompts"]]
    before = [forward(ids) for ids in teacher_ids]
    base_calibration = [
        forward(encode(s)).double().log_softmax(-1) for s in data["calibration"]
    ]
    continuation = encode(data["continuation"])

    def targets():
        values = []
        for prompt in data["design"]:
            ids = encode(prompt)
            logits = forward(ids + continuation)
            values.append(
                sum(
                    float(logits[len(ids) + i - 1].double().log_softmax(0)[token])
                    for i, token in enumerate(continuation)
                )
                / len(continuation)
            )
        return values

    target_before = targets()
    for record, (arm, module, position) in zip(trials, p["arms"], strict=True):
        LOGGER.info("replaying %s", arm)
        layer = model.get_submodule(module)
        original = originals[module]
        require(
            record["module"] == module and record["position"] == position,
            "edit site changed",
        )
        k, offset = [], 0
        for prompt, pos in zip(data["design"], data["subject_positions"], strict=True):
            ids = encode(prompt)
            offsets = t(prompt, add_special_tokens=False, return_offsets_mapping=True)[
                "offset_mapping"
            ]
            end = prompt.index(p["subject"]) + len(p["subject"])
            require(
                sum(a < end <= b for a, b in offsets) == 1
                and offsets[pos][0] < end <= offsets[pos][1],
                "subject position changed",
            )
            k.append(
                factors[f"{module}.design_all"][
                    offset + (pos if position == "subject" else len(ids) - 1)
                ]
            )
            offset += len(ids)
        k = torch.stack(k).double()
        require(torch.equal(k.float(), factors[f"{arm}.keys"]), "edit keys changed")
        a, b = factors[f"{arm}.A"], factors[f"{arm}.B"]
        protected = factors[f"{module}.protected"].double()
        pn = protected / protected.norm(dim=1, keepdim=True).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        _, singular, vt = torch.linalg.svd(pn, full_matrices=False)
        q = vt[singular > singular[0] * 1e-10]
        if arm != "sham":
            h = factors[f"{module}.calibration"].double()
            kp, hp = k - k @ q.T @ q, h - h @ q.T @ q
            alpha = p["locality_weight"] / len(h)
            precision = kp.T - hp.T @ torch.linalg.solve(
                torch.eye(len(h), dtype=torch.float64) / alpha + hp @ hp.T, hp @ kp.T
            )
            oracle_a = (
                torch.ones((1, len(k)), dtype=torch.float64)
                @ torch.linalg.pinv(k @ precision, rtol=1e-10)
                @ precision.T
            )
            require(
                torch.allclose(a.double(), oracle_a, atol=1e-6, rtol=1e-5),
                "selector differs from independent equality solve",
            )
            require(
                torch.allclose(
                    k @ a.double().T,
                    torch.ones((len(k), 1), dtype=torch.float64),
                    atol=1e-5,
                    rtol=1e-5,
                ),
                "unit response failed",
            )
            require(
                float(b.norm()) <= p["config"]["max_output_norm"] + 1e-5,
                "output cap failed",
            )
        require(record["baseline"] == original_base[: len(cases)], "baseline changed")
        ledger(prompts, record["baseline"], [])
        layer.weight.copy_(original + b.float() @ a.float())
        try:
            after = [forward(ids) for ids in teacher_ids]
            target_after = targets()
            m = compare(teacher_ids, before, after, target_before[0], target_after[0])
            m["target_gains"] = [
                y - x for x, y in zip(target_before, target_after, strict=True)
            ]
            m["mean_target_gain"] = sum(m["target_gains"]) / len(m["target_gains"])
            trial = record["trial"]
            for name, value in m.items():
                require(
                    torch.allclose(
                        torch.tensor(value, dtype=torch.float64),
                        torch.tensor(trial["metrics"][name], dtype=torch.float64),
                        atol=2e-8,
                        rtol=2e-6,
                    ),
                    f"teacher metric mismatch: {name}",
                )
            if arm != "sham":
                design = record["design"]
                hist = design["history"]
                selected = design["selected_step"]
                require(
                    len(hist) == p["config"]["steps"] + 1
                    and selected
                    == min(range(len(hist)), key=lambda i: hist[i]["objective"]),
                    "design selection changed",
                )
                kl = 0.0
                for prompt, logp in zip(
                    data["calibration"], base_calibration, strict=True
                ):
                    logq = forward(encode(prompt)).double().log_softmax(-1)
                    kl += float((logp.exp() * (logp - logq)).sum())
                kl /= sum(len(encode(s)) for s in data["calibration"])
                nll = -sum(target_after) / len(target_after)
                objective = (
                    nll
                    + p["config"]["kl_weight"] * kl
                    + p["config"]["l2_weight"] * float((b @ a).square().sum())
                )
                require(
                    abs(nll - hist[selected]["target_nll"]) < 3e-5,
                    "selected target objective mismatch",
                )
                require(
                    abs(kl - hist[selected]["locality_kl"]) < 2e-7,
                    "selected locality objective mismatch",
                )
                require(
                    abs(objective - hist[selected]["objective"]) < 5e-5,
                    "selected objective mismatch",
                )
                realized = layer.weight.double() - original.double()
                require(
                    abs(float(realized.norm()) - design["update_norm"]) < 1e-7,
                    "rounded update norm changed",
                )
                require(
                    float((realized @ q.T).norm()) <= p["config"]["preservation_atol"],
                    "protected span changed",
                )
                shift = (k.float() @ layer.weight.T - k.float() @ original.T) - b.T
                require(
                    float(shift.abs().max()) <= p["config"]["target_atol"],
                    "key response changed",
                )
            generated = generate(prompts, record["candidate"])
            ledger(prompts, generated, [trial["edit_id"]])
            check = answer_check(
                cases,
                data["prompt_ids"],
                record["baseline"],
                generated,
                p["answer_limits"],
                eos,
            )
            require(check == trial["answer_check"], "answer gate evidence changed")
            _, failed = decision(m, p["limits"])
            failed += ["generation:" + v for v in check["violations"]]
            require(
                trial["accepted"] == (not failed) and trial["violations"] == failed,
                "combined decision changed",
            )
            confirmed = generate(data["confirmation"], record["confirmation"])
            active = summary["generations"][len(ledgers)]["active_edit_ids"]
            require(
                len(active) == 1
                and (active == [trial["edit_id"]]) == trial["accepted"],
                "confirmation edit mismatch",
            )
            ledger(data["confirmation"], confirmed, active)
            complete = [
                full_name(o, c["answers"])
                for o, c in zip(generated, cases, strict=True)
            ]
            base_local = [
                full_name(o, c["answers"])
                for o, c in zip(record["baseline"][5:], cases[5:], strict=True)
            ]
            retained = sum(
                x and y for x, y in zip(base_local, complete[5:], strict=True)
            )
            scores = {
                "design_complete": sum(complete[:2]),
                "validation_complete": sum(complete[2:5]),
                "confirmation_complete": sum(
                    full_name(o, [p["target"]]) for o in confirmed
                ),
                "baseline_original_complete": sum(
                    full_name(o, p["original_answers"]) for o in record["baseline"][:5]
                ),
                "baseline_locality_complete": sum(base_local),
                "locality_complete_retained": retained,
            }
            require(scores == record["scores"], "complete-answer scores changed")
            strict = (
                all(complete[:5])
                and sum(base_local) >= 3
                and retained == sum(base_local)
            )
            require(
                record["strict_accepted"] == (strict and trial["accepted"]),
                "strict selection changed",
            )
        finally:
            layer.weight.copy_(original)
        require(
            record["rollback_exact"]
            and all(
                torch.equal(model.get_submodule(name).weight, w)
                for name, w in originals.items()
            ),
            "rollback changed",
        )
    require(ledgers == summary["generations"], "generation audit changed")
    designs = summary["edit_designs"]
    require(
        len(designs) == 2 and all(r["status"] == "completed" for r in designs),
        "missing design evidence",
    )
    design_forwards = (p["config"]["steps"] + 1) * (
        len(data["design"]) + len(data["calibration"])
    )
    design_backwards = p["config"]["steps"] * (
        len(data["design"]) + len(data["calibration"])
    )
    require(
        all(
            r["model_forwards"] == design_forwards
            and r["backward_calls"] == design_backwards
            for r in designs
        ),
        "design accounting changed",
    )
    forwards = (
        captures
        + len(designs) * design_forwards
        + len(trials) * 2 * (len(teacher_ids) + len(data["design"]))
        + sum(r["model_forwards"] for r in ledgers)
    )
    require(
        forwards == summary["model_forwards"] and summary["budget"]["used"] == 61,
        "forward/measurement accounting changed",
    )
    accepted = [r["arm"] for r in trials if r["strict_accepted"]]
    require(accepted == summary["accepted"], "accepted set changed")
    return {
        "passed": True,
        "trials": len(trials),
        "selectors_checked": 2,
        "selected_objectives_checked": 2,
        "generated_tokens_replayed": replayed,
        "retained_generated_tokens": sum(r["output_tokens"] for r in ledgers),
        "model_forwards_verified": forwards,
        "model_backward_calls": len(designs) * design_backwards,
        "measurement_units": 61,
        "artifact_hashes_checked": len(manifest),
        "accepted": accepted,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = verify(args.root, args.model)
    (args.root / "independent_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("verification=%s", report)
