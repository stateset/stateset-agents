"""Side effects of the weight edits outside thinking-mode math.

Runs on a GPU pod with this repository's wheel installed:

    python side_effect_job.py --out /workspace/out --neuron-spec conclude_neuron_spec.json

Models: base, W5T13 (Wait x0.5 + </think> x1.3), W5T15 (x1.5), and the designed
"answer computed -> conclude" neuron (+ Wait x0.5), each applied from an
EditSpec and verified tensor by tensor.

1. GSM8K-test[:500] with thinking OFF (the chat template closes an empty
   thinking block itself), 2,048-token cap: accuracy, and how often the reply
   contains a stray ``</think>`` — format damage in normal answers.
2. MMLU (1,000 questions from ``cais/mmlu`` "all" test, pinned), thinking off,
   zero-shot: the argmax over the A/B/C/D letter logits at the answer position
   (no generation), plus the probability the model puts on ``</think>`` there.
3. WikiText-2 test perplexity (65,536 tokens, 2,048-token windows) and the mean
   probability of ``</think>`` per position — general language-model damage.

Accuracies are compared to base per item (exact McNemar, bootstrap CI).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import pod_job as pj  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WAIT_IDS = (13784, 13428)
THINK_END = 248069


def mmlu_rows(n: int, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "cais/mmlu", "all", split="test", revision=MMLU_REVISION
    )  # nosec B615
    idx = random.Random(seed).sample(range(len(ds)), n)
    rows = []
    for i in idx:
        r = ds[i]
        choices = "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"]))
        rows.append(
            {
                "id": f"mmlu-{i}",
                "subject": r["subject"],
                "prompt": f"{r['question']}\n\n{choices}\n\nAnswer with the letter only.",
                "gold": "ABCD"[int(r["answer"])],
            }
        )
    return rows


def mmlu_eval(model, tok, rows: list[dict]) -> list[dict]:
    """Zero-shot letter choice at the first answer position, thinking off (batch 1)."""
    import torch

    letters = [tok.encode(c, add_special_tokens=False) for c in "ABCD"]
    assert all(len(t) == 1 for t in letters), letters
    letter_ids = [t[0] for t in letters]
    out = []
    for r in rows:
        text = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
            model.device
        )
        with torch.no_grad():
            logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        pred = "ABCD"[int(logits[letter_ids].argmax())]
        out.append(
            {
                "id": r["id"],
                "subject": r["subject"],
                "correct": pred == r["gold"],
                "p_think_end": float(probs[THINK_END]),
                "p_letters": float(probs[letter_ids].sum()),
            }
        )
    return out


def wikitext_eval(model, tok, n_tokens: int = 65536, window: int = 2048) -> dict:
    import torch
    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        revision=WIKITEXT_REVISION,
    )  # nosec B615
    ids = tok("\n\n".join(ds["text"]), add_special_tokens=False).input_ids[:n_tokens]
    nll, p_think, count = 0.0, 0.0, 0
    for s in range(0, len(ids) - 1, window):
        chunk = torch.tensor(ids[s : s + window + 1]).unsqueeze(0).to(model.device)
        if chunk.shape[1] < 2:
            break
        with torch.no_grad():
            logits = model(input_ids=chunk[:, :-1]).logits[0]
        targets = chunk[0, 1:]
        for a in range(0, logits.shape[0], 512):
            lp = torch.log_softmax(logits[a : a + 512].float(), -1)
            nll -= float(lp.gather(1, targets[a : a + 512, None]).sum())
            p_think += float(lp[:, THINK_END].exp().sum())
            count += lp.shape[0]
        del logits
    return {
        "tokens": count,
        "perplexity": round(math.exp(nll / count), 4),
        "mean_p_think_end": p_think / count,
    }


def summarize_gsm(rows: list[dict]) -> dict:
    n = len(rows)
    stray = [r["text"].count("</think>") > 0 for r in rows]
    clean = [r["finish"] == "stop" and not s for r, s in zip(rows, stray, strict=True)]
    return {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "strict": round(
            sum(r["correct"] and c for r, c in zip(rows, clean, strict=True)) / n, 4
        ),
        "stray_think_end": round(sum(stray) / n, 4),
        "hit_limit": round(sum(r["finish"] == "length" for r in rows) / n, 4),
        "mean_tokens": round(sum(r["tokens"] for r in rows) / n, 1),
    }


def main(argv: list[str] | None = None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stateset_agents.experimental.weight_space import (
        EditSpec,
        TokenRowScale,
        paired_compare,
    )
    from stateset_agents.experimental.weight_space.checkpoints import (
        apply_spec_to_checkpoint,
        resolve_checkpoint,
        verify_checkpoint,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--neuron-spec", default="/workspace/conclude_neuron_spec.json")
    ap.add_argument("--mmlu", type=int, default=1000)
    ap.add_argument("--gsm", type=int, default=500)
    ap.add_argument(
        "--models",
        default="base,W5T13,W5T15,neuron",
        help="comma list from base, W5T13, W5T15, neuron (base is always included)",
    )
    ap.add_argument(
        "--tag", default="", help="suffix for checkpoint dirs and output files"
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (pj.WORK / "data").mkdir(parents=True, exist_ok=True)

    wait = TokenRowScale(ids=WAIT_IDS, scale=0.5)
    every = {
        "W5T13": EditSpec("W5T13", (wait, TokenRowScale((THINK_END,), 1.3))),
        "W5T15": EditSpec("W5T15", (wait, TokenRowScale((THINK_END,), 1.5))),
        "neuron": EditSpec.from_json(Path(a.neuron_spec).read_text()),
    }
    wanted = {m.strip() for m in a.models.split(",")}
    specs = {k: v for k, v in every.items() if k in wanted}
    base = resolve_checkpoint(MODEL, REVISION)
    dirs = {"base": str(base)}
    for name, spec in specs.items():
        dst = pj.WORK / "ckpt" / f"side_{name}{a.tag}"
        manifest = apply_spec_to_checkpoint(spec, base, dst)
        verify_checkpoint(spec, base, dst)
        dirs[name] = str(dst)
        print(
            json.dumps(
                {
                    "event": "checkpoint_verified",
                    "name": name,
                    "hash": manifest["spec_hash"],
                }
            ),
            flush=True,
        )

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)  # nosec B615 - pinned
    mmlu = mmlu_rows(a.mmlu)
    result: dict = {"models": dirs, "mmlu": {}, "wikitext": {}, "gsm8k_no_thinking": {}}
    mmlu_rows_by_model = {}
    for name, d in dirs.items():
        model = (
            (
                AutoModelForCausalLM.from_pretrained(
                    d, dtype=torch.bfloat16
                )  # nosec B615 - local dir
            )
            .cuda()
            .eval()
        )
        rows = mmlu_eval(model, tok, mmlu)
        mmlu_rows_by_model[name] = rows
        result["mmlu"][name] = {
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "mean_p_think_end": sum(r["p_think_end"] for r in rows) / len(rows),
            "mean_p_letters": sum(r["p_letters"] for r in rows) / len(rows),
        }
        result["wikitext"][name] = wikitext_eval(model, tok)
        print(
            json.dumps(
                {
                    "event": "teacher_forced",
                    "model": name,
                    "mmlu": result["mmlu"][name],
                    "wikitext": result["wikitext"][name],
                }
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
    pj.write_jsonl(
        out / f"mmlu_rows{a.tag}.jsonl",
        [{"model": m, **r} for m, rs in mmlu_rows_by_model.items() for r in rs],
    )

    probs = pj.load_problems(256, 200, 500)
    gsm_rows = probs["heldout"][: a.gsm]
    gens = {}
    for name, d in dirs.items():
        model_dir = pj.MODEL if name == "base" else d
        gens[name] = pj.generate(
            f"side_{name}{a.tag}_gsm_nothink",
            model_dir,
            gsm_rows,
            2048,
            out,
            thinking=False,
        )
        result["gsm8k_no_thinking"][name] = summarize_gsm(gens[name])
        print(
            json.dumps(
                {
                    "event": "gsm_no_thinking",
                    "model": name,
                    **result["gsm8k_no_thinking"][name],
                }
            ),
            flush=True,
        )

    def strict(rows: list[dict]) -> list[dict]:
        return [
            {
                **r,
                "correct": bool(r["correct"])
                and r["finish"] == "stop"
                and "</think>" not in r["text"],
            }
            for r in rows
        ]

    result["paired_vs_base"] = {
        name: {
            "mmlu": paired_compare(
                mmlu_rows_by_model["base"], mmlu_rows_by_model[name]
            ),
            "gsm8k_no_thinking_strict": paired_compare(
                strict(gens["base"]), strict(gens[name])
            ),
        }
        for name in specs
    }
    (out / f"side_effects_summary{a.tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "side_effects_done"}), flush=True)


if __name__ == "__main__":
    main()
