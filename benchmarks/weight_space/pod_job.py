#!/usr/bin/env python3
"""Phase 1 (runs ON a GPU pod): loop-direction weight edit on Qwen3.5-0.8B.

Mechanism-level replication of Exp. 8 in "Sparks of Weight-Space Programming",
judged by REAL generation (thinking on), not a proxy:

  1. base traces on practice problems (GSM8K train)               [budgeted]
  2. read: MLP outputs at layers 15/19 inside real loops vs healthy
     reasoning -> one direction per layer (weight_space harness)   [budgeted]
  3. score base + loop-direction edits (k=64, alpha in 1/2/4) on a
     disjoint practice split; keep the best                        [budgeted]
  4. held-out GSM8K test + MATH-500 for base / kept / random-direction
     control at the kept k, alpha                        [evaluator side]

Subcommands: ``main`` (orchestrator) and ``generate`` (one vLLM run in its
own process so GPU memory is fully released between checkpoints).
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

MODEL = "Qwen/Qwen3.5-0.8B"
# Exact Hub revisions every run used (verified against the local snapshots).
REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
}
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
LAYERS = (15, 19)
K = 64
ALPHAS = (1.0, 2.0, 4.0)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
SEED = 1234
WORK = Path(os.environ.get("WS_WORK", "/workspace"))
T0 = time.time()


def log(**kw: object) -> None:
    kw["t"] = round(time.time() - T0, 1)
    print(json.dumps(kw, default=str), flush=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------- grading / loops
def post_think(text: str) -> str | None:
    return text.split("</think>", 1)[1] if "</think>" in text else None


def last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth, out = i + len("\\boxed{"), 1, []
    while j < len(s) and depth:
        ch = s[j]
        depth += (ch == "{") - (ch == "}")
        if depth:
            out.append(ch)
        j += 1
    return "".join(out) if depth == 0 else None


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def _norm(s: str) -> str:
    s = re.sub(r"\\(left|right|!|,|;|:)", "", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace(" ", "")
    return s.rstrip(".").removesuffix(".0")


def grade(rec: dict) -> dict:
    ans = post_think(rec["text"])
    if ans is None:
        return {"answer": None, "correct": False}
    boxed = last_boxed(ans)
    if rec["kind"] == "gsm8k":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", boxed if boxed is not None else ans)
        pred = nums[-1].replace(",", "").rstrip(".") if nums else None
        return {
            "answer": pred,
            "correct": pred is not None and _num_eq(pred, rec["gold"]),
        }
    try:
        from math_verify import parse, verify

        ok = bool(verify(parse("$" + rec["gold"] + "$"), parse(ans)))
    except Exception:  # noqa: BLE001 - grader fallback
        ok = boxed is not None and _norm(boxed) == _norm(rec["gold"])
    return {"answer": boxed, "correct": ok}


def loop_info(
    gen: list[int], n: int = 32, window: int = 2048, reps: int = 3
) -> dict | None:
    """A cyclic repeat: some n-gram occurs >= reps times in the last ``window`` tokens."""
    tail = gen[-window:]
    if len(tail) < n * reps:
        return None
    counts = Counter(tuple(tail[i : i + n]) for i in range(len(tail) - n + 1))
    gram, count = counts.most_common(1)[0]
    if count < reps:
        return None
    occ = [
        i
        for i in range(len(gen) - n + 1)
        if gen[i] == gram[0] and tuple(gen[i : i + n]) == gram
    ]
    return {"first": occ[0], "period": occ[1] - occ[0], "count": len(occ)}


def summarize(rows: list[dict]) -> dict:
    def agg(rs: list[dict]) -> dict:
        n = len(rs)
        if not n:
            return {"n": 0}
        return {
            "n": n,
            "accuracy": round(sum(r["correct"] for r in rs) / n, 4),
            "hit_limit": round(sum(r["finish"] == "length" for r in rs) / n, 4),
            "loop": round(sum(r["loop"] is not None for r in rs) / n, 4),
            "mean_tokens": round(sum(r["tokens"] for r in rs) / n, 1),
        }

    out = {"all": agg(rows)}
    for kind in sorted({r["kind"] for r in rows}):
        out[kind] = agg([r for r in rows if r["kind"] == kind])
    return out


def paired(base: list[dict], other: list[dict]) -> dict:
    by_id = {r["id"]: r for r in base}
    pairs = [(by_id[r["id"]], r) for r in other if r["id"] in by_id]
    return {
        "n": len(pairs),
        "newly_correct": sum(o["correct"] and not b["correct"] for b, o in pairs),
        "newly_wrong": sum(b["correct"] and not o["correct"] for b, o in pairs),
        "loops_fixed": sum(
            b["loop"] is not None and o["loop"] is None for b, o in pairs
        ),
        "loops_created": sum(
            b["loop"] is None and o["loop"] is not None for b, o in pairs
        ),
    }


# --------------------------------------------------------------- generation worker
def cmd_generate(a: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Hub ids are pinned via REVISIONS; edited checkpoints are local dirs (no revision).
    tok = AutoTokenizer.from_pretrained(
        a.model, revision=REVISIONS.get(a.model)
    )  # nosec B615
    rows = read_jsonl(Path(a.data))
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["question"] + SUFFIX}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for r in rows
    ]
    kw = {
        "model": a.model,
        "revision": REVISIONS.get(a.model),
        "dtype": "bfloat16",
        "max_model_len": a.max_tokens + 1536,
        "gpu_memory_utilization": float(os.environ.get("WS_GPU_UTIL", "0.88")),
        "seed": SEED,
        "max_num_seqs": 512,
    }
    fast = {
        "compilation_config": {
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        }
    }
    attempts = [
        ("fast+mm", {**kw, **fast, "limit_mm_per_prompt": {"image": 0, "video": 0}}),
        ("fast", {**kw, **fast}),
        ("plain+mm", {**kw, "limit_mm_per_prompt": {"image": 0, "video": 0}}),
        ("plain", dict(kw)),
    ]
    llm = None
    for name, args in attempts:
        try:
            llm = LLM(**args)
            log(event="vllm_config", variant=name)
            break
        except Exception as exc:  # noqa: BLE001
            log(event="vllm_retry", variant=name, error=str(exc)[:300])
    if llm is None:
        raise RuntimeError("vLLM could not be constructed with any variant")
    started = time.time()
    outs = llm.generate(
        prompts, SamplingParams(max_tokens=a.max_tokens, seed=SEED, **SAMPLING)
    )
    recs = []
    for r, o in zip(rows, outs, strict=True):
        c = o.outputs[0]
        rec = {
            "id": r["id"],
            "kind": r["kind"],
            "gold": r["gold"],
            "prompt_ids": list(o.prompt_token_ids),
            "gen_ids": list(c.token_ids),
            "text": c.text,
            "finish": c.finish_reason,
            "tokens": len(c.token_ids),
        }
        rec.update(grade(rec))
        rec["loop"] = loop_info(rec["gen_ids"])
        recs.append(rec)
    write_jsonl(Path(a.out), recs)
    gen_tokens = sum(r["tokens"] for r in recs)
    log(
        event="generated",
        label=a.label,
        gen_tokens=gen_tokens,
        tok_per_s=round(gen_tokens / max(1e-9, time.time() - started)),
        **summarize(recs)["all"],
    )
    # Results are on disk. Skip interpreter/vLLM teardown: on vLLM 0.33 the
    # EngineCore child sometimes never exits after a 1,000-prompt run, and the
    # orchestrator (subprocess.run(check=True)) then waits forever. Observed on
    # three L40S pods at once (2026-11-02), ~30 idle GPU-minutes each.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# --------------------------------------------------------------- orchestrator
class Budget:
    def __init__(self, limit: int) -> None:
        self.limit, self.ledger = limit, []

    def spend(self, why: str) -> None:
        if len(self.ledger) >= self.limit:
            raise RuntimeError(f"practice forward-pass budget exhausted: {self.ledger}")
        self.ledger.append(why)
        log(event="forward_pass", used=len(self.ledger), limit=self.limit, why=why)


def generate(
    label: str, model_dir: str, rows: list[dict], max_tokens: int, out: Path
) -> list[dict]:
    data = WORK / "data" / f"{label}.jsonl"
    write_jsonl(data, rows)
    dest = out / f"gen_{label}.jsonl.gz"
    cmd = [
        sys.executable,
        __file__,
        "generate",
        "--model",
        model_dir,
        "--data",
        str(data),
        "--out",
        str(dest),
        "--max-tokens",
        str(max_tokens),
        "--label",
        label,
    ]
    log(event="generate_start", label=label, n=len(rows), model=model_dir)
    subprocess.run(cmd, check=True)
    return read_jsonl(dest)


def load_problems(
    n_trace: int, n_select: int, n_held_gsm: int
) -> dict[str, list[dict]]:
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train", revision=GSM8K_REVISION)
    test = load_dataset("openai/gsm8k", "main", split="test", revision=GSM8K_REVISION)
    math500 = load_dataset(
        "HuggingFaceH4/MATH-500", split="test", revision=MATH500_REVISION
    )
    idx = list(range(len(train)))
    random.Random(0).shuffle(idx)

    def gsm(row: dict, i: int, split: str) -> dict:
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        return {
            "id": f"gsm8k-{split}-{i}",
            "kind": "gsm8k",
            "question": row["question"],
            "gold": gold,
        }

    return {
        "trace": [gsm(train[i], i, "train") for i in idx[:n_trace]],
        "select": [
            gsm(train[i], i, "train") for i in idx[n_trace : n_trace + n_select]
        ],
        "heldout": [gsm(test[i], i, "test") for i in range(n_held_gsm)]
        + [
            {
                "id": f"math500-{i}",
                "kind": "math",
                "question": r["problem"],
                "gold": r["answer"],
            }
            for i, r in enumerate(math500)
        ],
    }


def find_directions(trace: list[dict], budget: Budget, out: Path) -> dict:
    """Read: per-layer mean(MLP out | in-loop) - mean(MLP out | healthy)."""
    import torch
    from transformers import AutoModelForCausalLM
    from weight_space import WeightSpaceSession

    rng = random.Random(0)
    prefixes: dict[str, list[int]] = {}
    loop_keys: list[str] = []
    healthy_keys: list[str] = []
    for r in trace:
        P, G, info = r["prompt_ids"], r["gen_ids"], r["loop"]
        if info and r["finish"] == "length":
            start = info["first"] + info["period"]  # strictly inside a repetition
            for pos in rng.sample(
                range(start, len(G)), k=max(0, min(2, len(G) - start))
            ):
                key = f"L{len(prefixes)}"
                prefixes[key] = P + G[: pos + 1]
                loop_keys.append(key)
            if info["first"] > 256:  # healthy reasoning before the loop began
                for pos in rng.sample(range(64, info["first"] - 64), k=2):
                    key = f"H{len(prefixes)}"
                    prefixes[key] = P + G[: pos + 1]
                    healthy_keys.append(key)
        elif r["finish"] == "stop" and len(G) > 128:
            pos = rng.randrange(32, len(G) - 16)
            key = f"H{len(prefixes)}"
            prefixes[key] = P + G[: pos + 1]
            healthy_keys.append(key)
    loop_keys = rng.sample(loop_keys, k=min(96, len(loop_keys)))
    healthy_keys = rng.sample(healthy_keys, k=min(96, len(healthy_keys)))
    log(
        event="read_plan",
        loop_prefixes=len(loop_keys),
        healthy_prefixes=len(healthy_keys),
        mean_len=round(
            sum(len(prefixes[k]) for k in loop_keys + healthy_keys)
            / max(1, len(loop_keys + healthy_keys))
        ),
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, revision=REVISIONS[MODEL]
        )
        .to(os.environ.get("WS_DEVICE", "cuda"))
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    session = WeightSpaceSession(
        model,
        lambda key: prefixes[key],
        budget=len(LAYERS),
        base_model=MODEL,
        batch_size=8,
    )
    keys = sorted(
        loop_keys + healthy_keys, key=lambda k: len(prefixes[k])
    )  # less padding
    directions, stats = {}, {}
    seeded = out / "directions.pt"
    if seeded.exists():
        directions = {
            int(L): v.float().cpu()
            for L, v in torch.load(seeded, weights_only=True)["directions"].items()
        }
        for L in LAYERS:
            budget.spend(
                f"read: layer {L} MLP outputs (reused from an earlier run of this exact config)"
            )
            stats[L] = {"norm": round(float(directions[L].norm()), 4)}
        log(event="reused", what="directions", stats=stats)
    for L in LAYERS:
        if L in directions:
            continue
        module = f"model.layers.{L}.mlp.down_proj"
        budget.spend(f"read: layer {L} MLP outputs on {len(keys)} trace prefixes")
        outs = session.capture(module, keys, reason=f"read layer {L}")
        row = {k: i for i, k in enumerate(keys)}
        loop_mean = outs[[row[k] for k in loop_keys]].mean(0)
        healthy_mean = outs[[row[k] for k in healthy_keys]].mean(0)
        directions[L] = (loop_mean - healthy_mean).float()
        stats[L] = {"norm": round(float(directions[L].norm()), 4)}

    cos = torch.nn.functional.cosine_similarity(
        directions[LAYERS[0]], directions[LAYERS[1]], dim=0
    )
    log(event="directions", stats=stats, cos_between_layers=round(float(cos), 4))
    if not seeded.exists():
        torch.save(
            {
                "directions": directions,
                "loop_keys": len(loop_keys),
                "healthy_keys": len(healthy_keys),
            },
            out / "directions.pt",
        )

    from weight_space import direction_ablation

    g = torch.Generator().manual_seed(1)
    random_dirs = {L: torch.randn(directions[L].shape[0], generator=g) for L in LAYERS}
    deltas: dict[str, dict] = {}
    for alpha in ALPHAS:
        for tag, dirs in (("loop", directions), ("random", random_dirs)):
            name = f"{tag}_a{alpha:g}"
            deltas[name] = {
                L: direction_ablation(
                    session.weight(f"model.layers.{L}.mlp.down_proj"),
                    dirs[L],
                    k=K,
                    alpha=alpha,
                )
                for L in LAYERS
            }
            for L, d in deltas[name].items():
                session.apply(
                    f"model.layers.{L}.mlp.down_proj", d, note=f"{name} layer {L}"
                )
            session.export(out / "adapters" / name)
            while session.edits:
                session.revert()
    del session, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    names = [f"loop_a{alpha:g}" for alpha in ALPHAS]
    return {
        "deltas": deltas,
        "cos": float(cos),
        "stats": stats,
        "candidates": names,
        "control_of": {n: n.replace("loop_", "random_") for n in names},
        "layers": list(LAYERS),
    }


def find_wait_directions_and_deltas(
    trace: list[dict], budget: Budget, out: Path
) -> dict:
    """Wait-trigger mode: scan all layers in one pass, edit the two that separate best."""
    import torch
    from transformers import AutoModelForCausalLM
    from wait_direction import find_wait_directions
    from weight_space import WeightSpaceSession, direction_ablation

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, revision=REVISIONS[MODEL]
        )
        .to(os.environ.get("WS_DEVICE", "cuda"))
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = int(model.config.num_hidden_layers)
    # batch 2: the torch fallback for Qwen3.5 linear attention needs ~39 GB at
    # batch 8 x ~4k tokens (OOM on a 44 GB L40S, observed live).
    session = WeightSpaceSession(
        model, lambda s: s, budget=10**9, base_model=MODEL, batch_size=2
    )
    specs = {"k64_a1": (64, 1.0), "all_a1": (None, 1.0)}
    deltas: dict[str, dict] = {}
    try:
        directions, scores, info = find_wait_directions(
            trace, session, n_layers, budget
        )
    except Exception as exc:  # noqa: BLE001 - token candidates do not need the scan
        log(event="wait_scan_failed", error=str(exc)[:300])
        directions, scores, info, specs = {}, {}, {}, {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    layers = sorted(sorted(scores, key=lambda L: -scores[L])[:2]) if scores else []
    log(event="wait_directions", scores=scores, chosen_layers=layers, **info)
    if directions:
        torch.save(
            {"directions": directions, "scores": scores, "layers": layers},
            out / "wait_directions.pt",
        )
    g = torch.Generator().manual_seed(1)
    rand = {L: torch.randn(directions[L].shape[0], generator=g) for L in layers}
    for tag, (k, alpha) in specs.items():
        for kind, dirs in (("wait", directions), ("random", rand)):
            name = f"{kind}_{tag}"
            deltas[name] = {}
            for L in layers:
                W = session.weight(f"model.layers.{L}.mlp.down_proj")
                deltas[name][L] = direction_ablation(
                    W, dirs[L], k=k or W.shape[1], alpha=alpha
                )
                session.apply(
                    f"model.layers.{L}.mlp.down_proj",
                    deltas[name][L],
                    note=f"{name} layer {L}",
                )
            session.export(out / "adapters" / name)
            while session.edits:
                session.revert()
    del session, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # "Wait" starts 37.7% of in-loop lines (3.25x pre-onset): suppress it in the weights.
    # Control: "This" — a frequent line-starter with no loop enrichment (0.93x).
    token_edits = {
        "nowait_c0": {"ids": [13784, 13428], "scale": 0.0},
        "nowait_c05": {"ids": [13784, 13428], "scale": 0.5},
        "ctrlthis_c0": {"ids": [1919, 1061], "scale": 0.0},
        "ctrlthis_c05": {"ids": [1919, 1061], "scale": 0.5},
    }
    names = [f"wait_{t}" for t in specs] + ["nowait_c0", "nowait_c05"]
    control_of = {
        **{f"wait_{t}": f"random_{t}" for t in specs},
        "nowait_c0": "ctrlthis_c0",
        "nowait_c05": "ctrlthis_c05",
    }
    return {
        "deltas": deltas,
        "cos": None,
        "stats": {"separation": scores},
        "candidates": names,
        "control_of": control_of,
        "layers": layers,
        "token_edits": token_edits,
    }


def write_checkpoint(
    name: str,
    deltas: dict,
    model: str = MODEL,
    key_fmt: str = "model.language_model.layers.{L}.mlp.down_proj.weight",
    token_edit: dict | None = None,
    embed_key: str = "model.language_model.embed_tokens.weight",
) -> str:
    """Base checkpoint with W += dW on down_proj tensors and/or scaled embedding rows.

    ``token_edit={"ids": [...], "scale": c}`` multiplies those rows of the (tied)
    embedding by c: with tied input/output embeddings that scales the tokens'
    logits by c everywhere — "say this token less", written into the weights.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    src = Path(snapshot_download(model, revision=REVISIONS.get(model)))
    dst = WORK / "ckpt" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)
    index = dst / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else {}
    by_file: dict[str, list] = {}
    for L, d in deltas.items():
        key = key_fmt.format(L=L)
        by_file.setdefault(weight_map.get(key, "model.safetensors"), []).append(
            (key, d)
        )
    if token_edit:
        # Group entries by the tensor they touch: embedding rows or down_proj neuron columns.
        per_key: dict[str, list] = {}
        for entry in token_edit if isinstance(token_edit, list) else [token_edit]:
            if entry.get("kind", "embed_rows") == "down_cols":
                k = key_fmt.format(L=entry["layer"])
            else:
                k = embed_key
            per_key.setdefault(k, []).append(entry)
        for k, entries in per_key.items():
            by_file.setdefault(weight_map.get(k, "model.safetensors"), []).append(
                (k, entries)
            )
    changed = 0
    for fname, items in by_file.items():
        path = dst / fname
        tensors = load_file(str(path))
        for key, d in items:
            W = tensors[key]
            if isinstance(d, (dict, list)):  # token-row scaling (one or several groups)
                new = W.clone()
                for entry in d if isinstance(d, list) else [d]:
                    if entry.get("kind", "embed_rows") == "down_cols":
                        cols = list(entry["neurons"])
                        new[:, cols] = (
                            new[:, cols].float() * float(entry["scale"])
                        ).to(W.dtype)
                    else:
                        rows = list(entry["ids"])
                        new[rows] = (new[rows].float() * float(entry["scale"])).to(
                            W.dtype
                        )
                changed += int((new != W).sum())
                tensors[key] = new.contiguous()
                continue
            new = (W.float() + d.dense()).to(W.dtype).contiguous()
            changed += int((new != W).sum())
            tensors[key] = new
        path.unlink()
        save_file(tensors, str(path), metadata={"format": "pt"})
    log(event="checkpoint", name=name, changed_weights=changed)
    return str(dst)


def cmd_main(a: argparse.Namespace) -> None:
    import torch

    out = Path(a.out)
    (WORK / "data").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    import transformers
    import vllm

    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    log(event="env", **env)
    summary: dict = {
        "config": {
            "model": MODEL,
            "layers": LAYERS,
            "k": K,
            "alphas": ALPHAS,
            "sampling": SAMPLING,
            "seed": SEED,
            "practice_max_tokens": a.practice_max_tokens,
            "heldout_max_tokens": a.heldout_max_tokens,
            "env": env,
        }
    }

    def save() -> None:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    probs = load_problems(a.n_trace, a.n_select, a.n_heldout_gsm)
    token_spec = (
        json.loads(Path(a.token_spec).read_text())
        if a.direction_mode == "tokens"
        else None
    )
    budget = Budget(
        limit={
            "loop": 1 + len(LAYERS) + 1 + len(ALPHAS),
            "wait": 1 + 1 + 1 + 4,
            "tokens": 1 + 1 + len(token_spec["candidates"]) if token_spec else 0,
        }[a.direction_mode]
    )
    summary["config"]["direction_mode"] = a.direction_mode

    seeded = out / "gen_base_trace.jsonl.gz"
    if seeded.exists():
        budget.spend(
            "base traces on practice_trace (reused from an earlier run of this exact config)"
        )
        trace = read_jsonl(seeded)
        log(event="reused", what="base traces", n=len(trace))
    else:
        budget.spend("base traces on practice_trace")
        trace = generate(
            "base_trace", MODEL, probs["trace"], a.practice_max_tokens, out
        )
    summary["trace"] = summarize(trace)
    looped = sum(r["loop"] is not None and r["finish"] == "length" for r in trace)
    summary["trace"]["looped_at_limit"] = looped
    save()
    if looped < a.min_loops:
        summary["verdict"] = (
            f"premise not met: only {looped} looped traces (< {a.min_loops})"
        )
        summary["budget"] = budget.ledger
        save()
        log(event="stop", reason=summary["verdict"])
        return

    if token_spec:
        found = {
            "deltas": {},
            "cos": None,
            "stats": {},
            "layers": [],
            "token_edits": token_spec["edits"],
            "candidates": token_spec["candidates"],
            "control_of": token_spec["control_of"],
        }
        summary["token_spec"] = token_spec
    else:
        found = (
            find_directions
            if a.direction_mode == "loop"
            else find_wait_directions_and_deltas
        )(trace, budget, out)
    summary["directions"] = {
        "stats": found["stats"],
        "cos_between_layers": found["cos"],
        "layers": found["layers"],
    }
    save()

    seeded_select = out / "gen_base_select.jsonl.gz"
    if seeded_select.exists():
        budget.spend("select: base (reused from an earlier run of this exact config)")
        select = {"base": summarize(read_jsonl(seeded_select))}
    else:
        budget.spend("select: base")
        select = {
            "base": summarize(
                generate(
                    "base_select", MODEL, probs["select"], a.practice_max_tokens, out
                )
            )
        }
    for name in found["candidates"]:
        budget.spend(f"select: {name}")
        ckpt = write_checkpoint(
            name,
            found["deltas"].get(name, {}),
            token_edit=found.get("token_edits", {}).get(name),
        )
        select[name] = summarize(
            generate(
                f"{name}_select", ckpt, probs["select"], a.practice_max_tokens, out
            )
        )
        summary["select"] = select
        save()
    candidates = [n for n in select if n != "base"]
    kept = max(
        candidates,
        key=lambda n: (select[n]["all"]["accuracy"], -select[n]["all"]["loop"]),
    )
    control = found["control_of"][kept]
    summary["kept"] = kept
    summary["control"] = control
    summary["budget"] = budget.ledger
    save()
    log(
        event="kept",
        kept=kept,
        practice=select[kept]["all"],
        base=select["base"]["all"],
    )
    gain = select[kept]["all"]["accuracy"] - select["base"]["all"]["accuracy"]
    loop_drop = select["base"]["all"]["loop"] - select[kept]["all"]["loop"]
    if gain < a.min_practice_gain and loop_drop < a.min_loop_drop:
        summary["verdict"] = (
            f"no practice improvement (acc {gain:+.3f}, loop {-loop_drop:+.3f}); "
            "held-out skipped"
        )
        save()
        log(event="gate", verdict=summary["verdict"])
        small_model_check(a, probs, summary, save, out)
        summary["wall_seconds"] = round(time.time() - T0)
        save()
        log(event="done", verdict=summary["verdict"])
        return

    held = {}
    held["base"] = generate(
        "base_heldout", MODEL, probs["heldout"], a.heldout_max_tokens, out
    )
    summary["heldout"] = {"base": summarize(held["base"])}
    save()
    for name in (kept, control):
        ckpt = write_checkpoint(
            name,
            found["deltas"].get(name, {}),
            token_edit=found.get("token_edits", {}).get(name),
        )
        held[name] = generate(
            f"{name}_heldout", ckpt, probs["heldout"], a.heldout_max_tokens, out
        )
        summary["heldout"][name] = summarize(held[name])
        summary["heldout"][f"{name}_paired_vs_base"] = paired(held["base"], held[name])
        save()
    small_model_check(a, probs, summary, save, out)
    summary["wall_seconds"] = round(time.time() - T0)
    save()
    log(
        event="done",
        heldout={k: v.get("all", v) for k, v in summary["heldout"].items()},
    )


def small_model_check(
    a: argparse.Namespace, probs: dict, summary: dict, save, out: Path
) -> None:
    """Real generation for the Phase-0 (CPU, proxy-selected) Qwen3-0.6B edit + control."""
    if not a.small_edit or not Path(a.small_edit).exists():
        return
    if time.time() - T0 > a.small_cutoff_s:
        summary["small_model"] = {"skipped": f"past cutoff {a.small_cutoff_s}s"}
        save()
        return
    import torch
    from weight_space import LowRankDelta

    blob = torch.load(a.small_edit, weights_only=True)
    rows = [r for r in probs["heldout"] if r["kind"] == "gsm8k"]
    gens = {
        "base": generate("small_base", blob["model"], rows, a.practice_max_tokens, out)
    }
    report: dict = {
        "model": blob["model"],
        "layers": list(blob["layers"]),
        "k": blob["k"],
        "alpha": blob["alpha"],
        "base": summarize(gens["base"]),
    }
    for name, per_layer in blob["deltas"].items():
        deltas = {L: LowRankDelta(B=v["B"], A=v["A"]) for L, v in per_layer.items()}
        ckpt = write_checkpoint(
            name,
            deltas,
            model=blob["model"],
            key_fmt="model.layers.{L}.mlp.down_proj.weight",
        )
        gens[name] = generate(name, ckpt, rows, a.practice_max_tokens, out)
        report[name] = summarize(gens[name])
        report[f"{name}_paired_vs_base"] = paired(gens["base"], gens[name])
        summary["small_model"] = report
        save()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("main")
    m.add_argument("--out", default="/workspace/out")
    m.add_argument("--n-trace", type=int, default=256)
    m.add_argument("--n-select", type=int, default=200)
    m.add_argument("--n-heldout-gsm", type=int, default=500)
    m.add_argument("--practice-max-tokens", type=int, default=8192)
    m.add_argument("--heldout-max-tokens", type=int, default=16384)
    m.add_argument("--min-loops", type=int, default=8)
    m.add_argument(
        "--direction-mode", choices=["loop", "wait", "tokens"], default="loop"
    )
    m.add_argument("--token-spec", default="/workspace/token_spec.json")
    m.add_argument("--min-practice-gain", type=float, default=0.01)
    m.add_argument("--min-loop-drop", type=float, default=0.05)
    m.add_argument("--small-edit", default="/workspace/small_edit.pt")
    m.add_argument("--small-cutoff-s", type=int, default=8400)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-tokens", type=int, required=True)
    g.add_argument("--label", required=True)
    a = ap.parse_args()
    (cmd_main if a.cmd == "main" else cmd_generate)(a)


if __name__ == "__main__":
    main()
