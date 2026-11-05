#!/usr/bin/env python3
"""Arm B harness server (runs ON the pod): a budgeted weights kernel for an LLM.

The LLM (Claude Opus 5.5, driven from the operator's machine through an SSH
tunnel) sends Python to POST /python; it runs in a persistent namespace that
exposes the model's weights, an exact edit stack, and two BUDGETED ways to run
the network (``capture`` and ``evaluate``). POST /submit freezes the edits and
starts the held-out evaluation, which the LLM never sees.

Only localhost is bound; the operator reaches it through ``ssh -L``.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, "/workspace")
import pod_job as pj  # noqa: E402  (grading, loop detection, ckpt writer, vLLM worker)

PORT = 8765
os.environ.setdefault("WS_GPU_UTIL", "0.80")
OUT = Path("/workspace/out")
STATE: dict = {}
LOCK = threading.Lock()


def log(**kw: object) -> None:
    pj.log(**kw)


class BudgetExhausted(RuntimeError):
    pass


def setup(budget: int, n_trace: int, n_eval: int, max_tokens: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stateset_agents.experimental.weight_space import (
        WeightSpaceSession,
        direction_ablation,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (pj.WORK / "data").mkdir(parents=True, exist_ok=True)
    probs = pj.load_problems(n_trace, n_eval, 500)
    ledger: list[str] = []

    def spend(why: str) -> None:
        if len(ledger) >= budget:
            raise BudgetExhausted(
                f"forward-pass budget exhausted ({len(ledger)}/{budget}); ledger={ledger}"
            )
        ledger.append(why)
        log(event="forward_pass", used=len(ledger), limit=budget, why=why)

    spend(
        "initial: base model samples on the practice trace set (pre-generated for you)"
    )
    seeded = OUT / "gen_b_base_trace.jsonl.gz"
    traces = (
        pj.read_jsonl(seeded)
        if seeded.exists()
        else pj.generate("b_base_trace", pj.MODEL, probs["trace"], max_tokens, OUT)
    )

    tok = AutoTokenizer.from_pretrained(pj.MODEL, revision=pj.REVISIONS[pj.MODEL])
    model = (
        AutoModelForCausalLM.from_pretrained(
            pj.MODEL, dtype=torch.bfloat16, revision=pj.REVISIONS[pj.MODEL]
        )
        .to(os.environ.get("WS_DEVICE", "cuda"))
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    session = WeightSpaceSession(
        model,
        lambda s: s,
        budget=10**9,
        base_model=pj.MODEL,
        batch_size=int(os.environ.get("WS_CAPTURE_BATCH", "4")),
    )
    token_rows: list[dict] = []

    def scale_token_rows(ids: list, scale: float, note: str = "") -> dict:
        """Multiply rows of the TIED embedding/unembedding by `scale` (in memory and in every
        evaluated/held-out checkpoint). Free. Undo with clear_token_rows()."""
        W = model.get_input_embeddings().weight
        orig = W[list(ids)].detach().clone()
        with torch.no_grad():
            W[list(ids)] = (W[list(ids)].float() * float(scale)).to(W.dtype)
        token_rows.append(
            {"ids": list(ids), "scale": float(scale), "note": note, "_orig": orig}
        )
        return {
            "token_rows": [
                {k: v for k, v in e.items() if k != "_orig"} for e in token_rows
            ]
        }

    def clear_token_rows() -> dict:
        """Undo all token-row scalings (exactly, by dividing back in float32 then re-casting)."""
        W = model.get_input_embeddings().weight
        with torch.no_grad():
            for e in reversed(token_rows):
                W[e["ids"]] = e["_orig"]
        token_rows.clear()
        return {"token_rows": token_rows}

    def capture(module: str, prefixes: list, position: str = "last"):
        """Run the network ONCE on up to 512 prefixes (token-id lists or strings);
        return the output of `module` at each prefix's last token (or mean over
        tokens), as a float32 CPU tensor (n, out_features). Costs 1 forward pass."""
        if not 0 < len(prefixes) <= 512:
            raise ValueError("capture takes 1..512 prefixes per call")
        spend(f"capture {module} on {len(prefixes)} prefixes")
        ids = [
            p if isinstance(p, list) else tok(p, add_special_tokens=False)["input_ids"]
            for p in prefixes
        ]
        keyed = {f"k{i}": seq for i, seq in enumerate(ids)}
        session.encode = lambda key: keyed[key]
        order = sorted(keyed, key=lambda k: len(keyed[k]))
        outs = session.capture(module, order, position=position)
        back = {k: i for i, k in enumerate(order)}
        return outs[[back[f"k{i}"] for i in range(len(ids))]]

    def capture_many(modules: list, prefixes: list, position: str = "last") -> dict:
        """Like capture(), for several modules from ONE forward pass (costs 1).
        Returns {module: (n, out_features) float32 CPU tensor}."""
        if not 0 < len(prefixes) <= 512:
            raise ValueError("capture_many takes 1..512 prefixes per call")
        spend(f"capture_many {len(modules)} modules on {len(prefixes)} prefixes")
        ids = [
            p if isinstance(p, list) else tok(p, add_special_tokens=False)["input_ids"]
            for p in prefixes
        ]
        keyed = {f"k{i}": seq for i, seq in enumerate(ids)}
        session.encode = lambda key: keyed[key]
        order = sorted(keyed, key=lambda k: len(keyed[k]))
        outs = session.capture_many(list(modules), order, position=position)
        back = {k: i for i, k in enumerate(order)}
        idx = [back[f"k{i}"] for i in range(len(ids))]
        return {m: t[idx] for m, t in outs.items()}

    def apply_edit(module: str, B, A, note: str = "") -> str:
        """Add B @ A (B: out x r, A: r x in, any float tensors) to `module`.weight. Free."""
        from stateset_agents.experimental.weight_space import LowRankDelta

        return session.apply(
            module,
            LowRankDelta(B=B.detach().float().cpu(), A=A.detach().float().cpu()),
            note=note,
        )

    def ablate_direction(
        module: str, direction, k: int, alpha: float = 1.0, note: str = ""
    ) -> str:
        """Convenience: remove `direction` (len = out_features) from the k input
        columns of `module` that write most along it (rank-1 edit). Free."""
        d = direction_ablation(
            session.weight(module), direction.detach().float().cpu(), k=k, alpha=alpha
        )
        return session.apply(module, d, note=note or f"ablate k={k} alpha={alpha}")

    def revert():
        """Undo the most recent edit exactly. Free."""
        e = session.revert()
        return {"reverted": e.edit_id, "module": e.module, "note": e.note}

    def edits():
        """Active edits (bottom to top)."""
        return [
            {
                "edit_id": e.edit_id,
                "module": e.module,
                "rank": e.delta.rank,
                "note": e.note,
            }
            for e in session.edits
        ]

    def _deltas_by_module() -> dict:
        import torch as _t

        from stateset_agents.experimental.weight_space import LowRankDelta

        per: dict = {}
        for e in session.edits:
            per.setdefault(e.module, []).append(e.delta)
        return {
            m: LowRankDelta(
                B=_t.cat([d.B for d in ds], 1), A=_t.cat([d.A for d in ds], 0)
            )
            for m, ds in per.items()
        }

    def _ckpt(name: str, per_module: dict, with_token_rows: bool = True) -> str:
        def to_composite(m: str) -> str:
            return (
                m.replace("model.layers.", "model.language_model.layers.", 1)
                + ".weight"
            )

        import shutil

        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file, save_file

        src = Path(snapshot_download(pj.MODEL, revision=pj.REVISIONS[pj.MODEL]))
        dst = pj.WORK / "ckpt" / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=False)
        remaining = dict(per_module)
        for shard in sorted(dst.glob("*.safetensors")):
            tensors = load_file(str(shard))
            touched = False
            for m in list(remaining):
                for key in (to_composite(m), m + ".weight"):
                    if key in tensors:
                        W = tensors[key]
                        tensors[key] = (
                            (W.float() + remaining.pop(m).dense().cpu())
                            .to(W.dtype)
                            .contiguous()
                        )
                        touched = True
                        break
            if touched:
                shard.unlink()
                save_file(tensors, str(shard), metadata={"format": "pt"})
        if remaining:
            raise KeyError(f"modules not found in checkpoint: {sorted(remaining)}")
        if with_token_rows and token_rows:
            ekey = "model.language_model.embed_tokens.weight"
            for shard in sorted(dst.glob("*.safetensors")):
                tensors = load_file(str(shard))
                if ekey not in tensors:
                    continue
                E = tensors[ekey].clone()
                for e in token_rows:
                    E[e["ids"]] = (E[e["ids"]].float() * e["scale"]).to(E.dtype)
                tensors[ekey] = E.contiguous()
                shard.unlink()
                save_file(tensors, str(shard), metadata={"format": "pt"})
        return str(dst)

    eval_count = {"n": 0}

    def evaluate(n: int = 128) -> dict:
        """Sample the CURRENT (edited) model on n<=200 practice problems (thinking on,
        same decoding as the final test). Costs 1 forward pass. Returns accuracy,
        hit-token-limit rate, loop rate, mean tokens, and 3 example transcripts."""
        n = max(1, min(int(n), 200))
        spend(
            f"evaluate on {n} practice problems with {len(session.edits)} active edits"
        )
        eval_count["n"] += 1
        per = _deltas_by_module()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_dir = (
            _ckpt(f"b_eval{eval_count['n']}", per) if (per or token_rows) else pj.MODEL
        )
        rows = pj.generate(
            f"b_eval{eval_count['n']}", model_dir, probs["select"][:n], max_tokens, OUT
        )
        STATE["last_eval"] = rows
        report = {k: v for k, v in pj.summarize(rows)["all"].items() if k != "loop"}
        rng = random.Random(eval_count["n"])
        examples = []
        for want in ("hit_limit", "correct", "wrong"):
            pool = [
                r
                for r in rows
                if (
                    r["finish"] == "length"
                    if want == "hit_limit"
                    else (
                        r["correct"]
                        if want == "correct"
                        else not r["correct"] and r["finish"] == "stop"
                    )
                )
            ]
            if pool:
                r = rng.choice(pool)
                examples.append(
                    {
                        "kind": want,
                        "tokens": r["tokens"],
                        "head": r["text"][:500],
                        "tail": r["text"][-700:],
                    }
                )
        report["examples"] = examples
        return report

    def budget_status() -> dict:
        return {
            "limit": budget,
            "used": len(ledger),
            "remaining": budget - len(ledger),
            "ledger": list(ledger),
        }

    cfg = model.config
    ns = {
        "torch": torch,
        "tok": tok,
        "traces": [{k: v for k, v in r.items() if k != "loop"} for r in traces],
        "MODEL_ID": pj.MODEL,
        "config": cfg.to_dict(),
        "linear_modules": session.linear_modules,
        "weight": session.weight,
        "capture": capture,
        "capture_many": capture_many,
        "scale_token_rows": scale_token_rows,
        "clear_token_rows": clear_token_rows,
        "apply_edit": apply_edit,
        "ablate_direction": ablate_direction,
        "revert": revert,
        "edits": edits,
        "evaluate": evaluate,
        "budget": budget_status,
    }
    STATE.update(
        ns=ns,
        session=session,
        probs=probs,
        ledger=ledger,
        deltas=_deltas_by_module,
        ckpt=_ckpt,
        token_rows=token_rows,
        max_tokens=max_tokens,
        budget=budget,
    )
    log(event="ready", traces=len(traces), budget=budget_status())


def run_python(code: str) -> dict:
    buf = io.StringIO()
    err = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            # The kernel exists to run the programmer's Python: it runs on a disposable pod,
            # bound to 127.0.0.1 and reachable only through the operator's SSH tunnel.
            exec(compile(code, "<agent>", "exec"), STATE["ns"])  # nosec B102
        except BudgetExhausted as exc:
            err = f"BudgetExhausted: {exc}"
        except Exception:  # noqa: BLE001 - report to the agent
            err = traceback.format_exc(limit=6)
    out = buf.getvalue()
    if len(out) > 12000:
        out = (
            out[:6000] + f"\n...[{len(out) - 12000} chars truncated]...\n" + out[-6000:]
        )
    return {
        "stdout": out,
        "error": err,
        "budget_remaining": STATE["budget"] - len(STATE["ledger"]),
    }


def heldout(note: str) -> None:
    """Evaluator side: base / submitted edits / structure-matched random control."""
    import torch

    from stateset_agents.experimental.weight_space import LowRankDelta

    summary: dict = {
        "note": note,
        "edits": STATE["ns"]["edits"](),
        "budget": STATE["ns"]["budget"](),
        "token_rows": [
            {k: v for k, v in e.items() if k != "_orig"} for e in STATE["token_rows"]
        ],
    }
    per = STATE["deltas"]()
    g = torch.Generator().manual_seed(7)
    control = {}
    for (
        m,
        d,
    ) in (
        per.items()
    ):  # same A (same columns, same magnitudes), random output directions of equal norm
        R = torch.randn(d.B.shape, generator=g)
        R = R / R.norm(dim=0, keepdim=True) * d.B.norm(dim=0, keepdim=True)
        control[m] = LowRankDelta(B=R, A=d.A)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows = STATE["probs"]["heldout"]
    held = {"base": pj.generate("b_heldout_base", pj.MODEL, rows, 8192, OUT)}
    summary["heldout"] = {"base": pj.summarize(held["base"])}
    (OUT / "armB_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if per or STATE["token_rows"]:
        for name, deltas in (("claude_edit", per), ("random_control", control)):
            held[name] = pj.generate(
                f"b_heldout_{name}", STATE["ckpt"](f"b_{name}", deltas), rows, 8192, OUT
            )
            summary["heldout"][name] = pj.summarize(held[name])
            summary["heldout"][f"{name}_paired_vs_base"] = pj.paired(
                held["base"], held[name]
            )
            (OUT / "armB_summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
        STATE["session"].export(OUT / "armB_adapter")
    STATE["heldout_done"] = True
    log(
        event="heldout_done",
        heldout={k: v.get("all", v) for k, v in summary["heldout"].items()},
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # quiet
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._send(
            200,
            {
                "ready": "ns" in STATE,
                "submitted": STATE.get("submitted", False),
                "heldout_done": STATE.get("heldout_done", False),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        req = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
        )
        if "ns" not in STATE:
            return self._send(503, {"error": "harness not ready"})
        if self.path == "/python":
            if STATE.get("submitted"):
                return self._send(409, {"error": "already submitted"})
            with LOCK:
                result = run_python(req.get("code", ""))
            log(
                event="python",
                chars=len(req.get("code", "")),
                error=bool(result["error"]),
                budget_remaining=result["budget_remaining"],
            )
            return self._send(200, result)
        if self.path == "/submit":
            with LOCK:
                if STATE.get("submitted"):
                    return self._send(409, {"error": "already submitted"})
                STATE["submitted"] = True
            log(event="submit", note=req.get("note", ""), edits=STATE["ns"]["edits"]())
            threading.Thread(
                target=heldout, args=(req.get("note", ""),), daemon=True
            ).start()
            return self._send(200, {"submitted": True, "edits": STATE["ns"]["edits"]()})
        return self._send(404, {"error": "unknown endpoint"})


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--n-trace", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(event="listening", port=PORT)
    setup(a.budget, a.n_trace, a.n_eval, a.max_tokens)
    while not STATE.get("heldout_done"):
        time.sleep(5)
    log(event="server_done")


if __name__ == "__main__":
    main()
