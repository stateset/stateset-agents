"""Arm B: Claude Opus 5.5 as the weight programmer.

Provisions one RunPod GPU pod running ws_server.py (a budgeted weights kernel
bound to localhost), tunnels to it over SSH, and runs a Claude Opus 5.5 tool-use
loop locally (credentials never leave this machine). The agent gets the goal and
the kernel API only — no layers, no method. After it submits, the pod scores
base / Claude's edit / a structure-matched random edit on held-out GSM8K test +
MATH-500, which the agent never sees. The pod is always terminated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path

import anthropic
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from stateset_agents.remote.ledger import CostEntry, record_entry  # noqa: E402
from stateset_agents.remote.runpod import RunPodApi, SshTransport  # noqa: E402
from stateset_agents.remote.serve_session import (  # noqa: E402
    _FLASHINFER_PATCH_COMMAND,
    self_destruct_script,
)

HERE = Path(__file__).parent
IMAGE = "runpod/pytorch:1.1.0-cu1290-torch271-ubuntu2204"
MODEL = "claude-opus-5-5"
LOCAL_PORT = 18765

SYSTEM = """You are a weight-space programmer. You edit a neural network's weights \
directly — by reasoning about them — instead of training it.

You work in a persistent Python kernel on a GPU machine that holds the network. \
Reading weights and doing arithmetic on them is free. Running the network is the \
scarce resource: every call to `capture(...)` or `evaluate(...)` spends one forward \
pass from a fixed budget, so plan before you spend.

Rules:
- Do not use gradient descent or any optimizer on the weights (no autograd, no \
torch.optim, no least-squares fitting of weights to targets). Statistics, \
projections, SVD and contrasts over weights and activations are fine; the edit \
itself must be your own construction.
- Edits go through `apply_edit` / `ablate_direction` (exact, revertible with `revert`).
- Keep edits small and targeted; the model must stay good at everything else.
- A separate held-out evaluation you will never see decides the outcome. Do not \
try to access it.
- Call the `submit` tool when you are done (or when you judge further spending \
will not help). Whatever edits are active at submit time are what gets scored."""

TASK = """The network is {model_id} (a small hybrid-attention language model, \
loaded with Hugging Face transformers; module names like \
`model.layers.N.mlp.down_proj`). With thinking enabled it is weak at grade-school \
math word problems. Your goal: raise its accuracy on such problems by editing its \
weights.

Your budget is {budget} forward passes. One was already spent for you: `traces` \
holds the base model's sampled outputs on {n_traces} practice problems, using the \
exact decoding that the final test uses (thinking on, temperature 0.6, top_p 0.95, \
top_k 20, max {max_tokens} new tokens).

Kernel API (already imported; call `help(name)` for details):
- `traces`: list of dicts with keys id, kind, gold, prompt_ids, gen_ids, text, \
finish ('stop' or 'length'), tokens, answer, correct.
- `tok`: the tokenizer. `config`: the model config dict. `torch`: torch.
- `linear_modules(contains=None) -> {{name: [out, in]}}`  (free)
- `weight(name) -> float32 CPU tensor copy`  (free)
- `capture(module, prefixes, position='last'|'mean') -> (n, out) tensor`  \
(1 forward pass for up to 512 prefixes; a prefix is a list of token ids or a string)
- `capture_many(modules, prefixes, position='last'|'mean') -> {{module: (n, out) tensor}}`  \
(also 1 forward pass, for any number of modules at once)
- `apply_edit(module, B, A, note='')` adds B @ A to module.weight  (free)
- `ablate_direction(module, direction, k, alpha=1.0, note='')` removes a direction \
from the k input columns writing most along it  (free)
- `revert()`, `edits()`  (free)
- `evaluate(n=128) -> metrics + example transcripts` on separate practice problems \
with your current edits  (1 forward pass; n <= 200)
- `budget()`  (free)

Start by looking at the traces. Work in small steps; print compact summaries."""

TOOLS = [
    {
        "name": "python",
        "description": "Execute Python in the persistent weights kernel. Returns stdout "
        "(truncated to ~12k chars), any error traceback, and the remaining forward-pass budget.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."}
            },
            "required": ["code"],
        },
    },
    {
        "name": "submit",
        "description": "Finalize: freeze the currently active edits for the held-out evaluation. "
        "Call exactly once, when done.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "What the edit does and why."}
            },
            "required": ["note"],
        },
    },
]


def log(fh, **kw) -> None:
    kw["ts"] = time.strftime("%H:%M:%S")
    line = json.dumps(kw, default=str)
    print(line[:2000], flush=True)
    fh.write(line + "\n")
    fh.flush()


def run_agent(
    fh,
    client,
    base_url: str,
    budget: int,
    n_traces: int,
    max_tokens: int,
    max_turns: int,
) -> dict:
    messages: list = [
        {
            "role": "user",
            "content": TASK.format(
                model_id="Qwen/Qwen3.5-0.8B",
                budget=budget,
                n_traces=n_traces,
                max_tokens=max_tokens,
            ),
        }
    ]
    submitted = None
    nudged = False
    for turn in range(max_turns):
        try:
            with client.beta.messages.stream(
                model=MODEL,
                max_tokens=64000,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": "high"},
                cache_control={"type": "ephemeral"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                response = stream.get_final_message()
        except ValueError as exc:  # unparseable eager tool input: re-issue the turn
            log(fh, event="invalid_tool_json_reissue", error=str(exc)[:300])
            continue
        usage = response.usage
        log(
            fh,
            event="assistant",
            turn=turn,
            stop=response.stop_reason,
            model=response.model,
            in_tokens=usage.input_tokens,
            out_tokens=usage.output_tokens,
            cache_read=getattr(usage, "cache_read_input_tokens", None),
            content=[b.to_dict() for b in response.content],
        )
        if response.stop_reason == "refusal":
            log(fh, event="refusal", details=getattr(response, "stop_details", None))
            break
        messages.append({"role": "assistant", "content": response.content})
        calls = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason == "max_tokens" and calls:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": c.id,
                            "is_error": True,
                            "content": "Output hit max_tokens before this tool input finished; resend it shorter.",
                        }
                        for c in calls
                    ],
                }
            )
            continue
        if not calls:
            if nudged:
                break
            nudged = True
            messages.append(
                {
                    "role": "user",
                    "content": "Continue working, or call `submit` if you are done.",
                }
            )
            continue
        results = []
        for c in calls:
            inp = c.input if isinstance(c.input, dict) else {}
            key = "code" if c.name == "python" else "note"
            if not isinstance(inp.get(key), str) or (
                c.name not in ("python", "submit")
            ):
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": c.id,
                        "is_error": True,
                        "content": json.dumps({"INVALID_JSON": str(c.input)[:2000]}),
                    }
                )
                continue
            if c.name == "submit":
                r = requests.post(
                    f"{base_url}/submit", json={"note": inp["note"]}, timeout=120
                ).json()
                submitted = {"note": inp["note"], "server": r}
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": c.id,
                        "content": json.dumps(r),
                    }
                )
                log(fh, event="submitted", **submitted)
                continue
            r = requests.post(
                f"{base_url}/python", json={"code": inp["code"]}, timeout=3600
            ).json()
            log(fh, event="tool_result", turn=turn, result=r)
            text = r.get("stdout", "")
            if r.get("error"):
                text += f"\n[error]\n{r['error']}"
            text += f"\n[forward passes remaining: {r.get('budget_remaining')}]"
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": c.id,
                    "content": text,
                    **({"is_error": True} if r.get("error") else {}),
                }
            )
        messages.append({"role": "user", "content": results})
        if submitted:
            break
    if not submitted:
        r = requests.post(
            f"{base_url}/submit",
            json={"note": "auto-submitted: agent ended without submit"},
            timeout=120,
        ).json()
        submitted = {"note": "auto", "server": r}
        log(fh, event="auto_submitted", server=r)
    return submitted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--max-cost-usd", type=float, default=10.0)
    ap.add_argument("--lifetime-h", type=float, default=2.5)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument(
        "--anthropic-key-file",
        default=None,
        help="file holding the API key (read, never logged)",
    )
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    fh = open(out / "arm_b.log", "a")
    run_id = f"ws-armb-{uuid.uuid4().hex[:8]}"
    cfg = tomllib.loads(Path("~/.runpod/config.toml").expanduser().read_text())
    api_key = (cfg.get("default") or cfg.get("credentials") or {}).get("api_key", "")
    public_key = Path("~/.ssh/id_rsa.pub").expanduser().read_text().strip()
    api, ssh = RunPodApi(api_key), SshTransport()
    key = (
        Path(a.anthropic_key_file).expanduser().read_text().strip()
        if a.anthropic_key_file
        else None
    )
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    # Fail fast on missing/invalid credentials before renting anything.
    client.models.retrieve(MODEL)
    log(fh, event="anthropic_auth_ok", model=MODEL)

    staging = out / "upload"
    staging.mkdir()
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(
            (REPO / "stateset_agents" / "experimental" / "weight_space").glob("*.py")
        ):
            tar.add(
                p,
                arcname=f"weight_space/{p.name}",
                filter=lambda ti: (
                    setattr(ti, "uid", 0),
                    setattr(ti, "gid", 0),
                    setattr(ti, "uname", "root"),
                    setattr(ti, "gname", "root"),
                    ti,
                )[-1],
            )
    (staging / "wslib.tgz").write_bytes(buf.getvalue())
    (staging / "run_b.sh").write_text(f"""#!/bin/bash
set -uo pipefail
mkdir -p /workspace/out
trap 'rc=$?; cp /workspace/armb.log /workspace/out/ 2>/dev/null; echo $rc > /workspace/EXIT' EXIT
set -e
cd /workspace
mkdir -p wslib && tar --no-same-owner -xzf wslib.tgz -C wslib
export HF_HOME=/workspace/hf PIP_DISABLE_PIP_VERSION_CHECK=1
python -m pip install --quiet vllm datasets "math-verify[antlr4_13_2]" peft
{_FLASHINFER_PATCH_COMMAND}
PYTHONPATH=/workspace/wslib:/workspace python -u ws_server.py --budget {a.budget}
""")
    pod = api.create_pod(
        name=run_id,
        image=IMAGE,
        gpu_type_id=a.gpu,
        ports=["22/tcp"],
        env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
        container_disk_gb=100,
        cloud_type="SECURE",
    )
    pod_id, started, cost_per_hr = pod.get("id"), time.time(), pod.get("costPerHr")
    status, tunnel = "failed", None
    (out / "pod.json").write_text(json.dumps({"pod_id": pod_id, "run_id": run_id}))
    try:
        log(fh, event="pod_created", pod_id=pod_id, cost_per_hr=cost_per_hr)
        if (
            not pod_id
            or cost_per_hr is None
            or float(cost_per_hr) * a.lifetime_h > a.max_cost_usd
        ):
            log(fh, event="abort", reason="no pod id, unknown price, or over cost cap")
            return 4
        deadline = time.monotonic() + 1200
        while True:
            info = api.get_pod(pod_id)
            ip, port = info.get("publicIp"), (info.get("portMappings") or {}).get("22")
            if info.get("desiredStatus") == "RUNNING" and ip and port:
                break
            if time.monotonic() > deadline:
                log(fh, event="abort", reason="no ssh endpoint in 20 min")
                return 5
            time.sleep(10)
        ssh.wait_until_reachable(str(ip), int(port), 600)
        ssh.upload_secret(api_key, "/root/.ws_runpod_key")
        sd = staging / "self-destruct.sh"
        sd.write_text(
            self_destruct_script(pod_id, a.lifetime_h, key_file="/root/.ws_runpod_key")
        )
        ssh.upload(sd, "/workspace/self-destruct.sh")
        rc, msg = ssh.run(
            "chmod 600 /root/.ws_runpod_key && (nohup bash /workspace/self-destruct.sh > /workspace/sd.log 2>&1 < /dev/null &)"
        )
        if rc != 0:
            log(
                fh,
                event="abort",
                reason="self-destruct arming failed",
                detail=msg[-300:],
            )
            return 6
        log(fh, event="self_destruct_armed", hours=a.lifetime_h)
        for name in ("wslib.tgz", "run_b.sh"):
            ssh.upload(staging / name, f"/workspace/{name}")
        for name in ("pod_job.py", "ws_server.py"):
            ssh.upload(HERE / name, f"/workspace/{name}")
        ssh.run(
            "(nohup bash /workspace/run_b.sh > /workspace/armb.log 2>&1 < /dev/null &)"
        )
        log(fh, event="server_starting")

        tunnel = subprocess.Popen(
            [
                "ssh",
                *ssh._base_opts(),
                "-p",
                str(port),
                "-N",
                "-L",
                f"{LOCAL_PORT}:127.0.0.1:8765",
                f"root@{ip}",
            ]
        )
        base_url = f"http://127.0.0.1:{LOCAL_PORT}"
        ready_deadline = time.time() + 45 * 60
        while time.time() < ready_deadline:
            try:
                if requests.get(base_url, timeout=10).json().get("ready"):
                    break
            except requests.RequestException:
                pass
            rc, code = ssh.run("cat /workspace/EXIT 2>/dev/null")
            if rc == 0 and code.strip():
                _, tail = ssh.run("tail -c 1500 /workspace/armb.log")
                log(
                    fh,
                    event="abort",
                    reason=f"harness exited ({code.strip()}) before ready",
                    tail=tail,
                )
                return 8
            time.sleep(30)
        else:
            log(fh, event="abort", reason="harness never became ready")
            return 7
        log(fh, event="harness_ready")
        submitted = run_agent(fh, client, base_url, a.budget, 256, 8192, a.max_turns)
        heldout_deadline = started + a.lifetime_h * 3600 - 12 * 60
        while time.time() < heldout_deadline:
            try:
                if requests.get(base_url, timeout=10).json().get("heldout_done"):
                    break
            except requests.RequestException:
                pass
            time.sleep(30)
        files = ssh.download_dir("/workspace/out", out / "results")
        log(fh, event="downloaded", files=len(files), submitted=submitted)
        status = "succeeded"
        return 0
    finally:
        if tunnel:
            tunnel.terminate()
        if pod_id:
            for attempt in range(3):
                try:
                    api.terminate_pod(pod_id)
                    log(fh, event="terminated", pod_id=pod_id)
                    break
                except Exception as exc:  # noqa: BLE001
                    log(
                        fh,
                        event="terminate_retry",
                        attempt=attempt,
                        error=str(exc)[:300],
                    )
                    time.sleep(5)
            duration = time.time() - started
            cost = (
                float(cost_per_hr) * duration / 3600
                if cost_per_hr is not None
                else None
            )
            record_entry(
                CostEntry(
                    provider="runpod",
                    job_id=run_id,
                    base_model="Qwen/Qwen3.5-0.8B",
                    gpu=a.gpu,
                    gpu_count=1,
                    cost_per_hr=float(cost_per_hr) if cost_per_hr else None,
                    duration_s=duration,
                    cost_usd=cost,
                    status=status,
                )
            )
            log(
                fh,
                event="cost",
                duration_s=round(duration),
                cost_usd=round(cost, 3) if cost else None,
            )


if __name__ == "__main__":
    raise SystemExit(main())
