"""Launch pod_job.py on one RunPod GPU pod; poll, download, ALWAYS terminate.

Cost controls (fail closed): price must be known and costPerHr x lifetime must
fit --max-cost-usd; an in-pod self-destruct is armed before any other work;
the pod is terminated in ``finally``; spend is appended to the repo ledger.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from stateset_agents.remote.ledger import CostEntry, record_entry  # noqa: E402
from stateset_agents.remote.runpod import RunPodApi, SshTransport  # noqa: E402
from stateset_agents.remote.serve_session import (  # noqa: E402
    _FLASHINFER_PATCH_COMMAND,
    self_destruct_script,
)

HERE = Path(__file__).parent
WS_PKG = REPO / "stateset_agents" / "experimental" / "weight_space"
IMAGE = "runpod/pytorch:1.1.0-cu1290-torch271-ubuntu2204"
REMOTE_KEY = "/root/.ws_runpod_key"


def log(fh, **kw) -> None:
    kw["ts"] = time.strftime("%H:%M:%S")
    line = json.dumps(kw, default=str)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def with_retry(fh, what: str, fn, attempts: int = 5, wait_s: float = 15.0):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - transient ssh/scp failures
            log(fh, event="retry", what=what, attempt=attempt, error=str(exc)[:200])
            if attempt == attempts - 1:
                raise
            time.sleep(wait_s * (attempt + 1))


def api_call(fh, what: str, fn, attempts: int = 8):
    """Retry a read-only RunPod REST call through transient failures."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 5xx / SSL EOF / connection resets
            log(fh, event="api_retry", call=what, attempt=attempt, error=str(exc)[:160])
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 10 * (attempt + 1)))


def run_sh(job_args: str, entry: str | None = None) -> str:
    command = entry or f"pod_job.py main --out /workspace/out {job_args}"
    return f"""#!/bin/bash
set -uo pipefail
mkdir -p /workspace/out
trap 'rc=$?; cp /workspace/phase1.log /workspace/out/ 2>/dev/null; echo $rc > /workspace/EXIT' EXIT
set -e
cd /workspace
export HF_HOME=/workspace/hf PIP_DISABLE_PIP_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# A host with bad PyPI connectivity once hung this install for 26 idle,
# billed minutes; fail fast (exit 124) so the launcher terminates the pod.
echo '{{"event": "install_start"}}'
timeout 900 python -m pip install --quiet --timeout 30 --retries 5 vllm datasets "math-verify[antlr4_13_2]" peft
timeout 300 python -m pip install --quiet --timeout 30 --retries 5 /workspace/stateset_agents-*.whl
echo '{{"event": "installed"}}'
{_FLASHINFER_PATCH_COMMAND}
python -c "import torch, transformers, vllm; print('versions', torch.__version__, transformers.__version__, vllm.__version__, torch.cuda.get_device_name(0))"
PYTHONPATH=/workspace python -u {command}
"""


WHEEL_GLOB = "stateset_agents-*.whl"


def build_wheel(dest: Path) -> Path:
    """This checkout's stateset-agents wheel; pods pip-install it.

    (A standalone copy of ``weight_space`` no longer imports on its own: its
    exceptions derive from ``stateset_agents.core.errors.StateSetError``.)
    """
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--quiet",
            "-w",
            str(dest),
            str(REPO),
        ],
        check=True,
    )
    return max(dest.glob(WHEEL_GLOB), key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--max-cost-usd", type=float, default=15.0)
    ap.add_argument("--lifetime-h", type=float, default=3.0)
    ap.add_argument("--disk-gb", type=int, default=100)
    ap.add_argument("--ledger-model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--job-args", default="")
    ap.add_argument(
        "--entry",
        default=None,
        help="command run with python -u on the pod instead of 'pod_job.py main ...'",
    )
    ap.add_argument(
        "--resume-from",
        default=None,
        help="dir with gen_base_trace.jsonl.gz / directions.pt to seed",
    )
    ap.add_argument(
        "--upload",
        action="append",
        default=[],
        help="extra file uploaded to /workspace/<name>",
    )
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    fh = open(out / "launcher.log", "a")
    run_id = f"ws-phase1-{uuid.uuid4().hex[:8]}"
    cfg = tomllib.loads(Path("~/.runpod/config.toml").expanduser().read_text())
    api_key = (cfg.get("default") or cfg.get("credentials") or {}).get("api_key", "")
    if not api_key:
        log(fh, event="abort", reason="no RunPod api_key in ~/.runpod/config.toml")
        return 2
    public_key = Path("~/.ssh/id_rsa.pub").expanduser().read_text().strip()
    api, ssh = RunPodApi(api_key), SshTransport()

    staging = out / "upload"
    staging.mkdir()
    wheel = build_wheel(staging)
    (staging / "run.sh").write_text(run_sh(a.job_args, a.entry))
    (staging / "pod_job.py").write_text((HERE / "pod_job.py").read_text())
    (staging / "wait_direction.py").write_text((HERE / "wait_direction.py").read_text())
    small = HERE / "small_edit.pt"

    log(fh, event="create_pod", run_id=run_id, gpu=a.gpu, image=IMAGE)
    pod = None
    for attempt in range(4):
        try:
            pod = api.create_pod(
                name=run_id,
                image=IMAGE,
                gpu_type_id=a.gpu,
                ports=["22/tcp"],
                env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
                container_disk_gb=a.disk_gb,
                cloud_type="SECURE",
            )
            break
        except (
            Exception
        ) as exc:  # noqa: BLE001 - RunPod 500s are ambiguous: check before retrying
            log(fh, event="create_failed", attempt=attempt, error=str(exc)[:200])
            time.sleep(20)
            mine = [
                x
                for x in api_call(fh, "list_pods", api.list_pods)
                if x.get("name") == run_id
            ]
            if mine:
                pod = mine[0]
                log(fh, event="adopted_allocation", pod_id=pod.get("id"))
                break
    if pod is None:
        log(
            fh,
            event="abort",
            reason="pod creation failed 4 times; no allocation under our name",
        )
        return 2
    pod_id = pod.get("id")
    if not pod_id:
        log(
            fh,
            event="abort",
            reason="no pod id returned; inspect RunPod console",
            pod=pod,
        )
        return 3
    started = time.time()
    cost_per_hr = pod.get("costPerHr")
    status = "failed"
    (out / "pod.json").write_text(
        json.dumps({"pod_id": pod_id, "run_id": run_id, "created": started})
    )
    try:
        log(fh, event="pod_created", pod_id=pod_id, cost_per_hr=cost_per_hr)
        if cost_per_hr is None or float(cost_per_hr) * a.lifetime_h > a.max_cost_usd:
            log(
                fh,
                event="abort",
                reason=f"price {cost_per_hr}/hr x {a.lifetime_h}h exceeds ${a.max_cost_usd} or is unknown",
            )
            return 4
        deadline = time.monotonic() + 20 * 60
        while True:
            info = api_call(fh, "get_pod", lambda: api.get_pod(pod_id))
            ip, port = info.get("publicIp"), (info.get("portMappings") or {}).get("22")
            if info.get("desiredStatus") == "RUNNING" and ip and port:
                break
            if time.monotonic() > deadline:
                log(
                    fh,
                    event="abort",
                    reason="pod never published ssh within 20 min",
                    status=info.get("desiredStatus"),
                )
                return 5
            time.sleep(10)
        ssh.wait_until_reachable(str(ip), int(port), 600)
        log(fh, event="ssh_ready", host=ip, port=port)

        # Armed FIRST: nothing after this can leak a forever-billing pod.
        with_retry(fh, "upload_secret", lambda: ssh.upload_secret(api_key, REMOTE_KEY))
        sd = staging / "self-destruct.sh"
        sd.write_text(self_destruct_script(pod_id, a.lifetime_h, key_file=REMOTE_KEY))
        with_retry(
            fh,
            "upload self-destruct",
            lambda: ssh.upload(sd, "/workspace/self-destruct.sh"),
        )
        rc, msg = ssh.run(
            "chmod 600 /root/.ws_runpod_key && (nohup bash /workspace/self-destruct.sh > /workspace/sd.log 2>&1 < /dev/null &)"
        )
        if rc != 0:
            log(
                fh,
                event="abort",
                reason="self-destruct arming failed",
                detail=msg[-400:],
            )
            return 6
        log(fh, event="self_destruct_armed", hours=a.lifetime_h)

        for name in (wheel.name, "run.sh", "pod_job.py", "wait_direction.py"):
            with_retry(
                fh,
                f"upload {name}",
                lambda name=name: ssh.upload(staging / name, f"/workspace/{name}"),
            )
        if small.exists():
            with_retry(
                fh,
                "upload small_edit.pt",
                lambda: ssh.upload(small, "/workspace/small_edit.pt"),
            )
        for extra in a.upload:
            src = Path(extra)
            with_retry(
                fh,
                f"upload {src.name}",
                lambda src=src: ssh.upload(src, f"/workspace/{src.name}"),
            )
        if a.resume_from:
            ssh.run("mkdir -p /workspace/out")
            for name in (
                "gen_base_trace.jsonl.gz",
                "directions.pt",
                "gen_base_select.jsonl.gz",
            ):
                src = Path(a.resume_from) / name
                if src.exists():
                    with_retry(
                        fh,
                        f"seed {name}",
                        lambda src=src, name=name: ssh.upload(
                            src, f"/workspace/out/{name}"
                        ),
                    )
                    log(fh, event="seeded", file=name)
        rc, msg = ssh.run(
            "(nohup bash /workspace/run.sh > /workspace/phase1.log 2>&1 < /dev/null &)"
        )
        if rc != 0:
            log(fh, event="abort", reason="job start failed", detail=msg[-400:])
            return 7
        log(fh, event="job_started")

        job_deadline = started + a.lifetime_h * 3600 - 12 * 60
        seen: set[str] = set()
        failures = 0
        exit_code = None
        while time.time() < job_deadline:
            time.sleep(30)
            rc, text = ssh.run(
                # ASCII-only: a byte cut through a multi-byte progress-bar glyph made
                # the UTF-8 decode raise and killed a run before download (run 5b).
                "cat /workspace/EXIT 2>/dev/null; echo '<<SEP>>'; grep -a '^{' /workspace/phase1.log | tail -n 20 "
                "| tr -cd '\\11\\12\\15\\40-\\176'; echo '<<SEP>>'; tail -c 300 /workspace/phase1.log "
                "| tr -cd '\\11\\12\\15\\40-\\176'"
            )
            if rc != 0:
                failures += 1
                if failures >= 10:
                    state = api_call(fh, "get_pod", lambda: api.get_pod(pod_id)).get(
                        "desiredStatus"
                    )
                    log(fh, event="ssh_unreachable", pod_status=state)
                    if state != "RUNNING":
                        break
                continue
            failures = 0
            parts = text.split("<<SEP>>")
            for line in parts[1].strip().splitlines() if len(parts) > 1 else []:
                if line not in seen:
                    seen.add(line)
                    log(fh, event="pod", line=line[:1500])
            if parts[0].strip():
                exit_code = parts[0].strip()
                log(
                    fh,
                    event="job_exit",
                    code=exit_code,
                    tail=parts[2][-300:] if len(parts) > 2 else "",
                )
                break
        else:
            log(fh, event="deadline", note="downloading partial results")

        try:
            files = ssh.download_dir("/workspace/out", out / "results")
            log(fh, event="downloaded", files=len(files))
        except Exception as exc:  # noqa: BLE001
            log(fh, event="download_failed", error=str(exc)[:400])
        status = "succeeded" if exit_code == "0" else f"exit_{exit_code}"
        return 0 if exit_code == "0" else 1
    finally:
        for attempt in range(3):
            try:
                api.terminate_pod(pod_id)
                log(fh, event="terminated", pod_id=pod_id)
                break
            except Exception as exc:  # noqa: BLE001
                log(fh, event="terminate_retry", attempt=attempt, error=str(exc)[:300])
                time.sleep(5)
        duration = time.time() - started
        cost = float(cost_per_hr) * duration / 3600 if cost_per_hr is not None else None
        record_entry(
            CostEntry(
                provider="runpod",
                job_id=run_id,
                base_model=a.ledger_model,
                gpu=a.gpu,
                gpu_count=1,
                cost_per_hr=float(cost_per_hr) if cost_per_hr is not None else None,
                duration_s=duration,
                cost_usd=cost,
                status=status,
            )
        )
        log(
            fh,
            event="cost",
            duration_s=round(duration),
            cost_usd=round(cost, 3) if cost is not None else None,
            status=status,
        )


if __name__ == "__main__":
    raise SystemExit(main())
