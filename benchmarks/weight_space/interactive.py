"""Interactive weights-kernel session on one RunPod GPU (Claude in this session is the programmer).

Provisions a pod running ws_server.py, tunnels it to http://127.0.0.1:18765, and keeps
the session alive until the local STOP file exists, the held-out evaluation finishes
(after /submit), or the deadline. Then downloads /workspace/out and ALWAYS terminates.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import time
import tomllib
import uuid
from pathlib import Path

import requests

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("launch", HERE / "launch.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

LOCAL_PORT = 18765


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-cost-usd", type=float, default=3.5)
    ap.add_argument("--lifetime-h", type=float, default=3.0)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument(
        "--seed-trace",
        default=None,
        help="gen_base_trace.jsonl.gz to reuse instead of regenerating",
    )
    ap.add_argument(
        "--contrast",
        default=str(HERE / "answer_contrast.json"),
        help="contrast prefixes JSON",
    )
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    fh = open(out / "interactive.log", "a")
    run_id = f"ws-interactive-{uuid.uuid4().hex[:8]}"
    cfg = tomllib.loads(Path("~/.runpod/config.toml").expanduser().read_text())
    api_key = (cfg.get("default") or cfg.get("credentials") or {}).get("api_key", "")
    public_key = Path("~/.ssh/id_rsa.pub").expanduser().read_text().strip()
    api, ssh = L.RunPodApi(api_key), L.SshTransport()

    staging = out / "upload"
    staging.mkdir()
    (staging / "wslib.tgz").write_bytes(L.ws_tarball())
    (staging / "run_i.sh").write_text(f"""#!/bin/bash
set -uo pipefail
mkdir -p /workspace/out
trap 'rc=$?; cp /workspace/interactive_server.log /workspace/out/ 2>/dev/null; echo $rc > /workspace/EXIT' EXIT
set -e
cd /workspace
mkdir -p wslib && tar --no-same-owner -xzf wslib.tgz -C wslib
export HF_HOME=/workspace/hf PIP_DISABLE_PIP_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WS_CAPTURE_BATCH=4
python -m pip install --quiet vllm datasets "math-verify[antlr4_13_2]" peft
{L._FLASHINFER_PATCH_COMMAND}
PYTHONPATH=/workspace/wslib:/workspace python -u ws_server.py --budget {a.budget}
""")
    if a.seed_trace:
        shutil.copy(a.seed_trace, staging / "gen_b_base_trace.jsonl.gz")

    pod = None
    for attempt in range(4):
        try:
            pod = api.create_pod(
                name=run_id,
                image=L.IMAGE,
                gpu_type_id=a.gpu,
                ports=["22/tcp"],
                env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
                container_disk_gb=100,
                cloud_type="SECURE",
            )
            break
        except Exception as exc:  # noqa: BLE001 - ambiguous 500s: check before retrying
            L.log(fh, event="create_failed", attempt=attempt, error=str(exc)[:160])
            time.sleep(20)
            mine = [
                x
                for x in L.api_call(fh, "list_pods", api.list_pods)
                if x.get("name") == run_id
            ]
            if mine:
                pod = mine[0]
                break
    if pod is None:
        L.log(fh, event="abort", reason="create failed")
        return 2
    pod_id, started, cost_per_hr = pod["id"], time.time(), pod.get("costPerHr")
    tunnel, status = None, "failed"
    try:
        L.log(fh, event="pod_created", pod_id=pod_id, cost_per_hr=cost_per_hr)
        if cost_per_hr is None or float(cost_per_hr) * a.lifetime_h > a.max_cost_usd:
            L.log(fh, event="abort", reason="price unknown or over cap")
            return 4
        deadline = time.monotonic() + 1200
        while True:
            info = L.api_call(fh, "get_pod", lambda: api.get_pod(pod_id))
            ip, port = info.get("publicIp"), (info.get("portMappings") or {}).get("22")
            if info.get("desiredStatus") == "RUNNING" and ip and port:
                break
            if time.monotonic() > deadline:
                L.log(fh, event="abort", reason="no ssh endpoint in 20 min")
                return 5
            time.sleep(10)
        ssh.wait_until_reachable(str(ip), int(port), 600)
        L.with_retry(
            fh, "upload_secret", lambda: ssh.upload_secret(api_key, L.REMOTE_KEY)
        )
        sd = staging / "self-destruct.sh"
        sd.write_text(
            L.self_destruct_script(pod_id, a.lifetime_h, key_file=L.REMOTE_KEY)
        )
        L.with_retry(
            fh, "upload sd", lambda: ssh.upload(sd, "/workspace/self-destruct.sh")
        )
        rc, msg = ssh.run(
            "chmod 600 /root/.ws_runpod_key && (nohup bash /workspace/self-destruct.sh > /workspace/sd.log 2>&1 < /dev/null &)"
        )
        if rc != 0:
            L.log(fh, event="abort", reason="self-destruct arming failed")
            return 6
        L.log(fh, event="self_destruct_armed", hours=a.lifetime_h, host=ip, port=port)
        ssh.run("mkdir -p /workspace/out")
        uploads = [
            ("wslib.tgz", "/workspace/wslib.tgz"),
            ("run_i.sh", "/workspace/run_i.sh"),
        ]
        if a.seed_trace:
            uploads.append(
                (
                    "gen_b_base_trace.jsonl.gz",
                    "/workspace/out/gen_b_base_trace.jsonl.gz",
                )
            )
        for name, dest in uploads:
            L.with_retry(
                fh,
                f"upload {name}",
                lambda name=name, dest=dest: ssh.upload(staging / name, dest),
            )
        for name in ("pod_job.py", "ws_server.py"):
            L.with_retry(
                fh,
                f"upload {name}",
                lambda name=name: ssh.upload(HERE / name, f"/workspace/{name}"),
            )
        if Path(a.contrast).exists():
            L.with_retry(
                fh,
                "upload contrast",
                lambda: ssh.upload(Path(a.contrast), "/workspace/answer_contrast.json"),
            )
        ssh.run(
            "(nohup bash /workspace/run_i.sh > /workspace/interactive_server.log 2>&1 < /dev/null &)"
        )
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
        L.log(fh, event="server_starting", tunnel=base_url)
        stop = out / "STOP"
        end = started + a.lifetime_h * 3600 - 12 * 60
        ready_logged = False
        while time.time() < end and not stop.exists():
            try:
                st = requests.get(base_url, timeout=10).json()
                if st.get("ready") and not ready_logged:
                    L.log(fh, event="harness_ready")
                    ready_logged = True
                if st.get("heldout_done"):
                    L.log(fh, event="heldout_done")
                    break
            except requests.RequestException:
                if tunnel.poll() is not None:  # tunnel died: reopen it
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
            time.sleep(20)
        try:
            files = ssh.download_dir("/workspace/out", out / "results")
            L.log(fh, event="downloaded", files=len(files))
        except Exception as exc:  # noqa: BLE001
            L.log(fh, event="download_failed", error=str(exc)[:200])
        status = "succeeded"
        return 0
    finally:
        if tunnel:
            tunnel.terminate()
        for attempt in range(3):
            try:
                api.terminate_pod(pod_id)
                L.log(fh, event="terminated", pod_id=pod_id)
                break
            except Exception as exc:  # noqa: BLE001
                L.log(
                    fh, event="terminate_retry", attempt=attempt, error=str(exc)[:200]
                )
                time.sleep(5)
        duration = time.time() - started
        cost = float(cost_per_hr) * duration / 3600 if cost_per_hr is not None else None
        L.record_entry(
            L.CostEntry(
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
        L.log(
            fh,
            event="cost",
            duration_s=round(duration),
            cost_usd=round(cost, 3) if cost else None,
        )


if __name__ == "__main__":
    raise SystemExit(main())
