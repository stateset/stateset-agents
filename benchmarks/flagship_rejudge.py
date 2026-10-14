#!/usr/bin/env python3
"""Re-score existing flagship transcripts with the current judge and rubric.

No training and no policy rollouts: the tool takes a finished flagship seed's
``transcripts/{baseline,final}.jsonl`` (written by ``flagship_workload.py``),
rebuilds each scenario's judge reference material from the pinned dataset
revision, and runs only the workload's judge phase
(:func:`flagship_workload.phase_judge`) with the installed
:mod:`stateset_agents.evaluation.support_judge` rubric. It writes per-item
judge results, ``judge/{baseline,final}.jsonl``, and ``rejudge_summary.json``
(mean score per group, self-disagreement, findings counts, forced-check
counts) to ``--output-dir``.

Two execution modes:

``--local``
    Run the judge phase on this machine (it needs a GPU with ~44 GiB free for
    the 24B bf16 judge under vLLM). This is also what the pod runs.

RunPod (default)
    Rent exactly one single-GPU pod: by default the cheapest 80 GB-class GPU
    (A100 80GB / H100) in the free public catalog for the tier; upload the
    two transcript files; clone and install the exact harness commit; run
    ``--local`` there detached under ``nohup`` with an exit marker; download
    the outputs; terminate the pod unconditionally; record the authoritative
    cost in the ledger. Spending requires an explicit ceiling
    (``--confirm-max-cost-usd`` or ``STATESET_REJUDGE_CONFIRM_MAX_COST_USD``)
    that covers the worst case (the full pod lifetime at the catalog price
    and, after allocation, at the pod's own ``costPerHr``).

    # free: price the worst case from the public catalog
    python benchmarks/flagship_rejudge.py --input-dir RUN/artifacts/seed-42 \\
        --output-dir out/ --plan
    # free, offline: validate inputs and print the remote commands
    python benchmarks/flagship_rejudge.py --input-dir ... --output-dir out/ --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BENCHMARKS = Path(__file__).resolve().parent
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

import flagship_workload as workload  # noqa: E402
from flagship_runpod_driver import (  # noqa: E402
    DEFAULT_IMAGE,
    RUNTIME_PROFILES,
    FlagshipDriverError,
    resolve_harness_revision,
)
from runpod_backend_conformance import (  # noqa: E402
    _CATALOG_URL,
    _REMOTE_REPOSITORY,
    _REPOSITORY_URL,
    RunPodConformanceError,
    _arm_self_destruct,
    _positive_price,
    _public_key,
    _run_checked,
    _wait_for_ssh,
    _write_lease,
    fetch_catalog,
)

from stateset_agents.remote.executor import RemoteExecutionError  # noqa: E402
from stateset_agents.remote.ledger import (  # noqa: E402
    BudgetExceeded,
    CostEntry,
    check_budget,
    estimate_cost_usd,
    record_entry,
)
from stateset_agents.remote.runpod import (  # noqa: E402
    DEFAULT_RUNPOD_LEASE_DIR,
    RunPodApi,
    SshTransport,
)

SUMMARY_KIND = "stateset-flagship-rejudge-summary"
SUMMARY_NAME = "rejudge_summary.json"
PROVIDER_RECORD = "rejudge-provider.json"
GROUPS: tuple[str, ...] = ("baseline", "final")
#: The A+ gate's ceiling on judge self-disagreement (reported, not enforced).
SELF_DISAGREEMENT_GATE = 0.05
DEFAULT_MANIFEST = _BENCHMARKS / "flagship_manifest.json"
ENV_CEILING = "STATESET_REJUDGE_CONFIRM_MAX_COST_USD"
#: Hard upper bound on any ceiling: a rejudge is ~1 GPU-hour, never a seed.
MAX_CEILING_USD = 25.0
DEFAULT_TIER = "SECURE"
DEFAULT_DISK_GB = 150
DEFAULT_LIFETIME_SECONDS = 5400
MIN_LIFETIME_SECONDS = 1800
DEFAULT_POLL_SECONDS = 60.0
#: 80 GB-class GPUs that fit the 24B bf16 judge (~44 GiB of weights at
#: ``gpu_memory_utilization=0.85``), in tie-break preference order; the
#: cheapest available one in the tier wins. A100 PCIe comes last because its
#: RunPod networking has been unreliable for large snapshot downloads.
DEFAULT_GPU_CANDIDATES: tuple[str, ...] = (
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 NVL",
    "NVIDIA A100 80GB PCIe",
)
_ALLOWED_TIERS = frozenset({"SECURE", "COMMUNITY"})
_FINDING_KINDS = ("must_avoid_hits", "placeholders", "repeated_turns", "long_turns")
_REMOTE_INPUT = "/workspace/rejudge-input"
_REMOTE_OUTPUT = "/workspace/rejudge-output"
_REMOTE_EXIT = "/workspace/rejudge.exit"
_REMOTE_NOHUP_LOG = "/workspace/rejudge-nohup.log"
_REMOTE_HF_TOKEN = "/workspace/.stateset-rejudge-hf-token"
_REMOTE_LOG = f"{_REMOTE_OUTPUT}/rejudge.log"


class RejudgeError(RuntimeError):
    """Raised when a rejudge cannot run safely or produce honest results."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Inputs (local, free)
# --------------------------------------------------------------------------- #


def _transcript_path(input_dir: Path, group: str) -> Path:
    return Path(input_dir) / "transcripts" / f"{group}.jsonl"


def load_transcripts(input_dir: Path) -> dict[str, dict[str, list[list[str]]]]:
    """``{group: {scenario_id: [[role, text], ...]}}`` from a seed's artifacts.

    Both groups must exist, be non-empty, have unique ids, and cover the same
    scenario ids.
    """
    groups: dict[str, dict[str, list[list[str]]]] = {}
    for group in GROUPS:
        path = _transcript_path(input_dir, group)
        if not path.is_file():
            raise RejudgeError(f"missing transcripts file {path}")
        rows: dict[str, list[list[str]]] = {}
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                item_id = str(row["id"])
                transcript = [
                    [str(role), str(text)] for role, text in row["transcript"]
                ]
            except (ValueError, KeyError, TypeError) as exc:
                raise RejudgeError(f"{path}:{number + 1} is malformed: {exc}") from exc
            if not item_id or item_id in rows:
                raise RejudgeError(f"{path} has an empty or duplicate id {item_id!r}")
            rows[item_id] = transcript
        if not rows:
            raise RejudgeError(f"{path} has no transcripts")
        groups[group] = rows
    if set(groups["baseline"]) != set(groups["final"]):
        raise RejudgeError("baseline and final transcripts cover different scenarios")
    return groups


def transcript_digests(input_dir: Path) -> dict[str, str]:
    return {
        group: hashlib.sha256(
            _transcript_path(input_dir, group).read_bytes()
        ).hexdigest()
        for group in GROUPS
    }


def resolve_pins(args: argparse.Namespace) -> dict[str, Any]:
    """Judge model/revision and dataset revision: flags override the manifest."""
    manifest: dict[str, Any] = {}
    if args.manifest is not None:
        try:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RejudgeError(f"cannot read manifest {args.manifest}: {exc}") from exc
    judge = manifest.get("judge") if isinstance(manifest.get("judge"), Mapping) else {}
    pins = {
        "judge_model": args.judge_model or judge.get("model"),
        "judge_revision": args.judge_revision or judge.get("revision"),
        "dataset_revision": args.dataset_revision or manifest.get("dataset_revision"),
        "dataset_source": args.dataset_source or manifest.get("dataset"),
        "judge_loader": getattr(args, "judge_loader", None)
        or judge.get("loader")
        or workload.DEFAULT_JUDGE_LOADER,
    }
    if pins["judge_loader"] not in workload.JUDGE_LOADERS:
        raise RejudgeError(
            "judge_loader must be one of " + ", ".join(sorted(workload.JUDGE_LOADERS))
        )
    for key in ("judge_model", "judge_revision", "dataset_revision"):
        if not isinstance(pins[key], str) or not pins[key]:
            raise RejudgeError(f"{key} is required (flag or --manifest)")
    for key in ("judge_revision", "dataset_revision"):
        if pins[key] == "0" * 40:
            raise RejudgeError(f"{key} is the template zero revision")
    return pins


# --------------------------------------------------------------------------- #
# Judge phase and reporting
# --------------------------------------------------------------------------- #


def eval_records(
    transcripts: Mapping[str, Mapping[str, Any]],
    load_dataset: Callable[[str, str | None, str], list[Any]],
    source: str,
    revision: str,
) -> list[dict[str, Any]]:
    """The first ``n`` eval scenarios (``n`` = transcripts per group), as the
    workload selected them; their ids must be exactly the transcripts' ids."""
    count = len(transcripts["baseline"])
    selected = workload.select_examples(
        load_dataset(source, revision, workload.EVAL_SPLIT), count, workload.EVAL_SPLIT
    )
    records = [workload.scenario_record(scenario) for scenario in selected]
    if {r["id"] for r in records} != set(transcripts["baseline"]):
        raise RejudgeError(
            "transcript ids are not the first "
            f"{count} eval scenarios of dataset revision {revision}"
        )
    return records


def group_report(group: Mapping[str, Any]) -> dict[str, Any]:
    """Summary, findings counts and forced-check counts for one judged group."""
    results = list(group["results"])
    forced_false: dict[str, int] = {}
    overrides: dict[str, int] = {}
    for result in results:
        for check, value in (result.get("forced") or {}).items():
            if value is False:
                forced_false[check] = forced_false.get(check, 0) + 1
        for pass_overrides in result.get("overrides") or ():
            for check in pass_overrides:
                overrides[check] = overrides.get(check, 0) + 1
    return {
        "summary": dict(group["summary"]),
        "mean_score": float(group["summary"]["mean_score"]),
        "self_disagreement": float(group["summary"]["self_disagreement"]),
        "findings_summary": {
            kind: sum(1 for r in results if (r.get("findings") or {}).get(kind))
            for kind in _FINDING_KINDS
        },
        "forced_false_by_check": dict(sorted(forced_false.items())),
        "judge_overrides_by_check": dict(sorted(overrides.items())),
    }


def build_summary(
    phase_out: Mapping[str, Any], meta: Mapping[str, Any]
) -> dict[str, Any]:
    groups = {name: group_report(phase_out["groups"][name]) for name in GROUPS}
    worst = max(groups[name]["self_disagreement"] for name in GROUPS)
    return {
        "schema_version": 1,
        "kind": SUMMARY_KIND,
        "generated_at": _utc_now(),
        **dict(meta),
        "rubric_revision": phase_out.get("rubric_revision"),
        "judge_backend": phase_out.get("backend"),
        "peak_vram_mb": phase_out.get("peak_vram_mb"),
        "groups": groups,
        "baseline_score": groups["baseline"]["mean_score"],
        "final_score": groups["final"]["mean_score"],
        "judge_self_disagreement": worst,
        "self_disagreement_gate": SELF_DISAGREEMENT_GATE,
        "self_disagreement_within_gate": worst <= SELF_DISAGREEMENT_GATE,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def run_local(
    args: argparse.Namespace,
    pins: Mapping[str, Any],
    *,
    load_dataset: Callable[[str, str | None, str], list[Any]] | None = None,
    snapshot: Callable[[str, str | None], str] | None = None,
    phase: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Judge the transcripts on this machine and write every output file."""
    from stateset_agents import __version__

    load_dataset = load_dataset or workload._default_dataset_loader
    snapshot = snapshot or workload._default_snapshot
    phase = phase or workload.phase_judge
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    transcripts = load_transcripts(input_dir)
    source = str(pins["dataset_source"] or workload._default_dataset_name())
    records = eval_records(transcripts, load_dataset, source, pins["dataset_revision"])
    judge_path = snapshot(pins["judge_model"], pins["judge_revision"])
    # The same in-process engine settings as the workload's judge phase.
    for key, value in workload.PHASE_ENV.items():
        os.environ.setdefault(key, value)
    started = time.monotonic()
    phase_out = phase(
        {
            "model": pins["judge_model"],
            "revision": pins["judge_revision"],
            "model_path": judge_path,
            "backend": args.backend,
            "loader": pins.get("judge_loader", workload.DEFAULT_JUDGE_LOADER),
            "groups": {
                name: workload.judge_items(records, transcripts[name])
                for name in GROUPS
            },
        }
    )
    for name in GROUPS:
        results = phase_out["groups"][name]["results"]
        if len(results) != len(records):
            raise RejudgeError(f"judge scored the wrong number of {name} items")
        _write_jsonl(output_dir / "judge" / f"{name}.jsonl", results)
    summary = build_summary(
        phase_out,
        {
            "framework_version": str(__version__),
            "judge_model": pins["judge_model"],
            "judge_revision": pins["judge_revision"],
            "judge_loader": pins.get("judge_loader", workload.DEFAULT_JUDGE_LOADER),
            "dataset_source": source,
            "dataset_revision": pins["dataset_revision"],
            "eval_examples": len(records),
            "transcripts_sha256": transcript_digests(input_dir),
            "judge_seconds": round(time.monotonic() - started, 3),
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- #
# RunPod planning (free)
# --------------------------------------------------------------------------- #


def resolve_spend_ceiling(
    flag_value: float | None, environ: Mapping[str, str]
) -> float:
    """The explicit spend confirmation; absent, non-finite or above cap refuses."""
    raw: Any = flag_value
    if raw is None:
        text = environ.get(ENV_CEILING, "").strip()
        if not text:
            raise RejudgeError(
                "refusing to provision without an explicit spend ceiling: pass "
                f"--confirm-max-cost-usd or set {ENV_CEILING}"
            )
        try:
            raw = float(text)
        except ValueError as exc:
            raise RejudgeError(f"{ENV_CEILING} must be a number") from exc
    ceiling = float(raw)
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise RejudgeError("spend ceiling must be a positive finite number")
    if ceiling > MAX_CEILING_USD:
        raise RejudgeError(
            f"spend ceiling ${ceiling:.2f} exceeds the rejudge cap ${MAX_CEILING_USD:.2f}"
        )
    return ceiling


def execution_settings(args: argparse.Namespace) -> dict[str, Any]:
    tier = str(args.provider_tier)
    if tier not in _ALLOWED_TIERS:
        raise RejudgeError("--provider-tier must be SECURE or COMMUNITY")
    lifetime = int(args.max_lifetime_seconds)
    if lifetime < MIN_LIFETIME_SECONDS:
        raise RejudgeError(
            f"--max-lifetime-seconds must be at least {MIN_LIFETIME_SECONDS}"
        )
    return {
        "provider": "runpod",
        "provider_tier": tier,
        "gpu_candidates": [args.gpu] if args.gpu else list(DEFAULT_GPU_CANDIDATES),
        "gpu_count": 1,
        "container_image": args.container_image,
        "container_disk_gb": int(args.container_disk_gb),
        "max_lifetime_seconds": lifetime,
    }


def choose_gpu(
    catalog: Sequence[Mapping[str, Any]], tier: str, candidates: Sequence[str]
) -> tuple[str, float]:
    """Cheapest available candidate in ``tier`` (ties keep candidate order)."""
    availability_field = "secureCloud" if tier == "SECURE" else "communityCloud"
    price_field = "securePrice" if tier == "SECURE" else "communityPrice"
    options: list[tuple[float, int, str]] = []
    for rank, name in enumerate(candidates):
        rows = [row for row in catalog if row.get("id") == name]
        if len(rows) != 1 or rows[0].get(availability_field) is not True:
            continue
        try:
            price = _positive_price(rows[0].get(price_field), price_field)
        except RunPodConformanceError:
            continue
        options.append((price, rank, name))
    if not options:
        raise RejudgeError(
            f"none of {list(candidates)} is available in the {tier} catalog"
        )
    price, _rank, name = min(options)
    return name, price


def build_plan(
    settings: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    ceiling: float | None,
) -> dict[str, Any]:
    """Price the full pod lifetime from the public catalog (no authentication).

    With a ``ceiling`` the worst case must fit it; without one (``--plan``
    only) the worst case is reported and nothing can be provisioned.
    """
    gpu, price = choose_gpu(
        catalog, str(settings["provider_tier"]), settings["gpu_candidates"]
    )
    lifetime = int(settings["max_lifetime_seconds"])
    worst_case = round(price * lifetime / 3600.0, 4)
    if ceiling is not None:
        try:
            check_budget(price, lifetime, ceiling, gpu_count=1)
        except BudgetExceeded as exc:
            raise RejudgeError(str(exc)) from exc
    return {
        "schema_version": 1,
        "kind": "stateset-flagship-rejudge-plan",
        "generated_at": _utc_now(),
        **{key: settings[key] for key in sorted(settings)},
        "gpu_type_id": gpu,
        "catalog_cost_per_hr_usd": price,
        "worst_case_cost_usd": worst_case,
        "max_cost_usd": ceiling,
        "catalog_source": _CATALOG_URL,
        "catalog_quote_is_authoritative": False,
        "provisions_hardware": False,
    }


def harness_is_published(root: Path, revision: str) -> bool:
    """Whether a remote-tracking branch contains ``revision`` (the pod clones
    the public repository, so an unpushed commit would fail after paying)."""
    result = subprocess.run(
        ["git", "branch", "-r", "--contains", revision],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def local_argv(pins: Mapping[str, Any]) -> list[str]:
    """The on-pod ``--local`` command (argv; quoted by :func:`remote_commands`)."""
    argv = [
        "python",
        "-u",
        "benchmarks/flagship_rejudge.py",
        "--local",
        "--input-dir",
        _REMOTE_INPUT,
        "--output-dir",
        _REMOTE_OUTPUT,
        "--manifest",
        "",
        "--judge-model",
        str(pins["judge_model"]),
        "--judge-revision",
        str(pins["judge_revision"]),
        "--dataset-revision",
        str(pins["dataset_revision"]),
        "--backend",
        "vllm",
    ]
    if pins.get("dataset_source"):
        argv += ["--dataset-source", str(pins["dataset_source"])]
    loader = pins.get("judge_loader", workload.DEFAULT_JUDGE_LOADER)
    if loader != workload.DEFAULT_JUDGE_LOADER:
        argv += ["--judge-loader", str(loader)]
    return argv


#: pip extras per judge loader: the Gemma 4 judge needs the flagship-v2
#: dependency path (Transformers >= 5.5, vLLM >= 0.29).
JUDGE_LOADER_EXTRAS: dict[str, str] = {
    "mistral": "training,vllm",
    "gemma4": "training,vllm,flagship-v2",
}


def remote_commands(
    pins: Mapping[str, Any], harness_revision: str, *, with_hf_token: bool
) -> dict[str, str]:
    """Setup and detached-launch shell commands (pure; no secrets inline)."""
    repo = shlex.quote(_REMOTE_REPOSITORY)
    loader = str(pins.get("judge_loader", "mistral"))
    # The Gemma 4 judge needs the v2 stack, CUDA 12.9 builds included.
    profile = RUNTIME_PROFILES["flagship-v2" if loader == "gemma4" else "flagship-v1"]
    setup = " && ".join(
        [
            f"git clone --filter=blob:none {shlex.quote(_REPOSITORY_URL)} {repo}",
            f"git -C {repo} checkout --detach {shlex.quote(harness_revision)}",
            *profile.get("pre_install", []),
            "python -m pip install --quiet -e "
            f"{repo}'[{JUDGE_LOADER_EXTRAS[str(pins.get('judge_loader', 'mistral'))]}]'",
            *profile.get("post_install", []),
            f"mkdir -p {shlex.quote(_REMOTE_INPUT + '/transcripts')} "
            f"{shlex.quote(_REMOTE_OUTPUT)}",
        ]
    )
    command = " ".join(shlex.quote(part) for part in local_argv(pins))
    exports = (
        f'export HF_TOKEN="$(cat {shlex.quote(_REMOTE_HF_TOKEN)})"; '
        if with_hf_token
        else ""
    )
    inner = (
        f"{exports}cd {repo} && {command} > {shlex.quote(_REMOTE_LOG)} 2>&1; "
        f"echo $? > {shlex.quote(_REMOTE_EXIT)}"
    )
    launch = (
        f"rm -f {shlex.quote(_REMOTE_EXIT)} && "
        f"(nohup bash -c {shlex.quote(inner)} > {shlex.quote(_REMOTE_NOHUP_LOG)} "
        "2>&1 < /dev/null &)"
    )
    return {"setup": setup, "launch": launch}


# --------------------------------------------------------------------------- #
# Paid execution
# --------------------------------------------------------------------------- #


def _remaining(lifetime: int, started_at: float, stage: str) -> int:
    remaining = lifetime - max(1, math.ceil(time.time() - started_at))
    if remaining < 1:
        raise RejudgeError(f"pod exhausted its billable lifetime before {stage}")
    return remaining


_TERMINAL_POD_STATES = frozenset({"EXITED", "TERMINATED", "DEAD"})


def _poll_until_exit(
    ssh: Any,
    *,
    deadline_s: int,
    poll_s: float,
    pod_status: Callable[[], str | None] | None = None,
) -> int | None:
    deadline = time.monotonic() + max(1, int(deadline_s))
    while True:
        if pod_status is not None:
            status = pod_status()
            if status in _TERMINAL_POD_STATES:
                raise RejudgeError(
                    f"provider reports the pod {status} before the rejudge finished"
                )
        _, tail = ssh.run(f"tail -n 1 {shlex.quote(_REMOTE_LOG)} 2>/dev/null")
        print(f"{datetime.now(timezone.utc):%H:%M:%SZ} | {tail.strip()[-160:]}")
        code, text = ssh.run(f"cat {shlex.quote(_REMOTE_EXIT)} 2>/dev/null")
        if code == 0 and text.strip():
            try:
                return int(text.strip().splitlines()[-1])
            except ValueError:
                return 1
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.0, min(float(poll_s), deadline - time.monotonic())))


def execute_remote(
    args: argparse.Namespace,
    pins: Mapping[str, Any],
    settings: Mapping[str, Any],
    plan: Mapping[str, Any],
    harness_revision: str,
    *,
    api: Any,
    ssh: Any,
    public_key: str,
    hf_token: str | None,
    lease_dir: Path = DEFAULT_RUNPOD_LEASE_DIR,
    ledger_path: Path | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> dict[str, Any]:
    """Provision one pod, rejudge, download outputs, terminate unconditionally.

    Never raises for rejudge failures: the outcome records status, exit code,
    authoritative price, billed seconds, cost and termination. Raises only
    when termination of a created pod could not be confirmed.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pod_id = ""
    lease: Path | None = None
    started_at = time.time()
    price: float | None = None
    termination_confirmed = False
    termination_error: Exception | None = None
    remote_code: int | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    status = "failed"
    commands = remote_commands(pins, harness_revision, with_hf_token=bool(hf_token))
    try:
        pod = api.create_pod(
            name=f"stateset-rejudge-{uuid.uuid4().hex[:8]}",
            image=str(settings["container_image"]),
            gpu_type_id=str(plan["gpu_type_id"]),
            gpu_count=1,
            ports=["22/tcp"],
            env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
            container_disk_gb=int(settings["container_disk_gb"]),
            cloud_type=str(settings["provider_tier"]),
        )
        pod_id = str(pod.get("id") or "")
        if not pod_id:
            raise RejudgeError("RunPod create response omitted pod id")
        started_at = time.time()
        lease = _write_lease(lease_dir, pod_id, dict(plan), started_at)
        raw_price = pod.get("costPerHr")
        if raw_price is None:
            raw_price = api.get_pod(pod_id).get("costPerHr")
        price = _positive_price(raw_price, "pod costPerHr")
        lifetime = int(settings["max_lifetime_seconds"])
        try:
            check_budget(price, lifetime, float(plan["max_cost_usd"]), gpu_count=1)
        except BudgetExceeded as exc:
            raise RejudgeError(str(exc)) from exc
        host, port, _ = _wait_for_ssh(
            api, pod_id, timeout_s=min(900, _remaining(lifetime, started_at, "SSH"))
        )
        ssh.wait_until_reachable(
            host, port, min(300, _remaining(lifetime, started_at, "SSH setup"))
        )
        _arm_self_destruct(
            ssh, api, pod_id, _remaining(lifetime, started_at, "watchdog arming")
        )
        if hf_token:
            ssh.upload_secret(hf_token, _REMOTE_HF_TOKEN)
        _run_checked(ssh, commands["setup"], "harness checkout and install")
        for group in GROUPS:
            ssh.upload(
                _transcript_path(Path(args.input_dir), group),
                f"{_REMOTE_INPUT}/transcripts/{group}.jsonl",
            )
        _run_checked(ssh, commands["launch"], "rejudge launch")

        def provider_status() -> str | None:
            try:
                return str(api.get_pod(pod_id).get("desiredStatus") or "") or None
            except Exception:  # noqa: BLE001 - an API hiccup is not a pod failure
                return None

        remote_code = _poll_until_exit(
            ssh,
            deadline_s=_remaining(lifetime, started_at, "rejudge"),
            poll_s=poll_seconds,
            pod_status=provider_status,
        )
        try:
            ssh.download_dir(_REMOTE_OUTPUT, output_dir)
        except Exception as exc:  # noqa: BLE001
            raise RejudgeError(f"could not retrieve outputs: {exc}") from exc
        summary_path = output_dir / SUMMARY_NAME
        if summary_path.is_file():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                summary = loaded if isinstance(loaded, dict) else None
            except json.JSONDecodeError:
                summary = None
        if remote_code is None:
            raise RejudgeError("rejudge did not finish within the pod lifetime")
        if remote_code != 0:
            raise RejudgeError(f"remote rejudge exited {remote_code}")
        if summary is None:
            raise RejudgeError("rejudge wrote no summary")
        status = "completed"
    except (RejudgeError, RunPodConformanceError, RemoteExecutionError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - still terminate and account
        error = f"unexpected {type(exc).__name__}: {exc}"
    finally:
        if pod_id:
            try:
                api.terminate_pod(pod_id)
            except Exception as exc:  # noqa: BLE001
                termination_error = exc
            else:
                termination_confirmed = True
                if lease is not None:
                    lease.unlink(missing_ok=True)
        billed_seconds = float(max(1, math.ceil(time.time() - started_at)))
        cost = estimate_cost_usd(price, billed_seconds) if pod_id else None
        if status == "completed" and not termination_confirmed:
            status = "cleanup-pending"
        record_entry(
            CostEntry(
                provider="runpod",
                job_id=pod_id or "creation-failed",
                base_model=str(pins["judge_model"]),
                gpu=str(plan["gpu_type_id"]),
                gpu_count=1,
                cost_per_hr=price,
                duration_s=billed_seconds if pod_id else 0.0,
                cost_usd=cost,
                status=status,
            ),
            path=ledger_path,
        )
    outcome = {
        "schema_version": 1,
        "kind": "stateset-flagship-rejudge-provider-record",
        "status": status,
        "error": error,
        "remote_exit_code": remote_code,
        "pod_id": pod_id,
        "gpu_type_id": plan.get("gpu_type_id"),
        "harness_revision": harness_revision,
        "authoritative_cost_per_hr_usd": price,
        "billed_seconds": billed_seconds if pod_id else 0.0,
        "cost_usd": cost,
        "cost_source": "runpod-api",
        "max_cost_usd": plan.get("max_cost_usd"),
        "termination_confirmed": termination_confirmed,
        "cleanup_lease_retained": bool(lease and lease.exists()),
        "recorded_at": _utc_now(),
    }
    try:
        (output_dir / PROVIDER_RECORD).write_text(
            json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"provider record not written: {exc}", file=sys.stderr)
    if pod_id and not termination_confirmed:
        raise RejudgeError(
            f"termination of pod {pod_id} was not confirmed; cleanup lease "
            f"retained: {termination_error}"
        ) from termination_error
    return {**outcome, "summary": summary}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="seed artifact dir containing transcripts/{baseline,final}.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=lambda value: Path(value) if value else None,
        default=DEFAULT_MANIFEST,
        help="flagship manifest supplying judge/dataset pins ('' for none)",
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-revision")
    parser.add_argument(
        "--judge-loader",
        choices=sorted(workload.JUDGE_LOADERS),
        default=None,
        help="judge engine loader (default: the manifest's judge.loader or mistral)",
    )
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-source")
    parser.add_argument("--local", action="store_true", help="judge on this machine")
    parser.add_argument(
        "--backend", choices=("vllm", "hf", "auto"), default="vllm", help="--local"
    )
    parser.add_argument("--plan", action="store_true", help="price only (free)")
    parser.add_argument("--dry-run", action="store_true", help="offline preview")
    parser.add_argument("--confirm-max-cost-usd", type=float)
    parser.add_argument(
        "--gpu", default=None, help="exact catalog GPU id (default: cheapest 80 GB)"
    )
    parser.add_argument("--provider-tier", default=DEFAULT_TIER)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-disk-gb", type=int, default=DEFAULT_DISK_GB)
    parser.add_argument(
        "--max-lifetime-seconds", type=int, default=DEFAULT_LIFETIME_SECONDS
    )
    parser.add_argument("--harness-revision", default=None)
    parser.add_argument(
        "--allow-unpublished-harness",
        action="store_true",
        help="skip the check that a remote branch contains the harness commit",
    )
    parser.add_argument("--ssh-public-key", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--lease-dir", type=Path, default=DEFAULT_RUNPOD_LEASE_DIR)
    parser.add_argument("--ledger", type=Path, default=None)
    return parser


def _print_summary(summary: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {
                key: summary.get(key)
                for key in (
                    "rubric_revision",
                    "baseline_score",
                    "final_score",
                    "judge_self_disagreement",
                    "self_disagreement_within_gate",
                )
            },
            sort_keys=True,
        )
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    catalog_fetcher: Any = None,
    api_factory: Any = None,
    ssh_factory: Any = None,
    local_runner: Any = None,
    published_check: Any = None,
) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if environ is None else environ
    fetch = catalog_fetcher or fetch_catalog
    make_api = api_factory or RunPodApi
    make_ssh = ssh_factory or SshTransport
    is_published = published_check or harness_is_published
    root = _BENCHMARKS.parent
    try:
        pins = resolve_pins(args)
        load_transcripts(Path(args.input_dir))
        if args.local:
            summary = (local_runner or run_local)(args, pins)
            _print_summary(summary)
            return 0
        settings = execution_settings(args)
        if args.dry_run:
            revision = args.harness_revision or "<HEAD at execution>"
            print(
                json.dumps(
                    {
                        "kind": "stateset-flagship-rejudge-dry-run",
                        "settings": settings,
                        "pins": pins,
                        "transcripts_sha256": transcript_digests(args.input_dir),
                        "commands": remote_commands(
                            pins, revision, with_hf_token=bool(env.get("HF_TOKEN"))
                        ),
                        "provisions_hardware": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.plan:
            ceiling = (
                resolve_spend_ceiling(args.confirm_max_cost_usd, env)
                if args.confirm_max_cost_usd is not None or env.get(ENV_CEILING)
                else None
            )
            print(json.dumps(build_plan(settings, fetch(), ceiling), indent=2))
            return 0
        ceiling = resolve_spend_ceiling(args.confirm_max_cost_usd, env)
        try:
            revision = resolve_harness_revision(root, args.harness_revision)
        except FlagshipDriverError as exc:
            raise RejudgeError(str(exc)) from exc
        if not args.allow_unpublished_harness and not is_published(root, revision):
            raise RejudgeError(
                f"harness commit {revision} is not on any remote branch; push it "
                "first (the pod clones the public repository)"
            )
        plan = build_plan(settings, fetch(), ceiling)
        api_key = str(env.get("RUNPOD_API_KEY", "")).strip()
        if not api_key:
            raise RejudgeError("RUNPOD_API_KEY is not set")
        public_key, private_key = _public_key(args.ssh_public_key)
        outcome = execute_remote(
            args,
            pins,
            settings,
            plan,
            revision,
            api=make_api(api_key),
            ssh=make_ssh(key_path=private_key),
            public_key=public_key,
            hf_token=str(env.get("HF_TOKEN", "")).strip() or None,
            lease_dir=args.lease_dir,
            ledger_path=args.ledger,
            poll_seconds=args.poll_seconds,
        )
        print(
            json.dumps(
                {k: outcome[k] for k in ("status", "error", "cost_usd", "pod_id")},
                sort_keys=True,
            )
        )
        if outcome["status"] != "completed" or outcome["summary"] is None:
            return 1
        _print_summary(outcome["summary"])
        return 0
    except (
        RejudgeError,
        workload.WorkloadError,
        RunPodConformanceError,
        RemoteExecutionError,
        OSError,
    ) as exc:
        print(f"flagship rejudge rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
