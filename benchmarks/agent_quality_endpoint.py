#!/usr/bin/env python3
"""Serve paired agent-quality policies behind vLLM and bind evaluators to them.

The official τ³-bench and BFCL V4 CLIs talk to an OpenAI-compatible endpoint
by *name*. Nothing in those CLIs proves which weights answered. This module
closes that gap with four subcommands:

``serve``
    Launch vLLM's OpenAI server for the base model at a pinned revision and,
    optionally, the trained StateSet LoRA adapter (served as a LoRA module or
    merged into a full checkpoint via ``serving_artifacts``). It writes an
    endpoint record naming every served policy, its immutable revision and
    adapter digest, runs the same greedy adapter-effect probe as
    ``serve-remote``, and tears every server down on exit. With a trailing
    ``-- command`` it runs that command (normally the agent-quality matrix)
    against the endpoint and then tears down; otherwise it serves until
    SIGINT/SIGTERM.
``exec``
    The binding shim official pipeline stages run through. It receives the
    manifest ``{model}`` and ``{model_revision}``, resolves the matching served
    policy from the endpoint record, verifies the revision marker, the adapter
    digest and the live ``/v1/models`` listing, fails closed when a required
    secret environment variable (for example the τ³ user simulator's API key)
    is absent, then runs the upstream argv without a shell. ``@@SERVED_MODEL@@``
    / ``@@POLICY_DIR@@`` / ``@@API_BASE@@`` tokens in the argv are substituted
    and the endpoint URL and key are exported for litellm (``HOSTED_VLLM_*``)
    and BFCL (``REMOTE_OPENAI_*``). It retains a secret-free attestation.
``meter``
    Write the exact per-task ``task_id``/``cost_usd`` JSONL the normalizer
    requires: measured endpoint GPU time (wall seconds x the declared hourly
    rate, apportioned equally across tasks) plus any per-task provider cost
    the suite itself recorded (the τ³ user simulator).
``check-env``
    List the secret environment variables a manifest's pipelines require and
    fail closed when any is unset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.agent_quality_evidence import (  # noqa: E402
    AgentQualityEvidenceError,
    hash_artifact,
)


class EndpointError(RuntimeError):
    """Raised when a served policy cannot be bound to its immutable identity."""


RECORD_ENV = "STATESET_AQ_ENDPOINT_RECORD"
RECORD_KIND = "stateset-agent-quality-endpoint"
ATTESTATION_KIND = "stateset-agent-quality-endpoint-attestation"
REVISION_MARKER = ".stateset-model-revision"
ADAPTER_MARKER = ".stateset-adapter-sha256"
BASELINE_ALIAS = "stateset-aq-baseline"
TRAINED_ALIAS = "stateset-aq-trained"
#: Files copied from the base snapshot into each policy directory. Upstream
#: evaluators (BFCL's local-model handlers) load the tokenizer and config from
#: a local path and also send that path as the request's model name.
POLICY_VIEW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
)
SUBSTITUTION_TOKENS = ("@@SERVED_MODEL@@", "@@POLICY_DIR@@", "@@API_BASE@@")
IMMUTABLE_HEX = frozenset("0123456789abcdef")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_revision(value: str, label: str) -> str:
    if (
        len(value) != 40
        or any(char not in IMMUTABLE_HEX for char in value)
        or value == "0" * 40
    ):
        raise EndpointError(
            f"{label} must be a non-zero 40-character lowercase hex revision"
        )
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


# -- HTTP (stdlib only, so the exec shim stays dependency-free) ---------------


def _require_http(url: str) -> str:
    if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
        raise EndpointError(f"endpoint URL must be http(s): {url}")
    return url


def http_get_json(url: str, token: str, *, timeout: float = 10.0) -> Any:
    """GET ``url`` with a bearer token and return the decoded JSON body."""
    _require_http(url)
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"}, method="GET"
    )
    with urllib.request.urlopen(
        request, timeout=timeout
    ) as response:  # noqa: S310  # nosec B310 - scheme checked
        return json.loads(response.read().decode("utf-8"))


def http_post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    """POST JSON and return the decoded object (``serve_session`` seam shape)."""
    _require_http(url)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(
        request, timeout=120
    ) as response:  # noqa: S310  # nosec B310 - scheme checked
        return dict(json.loads(response.read().decode("utf-8")))


def live_model_ids(base_url: str, token: str) -> dict[str, dict[str, Any]]:
    """Return ``/v1/models`` entries keyed by served model id."""
    body = http_get_json(f"{base_url}/models", token)
    data = body.get("data") if isinstance(body, Mapping) else None
    if not isinstance(data, list):
        raise EndpointError(f"{base_url}/models returned no model list")
    return {
        str(item["id"]): dict(item)
        for item in data
        if isinstance(item, Mapping) and "id" in item
    }


# -- endpoint record -----------------------------------------------------------


def load_record(path: Path | None) -> tuple[Path, dict[str, Any]]:
    """Load the endpoint record named by ``path`` or ``$STATESET_AQ_ENDPOINT_RECORD``."""
    if path is None:
        raw = os.environ.get(RECORD_ENV)
        if not raw:
            raise EndpointError(
                f"no endpoint record: pass --endpoint-record or set {RECORD_ENV} "
                "(agent_quality_endpoint.py serve sets it for its child command)"
            )
        path = Path(raw)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EndpointError(f"endpoint record is unreadable: {path}") from exc
    if not isinstance(record, Mapping) or record.get("kind") != RECORD_KIND:
        raise EndpointError(f"{path} is not a {RECORD_KIND} record")
    if record.get("schema_version") != 1:
        raise EndpointError(f"{path} must have schema_version=1")
    policies = record.get("policies")
    if not isinstance(policies, list) or not policies:
        raise EndpointError(f"{path} lists no served policies")
    return path, dict(record)


def resolve_policy(
    record: Mapping[str, Any], model: str, model_revision: str
) -> dict[str, Any]:
    """Return the single served policy bound to ``model`` at ``model_revision``."""
    matches = [
        dict(policy)
        for policy in record["policies"]
        if policy.get("model") == model
        and policy.get("model_revision") == model_revision
    ]
    if len(matches) != 1:
        served = [
            f"{policy.get('model')}@{policy.get('model_revision')}"
            for policy in record["policies"]
        ]
        raise EndpointError(
            f"endpoint serves no unique policy for {model}@{model_revision}; "
            f"served={served}"
        )
    return matches[0]


def read_api_key(record: Mapping[str, Any]) -> str:
    key_file = record.get("api_key_file")
    if not isinstance(key_file, str):
        raise EndpointError("endpoint record names no api_key_file")
    try:
        key = Path(key_file).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise EndpointError(f"endpoint API key file is unreadable: {key_file}") from exc
    if not key:
        raise EndpointError("endpoint API key file is empty")
    return key


def verify_policy(
    policy: Mapping[str, Any],
    *,
    api_key: str,
    model_lister: Callable[[str, str], dict[str, dict[str, Any]]] = live_model_ids,
) -> dict[str, Any]:
    """Prove the served policy matches its revision, digest and live listing."""
    policy_dir = Path(str(policy["policy_dir"]))
    marker = policy_dir / REVISION_MARKER
    try:
        marked = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise EndpointError(f"policy directory has no {REVISION_MARKER}") from exc
    if marked != policy["model_revision"]:
        raise EndpointError(
            f"{marker} records {marked!r}, expected {policy['model_revision']}"
        )
    adapter_path = policy.get("adapter_path")
    if adapter_path is not None:
        try:
            digest = hash_artifact(Path(str(adapter_path)))
        except AgentQualityEvidenceError as exc:
            raise EndpointError(str(exc)) from exc
        if digest != policy.get("artifact_sha256"):
            raise EndpointError(
                "served adapter digest does not match the endpoint record "
                f"({digest} != {policy.get('artifact_sha256')})"
            )
    base_url = str(policy["base_url"])
    try:
        models = model_lister(base_url, api_key)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise EndpointError(f"endpoint {base_url} is not answering: {exc}") from exc
    for name in (policy["served_model_name"], str(policy_dir)):
        if name not in models:
            raise EndpointError(
                f"endpoint {base_url} does not serve {name!r}; live={sorted(models)}"
            )
        root = models[name].get("root")
        if adapter_path is not None and root is not None and root != adapter_path:
            raise EndpointError(
                f"served model {name!r} is rooted at {root!r}, not the recorded "
                f"adapter {adapter_path!r}"
            )
    return {
        name: models[name]
        for name in (str(policy["served_model_name"]), str(policy_dir))
    }


# -- exec shim -------------------------------------------------------------------


def _substitute(value: str, tokens: Mapping[str, str]) -> str:
    for token, replacement in tokens.items():
        value = value.replace(token, replacement)
    return value


def _resolve_executable(
    argv0: str, upstream: Path, venv: str | None
) -> tuple[str, dict[str, str]]:
    if venv is None or "/" in argv0:
        return argv0, {}
    venv_dir = (upstream / venv).resolve()
    candidate = venv_dir / "bin" / argv0
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise EndpointError(
            f"{argv0} is not installed in the pinned upstream environment "
            f"{venv_dir}; run benchmarks/agent_quality_upstream.py install"
        )
    path = f"{venv_dir / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    return str(candidate), {"VIRTUAL_ENV": str(venv_dir), "PATH": path}


def require_env(names: Sequence[str]) -> None:
    """Fail closed when a secret the upstream suite needs is not exported."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise EndpointError(
            "required environment variable(s) not set: "
            + ", ".join(missing)
            + ". Manifests reference secrets (for example the tau3-bench user "
            "simulator's provider API key) by name only; export them before "
            "running and never write them into a manifest."
        )


def run_exec(args: argparse.Namespace) -> int:
    """Bind the policy, run the upstream argv, and retain an attestation."""
    require_env(args.require_env)
    record_path, record = load_record(args.endpoint_record)
    _require_revision(args.model_revision, "model_revision")
    policy = resolve_policy(record, args.model, args.model_revision)
    api_key = read_api_key(record)
    live = verify_policy(policy, api_key=api_key)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise EndpointError("exec requires an upstream command after --")
    upstream = Path(args.upstream_repository).resolve()
    if not upstream.is_dir():
        raise EndpointError(f"upstream repository does not exist: {upstream}")
    base_url = str(policy["base_url"])
    tokens = {
        "@@SERVED_MODEL@@": str(policy["served_model_name"]),
        "@@POLICY_DIR@@": str(policy["policy_dir"]),
        "@@API_BASE@@": base_url,
    }
    argv = [_substitute(part, tokens) for part in command]
    argv[0], venv_env = _resolve_executable(argv[0], upstream, args.upstream_venv)
    env = os.environ.copy()
    env.update(venv_env)
    env.update(
        {
            # litellm's hosted_vllm/ provider (tau3-bench agent LLM).
            "HOSTED_VLLM_API_BASE": base_url,
            "HOSTED_VLLM_API_KEY": api_key,
            # BFCL's local-model handlers with --skip-server-setup.
            "REMOTE_OPENAI_BASE_URL": base_url,
            "REMOTE_OPENAI_API_KEY": api_key,
        }
    )
    for name, value in args.env:
        env[name] = _substitute(value, tokens)
    started_at = _utc_now()
    started = time.monotonic()
    completed = subprocess.run(argv, cwd=upstream, env=env, check=False)
    wall_seconds = max(time.monotonic() - started, 1e-9)
    if args.attestation_output is not None:
        public_record = {
            key: value for key, value in record.items() if key != "api_key_file"
        }
        attestation = {
            "schema_version": 1,
            "kind": ATTESTATION_KIND,
            "model": args.model,
            "model_revision": args.model_revision,
            "policy": policy,
            "endpoint_record_sha256": _sha256_json(public_record),
            "endpoint_record": public_record,
            "live_models": live,
            "argv": argv,
            "env_overrides": sorted(name for name, _ in args.env),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "wall_seconds": wall_seconds,
            "returncode": completed.returncode,
            "gpu_hourly_usd": record.get("gpu_hourly_usd"),
            "record_path": str(record_path),
        }
        output = Path(args.attestation_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(attestation, indent=2) + "\n", encoding="utf-8")
    return completed.returncode


# -- cost meter ------------------------------------------------------------------


def _finite_cost(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EndpointError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise EndpointError(f"{label} must be finite and non-negative")
    return result


def _tau3_provider_costs(results: Path) -> dict[str, float]:
    source = results / "results.json" if results.is_dir() else results
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EndpointError(f"tau3 results are unreadable: {source}") from exc
    simulations = raw.get("simulations") if isinstance(raw, Mapping) else None
    if not isinstance(simulations, list) or not simulations:
        raise EndpointError(f"tau3 results contain no simulations: {source}")
    costs: dict[str, float] = {}
    for simulation in simulations:
        if not isinstance(simulation, Mapping):
            raise EndpointError(f"invalid tau3 simulation in {source}")
        task_id = str(simulation.get("task_id", "")).strip()
        if not task_id or task_id in costs:
            raise EndpointError(f"missing or duplicate tau3 task_id in {source}")
        # The agent runs on the metered GPU endpoint; the user simulator is a
        # provider API priced by tau3's own litellm accounting. Unpriced user
        # turns fail closed rather than silently counting as free.
        costs[task_id] = _finite_cost(
            simulation.get("user_cost"), f"user_cost for tau3 task {task_id}"
        )
    return costs


def _bfcl_task_ids(results: Path) -> dict[str, float]:
    from benchmarks.adapters.official_result_normalizer import (
        OfficialResultError,
        _json_files,
        _load_json_records,
    )

    ids: dict[str, float] = {}
    try:
        for path in _json_files(results):
            for record in _load_json_records(path):
                task_id = str(record.get("id", "")).strip()
                if not task_id or task_id in ids:
                    raise EndpointError(f"missing or duplicate BFCL id in {path}")
                ids[task_id] = 0.0
    except OfficialResultError as exc:
        raise EndpointError(str(exc)) from exc
    return ids


def meter_costs(
    suite: str, results: Path, attestation_path: Path
) -> list[dict[str, Any]]:
    """Return one measured cost row per task, in upstream result order."""
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EndpointError(f"attestation is unreadable: {attestation_path}") from exc
    if attestation.get("kind") != ATTESTATION_KIND:
        raise EndpointError(f"{attestation_path} is not an endpoint attestation")
    if attestation.get("returncode") != 0:
        raise EndpointError("the metered upstream stage did not complete")
    wall = _finite_cost(attestation.get("wall_seconds"), "wall_seconds")
    rate = _finite_cost(attestation.get("gpu_hourly_usd"), "gpu_hourly_usd")
    if suite == "tau3-bench":
        provider = _tau3_provider_costs(results)
    elif suite == "bfcl-v4":
        provider = _bfcl_task_ids(results)
    else:
        raise EndpointError(f"no meter for suite {suite}")
    gpu_share = wall / 3600.0 * rate / len(provider)
    return [
        {
            "task_id": task_id,
            "cost_usd": gpu_share + provider_cost,
            "gpu_cost_usd": gpu_share,
            "provider_cost_usd": provider_cost,
            "method": "endpoint-wall-seconds-x-hourly-rate-equal-split",
        }
        for task_id, provider_cost in provider.items()
    ]


def run_meter(args: argparse.Namespace) -> int:
    rows = meter_costs(args.suite, Path(args.results), Path(args.attestation))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return 0


# -- secret requirements -----------------------------------------------------------


def manifest_required_env(manifest: Mapping[str, Any]) -> list[str]:
    """Collect every ``--require-env NAME`` declared by the manifest pipelines."""
    pipelines = manifest.get("evaluation_config", {}).get(
        "official_suite_pipelines", {}
    )
    names: list[str] = []
    for pipeline in pipelines.values():
        for command in pipeline.get("commands", []):
            for flag, value in zip(command, command[1:], strict=False):
                if flag == "--require-env" and value not in names:
                    names.append(value)
    return names


def run_check_env(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    names = manifest_required_env(manifest)
    require_env(names)
    print("required environment present: " + (", ".join(names) or "(none)"))
    return 0


# -- serve -------------------------------------------------------------------------


def _base_snapshot(base_model: str, revision: str, *, full: bool) -> Path:
    """Return a local directory holding the base model at ``revision``."""
    local = Path(base_model)
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download

    patterns = None if full else ["*.json", "*.txt", "*.model", "*.jinja"]
    return Path(
        snapshot_download(
            repo_id=base_model, revision=revision, allow_patterns=patterns
        )
    )


def build_policy_dir(
    destination: Path,
    source: Path,
    *,
    revision: str,
    artifact_sha256: str | None = None,
) -> Path:
    """Create the tokenizer/config view an evaluator loads for one policy."""
    destination.mkdir(parents=True, exist_ok=False)
    copied = 0
    for name in POLICY_VIEW_PATTERNS:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)
            copied += 1
    if (
        not (destination / "config.json").is_file()
        or not (destination / "tokenizer_config.json").is_file()
    ):
        raise EndpointError(
            f"{source} lacks config.json/tokenizer_config.json for a policy view"
        )
    (destination / REVISION_MARKER).write_text(revision + "\n", encoding="utf-8")
    if artifact_sha256 is not None:
        (destination / ADAPTER_MARKER).write_text(
            artifact_sha256 + "\n", encoding="utf-8"
        )
    return destination


class ServedEndpoint:
    """Own one or two vLLM OpenAI servers and the record describing them."""

    def __init__(
        self,
        *,
        base_model: str,
        base_revision: str,
        baseline_model: str,
        baseline_revision: str,
        workdir: Path,
        gpu_hourly_usd: float,
        host: str = "127.0.0.1",
        port: int = 8000,
        vllm_executable: str = "vllm",
        vllm_args: Sequence[str] = (),
        adapter: Path | None = None,
        trained_model: str | None = None,
        trained_revision: str | None = None,
        trained_mode: str = "lora",
        expected_artifact_sha256: str | None = None,
        ready_timeout_s: float = 1800.0,
        poll_interval_s: float = 2.0,
        strict_effect: bool = True,
    ) -> None:
        self.base_model = base_model
        self.base_revision = _require_revision(base_revision, "base revision")
        self.baseline_model = baseline_model
        self.baseline_revision = _require_revision(
            baseline_revision, "baseline revision"
        )
        if not math.isfinite(gpu_hourly_usd) or gpu_hourly_usd < 0:
            raise EndpointError("gpu hourly rate must be finite and non-negative")
        self.workdir = workdir.resolve()
        self.gpu_hourly_usd = gpu_hourly_usd
        self.host = host
        self.port = port
        self.vllm_executable = vllm_executable
        self.vllm_args = list(vllm_args)
        self.adapter = adapter.resolve() if adapter is not None else None
        self.trained_model = trained_model
        self.trained_revision = trained_revision
        if trained_mode not in {"lora", "merged"}:
            raise EndpointError("trained mode must be 'lora' or 'merged'")
        self.trained_mode = trained_mode
        self.expected_artifact_sha256 = expected_artifact_sha256
        self.ready_timeout_s = ready_timeout_s
        self.poll_interval_s = poll_interval_s
        self.strict_effect = strict_effect
        self.token = secrets.token_urlsafe(24)
        self.processes: list[subprocess.Popen[bytes]] = []
        self.record_path = self.workdir / "endpoint.json"
        self.servers: list[dict[str, Any]] = []
        self.policies: list[dict[str, Any]] = []
        if self.adapter is not None:
            if not trained_model or trained_revision is None:
                raise EndpointError("an adapter requires trained model and revision")
            _require_revision(trained_revision, "trained revision")

    # -- lifecycle --------------------------------------------------------------

    def _launch(self, argv: list[str], log_name: str) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
        env["VLLM_API_KEY"] = self.token  # never on the command line
        log = (self.workdir / log_name).open("wb")
        process = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        log.close()
        self.processes.append(process)
        return process

    def _await_ready(
        self, process: subprocess.Popen[bytes], base_url: str, names: Sequence[str]
    ) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        last = "no response"
        while True:
            if process.poll() is not None:
                raise EndpointError(
                    f"vLLM exited {process.returncode} before serving {base_url}; "
                    f"see {self.workdir}"
                )
            try:
                models = live_model_ids(base_url, self.token)
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    raise EndpointError(
                        "vLLM rejected the generated API token; refusing to "
                        "serve unauthenticated"
                    ) from exc
                last = f"HTTP {exc.code}"
            except (OSError, ValueError, EndpointError) as exc:
                last = str(exc)
            else:
                missing = [name for name in names if name not in models]
                if not missing:
                    return
                last = f"missing served models {missing}"
            if time.monotonic() >= deadline:
                raise EndpointError(
                    f"vLLM at {base_url} not ready within {self.ready_timeout_s}s "
                    f"({last})"
                )
            time.sleep(self.poll_interval_s)

    def start(self) -> Path:
        """Launch servers, verify them, and write the endpoint record."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        key_file = self.workdir / "api_key"
        key_file.write_text(self.token + "\n", encoding="utf-8")
        key_file.chmod(0o600)
        merged = self.adapter is not None and self.trained_mode == "merged"
        snapshot = _base_snapshot(self.base_model, self.base_revision, full=merged)
        policy_root = self.workdir / "policies"
        baseline_dir = build_policy_dir(
            policy_root / "baseline", snapshot, revision=self.baseline_revision
        )
        base_url = f"http://{self.host}:{self.port}/v1"
        base_source = (
            str(snapshot) if Path(self.base_model).is_dir() else self.base_model
        )
        argv = [
            self.vllm_executable,
            "serve",
            base_source,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--served-model-name",
            BASELINE_ALIAS,
            str(baseline_dir),
        ]
        if not Path(self.base_model).is_dir():
            argv += ["--revision", self.base_revision]
        adapter_digest: str | None = None
        trained_dir: Path | None = None
        trained_url = base_url
        trained_argv: list[str] | None = None
        if self.adapter is not None:
            try:
                adapter_digest = hash_artifact(self.adapter)
            except AgentQualityEvidenceError as exc:
                raise EndpointError(str(exc)) from exc
            if (
                self.expected_artifact_sha256 is not None
                and adapter_digest != self.expected_artifact_sha256
            ):
                raise EndpointError(
                    "adapter digest does not match the manifest "
                    f"trained_policy.artifact_sha256 ({adapter_digest})"
                )
            assert self.trained_revision is not None
            if merged:
                trained_dir = self._merge(snapshot, policy_root / "trained")
                (trained_dir / REVISION_MARKER).write_text(
                    self.trained_revision + "\n", encoding="utf-8"
                )
                (trained_dir / ADAPTER_MARKER).write_text(
                    adapter_digest + "\n", encoding="utf-8"
                )
                trained_url = f"http://{self.host}:{self.port + 1}/v1"
                trained_argv = [
                    self.vllm_executable,
                    "serve",
                    str(trained_dir),
                    "--host",
                    self.host,
                    "--port",
                    str(self.port + 1),
                    "--served-model-name",
                    TRAINED_ALIAS,
                    str(trained_dir),
                    *self.vllm_args,
                ]
            else:
                trained_dir = build_policy_dir(
                    policy_root / "trained",
                    snapshot,
                    revision=self.trained_revision,
                    artifact_sha256=adapter_digest,
                )
                argv += [
                    "--enable-lora",
                    "--lora-modules",
                    f"{TRAINED_ALIAS}={self.adapter}",
                    f"{trained_dir}={self.adapter}",
                ]
        argv += self.vllm_args
        process = self._launch(argv, "vllm-baseline.log")
        self.servers.append({"base_url": base_url, "argv": argv, "pid": process.pid})
        names = [BASELINE_ALIAS, str(baseline_dir)]
        if trained_dir is not None and not merged:
            names += [TRAINED_ALIAS, str(trained_dir)]
        self._await_ready(process, base_url, names)
        if trained_argv is not None:
            trained_process = self._launch(trained_argv, "vllm-trained.log")
            self.servers.append(
                {
                    "base_url": trained_url,
                    "argv": trained_argv,
                    "pid": trained_process.pid,
                }
            )
            self._await_ready(
                trained_process, trained_url, [TRAINED_ALIAS, str(trained_dir)]
            )
        self.policies = [
            {
                "policy": "baseline",
                "model": self.baseline_model,
                "model_revision": self.baseline_revision,
                "served_model_name": BASELINE_ALIAS,
                "policy_dir": str(baseline_dir),
                "base_url": base_url,
                "mode": "base",
            }
        ]
        if trained_dir is not None:
            policy: dict[str, Any] = {
                "policy": "trained",
                "model": self.trained_model,
                "model_revision": self.trained_revision,
                "served_model_name": TRAINED_ALIAS,
                "policy_dir": str(trained_dir),
                "base_url": trained_url,
                "mode": self.trained_mode,
                "artifact_sha256": adapter_digest,
            }
            if not merged:
                policy["adapter_path"] = str(self.adapter)
            self.policies.append(policy)
            self._probe_effect(base_url, trained_url)
        record = {
            "schema_version": 1,
            "kind": RECORD_KIND,
            "created_at": _utc_now(),
            "backend": "local-vllm",
            "vllm_executable": self.vllm_executable,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "gpu_hourly_usd": self.gpu_hourly_usd,
            "api_key_file": str(key_file),
            "servers": self.servers,
            "policies": self.policies,
        }
        self.record_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        return self.record_path

    def _merge(self, snapshot: Path, destination: Path) -> Path:
        from stateset_agents.training.serving_artifacts import (
            export_merged_model_for_serving,
        )

        assert self.adapter is not None
        merged = export_merged_model_for_serving(
            base_model_name=str(snapshot),
            adapter_dir=str(self.adapter),
            output_dir=str(destination),
        )
        if not merged:
            raise EndpointError("merging the adapter into the base model failed")
        return Path(merged).resolve()

    def _probe_effect(self, base_url: str, trained_url: str) -> None:
        from stateset_agents.remote.serve_session import (
            adapter_no_effect_message,
            greedy_completion,
        )

        root = base_url.removesuffix("/v1")
        trained_root = trained_url.removesuffix("/v1")
        base = greedy_completion(
            root, self.token, BASELINE_ALIAS, http_post_json=http_post_json
        )
        trained = greedy_completion(
            trained_root, self.token, TRAINED_ALIAS, http_post_json=http_post_json
        )
        if base == trained:
            message = adapter_no_effect_message(TRAINED_ALIAS)
            if self.strict_effect:
                raise EndpointError(message)
            print(f"WARNING: {message}", file=sys.stderr)

    def stop(self, grace_s: float = 30.0) -> None:
        """Terminate every server process group, escalating to SIGKILL."""
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace_s
        for process in self.processes:
            remaining = max(deadline - time.monotonic(), 0.1)
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        self.processes.clear()

    def __enter__(self) -> ServedEndpoint:
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _manifest_identities(args: argparse.Namespace) -> None:
    """Fill policy identities from the manifest and enforce its secrets."""
    if args.manifest is None:
        return
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    require_env(manifest_required_env(manifest))
    baseline = manifest["baseline_policy"]
    trained = manifest["trained_policy"]
    if args.base_revision != baseline["revision"]:
        raise EndpointError(
            "--base-revision must equal manifest baseline_policy.revision"
        )
    args.baseline_model = args.baseline_model or baseline["model"]
    args.baseline_revision = args.baseline_revision or baseline["revision"]
    if args.adapter is not None:
        args.trained_model = args.trained_model or trained["model"]
        args.trained_revision = args.trained_revision or trained["revision"]
        args.expected_artifact_sha256 = trained["artifact_sha256"]


def run_serve(args: argparse.Namespace) -> int:
    args.expected_artifact_sha256 = None
    _manifest_identities(args)
    endpoint = ServedEndpoint(
        base_model=args.base_model,
        base_revision=args.base_revision,
        baseline_model=args.baseline_model or args.base_model,
        baseline_revision=args.baseline_revision or args.base_revision,
        workdir=Path(args.workdir),
        gpu_hourly_usd=args.gpu_hourly_usd,
        host=args.host,
        port=args.port,
        vllm_executable=args.vllm_executable,
        vllm_args=args.vllm_arg,
        adapter=Path(args.adapter) if args.adapter else None,
        trained_model=args.trained_model,
        trained_revision=args.trained_revision,
        trained_mode=args.trained_mode,
        expected_artifact_sha256=args.expected_artifact_sha256,
        ready_timeout_s=args.ready_timeout,
        strict_effect=not args.allow_no_effect,
    )
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    stop_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        raise KeyboardInterrupt

    previous = {
        sig: signal.signal(sig, _request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with endpoint:
            print(f"endpoint record: {endpoint.record_path}", flush=True)
            if command:
                env = os.environ.copy()
                env[RECORD_ENV] = str(endpoint.record_path)
                return subprocess.run(command, env=env, check=False).returncode
            print("serving; send SIGINT/SIGTERM to tear down", flush=True)
            while not stop_requested:
                time.sleep(1.0)
    except KeyboardInterrupt:
        return 130 if not command else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


# -- CLI -------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)

    serve = sub.add_parser("serve", help="serve base (+ trained) policies via vLLM")
    serve.add_argument("--base-model", required=True)
    serve.add_argument("--base-revision", required=True)
    serve.add_argument("--manifest", type=Path)
    serve.add_argument("--baseline-model")
    serve.add_argument("--baseline-revision")
    serve.add_argument("--adapter")
    serve.add_argument("--trained-model")
    serve.add_argument("--trained-revision")
    serve.add_argument("--trained-mode", choices=("lora", "merged"), default="lora")
    serve.add_argument("--workdir", required=True)
    serve.add_argument("--gpu-hourly-usd", type=float, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--vllm-executable", default="vllm")
    serve.add_argument("--vllm-arg", action="append", default=[])
    serve.add_argument("--ready-timeout", type=float, default=1800.0)
    serve.add_argument("--allow-no-effect", action="store_true")
    serve.add_argument("command", nargs=argparse.REMAINDER)

    execute = sub.add_parser("exec", help="bind a served policy and run argv")
    execute.add_argument("--model", required=True)
    execute.add_argument("--model-revision", required=True)
    execute.add_argument("--upstream-repository", required=True)
    execute.add_argument("--upstream-venv")
    execute.add_argument("--endpoint-record", type=Path)
    execute.add_argument("--require-env", action="append", default=[])
    execute.add_argument(
        "--env", nargs=2, action="append", default=[], metavar=("NAME", "VALUE")
    )
    execute.add_argument("--attestation-output")
    execute.add_argument("command", nargs=argparse.REMAINDER)

    meter = sub.add_parser("meter", help="write per-task measured cost JSONL")
    meter.add_argument("--suite", required=True)
    meter.add_argument("--results", required=True)
    meter.add_argument("--attestation", required=True)
    meter.add_argument("--output", required=True)

    check = sub.add_parser("check-env", help="verify a manifest's required secrets")
    check.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "serve": run_serve,
        "exec": run_exec,
        "meter": run_meter,
        "check-env": run_check_env,
    }
    try:
        return handlers[args.action](args)
    except EndpointError as exc:
        print(f"agent-quality endpoint rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
