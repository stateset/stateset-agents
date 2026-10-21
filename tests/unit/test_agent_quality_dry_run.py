"""Offline end-to-end dry run of the agent-quality gate.

Everything a measured tau3-bench / BFCL V4 run executes is exercised for real
here -- the committed manifest and harness config, the matrix runner, the
paired harness, the official-suite pipeline, the endpoint shim, the cost
meter, the normalizer and the evidence/A+ validators -- with three
substitutions: a stub ``vllm serve`` (an authenticated local HTTP server),
stub ``tau2``/``bfcl`` executables inside throwaway git checkouts, and a fake
user-simulator key. No network beyond localhost, no GPU, no spend.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks import a_plus_gate
from benchmarks.agent_quality_endpoint import (
    ATTESTATION_KIND,
    RECORD_ENV,
    EndpointError,
    ServedEndpoint,
    build_parser,
)
from benchmarks.agent_quality_endpoint import main as endpoint_main
from benchmarks.agent_quality_endpoint import (
    manifest_required_env,
    meter_costs,
    read_api_key,
    resolve_policy,
    run_exec,
    verify_policy,
)
from benchmarks.agent_quality_evidence import (
    AgentQualityEvidenceError,
    hash_artifact,
    load_runs,
    validate_matrix,
)
from benchmarks.agent_quality_upstream import (
    UpstreamError,
    install_checkout,
    load_pins,
    verify_manifest,
)
from benchmarks.run_agent_quality_matrix import (
    AgentQualityRunnerError,
    load_manifest,
)
from benchmarks.run_agent_quality_matrix import main as matrix_main
from benchmarks.run_agent_quality_matrix import (
    validate_budget,
    validate_execution_ready,
)
from stateset_agents import __version__ as PACKAGE_VERSION

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "agent_quality"
MANIFEST = ROOT / "benchmarks" / "agent_quality_manifest.json"
HARNESSES = ROOT / "benchmarks" / "agent_quality_harnesses.json"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TRAINED_MODEL = "stateset/support-adapter-dry-run"
TRAINED_REVISION = "e" * 40
HARNESS_COMMIT = "f" * 40


# The agent-quality harness runs on Linux GPU hosts (vLLM, upstream virtualenv
# binaries). These tests exec POSIX stub executables via their shebang, which
# Windows cannot run, so they are POSIX-only; the pure validation tests in this
# file still run everywhere.
posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="execs POSIX stub executables (Linux-only harness)"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _install_stub(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = source.read_text(encoding="utf-8")
    destination.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    destination.chmod(0o755)
    return destination


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "dry-run",
        "GIT_AUTHOR_EMAIL": "dry-run@example.invalid",
        "GIT_COMMITTER_NAME": "dry-run",
        "GIT_COMMITTER_EMAIL": "dry-run@example.invalid",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fake_upstream(root: Path, name: str, executable: str, stub: str) -> str:
    checkout = root / name
    checkout.mkdir(parents=True)
    _git(checkout, "init", "-q")
    (checkout / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (checkout / "README.md").write_text(f"stub {name}\n", encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-q", "-m", "stub upstream")
    _install_stub(FIXTURES / stub, checkout / ".venv" / "bin" / executable)
    return _git(checkout, "rev-parse", "HEAD")


@pytest.fixture()
def models(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "models" / "base"
    base.mkdir(parents=True)
    (base / "config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")
    (base / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "models" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00stateset-lora\x01")
    return {"base": base, "adapter": adapter}


@pytest.fixture()
def fake_vllm(tmp_path: Path) -> Path:
    return _install_stub(FIXTURES / "fake_vllm.py", tmp_path / "bin" / "vllm")


def _endpoint(
    tmp_path: Path, models: dict[str, Path], fake_vllm: Path, **overrides: Any
) -> ServedEndpoint:
    options: dict[str, Any] = {
        "base_model": str(models["base"]),
        "base_revision": BASE_REVISION,
        "baseline_model": "Qwen/Qwen2.5-7B-Instruct",
        "baseline_revision": BASE_REVISION,
        "workdir": tmp_path / "endpoint",
        "gpu_hourly_usd": 3.6,
        "port": _free_port(),
        "vllm_executable": str(fake_vllm),
        "vllm_args": ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
        "adapter": models["adapter"],
        "trained_model": TRAINED_MODEL,
        "trained_revision": TRAINED_REVISION,
        "ready_timeout_s": 30.0,
        "poll_interval_s": 0.05,
    }
    options.update(overrides)
    return ServedEndpoint(**options)


def _planned_manifest(
    tmp_path: Path, models: dict[str, Path], revisions: dict[str, str]
) -> Path:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["trained_policy"] = {
        "model": TRAINED_MODEL,
        "revision": TRAINED_REVISION,
        "artifact_sha256": hash_artifact(models["adapter"]),
    }
    manifest["budget"]["max_total_cost_usd"] = 500.0
    for suite in manifest["suites"]:
        suite["revision"] = revisions[suite["name"]]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# -- committed contract ---------------------------------------------------------


def test_committed_manifest_matches_pinned_upstreams_and_fails_closed() -> None:
    manifest = load_manifest(MANIFEST)
    pins = load_pins(HARNESSES)
    assert {name: pin["revision"] for name, pin in pins.items()} == {
        "tau3-bench": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
        "bfcl-v4": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
    }
    verify_manifest(MANIFEST, pins)
    assert a_plus_gate.pinned_suite_revisions() == {
        name: pin["revision"] for name, pin in pins.items()
    }
    assert manifest_required_env(manifest) == ["OPENAI_API_KEY"]
    serialized = json.dumps(manifest)
    assert re.search(r"\bsk-[A-Za-z0-9_-]{16,}", serialized) is None
    with pytest.raises(AgentQualityRunnerError) as excinfo:
        validate_execution_ready(manifest, PACKAGE_VERSION)
    message = str(excinfo.value)
    assert "manifest.trained_policy.model" in message
    assert "manifest.budget.max_total_cost_usd" in message


def test_budget_requires_ceiling_covering_planned_worst_case() -> None:
    manifest = load_manifest(MANIFEST)
    manifest["budget"] = {"estimated_cost_usd_per_attempt": 30.0}
    with pytest.raises(AgentQualityRunnerError, match="max_total_cost_usd"):
        validate_budget(manifest)
    manifest["budget"]["max_total_cost_usd"] = 100.0
    with pytest.raises(AgentQualityRunnerError, match="planned worst case"):
        validate_budget(manifest)
    manifest["budget"]["max_total_cost_usd"] = 180.0
    assert validate_budget(manifest) == 180.0


def test_upstream_install_pins_clean_checkout(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q")
    (origin / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    _git(origin, "add", ".")
    _git(origin, "commit", "-q", "-m", "pinned")
    revision = _git(origin, "rev-parse", "HEAD")
    pin = {
        "url": str(origin),
        "revision": revision,
        "checkout_dir": "suite",
        "install": [
            [
                sys.executable,
                "-c",
                "import pathlib; p = pathlib.Path('.venv/bin/tool'); "
                "p.parent.mkdir(parents=True, exist_ok=True); p.write_text('')",
            ]
        ],
        "executables": [".venv/bin/tool"],
    }
    checkout = install_checkout("tau3-bench", pin, tmp_path / "upstream")
    assert _git(checkout, "rev-parse", "HEAD") == revision
    (checkout / "dirty.txt").write_text("x", encoding="utf-8")
    with pytest.raises(UpstreamError, match="local modifications"):
        install_checkout("tau3-bench", pin, tmp_path / "upstream")


# -- endpoint wrapper -----------------------------------------------------------


@posix_only
def test_endpoint_serves_binds_and_tears_down(
    tmp_path: Path, models: dict[str, Path], fake_vllm: Path
) -> None:
    endpoint = _endpoint(tmp_path, models, fake_vllm)
    with endpoint:
        record = json.loads(endpoint.record_path.read_text(encoding="utf-8"))
        assert record["kind"] == "stateset-agent-quality-endpoint"
        assert endpoint.token not in endpoint.record_path.read_text(encoding="utf-8")
        argv = record["servers"][0]["argv"]
        assert "--enable-lora" in argv and "hermes" in argv
        trained = resolve_policy(record, TRAINED_MODEL, TRAINED_REVISION)
        assert trained["artifact_sha256"] == hash_artifact(models["adapter"])
        live = verify_policy(trained, api_key=read_api_key(record))
        assert live["stateset-aq-trained"]["root"] == str(models["adapter"])
        baseline = resolve_policy(record, "Qwen/Qwen2.5-7B-Instruct", BASE_REVISION)
        verify_policy(baseline, api_key=read_api_key(record))
        (models["adapter"] / "adapter_model.safetensors").write_bytes(b"swapped")
        with pytest.raises(EndpointError, match="adapter digest"):
            verify_policy(trained, api_key=read_api_key(record))
        process = endpoint.processes[0]
    assert process.poll() is not None
    with pytest.raises(EndpointError, match="no unique policy"):
        resolve_policy(record, TRAINED_MODEL, "d" * 40)


@posix_only
def test_endpoint_merged_mode_runs_second_server(
    tmp_path: Path,
    models: dict[str, Path],
    fake_vllm: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_merge(self: ServedEndpoint, snapshot: Path, destination: Path) -> Path:
        destination.mkdir(parents=True)
        for item in snapshot.iterdir():
            (destination / item.name).write_bytes(item.read_bytes())
        return destination

    monkeypatch.setattr(ServedEndpoint, "_merge", fake_merge)
    with _endpoint(tmp_path, models, fake_vllm, trained_mode="merged") as endpoint:
        record = json.loads(endpoint.record_path.read_text(encoding="utf-8"))
        assert len(record["servers"]) == 2
        trained = resolve_policy(record, TRAINED_MODEL, TRAINED_REVISION)
        assert trained["base_url"] != record["policies"][0]["base_url"]
        assert "adapter_path" not in trained
        verify_policy(trained, api_key=read_api_key(record))
        processes = list(endpoint.processes)
    assert all(process.poll() is not None for process in processes)


def test_endpoint_rejects_manifest_digest_mismatch(
    tmp_path: Path, models: dict[str, Path], fake_vllm: Path
) -> None:
    endpoint = _endpoint(tmp_path, models, fake_vllm, expected_artifact_sha256="0" * 64)
    with pytest.raises(EndpointError, match="artifact_sha256"):
        with endpoint:
            pass
    assert not endpoint.processes


@posix_only
def test_endpoint_rejects_adapter_without_effect(
    tmp_path: Path,
    models: dict[str, Path],
    fake_vllm: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stateset_agents.remote.serve_session.greedy_completion",
        lambda *_a, **_k: "identical",
    )
    endpoint = _endpoint(tmp_path, models, fake_vllm)
    with pytest.raises(EndpointError, match="NO EFFECT"):
        with endpoint:
            pass
    assert not endpoint.processes


def test_exec_fails_closed_without_user_simulator_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = endpoint_main(
        [
            "exec",
            "--model",
            TRAINED_MODEL,
            "--model-revision",
            TRAINED_REVISION,
            "--upstream-repository",
            str(tmp_path),
            "--endpoint-record",
            str(tmp_path / "missing.json"),
            "--require-env",
            "OPENAI_API_KEY",
            "--",
            "tau2",
            "run",
        ]
    )
    assert code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_exec_requires_endpoint_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(RECORD_ENV, raising=False)
    args = build_parser().parse_args(
        [
            "exec",
            "--model",
            "m",
            "--model-revision",
            "a" * 40,
            "--upstream-repository",
            str(tmp_path),
            "--",
            "true",
        ]
    )
    with pytest.raises(EndpointError, match=RECORD_ENV):
        run_exec(args)


def test_meter_rejects_unpriced_user_simulator(tmp_path: Path) -> None:
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "kind": ATTESTATION_KIND,
                "returncode": 0,
                "wall_seconds": 3600.0,
                "gpu_hourly_usd": 2.0,
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "tau3"
    results.mkdir()
    simulations = [
        {"task_id": "1", "user_cost": 0.5},
        {"task_id": "2", "user_cost": 0.25},
    ]
    (results / "results.json").write_text(
        json.dumps({"simulations": simulations}), encoding="utf-8"
    )
    rows = meter_costs("tau3-bench", results, attestation)
    assert [row["cost_usd"] for row in rows] == [1.5, 1.25]
    simulations[1]["user_cost"] = None
    (results / "results.json").write_text(
        json.dumps({"simulations": simulations}), encoding="utf-8"
    )
    with pytest.raises(EndpointError, match="user_cost"):
        meter_costs("tau3-bench", results, attestation)


@posix_only
def test_serve_cli_runs_child_against_record_then_tears_down(
    tmp_path: Path, models: dict[str, Path], fake_vllm: Path
) -> None:
    probe = tmp_path / "child.json"
    child = (
        "import json, os, pathlib, sys; "
        f"r = json.load(open(os.environ['{RECORD_ENV}'])); "
        f"pathlib.Path({str(probe)!r}).write_text(json.dumps(r['servers']))"
    )
    code = endpoint_main(
        [
            "serve",
            "--base-model",
            str(models["base"]),
            "--base-revision",
            BASE_REVISION,
            "--baseline-model",
            "Qwen/Qwen2.5-7B-Instruct",
            "--workdir",
            str(tmp_path / "endpoint"),
            "--gpu-hourly-usd",
            "1.0",
            "--port",
            str(_free_port()),
            "--vllm-executable",
            str(fake_vllm),
            "--ready-timeout",
            "30",
            "--",
            sys.executable,
            "-c",
            child,
        ]
    )
    assert code == 0
    pid = json.loads(probe.read_text(encoding="utf-8"))[0]["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# -- the full offline matrix -----------------------------------------------------


@posix_only
def test_offline_matrix_produces_bundle_the_gates_validate(
    tmp_path: Path,
    models: dict[str, Path],
    fake_vllm: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = tmp_path / "upstream"
    revisions = {
        "tau3-bench": _fake_upstream(upstream, "tau2-bench", "tau2", "fake_tau2.py"),
        "bfcl-v4": _fake_upstream(upstream, "gorilla", "bfcl", "fake_bfcl.py"),
    }
    manifest_path = _planned_manifest(tmp_path, models, revisions)
    monkeypatch.setenv("STATESET_AQ_UPSTREAM_ROOT", str(upstream))
    monkeypatch.setenv("OPENAI_API_KEY", "dry-run-not-a-real-key")
    monkeypatch.setattr(
        "benchmarks.run_agent_quality_matrix.git_commit", lambda _root: HARNESS_COMMIT
    )
    output = tmp_path / "out"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with _endpoint(
        tmp_path,
        models,
        fake_vllm,
        expected_artifact_sha256=manifest["trained_policy"]["artifact_sha256"],
    ) as endpoint:
        monkeypatch.setenv(RECORD_ENV, str(endpoint.record_path))
        code = matrix_main(
            [str(manifest_path), "--output-dir", str(output), "--root", str(ROOT)]
        )
    accounting = json.loads(
        (output / "_accounting" / "matrix-summary.json").read_text(encoding="utf-8")
    )
    assert code == 0, accounting
    assert accounting["completed"] == 6
    assert accounting["attempts"][-1]["cumulative_cost_usd"] > 0

    runs = load_runs([output / "evidence"])
    validate_matrix(runs, min_seeds=3, expected_suites=["tau3-bench", "bfcl-v4"])
    by_suite = {(run["suite"], run["seed"]): run for run in runs}
    tau = by_suite[("tau3-bench", 1337)]
    assert (tau["baseline_score"], tau["trained_score"]) == (0.4, 0.6)
    bfcl = by_suite[("bfcl-v4", 42)]
    assert (bfcl["baseline_score"], bfcl["trained_score"]) == (0.5, 0.75)
    assert {run["cost_source"] for run in runs} == {"local-meter"}
    assert all(
        run["training_artifact_sha256"] == hash_artifact(models["adapter"])
        for run in runs
    )

    artifact = output / "runs" / "tau3-bench-seed42" / "artifact" / "trained"
    attestation = json.loads(
        (artifact / "official" / "endpoint-attestation.json").read_text("utf-8")
    )
    assert attestation["policy"]["model_revision"] == TRAINED_REVISION
    assert attestation["argv"][attestation["argv"].index("--agent-llm") + 1] == (
        "hosted_vllm/stateset-aq-trained"
    )
    assert endpoint.token not in json.dumps(attestation)
    execution = json.loads((artifact / "execution-manifest.json").read_text("utf-8"))
    assert execution["model_revision_binding"] == "command-argument"
    assert TRAINED_REVISION in execution["commands"][0]

    # The A+ gate's agent-quality branch accepts the bundle's shape and
    # provenance, then rejects it because stub suites are not the pinned
    # official upstream revisions -- a dry run can never pass for real.
    with pytest.raises(a_plus_gate.APlusGateError, match="pinned official upstream"):
        a_plus_gate.validate_agent_quality(
            output / "evidence", HARNESS_COMMIT, PACKAGE_VERSION
        )
    summary = a_plus_gate.validate_agent_quality(
        output / "evidence", HARNESS_COMMIT, PACKAGE_VERSION, pins=revisions
    )
    assert summary["passed"] is True
    assert summary["suites"]["tau3-bench"]["mean_improvement"] == pytest.approx(0.8 / 3)
    with pytest.raises(a_plus_gate.APlusGateError, match="do not match"):
        a_plus_gate.validate_agent_quality(
            output / "evidence", "a" * 40, PACKAGE_VERSION, pins=revisions
        )

    retained = (
        output / "runs" / "bfcl-v4-seed42" / "artifact" / "trained" / "tasks.jsonl"
    )
    retained.write_text(retained.read_text("utf-8").replace("false", "true"), "utf-8")
    with pytest.raises(AgentQualityEvidenceError, match="digest"):
        load_runs([output / "evidence"])


def test_matrix_stops_at_the_approved_spend_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["budget"] = {
        "max_total_cost_usd": 180.0,
        "estimated_cost_usd_per_attempt": 30.0,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    calls: list[str] = []

    def fake_run_suite(_manifest: Any, suite: Any, seed: int, **kwargs: Any) -> Path:
        calls.append(f"{suite['name']}-{seed}")
        evidence = kwargs["output_dir"] / f"{suite['name']}-{seed}.json"
        evidence.write_text(json.dumps({"evaluation_cost_usd": 100.0}), "utf-8")
        return evidence

    module = "benchmarks.run_agent_quality_matrix"
    monkeypatch.setattr(f"{module}.validate_execution_ready", lambda *_: None)
    monkeypatch.setattr(f"{module}.git_commit", lambda _root: HARNESS_COMMIT)
    monkeypatch.setattr(f"{module}.run_suite", fake_run_suite)
    out = tmp_path / "out"
    assert matrix_main([str(path), "--output-dir", str(out)]) == 2
    assert calls == ["tau3-bench-42", "bfcl-v4-42"]
    accounting = json.loads(
        (out / "_accounting" / "matrix-summary.json").read_text("utf-8")
    )
    assert accounting["failed"] == 4
    assert "approved ceiling" in accounting["attempts"][-1]["error"]
