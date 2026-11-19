"""Tests for the budget-bounded RunPod shootout launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _name in (
    "backend_conformance",
    "runpod_backend_conformance",
    "framework_comparison",
    "shootout",
):
    if _name not in sys.modules:
        _load(_name, BENCHMARKS / f"{_name}.py")
launcher = _load("runpod_shootout", BENCHMARKS / "runpod_shootout.py")

SHA = "a" * 40


def _shootout_manifest() -> dict[str, Any]:
    example = json.loads((BENCHMARKS / "shootout_manifest.example.json").read_text())
    example["hardware"] = {"gpu": "NVIDIA A40", "gpu_count": 1}
    return example


def _write_inputs(tmp_path: Path, **execution_updates: Any) -> Path:
    (tmp_path / "benchmarks").mkdir(exist_ok=True)
    (tmp_path / "benchmarks" / "shootout.json").write_text(
        json.dumps(_shootout_manifest())
    )
    execution = {
        "provider": "runpod",
        "provider_tier": "SECURE",
        "container_image": "runpod/pytorch:2.4.0",
        "gpu_name": "NVIDIA A40",
        "gpu_count": 1,
        "container_disk_gb": 60,
        "timeout_seconds": 1800,
        "max_lifetime_seconds": 2400,
        "max_cost_usd": 1.0,
    }
    execution.update(execution_updates)
    manifest = {
        "schema_version": 1,
        "harness_revision": SHA,
        "shootout_manifest": "benchmarks/shootout.json",
        "execution": execution,
        "extra_pip": ["trl==1.9.1"],
        "prewarm_command": "python -c 'print(1)'",
    }
    path = tmp_path / "launcher.json"
    path.write_text(json.dumps(manifest))
    return path


def _catalog(**updates: Any) -> list[dict[str, Any]]:
    row = {
        "id": "NVIDIA A40",
        "displayName": "A40",
        "secureCloud": True,
        "communityCloud": True,
        "securePrice": 0.44,
        "communityPrice": 0.39,
    }
    row.update(updates)
    return [row]


class _Api:
    api_key = "test-secret"
    root = "https://rest.runpod.test/v1"

    def __init__(self, price: float = 0.44) -> None:
        self.price = price
        self.terminated: list[str] = []

    def create_pod(self, **kwargs: Any) -> dict[str, Any]:
        self.create_kwargs = kwargs
        return {"id": "pod-123", "costPerHr": self.price}

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return {
            "id": pod_id,
            "costPerHr": self.price,
            "desiredStatus": "RUNNING",
            "publicIp": "127.0.0.1",
            "portMappings": {"22": 22022},
        }

    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)


class _TerminationFailsApi(_Api):
    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)
        raise RuntimeError("provider unavailable")


class _Ssh:
    """Fake transport: the detached shootout "finishes" after ``polls_to_exit``
    exit-marker checks, and each evidence download lands one more file."""

    def __init__(self, exit_code: int = 0, polls_to_exit: int = 2) -> None:
        self.exit_code = exit_code
        self.polls_to_exit = polls_to_exit
        self.commands: list[str] = []
        self.secrets: list[str] = []
        self.downloads = 0
        self.exit_checks = 0

    def wait_until_reachable(self, host: str, port: int, timeout: int) -> None:
        self.endpoint = (host, port, timeout)

    def upload(self, local: Path, remote: str) -> None:
        pass

    def upload_secret(self, secret: str, remote: str) -> None:
        self.secrets.append(remote)

    def run(self, command: str) -> tuple[int, str]:
        self.commands.append(command)
        if "cat " in command and "shootout.exit" in command:
            self.exit_checks += 1
            if self.exit_checks >= self.polls_to_exit:
                return 0, f"{self.exit_code}\n"
            return 1, ""
        if "tail -c" in command:
            return 0, "shootout output tail"
        return 0, "ok"

    def download_dir(self, remote: str, local: Path) -> list[Path]:
        self.downloads += 1
        local.mkdir(parents=True, exist_ok=True)
        path = local / f"stateset-agents-seed{self.downloads}.json"
        path.write_text("{}", encoding="utf-8")
        return [path]


def test_plan_prices_full_lifetime_and_counts_runs(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    assert plan["provisions_hardware"] is False
    assert plan["planned_runs"] == 3 * 2
    assert plan["worst_case_cost_usd"] == pytest.approx(0.44 * 2400 / 3600, abs=1e-3)
    assert plan["frameworks"] == ["stateset-agents", "trl"]


def test_plan_fails_closed_when_lifetime_exceeds_ceiling(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(
        _write_inputs(tmp_path, max_cost_usd=0.01), tmp_path
    )
    with pytest.raises(launcher.RunPodShootoutError):
        launcher.build_plan(manifest, _catalog())


def test_manifest_requires_lifetime_headroom_and_matching_gpu(tmp_path: Path) -> None:
    with pytest.raises(launcher.RunPodShootoutError, match="headroom"):
        launcher.load_launcher_manifest(
            _write_inputs(tmp_path, max_lifetime_seconds=1800), tmp_path
        )
    with pytest.raises(launcher.RunPodShootoutError, match="hardware"):
        launcher.load_launcher_manifest(
            _write_inputs(tmp_path, gpu_name="NVIDIA H100 80GB HBM3"), tmp_path
        )


def test_manifest_rejects_template_harness_revision(tmp_path: Path) -> None:
    path = _write_inputs(tmp_path)
    raw = json.loads(path.read_text())
    raw["harness_revision"] = "0" * 40
    path.write_text(json.dumps(raw))
    with pytest.raises(launcher.RunPodShootoutError, match="non-zero"):
        launcher.load_launcher_manifest(path, tmp_path)


def test_cli_refuses_execution_without_exact_spend_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    path = _write_inputs(tmp_path)
    monkeypatch.setattr(launcher, "fetch_catalog", lambda: _catalog())
    code = launcher.main(
        [
            str(path),
            "--root",
            str(tmp_path),
            "--execute",
            "--confirm-max-cost-usd",
            "0.5",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert "exactly equal" in capsys.readouterr().err
    assert launcher.main([str(path), "--root", str(tmp_path)]) == 0


def test_execute_runs_shootout_downloads_records_and_terminates(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    api, ssh = _Api(), _Ssh()
    out = tmp_path / "evidence"
    result = launcher.execute(
        manifest,
        out,
        plan,
        api=api,
        ssh=ssh,
        public_key="ssh-ed25519 AAAA",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_seconds=0,
    )
    assert result == out and (out / "stateset-agents-seed1.json").exists()
    joined = "\n".join(ssh.commands)
    assert f"checkout --detach {SHA}" in joined
    assert "pip install --quiet -e" in joined and "trl==1.9.1" in joined
    assert "print(1)" in joined  # prewarm ran inside the repo
    run_cmd = [c for c in ssh.commands if "benchmarks/shootout.py" in c][0]
    assert "--required-framework stateset-agents" in run_cmd
    assert "--required-framework trl" in run_cmd
    assert "--timeout-seconds 1800" in run_cmd
    assert "nohup" in run_cmd and "shootout.exit" in run_cmd  # detached + marker
    # evidence was pulled while the remote run was still going, then once more
    assert ssh.downloads >= 3 and ssh.exit_checks == 2
    assert launcher._REMOTE_SHOOTOUT_MANIFEST in ssh.secrets
    assert api.terminated == ["pod-123"]
    record = json.loads((out / "runpod-provider.json").read_text())
    assert record["status"] == "completed" and record["termination_confirmed"]
    assert record["schema_version"] == 2
    assert record["provider"] == "runpod"
    assert record["cost_source"] == ("authoritative-provider-rate-x-observed-lifetime")
    assert record["harness_revision"] == SHA
    ledger = (tmp_path / "ledger.jsonl").read_text()
    assert "pod-123" in ledger and "runpod" in ledger
    assert not (tmp_path / "leases").exists() or not any(
        (tmp_path / "leases").iterdir()
    )


def test_remote_failure_keeps_evidence_terminates_and_raises(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    api = _Api()
    out = tmp_path / "evidence"
    with pytest.raises(launcher.RunPodShootoutError, match="remote shootout failed"):
        launcher.execute(
            manifest,
            out,
            plan,
            api=api,
            ssh=_Ssh(exit_code=3),
            public_key="k",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
            poll_seconds=0,
        )
    assert api.terminated == ["pod-123"]
    record = json.loads((out / "runpod-provider.json").read_text())
    assert record["status"] == "failed" and record["remote_exit_code"] == 3


def test_price_drift_after_allocation_terminates_before_running(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    api, ssh = _Api(price=40.0), _Ssh()
    with pytest.raises(launcher.RunPodShootoutError):
        launcher.execute(
            manifest,
            tmp_path / "evidence",
            plan,
            api=api,
            ssh=ssh,
            public_key="k",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
            poll_seconds=0,
        )
    assert api.terminated == ["pod-123"] and ssh.commands == []


def test_unconfirmed_termination_fails_closed_with_lease(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    with pytest.raises(launcher.RunPodShootoutError, match="not confirmed"):
        launcher.execute(
            manifest,
            tmp_path / "evidence",
            plan,
            api=_TerminationFailsApi(),
            ssh=_Ssh(),
            public_key="k",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
            poll_seconds=0,
        )
    assert any((tmp_path / "leases").iterdir())


def test_lifetime_exhaustion_keeps_partial_evidence_and_terminates(
    tmp_path: Path,
) -> None:
    manifest = launcher.load_launcher_manifest(
        _write_inputs(tmp_path, timeout_seconds=1, max_lifetime_seconds=2), tmp_path
    )
    plan = launcher.build_plan(manifest, _catalog())
    api = _Api()
    out = tmp_path / "evidence"
    with pytest.raises(launcher.RunPodShootoutError, match="did not finish"):
        launcher.execute(
            manifest,
            out,
            plan,
            api=api,
            ssh=_Ssh(polls_to_exit=10_000_000),
            public_key="k",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
            poll_seconds=0.05,
        )
    assert api.terminated == ["pod-123"]
    assert any(out.glob("stateset-agents-seed*.json"))  # partial evidence retained


def test_poll_prints_a_progress_line_per_poll_with_newest_run_output(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    class _StreamingSsh(_Ssh):
        def download_dir(self, remote: str, local: Path) -> list[Path]:
            paths = super().download_dir(remote, local)
            run = local / "runs" / "trl-seed42"
            run.mkdir(parents=True, exist_ok=True)
            (run / "stdout.log").write_text(
                f"step {self.downloads}/200 loss=0.5\n", encoding="utf-8"
            )
            return paths

    ssh = _StreamingSsh(polls_to_exit=3)
    code = launcher._poll_until_exit(ssh, tmp_path, deadline_s=60, poll_s=0.01)
    assert code == 0
    lines = [
        ln for ln in capsys.readouterr().out.splitlines() if " poll elapsed=" in ln
    ]
    assert len(lines) == 3
    assert "evidence=3 runs_started=1 | trl-seed42: step 3/200 loss=0.5" in lines[-1]


def test_remote_shootout_runs_unbuffered_for_live_logs(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    ssh = _Ssh()
    launcher.execute(
        manifest,
        tmp_path / "evidence",
        launcher.build_plan(manifest, _catalog()),
        api=_Api(),
        ssh=ssh,
        public_key="ssh-ed25519 AAAA",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_seconds=0,
    )
    launch = next(c for c in ssh.commands if "shootout.py" in c and "nohup" in c)
    assert "python -u benchmarks/shootout.py" in launch


class _UploadSsh(_Ssh):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.uploads: list[tuple[str, str]] = []

    def upload(self, local: Path, remote: str) -> None:
        self.uploads.append((local.name, remote))


def test_resume_uploads_completed_evidence_and_reuses_the_output_dir(
    tmp_path: Path,
) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    out = tmp_path / "evidence"
    out.mkdir()
    shootout = manifest["_shootout"]
    trl_version = next(
        item["version"] for item in shootout["implementations"] if item["name"] == "trl"
    )
    run_dir = out / "runs" / "trl-seed42"
    artifact = run_dir / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "weights").write_bytes(b"retained")
    (run_dir / "stdout.log").write_text("first pod log")
    comparison = sys.modules["framework_comparison"]
    completed = {
        "schema_version": 2,
        "measured": True,
        "framework": "trl",
        "framework_version": trl_version,
        "seed": 42,
        "harness_commit": SHA,
        "manifest_sha256": launcher.canonical_digest(shootout),
        **{field: shootout[field] for field in sys.modules["shootout"].MANIFEST_FIELDS},
        "timestamp": "2026-09-10T00:00:00+00:00",
        "command": "trl adapter",
        "config": shootout["config"],
        "hardware": {**shootout["hardware"], "cuda": "12.8"},
        "metrics": {
            "samples_per_second": 1.0,
            "wall_clock_seconds": 1.0,
            "peak_vram_mb": 1.0,
            "eval_score_baseline": 0.0,
            "eval_score_final": 0.1,
        },
        "artifact_path": "runs/trl-seed42/artifact",
        "artifact_sha256": comparison.hash_artifact(artifact),
    }
    (out / "trl-seed42.json").write_text(json.dumps(completed))
    (out / "trl-seed1337.json").write_text(
        json.dumps({**completed, "seed": 1337, "harness_commit": "f" * 40})
    )
    (out / "runpod-provider.json").write_text(json.dumps({"kind": "provider"}))
    (out / "_accounting").mkdir()
    (out / "_accounting" / "shootout-summary.json").write_text("{}")
    ssh = _UploadSsh()
    launcher.execute(
        manifest,
        out,
        plan,
        api=_Api(),
        ssh=ssh,
        public_key="ssh-ed25519 AAAA",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_seconds=0,
        resume=True,
    )
    evidence_uploads = [u for u in ssh.uploads if u[0].endswith(".json")]
    assert evidence_uploads == [
        ("trl-seed42.json", f"{launcher._REMOTE_OUTPUT}/trl-seed42.json")
    ]
    assert (
        "weights",
        f"{launcher._REMOTE_OUTPUT}/runs/trl-seed42/artifact/weights",
    ) in ssh.uploads
    assert (
        "stdout.log",
        f"{launcher._REMOTE_OUTPUT}/runs/trl-seed42/stdout.log",
    ) in ssh.uploads
    mkdir_index = next(
        i for i, c in enumerate(ssh.commands) if c.startswith("mkdir -p ")
    )
    launch_index = next(
        i for i, c in enumerate(ssh.commands) if "nohup" in c and "shootout.py" in c
    )
    assert mkdir_index < launch_index
    assert (out / "stateset-agents-seed1.json").exists()  # new evidence landed too
    # the first pod's run logs and summary were moved aside, not clobbered
    assert (out / "runs.before-resume-1" / "trl-seed42" / "stdout.log").read_text() == (
        "first pod log"
    )
    assert (out / "_accounting.before-resume-1" / "shootout-summary.json").exists()
    # the first pod's record is kept; this pod gets its own
    first = json.loads((out / "runpod-provider.json").read_text())
    second = json.loads((out / "runpod-provider-2.json").read_text())
    assert first == {"kind": "provider"}
    assert second["status"] == "completed" and second["pod_id"] == "pod-123"


def test_existing_output_dir_still_refused_without_resume(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    out = tmp_path / "evidence"
    out.mkdir()
    with pytest.raises(launcher.RunPodShootoutError, match="--resume"):
        launcher.execute(
            manifest,
            out,
            launcher.build_plan(manifest, _catalog()),
            api=_Api(),
            ssh=_Ssh(),
            public_key="ssh-ed25519 AAAA",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
        )


def test_cli_accepts_resume_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(launcher, "fetch_catalog", lambda: _catalog())
    monkeypatch.setattr(launcher, "RunPodApi", lambda key: _Api())
    monkeypatch.setattr(launcher, "SshTransport", lambda key_path=None: _Ssh())
    monkeypatch.setattr(
        launcher, "_public_key", lambda path: ("ssh-ed25519 AAAA", None)
    )
    monkeypatch.setattr(
        launcher, "execute", lambda *a, **kw: seen.update(kw) or (tmp_path / "x")
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    code = launcher.main(
        [
            str(_write_inputs(tmp_path)),
            "--root",
            str(tmp_path),
            "--execute",
            "--confirm-max-cost-usd",
            "1.0",
            "--output-dir",
            str(tmp_path / "evidence"),
            "--resume",
        ]
    )
    assert code == 0 and seen["resume"] is True


def _source_bundle(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "source.py").write_text("print('candidate')\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-m",
            "candidate",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    bundle = tmp_path / "candidate.bundle"
    subprocess.run(
        ["git", "-C", str(repo), "bundle", "create", str(bundle), "HEAD"],
        check=True,
        capture_output=True,
    )
    return bundle, revision


@pytest.mark.parametrize("invalid", ["missing", "corrupt", "wrong-commit", "symlink"])
def test_invalid_source_bundle_rejected_before_allocation(
    tmp_path: Path, invalid: str
) -> None:
    bundle, revision = _source_bundle(tmp_path)
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    manifest["harness_revision"] = revision
    if invalid == "missing":
        bundle = tmp_path / "missing.bundle"
    elif invalid == "corrupt":
        bundle.write_bytes(b"not a git bundle")
    elif invalid == "wrong-commit":
        manifest["harness_revision"] = SHA
    else:
        link = tmp_path / "linked.bundle"
        link.symlink_to(bundle)
        bundle = link
    api = _Api()
    with pytest.raises(launcher.RunPodShootoutError, match="source bundle"):
        launcher.execute(
            manifest,
            tmp_path / "out",
            launcher.build_plan(manifest, _catalog()),
            api=api,
            ssh=_Ssh(),
            public_key="ssh-ed25519 AAAA",
            source_bundle=bundle,
        )
    assert not hasattr(api, "create_kwargs")


def test_source_bundle_is_uploaded_verified_and_recorded(tmp_path: Path) -> None:
    bundle, revision = _source_bundle(tmp_path)
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    manifest["harness_revision"] = revision
    ssh, api = _UploadSsh(), _Api()
    out = tmp_path / "out"
    launcher.execute(
        manifest,
        out,
        launcher.build_plan(manifest, _catalog()),
        api=api,
        ssh=ssh,
        public_key="ssh-ed25519 AAAA",
        source_bundle=bundle,
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_seconds=0,
    )
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert (bundle.name, launcher._REMOTE_SOURCE_BUNDLE) in ssh.uploads
    checksum_index = next(i for i, c in enumerate(ssh.commands) if digest in c)
    clone_index = next(i for i, c in enumerate(ssh.commands) if "git clone" in c)
    assert checksum_index < clone_index
    assert launcher._REMOTE_SOURCE_BUNDLE in ssh.commands[clone_index]
    assert launcher._REPOSITORY_URL not in ssh.commands[clone_index]
    record = json.loads((out / "runpod-provider.json").read_text())
    assert record["source_bundle_sha256"] == digest
    assert record["harness_revision"] == revision
    assert api.terminated == ["pod-123"]


def test_prewarm_failure_retains_diagnostics_and_terminates(tmp_path: Path) -> None:
    class PrewarmFails(_Ssh):
        def run(self, command: str) -> tuple[int, str]:
            if "print(1)" in command:
                return 1, "GPU contract failed"
            return super().run(command)

    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    api, ssh = _Api(), PrewarmFails()
    out = tmp_path / "out"
    with pytest.raises(launcher.RunPodConformanceError, match="GPU contract failed"):
        launcher.execute(
            manifest,
            out,
            launcher.build_plan(manifest, _catalog()),
            api=api,
            ssh=ssh,
            public_key="ssh-ed25519 AAAA",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
        )
    assert ssh.downloads == 1
    assert api.terminated == ["pod-123"]
    record = json.loads((out / "runpod-provider.json").read_text())
    assert record["status"] == "failed" and record["termination_confirmed"]
    assert not any("nohup" in c and "shootout.py" in c for c in ssh.commands)
