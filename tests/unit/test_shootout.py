"""Tests for the framework-neutral measured shootout orchestrator."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location("shootout", BENCHMARKS / "shootout.py")
assert SPEC is not None and SPEC.loader is not None
shootout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shootout
SPEC.loader.exec_module(shootout)

ShootoutError = shootout.ShootoutError


def _manifest(**overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "agent-rl-shootout-v1",
        "cache_policy": "prewarmed-v1",
        "algorithm": "grpo",
        "algorithm_revision": "objective-v1",
        "model": "Qwen/Qwen3.5-0.8B",
        "model_revision": "a" * 40,
        "task": "customer-support-v1",
        "dataset_revision": "b" * 40,
        "config": {"steps": 10, "global_batch_size": 8},
        "seeds": [42, 1337, 2026],
        "hardware": {"gpu": "NVIDIA H100", "gpu_count": 1},
        "implementations": [
            {
                "name": "stateset-agents",
                "version": "0.42.3",
                "command": [
                    "one",
                    "{seed}",
                    "{adapter_output}",
                    "{artifact_dir}",
                    "{model}",
                    "{model_revision}",
                    "{dataset_revision}",
                    "{task}",
                    "{config_json}",
                ],
            },
            {
                "name": "trl",
                "version": "1.7.0",
                "command": [
                    "two",
                    "{seed}",
                    "{adapter_output}",
                    "{artifact_dir}",
                    "{model}",
                    "{model_revision}",
                    "{dataset_revision}",
                    "{task}",
                    "{config_json}",
                ],
            },
        ],
    }
    manifest.update(overrides)
    return manifest


def test_manifest_requires_matched_three_seed_matrix(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    loaded = shootout.load_manifest(path)
    assert loaded["seeds"] == [42, 1337, 2026]

    path.write_text(json.dumps(_manifest(seeds=[42])), encoding="utf-8")
    with pytest.raises(ShootoutError, match="at least three unique"):
        shootout.load_manifest(path)


def test_manifest_requires_commands_to_receive_neutral_protocol(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["implementations"][0]["command"].remove("{config_json}")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ShootoutError, match="missing protocol placeholders"):
        shootout.load_manifest(path)


def test_execution_order_rotates_to_reduce_bias() -> None:
    implementations = _manifest()["implementations"]
    assert shootout.execution_order(implementations, 0)[0]["name"] == "stateset-agents"
    assert shootout.execution_order(implementations, 1)[0]["name"] == "trl"


def test_hash_artifact_covers_names_and_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"weights")
    first = shootout.hash_artifact(artifact)
    (artifact / "weights.bin").write_bytes(b"changed")
    second = shootout.hash_artifact(artifact)
    assert len(first) == 64
    assert first != second


def test_adapter_result_requires_measured_matching_hardware(tmp_path: Path) -> None:
    raw = {
        "status": "completed",
        "measured": True,
        "config_sha256": shootout.canonical_digest(_manifest()["config"]),
        "framework_version": "0.42.3",
        "artifact_path": "/tmp/artifact",
        "hardware": {"gpu": "NVIDIA H100", "gpu_count": 1, "cuda": "12.8"},
        "metrics": {
            "samples_processed": 10,
            "peak_vram_mb": 100,
            "eval_score_baseline": -0.2,
            "eval_score_final": 0.1,
        },
    }
    assert shootout.validate_adapter_result(
        raw,
        {"gpu": "NVIDIA H100", "gpu_count": 1},
        shootout.canonical_digest(_manifest()["config"]),
        "0.42.3",
        tmp_path / "result.json",
    )
    raw["measured"] = False
    with pytest.raises(ShootoutError, match="measured completion"):
        shootout.validate_adapter_result(
            raw,
            {"gpu": "NVIDIA H100", "gpu_count": 1},
            shootout.canonical_digest(_manifest()["config"]),
            "0.42.3",
            tmp_path / "result.json",
        )


def test_run_implementation_emits_valid_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "output"
    adapter_code = (
        "import json,pathlib,sys; out=pathlib.Path(sys.argv[1]); "
        "artifact=pathlib.Path(sys.argv[2]); (artifact/'weights').write_bytes(b'x'); "
        "out.write_text(json.dumps({'status':'completed','measured':True,"
        "'artifact_path':str(artifact),'hardware':{'gpu':'NVIDIA H100',"
        "'gpu_count':1,'cuda':'12.8'},'metrics':{'samples_processed':10,"
        "'peak_vram_mb':100,'eval_score_baseline':0.2,'eval_score_final':0.3},"
        "'config_sha256':sys.argv[3],'framework_version':'0.42.3'}))"
    )
    implementation = {
        "name": "stateset-agents",
        "version": "0.42.3",
        "command": [
            sys.executable,
            "-c",
            adapter_code,
            "{adapter_output}",
            "{artifact_dir}",
            shootout.canonical_digest(_manifest()["config"]),
        ],
    }
    manifest = _manifest(
        implementations=[implementation, _manifest()["implementations"][1]]
    )
    monkeypatch.setattr(shootout, "git_commit", lambda _root: "c" * 40)

    evidence_path = shootout.run_implementation(
        manifest, implementation, 42, output, root, timeout_seconds=10
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["measured"] is True
    assert evidence["framework"] == "stateset-agents"
    assert evidence["metrics"]["samples_per_second"] > 0
    assert len(evidence["artifact_sha256"]) == 64


def test_unknown_placeholder_fails_closed() -> None:
    with pytest.raises(ShootoutError, match="unknown command placeholder"):
        shootout._format_command(["{secret}"], {"seed": 42})


def test_required_framework_roster_fails_before_execution() -> None:
    manifest = _manifest()
    shootout.validate_required_frameworks(manifest, ["stateset-agents", "trl"])
    with pytest.raises(ShootoutError, match="nemo-rl, openrlhf, verl"):
        shootout.validate_required_frameworks(
            manifest,
            ["stateset-agents", "trl", "verl", "nemo-rl", "openrlhf"],
        )


def test_main_attempts_full_matrix_and_accounts_for_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    output = tmp_path / "evidence"
    calls: list[tuple[str, int]] = []

    def fake_run(
        manifest: dict[str, Any],
        implementation: dict[str, Any],
        seed: int,
        output_dir: Path,
        root: Path,
        timeout_seconds: int,
    ) -> Path:
        del manifest, root, timeout_seconds
        framework = implementation["name"]
        calls.append((framework, seed))
        if framework == "stateset-agents" and seed == 42:
            raise ShootoutError("deliberate failure")
        return output_dir / f"{framework}-seed{seed}.json"

    monkeypatch.setattr(shootout, "run_implementation", fake_run)
    assert (
        shootout.main(
            [str(manifest_path), "--output-dir", str(output), "--root", str(tmp_path)]
        )
        == 2
    )
    assert len(calls) == 6
    summary = json.loads((output / "_accounting" / "shootout-summary.json").read_text())
    assert summary["attempted"] == 6
    assert summary["completed"] == 5
    assert summary["failed"] == 1
    assert summary["attempts"][0]["error"] == "deliberate failure"


def test_preflight_runs_every_framework_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    output = tmp_path / "preflight"
    calls: list[tuple[str, int]] = []

    def fake_run(
        manifest: dict[str, Any],
        implementation: dict[str, Any],
        seed: int,
        output_dir: Path,
        root: Path,
        timeout_seconds: int,
    ) -> Path:
        del manifest, root, timeout_seconds
        framework = implementation["name"]
        calls.append((framework, seed))
        return output_dir / f"{framework}-seed{seed}.json"

    monkeypatch.setattr(shootout, "run_implementation", fake_run)
    assert (
        shootout.main(
            [
                str(manifest_path),
                "--output-dir",
                str(output),
                "--root",
                str(tmp_path),
                "--preflight",
            ]
        )
        == 0
    )
    assert calls == [("stateset-agents", 42), ("trl", 42)]
    summary = json.loads((output / "_accounting" / "shootout-summary.json").read_text())
    assert summary["mode"] == "preflight"
    assert summary["attempted"] == 2


def test_accounting_is_not_an_evidence_candidate(tmp_path: Path) -> None:
    shootout.write_run_summary(
        tmp_path,
        mode="measured",
        manifest=tmp_path / "manifest.json",
        attempts=[],
    )
    assert list(tmp_path.glob("*.json")) == []
    assert (tmp_path / "_accounting" / "shootout-summary.json").is_file()


def test_run_logs_stream_live_and_timeouts_kill_the_child(tmp_path: Path) -> None:
    """stdout.log is written while the run is in progress (not only at exit),
    and a run that exceeds the timeout is killed and recorded as such."""
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "output"
    marker = tmp_path / "started"
    child = (
        "import pathlib,sys,time; print('step 1', flush=True); "
        f"pathlib.Path({str(marker)!r}).write_text('x'); time.sleep(30)"
    )
    implementation = {
        "name": "stateset-agents",
        "version": "0.42.3",
        "command": [sys.executable, "-c", child],
    }
    started = time.monotonic()
    with pytest.raises(ShootoutError, match="timed out"):
        shootout.run_implementation(
            _manifest(), implementation, 42, output, root, timeout_seconds=2
        )
    assert time.monotonic() - started < 20  # killed, not left to sleep out
    run_dir = output / "runs" / "stateset-agents-seed42"
    assert marker.exists()
    assert (run_dir / "stdout.log").read_text(encoding="utf-8").startswith("step 1")
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["kind"] == "timeout"


def test_main_prints_flushed_progress_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    output = tmp_path / "evidence"

    def fake_run(manifest, implementation, seed, output_dir, root, timeout_seconds):
        if seed == 42 and implementation["name"] == "trl":
            raise ShootoutError("boom")
        return output_dir / f"{implementation['name']}-seed{seed}.json"

    monkeypatch.setattr(shootout, "run_implementation", fake_run)
    shootout.main(
        [str(manifest_path), "--output-dir", str(output), "--root", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert out.count("run start seed=") == 6
    assert out.count("run done seed=") == 5
    assert "run failed seed=42 framework=trl elapsed=0s: boom" in out


def test_main_skips_pairs_whose_evidence_is_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A resumed matrix reruns only the missing seed/framework pairs."""
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "evidence"
    adapter_code = (
        "import json,pathlib,sys; out=pathlib.Path(sys.argv[1]); "
        "artifact=pathlib.Path(sys.argv[2]); (artifact/'weights').write_bytes(b'x'); "
        "out.write_text(json.dumps({'status':'completed','measured':True,"
        "'artifact_path':str(artifact),'hardware':{'gpu':'NVIDIA H100',"
        "'gpu_count':1,'cuda':'12.8'},'metrics':{'samples_processed':10,"
        "'peak_vram_mb':100,'eval_score_baseline':0.2,'eval_score_final':0.3},"
        "'config_sha256':sys.argv[3],'framework_version':'0.42.3'}))"
    )
    implementation = {
        "name": "stateset-agents",
        "version": "0.42.3",
        "command": [
            sys.executable,
            "-c",
            adapter_code,
            "{adapter_output}",
            "{artifact_dir}",
            shootout.canonical_digest(_manifest()["config"]),
            # protocol placeholders the manifest loader insists on
            "{seed}",
            "{model}",
            "{model_revision}",
            "{dataset_revision}",
            "{task}",
            "{config_json}",
        ],
    }
    manifest = _manifest(
        implementations=[implementation, _manifest()["implementations"][1]]
    )
    monkeypatch.setattr(shootout, "git_commit", lambda _root: "c" * 40)
    # a real earlier run left validated evidence for (stateset-agents, 42)
    shootout.run_implementation(
        manifest, implementation, 42, output, root, timeout_seconds=10
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls: list[tuple[str, int]] = []

    def fake_run(manifest, implementation, seed, output_dir, root, timeout_seconds):
        calls.append((implementation["name"], seed))
        return output_dir / f"{implementation['name']}-seed{seed}.json"

    monkeypatch.setattr(shootout, "run_implementation", fake_run)
    assert (
        shootout.main(
            [str(manifest_path), "--output-dir", str(output), "--root", str(root)]
        )
        == 0
    )
    assert ("stateset-agents", 42) not in calls and len(calls) == 5
    out = capsys.readouterr().out
    assert "run skipped seed=42 framework=stateset-agents" in out
    assert "wrote 5 measured evidence documents (1 already present)" in out
    summary = json.loads((output / "_accounting" / "shootout-summary.json").read_text())
    assert summary["attempted"] == 6
    assert summary["completed"] == 5 and summary["skipped"] == 1
    assert summary["failed"] == 0
    assert summary["attempts"][0]["status"] == "skipped"


def test_existing_evidence_fails_closed_on_a_corrupt_file(tmp_path: Path) -> None:
    implementation = _manifest()["implementations"][0]
    path = tmp_path / "stateset-agents-seed42.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ShootoutError, match="unreadable"):
        shootout.existing_evidence(tmp_path, implementation, 42)
    path.write_text(json.dumps({"measured": True}), encoding="utf-8")
    with pytest.raises((ShootoutError, shootout.EvidenceError)):
        shootout.existing_evidence(tmp_path, implementation, 42)
    assert shootout.existing_evidence(tmp_path, implementation, 1337) is None
