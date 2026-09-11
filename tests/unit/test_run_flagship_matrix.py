"""Contract tests for the flagship benchmark collector."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from stateset_agents import __version__ as PACKAGE_VERSION

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "run_flagship_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_flagship_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "stateset-flagship-manifest",
        "protocol": "stateset-flagship-multiturn-v1",
        "framework_version": PACKAGE_VERSION,
        "provider": "runpod",
        "cost_source": "runpod-graphql",
        "model": "org/model-8b",
        "model_revision": "a" * 40,
        "model_parameter_count": 8_000_000_000,
        "dataset": "stateset/customer-support-multiturn-v1",
        "dataset_revision": "b" * 40,
        "trainer": "gspo",
        "task": "customer_support",
        "seeds": [42, 1337, 2026],
        "config": {
            "num_train_examples": 500,
            "num_eval_examples": 200,
            "max_wall_clock_seconds": 14400,
            "max_cost_usd_per_seed": 25.0,
        },
        "judge": {
            "model": "other-org/judge",
            "revision": "c" * 40,
            "rubric_revision": "d" * 40,
        },
        "hardware": {"gpu": "NVIDIA H100 80GB HBM3", "gpu_count": 1},
        "command": [
            "python",
            "driver.py",
            "{seed}",
            "{mode}",
            "{framework_version}",
            "{model}",
            "{model_revision}",
            "{dataset_revision}",
            "{config_json}",
            "{config_sha256}",
            "{judge_json}",
            "{judge_sha256}",
            "{adapter_output}",
            "{artifact_dir}",
        ],
    }


def write_manifest(tmp_path: Path, value: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value or manifest()), encoding="utf-8")
    return path


def run(seed: int, improvement: float = 0.05) -> dict[str, Any]:
    return {
        "seed": seed,
        "preflight": False,
        "metrics": {
            "baseline_score": 0.50,
            "final_score": 0.50 + improvement,
            "judge_self_disagreement": 0.02,
            "cost_usd": 4.0,
        },
        "wall_clock_seconds": 100.0,
    }


def retained_run(root: Path, value: dict[str, Any], seed: int) -> dict[str, Any]:
    artifact = root / "artifacts" / f"seed-{seed}"
    artifact.mkdir(parents=True)
    (artifact / "weights.bin").write_bytes(f"weights-{seed}".encode())
    return {
        "schema_version": 2,
        "kind": "stateset-flagship-evidence",
        "measured": True,
        "preflight": False,
        "seed": seed,
        "harness_commit": "e" * 40,
        "manifest_sha256": runner.digest_json(value),
        "framework_version": value["framework_version"],
        "provider": value["provider"],
        "cost_source": value["cost_source"],
        "model": value["model"],
        "model_revision": value["model_revision"],
        "model_parameter_count": value["model_parameter_count"],
        "dataset": value["dataset"],
        "dataset_revision": value["dataset_revision"],
        "trainer": value["trainer"],
        "task": value["task"],
        "config_sha256": runner.digest_json(value["config"]),
        "judge": value["judge"],
        "judge_sha256": runner.digest_json(value["judge"]),
        "hardware": {**value["hardware"], "cuda": "13.0", "driver": "580.1"},
        "metrics": {
            "baseline_score": 0.50,
            "final_score": 0.55,
            "judge_self_disagreement": 0.01,
            "train_examples": 500,
            "eval_examples": 200,
            "peak_vram_mb": 40000,
            "cost_usd": 4.0,
        },
        "artifact_sha256": runner.hash_artifact(artifact),
        "artifact_path": f"artifacts/seed-{seed}",
        "wall_clock_seconds": 100.0,
        "collected_at": "2026-09-09T12:00:00+00:00",
    }


def test_example_manifest_is_valid() -> None:
    loaded = runner.load_manifest(ROOT / "benchmarks/flagship_manifest.example.json")
    assert loaded["seeds"] == [42, 1337, 2026]
    assert loaded["model_parameter_count"] == 8_000_000_000


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(model_parameter_count=900_000_000),
            "between 7B and 9B",
        ),
        (lambda value: value.update(seeds=[42, 42, 2026]), "three unique"),
        (lambda value: value.update(trainer="grpo"), "requires gspo"),
        (lambda value: value["config"].update(num_eval_examples=199), ">= 200"),
        (
            lambda value: value.update(api_key="do-not-store"),
            "may not contain credentials",
        ),
    ],
)
def test_manifest_rejects_non_publishable_contracts(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    value = manifest()
    mutation(value)
    with pytest.raises(runner.FlagshipError, match=message):
        runner.load_manifest(write_manifest(tmp_path, value))


def test_manifest_requires_every_command_placeholder(tmp_path: Path) -> None:
    value = manifest()
    value["command"].remove("{artifact_dir}")
    with pytest.raises(runner.FlagshipError, match="artifact_dir"):
        runner.load_manifest(write_manifest(tmp_path, value))


def test_measured_execution_rejects_templates_and_stale_frameworks() -> None:
    value = manifest()
    runner.validate_execution_ready(value, PACKAGE_VERSION)

    value["provider"] = "REPLACE_WITH_PROVIDER"
    with pytest.raises(runner.FlagshipError, match="template placeholders"):
        runner.validate_execution_ready(value, PACKAGE_VERSION)

    value = manifest()
    value["model_revision"] = "0" * 40
    with pytest.raises(runner.FlagshipError, match="template zero revision"):
        runner.validate_execution_ready(value, PACKAGE_VERSION)

    value = manifest()
    value["framework_version"] = "0.1.0"
    with pytest.raises(runner.FlagshipError, match="installed StateSet version"):
        runner.validate_execution_ready(value, PACKAGE_VERSION)


def test_matrix_passes_paired_significance_and_cost_summary() -> None:
    summary = runner.validate_matrix(
        [run(42, 0.05), run(1337, 0.051), run(2026, 0.049)], manifest()
    )
    assert summary["passed"] is True
    assert summary["mean_improvement"] == pytest.approx(0.05)
    assert summary["total_cost_usd"] == pytest.approx(12.0)


def test_matrix_rejects_seed_selection_and_preflight() -> None:
    with pytest.raises(runner.FlagshipError, match="every declared seed"):
        runner.validate_matrix([run(42), run(1337)], manifest())
    values = [run(42), run(1337), run(2026)]
    values[0]["preflight"] = True
    with pytest.raises(runner.FlagshipError, match="preflight"):
        runner.validate_matrix(values, manifest())


def test_matrix_rejects_mixed_harnesses_and_reused_artifact_paths() -> None:
    values = [run(42), run(1337), run(2026)]
    for value in values:
        value.update(
            harness_commit="a" * 40,
            manifest_sha256="b" * 64,
            artifact_path=f"artifacts/{value['seed']}",
        )
    values[-1]["harness_commit"] = "c" * 40
    with pytest.raises(runner.FlagshipError, match="multiple harness"):
        runner.validate_matrix(values, manifest())

    values[-1]["harness_commit"] = "a" * 40
    values[-1]["artifact_path"] = values[0]["artifact_path"]
    with pytest.raises(runner.FlagshipError, match="distinct artifact"):
        runner.validate_matrix(values, manifest())


def test_matrix_rejects_unstable_or_insignificant_results() -> None:
    with pytest.raises(runner.FlagshipError, match="confidence"):
        runner.validate_matrix(
            [run(42, 0.20), run(1337, 0.04), run(2026, -0.05)], manifest()
        )
    values = [run(42), run(1337), run(2026)]
    values[1]["metrics"]["judge_self_disagreement"] = 0.051
    with pytest.raises(runner.FlagshipError, match="self-disagreement"):
        runner.validate_matrix(values, manifest())


def test_adapter_requires_cost_topology_and_owned_artifact(tmp_path: Path) -> None:
    value = manifest()
    config_sha = runner.digest_json(value["config"])
    raw = {
        "status": "completed",
        "measured": True,
        "framework_version": value["framework_version"],
        "model_revision": value["model_revision"],
        "dataset_revision": value["dataset_revision"],
        "config_sha256": config_sha,
        "judge_sha256": runner.digest_json(value["judge"]),
        "cost_source": value["cost_source"],
        "artifact_path": str(tmp_path / "artifact"),
        "hardware": {
            **value["hardware"],
            "cuda": "13.0",
            "driver": "580.1",
        },
        "metrics": {
            "baseline_score": 0.5,
            "final_score": 0.56,
            "judge_self_disagreement": 0.01,
            "train_examples": 500,
            "eval_examples": 200,
            "peak_vram_mb": 40000,
            "cost_usd": 4.2,
        },
    }
    assert runner.validate_adapter(raw, value, config_sha, tmp_path / "a.json")
    raw["judge_sha256"] = "0" * 64
    with pytest.raises(runner.FlagshipError, match="judge_sha256"):
        runner.validate_adapter(raw, value, config_sha, tmp_path / "a.json")
    raw["judge_sha256"] = runner.digest_json(value["judge"])
    raw["cost_source"] = "estimate"
    with pytest.raises(runner.FlagshipError, match="cost_source"):
        runner.validate_adapter(raw, value, config_sha, tmp_path / "a.json")


def test_run_seed_measures_externally_and_hashes_owned_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = manifest()

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        adapter_path = Path(command[-2])
        artifact_dir = Path(command[-1])
        (artifact_dir / "weights.bin").write_bytes(b"weights")
        adapter = {
            "status": "completed",
            "measured": True,
            "framework_version": value["framework_version"],
            "model_revision": value["model_revision"],
            "dataset_revision": value["dataset_revision"],
            "config_sha256": runner.digest_json(value["config"]),
            "judge_sha256": runner.digest_json(value["judge"]),
            "cost_source": value["cost_source"],
            "artifact_path": str(artifact_dir),
            "hardware": {**value["hardware"], "cuda": "13.0", "driver": "580.1"},
            "metrics": {
                "baseline_score": 0.5,
                "final_score": 0.55,
                "judge_self_disagreement": 0.01,
                "train_examples": 500,
                "eval_examples": 200,
                "peak_vram_mb": 40000,
                "cost_usd": 4.0,
            },
        }
        adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    evidence = runner.run_seed(value, 42, tmp_path / "out", "e" * 40, False)
    artifact = tmp_path / "out/evidence/artifacts/seed-42"
    evidence_path = tmp_path / "out/evidence/seed-42.json"
    assert evidence["schema_version"] == 2
    assert evidence["artifact_path"] == "artifacts/seed-42"
    assert evidence["artifact_sha256"] == runner.hash_artifact(artifact)
    assert runner.validate_retained_evidence(evidence, value, evidence_path)
    assert evidence["wall_clock_seconds"] >= 0
    assert (tmp_path / "out/attempts/seed-42/stdout.log").read_text() == "ok"

    (artifact / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(runner.FlagshipError, match="digest mismatch"):
        runner.validate_retained_evidence(evidence, value, evidence_path)


def test_retained_artifact_rejects_escape_and_symlink(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence" / "seed-42.json"
    evidence_path.parent.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"weights")
    data = {"artifact_path": "../outside.bin", "artifact_sha256": "0" * 64}
    with pytest.raises(runner.FlagshipError, match="escapes"):
        runner.verify_retained_artifact(data, evidence_path)

    artifact = evidence_path.parent / "artifact"
    artifact.symlink_to(outside)
    data["artifact_path"] = "artifact"
    with pytest.raises(runner.FlagshipError, match="escapes|unsafe"):
        runner.verify_retained_artifact(data, evidence_path)


def test_validate_existing_rehashes_complete_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    value = manifest()
    manifest_path = write_manifest(tmp_path, value)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    for seed in value["seeds"]:
        document = retained_run(evidence_dir, value, seed)
        (evidence_dir / f"seed-{seed}.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
    report_dir = tmp_path / "report"

    assert (
        runner.main(
            [
                str(manifest_path),
                "--validate-existing",
                str(evidence_dir),
                "--output-dir",
                str(report_dir),
            ]
        )
        == 0
    )
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["seed_count"] == 3
    assert "validated 3 retained flagship runs" in capsys.readouterr().out


def test_dry_run_is_non_allocating(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "evidence"
    result = runner.main(
        [str(write_manifest(tmp_path)), "--output-dir", str(output), "--dry-run"]
    )
    assert result == 0
    assert not output.exists()
    assert capsys.readouterr().out.count("mode=measured") == 3


def test_measured_main_rejects_the_example_template_before_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "must-not-exist"
    result = runner.main(
        [
            str(ROOT / "benchmarks/flagship_manifest.example.json"),
            "--output-dir",
            str(output),
        ]
    )
    assert result == 1
    assert not output.exists()
    assert "template placeholders" in capsys.readouterr().err


def test_failed_publication_gate_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "git_commit", lambda _root: "e" * 40)
    monkeypatch.setattr(
        runner,
        "run_seed",
        lambda _manifest, seed, _output, _commit, _preflight: run(seed, 0.0),
    )
    output = tmp_path / "failed"
    result = runner.main([str(write_manifest(tmp_path)), "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert report["passed"] is False
    assert "below +0.0300" in report["failures"][0]
