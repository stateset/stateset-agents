"""Production promotion revalidates raw evidence and actual checkpoint bytes."""

import copy
import json

import pytest

from stateset_agents.evaluation.agent_runs import evaluation_report
from stateset_agents.remote.deployment import DeploymentSpec
from stateset_agents.training.lineage import AdapterManifest, write_manifest
from stateset_agents.training.promotion import (
    artifact_digest,
    evaluate_promotion,
    verify_promotion,
)


def promotion(tmp_path):
    artifact = tmp_path / "policy"
    artifact.mkdir()
    (artifact / "pytorch_model.bin").write_bytes(b"unit-test-placeholder-weights")
    write_manifest(
        artifact,
        AdapterManifest(
            base_model="base",
            model_revision="a" * 40,
            dataset_sha256="b" * 64,
            reward_sha256="c" * 64,
        ),
        required=True,
    )
    digest = artifact_digest(artifact)

    def report(seed, success):
        return evaluation_report(
            environment="fixture",
            base_model="base",
            seed=seed,
            cases={str(index): {"id": index} for index in range(32)},
            settings={
                "temperature": 1,
                "model_revision": "a" * 40,
                "dataset_sha256": "b" * 64,
                "reward_sha256": "c" * 64,
            },
            checkpoint={
                "path": f"candidate-{seed}" if success else f"baseline-{seed}",
                "inventory_sha256": (
                    digest
                    if seed == 0 and success
                    else f"{seed + int(success) + 1:064x}"
                ),
            },
            outcomes=[
                {
                    "case_id": str(index),
                    "success": success,
                    "reward": float(success),
                    "policy_violations": 0,
                    "truncated": None,
                    "generated_tokens": 3,
                    "tool_calls": 1,
                    "elapsed_seconds": 0.1,
                }
                for index in range(32)
            ],
        )

    evidence = {
        "artifact_sha256": digest,
        "dataset_sha256": "b" * 64,
        "reward_sha256": "c" * 64,
        "review": {"source": "reviewed", "reviewer": "unit-test fixture"},
        "baselines": [report(seed, False) for seed in range(6)],
        "candidates": [report(seed, True) for seed in range(6)],
    }
    return artifact, evidence


def test_production_deployment_revalidates_a_portable_receipt(tmp_path):
    artifact, evidence = promotion(tmp_path)
    receipt = evaluate_promotion(artifact, evidence)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"receipt": receipt, "evidence": evidence}))
    assert verify_promotion(artifact, path)["passed"]
    spec = DeploymentSpec(
        name="deployed",
        model_name="base",
        weights_uri=f"s3://bucket/{receipt['artifact_sha256']}/",
        gpu="test",
        production=True,
        promotion_receipt=str(path),
        artifact_directory=str(artifact),
    )
    assert spec.production
    (artifact / "pytorch_model.bin").write_bytes(b"changed after evaluation")
    with pytest.raises(ValueError, match="hash differs"):
        verify_promotion(artifact, path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_review",
        "changed_reward",
        "empty_seeds",
        "wrong_artifact",
        "wrong_protocol",
        "duplicate_cases",
        "duplicate_checkpoints",
    ],
)
def test_incomplete_or_mismatched_evidence_cannot_be_promoted(tmp_path, mutation):
    artifact, evidence = promotion(tmp_path)
    altered = copy.deepcopy(evidence)
    if mutation == "missing_review":
        altered.pop("review")
    elif mutation == "changed_reward":
        altered["reward_sha256"] = "d" * 64
    elif mutation == "empty_seeds":
        altered["candidates"] = []
    elif mutation == "wrong_protocol":
        for report in altered["baselines"] + altered["candidates"]:
            report["settings"]["model_revision"] = "d" * 40
    elif mutation == "duplicate_cases":
        for report in altered["baselines"] + altered["candidates"]:
            report["case_hashes"] = dict.fromkeys(report["case_hashes"], "e" * 64)
    elif mutation == "duplicate_checkpoints":
        altered["candidates"][2]["selected_checkpoint"]["inventory_sha256"] = altered[
            "candidates"
        ][1]["selected_checkpoint"]["inventory_sha256"]
    else:
        altered["artifact_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        evaluate_promotion(artifact, altered)


@pytest.mark.parametrize("wrong_model", [True, False])
def test_deployment_rejects_wrong_model_or_digest_only_in_query(tmp_path, wrong_model):
    artifact, evidence = promotion(tmp_path)
    receipt = evaluate_promotion(artifact, evidence)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"receipt": receipt, "evidence": evidence}))
    digest = receipt["artifact_sha256"]
    with pytest.raises(ValueError):
        DeploymentSpec(
            name="deployed",
            model_name="different" if wrong_model else "base",
            weights_uri=(
                f"s3://bucket/{digest}/"
                if wrong_model
                else f"s3://bucket/policy?digest={digest}"
            ),
            gpu="test",
            production=True,
            promotion_receipt=str(path),
            artifact_directory=str(artifact),
        )


def test_production_requires_evidence_before_a_provider_can_allocate():
    with pytest.raises(ValueError, match="promotion receipt"):
        DeploymentSpec(
            name="x",
            model_name="base",
            weights_uri="s3://x",
            gpu="test",
            production=True,
        )


def test_required_lineage_write_failure_is_fatal(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("blocked")
    manifest = AdapterManifest(
        base_model="base",
        model_revision="a" * 40,
        dataset_sha256="b" * 64,
        reward_sha256="c" * 64,
    )
    with pytest.raises(OSError):
        write_manifest(blocker / "child", manifest, required=True)


def test_required_lineage_rejects_mutable_revision_before_writing(tmp_path):
    with pytest.raises(ValueError, match="immutable model revision"):
        write_manifest(
            tmp_path / "policy",
            AdapterManifest(
                base_model="base",
                model_revision="main",
                dataset_sha256="b" * 64,
                reward_sha256="c" * 64,
            ),
            required=True,
        )
    assert not (tmp_path / "policy").exists()


def test_cli_reports_missing_promotion_without_loading_provider(monkeypatch):
    from typer.testing import CliRunner

    from stateset_agents.cli_remote import app

    def forbidden_provider(*args):
        pytest.fail("Production validation must precede provider construction")

    monkeypatch.setattr(
        "stateset_agents.remote.deployment_registry.get_deployment_provider",
        forbidden_provider,
    )
    result = CliRunner().invoke(
        app,
        [
            "inference-deploy",
            "--provider",
            "nebius",
            "--name",
            "example",
            "--model-name",
            "base",
            "--weights-uri",
            "s3://bucket/policy",
            "--gpu",
            "test",
            "--production",
        ],
    )
    assert result.exit_code == 1
    assert "promotion receipt" in result.output
