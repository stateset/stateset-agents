"""Production promotion bound to actual policy bytes and paired outcome evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import compare_runs, content_hash

from .lineage import read_manifest


def artifact_digest(directory: Path) -> str:
    """Hash a nonempty reusable policy tree, rejecting symlinks and special files."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Policy artifact must be a real directory")
    inventory = {}
    weights = False
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Policy artifact cannot contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Policy artifact must contain only regular files")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory[path.relative_to(directory).as_posix()] = digest.hexdigest()
        weights |= (
            path.suffix in {".safetensors", ".bin", ".pt"} and path.stat().st_size > 0
        )
    if not inventory or not weights:
        raise ValueError("Policy artifact has no nonempty reusable weights")
    return content_hash(inventory)


def evaluate_promotion(artifact: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Recompute an empirical six-seed quality gate and bind the selected artifact.

    Review attestations and experiment independence remain the reviewer's
    responsibility. This checks recorded evidence, not remote inference output.
    """
    if not isinstance(evidence, dict):
        raise ValueError("Promotion evidence must be a JSON object")
    lineage = read_manifest(artifact)
    if (
        not lineage
        or not isinstance(lineage.get("base_model"), str)
        or not lineage["base_model"].strip()
        or not re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})", str(lineage.get("model_revision", ""))
        )
    ):
        raise ValueError("Production policy requires complete model lineage")
    for field in ("dataset_sha256", "reward_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(lineage.get(field, ""))):
            raise ValueError(f"Production lineage requires {field}")
        if evidence.get(field) != lineage[field]:
            raise ValueError(f"Evaluation and lineage differ: {field}")
    review = evidence.get("review")
    if (
        not isinstance(review, dict)
        or review.get("source") != "reviewed"
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
    ):
        raise ValueError("Promotion requires reviewed evaluation provenance")
    digest = artifact_digest(artifact)
    if evidence.get("artifact_sha256") != digest:
        raise ValueError("Evaluation artifact hash differs from policy bytes")
    baselines, candidates = evidence.get("baselines"), evidence.get("candidates")
    if not isinstance(baselines, list) or not isinstance(candidates, list):
        raise ValueError("Promotion requires complete paired raw evaluation reports")
    result = compare_runs(baselines, candidates, min_seeds=6, require_significance=True)
    if any(len(set(report["case_hashes"].values())) < 32 for report in candidates):
        raise ValueError(
            "Promotion requires at least 32 distinct held-out cases per seed"
        )
    checkpoint_hashes = [
        report["selected_checkpoint"].get("inventory_sha256") for report in candidates
    ]
    if any(
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in checkpoint_hashes
    ) or len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("Candidate seeds require distinct checkpoint content hashes")
    for report in baselines + candidates:
        for field in ("model_revision", "dataset_sha256", "reward_sha256"):
            if report["settings"].get(field) != lineage[field]:
                raise ValueError(
                    f"Raw evaluation protocol differs from lineage: {field}"
                )
    if not result["passed"]:
        raise ValueError("Policy did not pass paired quality and safety gates")
    selected = [
        report
        for report in candidates
        if report["selected_checkpoint"].get("inventory_sha256") == digest
    ]
    if len(selected) != 1:
        raise ValueError(
            "Selected policy must match exactly one evaluated training checkpoint"
        )
    if selected[0]["base_model"] != lineage["base_model"]:
        raise ValueError("Evaluation base model differs from lineage")
    return {
        "schema_version": 1,
        "base_model": lineage["base_model"],
        "model_revision": lineage["model_revision"],
        "artifact_sha256": digest,
        "evidence_sha256": content_hash(evidence),
        "lineage_sha256": content_hash(lineage),
        "quality_gate": result,
        "passed": True,
    }


def verify_promotion(artifact: Path, receipt_path: Path) -> dict[str, Any]:
    """Revalidate policy bytes and raw evidence before production deployment."""
    document = json.loads(receipt_path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("evidence"), dict):
        raise ValueError("Invalid promotion receipt")
    actual = evaluate_promotion(artifact, document["evidence"])
    if document.get("receipt") != actual:
        raise ValueError("Promotion receipt differs from recomputed evidence")
    return actual


def main() -> None:
    """Create a new portable promotion receipt after validating all raw evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.artifact.resolve()):
        parser.error("Promotion receipt must live outside the policy artifact")
    evidence = json.loads(args.evidence.read_text())
    receipt = evaluate_promotion(args.artifact, evidence)
    with args.output.open("x") as stream:
        json.dump(
            {"receipt": receipt, "evidence": evidence},
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
