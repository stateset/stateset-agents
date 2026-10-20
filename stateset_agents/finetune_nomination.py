"""Explicit checkpoint nominations for a separate, reviewed test assessment.

Receipts select bytes already verified by a validation comparison. They do not
copy weights, replace training completion, launch inference or establish quality.
"""

from __future__ import annotations

import re
import time
import uuid
from itertools import islice
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.finetune_validation import (
    MAX_COMPARISON_RECORDS,
    RECORD_ID,
    comparison_path,
    comparison_status,
)
from stateset_agents.finetune_workspace import read_json, read_run_plan, write_json


def nomination_plan(run: Path, comparison_id: str) -> dict[str, Any]:
    """Capture a metadata-only nomination review; no writes or weight reads."""
    return _selection_plan(
        run.resolve(), comparison_id, content_hash(package_implementation())
    )


def _selection_plan(
    run: Path, comparison_id: str, implementation_sha256: str
) -> dict[str, Any]:
    run = run.resolve()
    path = comparison_path(run, comparison_id)
    status = comparison_status(path)
    result = status.get("result")
    if not result or not result["comparison_complete"]:
        raise ValueError("A completed, complete validation comparison is required")
    choice = result["recommendation"]
    if choice["kind"] != "checkpoint":
        raise ValueError("The comparison does not recommend an adapter checkpoint")
    captured = read_json(path / "plan.json")
    _, parent_sha = read_run_plan(run)
    if (
        captured["parent_plan_sha256"] != parent_sha
        or captured["implementation_sha256"] != implementation_sha256
    ):
        raise ValueError("Comparison code or parent changed; compare again")
    selected = next(
        row for row in result["rows"] if row["checkpoint"] == choice["checkpoint"]
    )
    evaluation_id = selected["evaluation_id"]
    snapshot = next(
        row for row in captured["evaluations"] if row["evaluation_id"] == evaluation_id
    )
    evaluation = run / "evaluations" / evaluation_id
    if evaluation.is_symlink() or evaluation.parent.is_symlink():
        raise ValueError("Selection evidence must belong to the retained run")
    plan = read_json(evaluation / "plan.json")
    if content_hash(plan) != snapshot["plan_sha256"]:
        raise ValueError("Selected evaluation changed after comparison")
    if (
        plan["config"]["checkpoint"] != choice["checkpoint"]
        or plan["config"]["split"] != "validation"
    ):
        raise ValueError("Selection must name a validation checkpoint")
    if plan["binding"]["artifact_sha256"] != selected["artifact_sha256"]:
        raise ValueError("Selected checkpoint differs from the comparison")
    return {
        "schema_version": 1,
        "kind": "stateset-checkpoint-nomination",
        "run_id": run.name,
        "parent_plan_sha256": parent_sha,
        "implementation_sha256": captured["implementation_sha256"],
        "comparison_id": comparison_id,
        "comparison_plan_sha256": result["comparison_plan_sha256"],
        "comparison_result_sha256": read_json(path / "state.json")["result_sha256"],
        "checkpoint": choice["checkpoint"],
        "artifact_sha256": selected["artifact_sha256"],
        "checkpoint_receipt_sha256": plan["binding"]["checkpoint_receipt_sha256"],
        "selection_evaluation_id": evaluation_id,
        "selection_plan_sha256": snapshot["plan_sha256"],
        "selection_cases_sha256": plan["cases_sha256"],
    }


def prepare_nomination(
    run: Path, comparison_id: str, *, expected_plan_sha256: str
) -> dict[str, Any]:
    """Retain exactly the reviewed nomination without changing training artifacts."""
    plan = nomination_plan(run, comparison_id)
    digest = content_hash(plan)
    if digest != expected_plan_sha256:
        raise ValueError("Nomination changed; capture a fresh review")
    directory = run.resolve() / "nominations"
    if directory.is_symlink():
        raise ValueError("Nomination history must be a regular directory")
    directory.mkdir(mode=0o700, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    path = directory / name
    path.mkdir(mode=0o700)
    # Publish one complete JSON record atomically; an interrupted directory has
    # no valid receipt and cannot authorize a test evaluation.
    write_json(path / "receipt.json", {"plan": plan, "plan_sha256": digest})
    return {
        "nomination_id": name,
        "plan_sha256": digest,
        "checkpoint": plan["checkpoint"],
        "comparison_id": comparison_id,
        "quality_gain_verified": False,
        "training_started": False,
        "inference_started": False,
    }


def verify_nomination(
    run: Path, nomination_id: str, binding: dict[str, Any]
) -> dict[str, str]:
    """Recheck the nomination's lineage and bind it to freshly hashed weights."""
    receipt = _read_receipt(run, nomination_id)
    plan = receipt["plan"]
    if (
        not isinstance(plan, dict)
        or content_hash(plan) != receipt["plan_sha256"]
        or plan != nomination_plan(run, plan["comparison_id"])
        or any(
            plan[key] != binding[key]
            for key in ("checkpoint", "artifact_sha256", "parent_plan_sha256")
        )
        or (
            "checkpoint_receipt_sha256" in binding
            and plan["checkpoint_receipt_sha256"]
            != binding["checkpoint_receipt_sha256"]
        )
    ):
        raise ValueError("Nomination or selected checkpoint changed; review again")
    return {"nomination_id": nomination_id, "plan_sha256": receipt["plan_sha256"]}


def _read_receipt(run: Path, nomination_id: str) -> dict[str, Any]:
    if not isinstance(nomination_id, str) or not re.fullmatch(RECORD_ID, nomination_id):
        raise ValueError("Choose a retained nomination ID")
    directory = run / "nominations"
    path = directory / nomination_id
    if directory.is_symlink() or path.is_symlink() or not path.is_dir():
        raise ValueError("Nomination must belong to the retained run")
    receipt = read_json(path / "receipt.json")
    plan = receipt.get("plan")
    if not isinstance(plan, dict) or content_hash(plan) != receipt.get("plan_sha256"):
        raise ValueError("Nomination receipt changed or is unreadable")
    return receipt


def _public_nomination(nomination_id: str, receipt: dict[str, Any]) -> dict[str, Any]:
    plan = receipt["plan"]
    return {
        "nomination_id": nomination_id,
        "state": "available",
        "plan_sha256": receipt["plan_sha256"],
        "checkpoint": plan["checkpoint"],
        "comparison_id": plan["comparison_id"],
        "selection_evaluation_id": plan["selection_evaluation_id"],
        "bytes_verified": False,
        "quality_gain_verified": False,
        "training_started": False,
        "inference_started": False,
    }


def nomination_status(run: Path, nomination_id: str) -> dict[str, Any]:
    """Verify and project selection metadata, without reading weights or cases."""
    run = run.resolve()
    receipt = _read_receipt(run, nomination_id)
    if receipt["plan"] != nomination_plan(run, receipt["plan"]["comparison_id"]):
        raise ValueError("Nomination lineage changed; inspect its retained evidence")
    return _public_nomination(nomination_id, receipt)


def list_nominations(run: Path) -> dict[str, Any]:
    """List bounded metadata; preserve broken rows without exporting private errors."""
    run = run.resolve()
    read_run_plan(run)
    directory = run / "nominations"
    if directory.is_symlink():
        raise ValueError("Nomination history must be a regular directory")
    if not directory.exists():
        return {"nominations": [], "bytes_verified": False}
    paths = list(islice(directory.iterdir(), MAX_COMPARISON_RECORDS + 1))
    if len(paths) > MAX_COMPARISON_RECORDS:
        raise ValueError(
            "Nomination history exceeds 100 entries; inspect a specific ID"
        )
    identity = content_hash(package_implementation())
    selections: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, reverse=True):
        if not re.fullmatch(RECORD_ID, path.name):
            continue
        try:
            receipt = _read_receipt(run, path.name)
            comparison_id = receipt["plan"]["comparison_id"]
            if comparison_id not in selections:
                selections[comparison_id] = _selection_plan(
                    run, comparison_id, identity
                )
            if receipt["plan"] != selections[comparison_id]:
                raise ValueError("Nomination lineage changed")
            rows.append(_public_nomination(path.name, receipt))
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            StopIteration,
            RecursionError,
        ):
            rows.append(
                {
                    "nomination_id": path.name,
                    "state": "unavailable",
                    "bytes_verified": False,
                }
            )
    return {"nominations": rows, "bytes_verified": False}


def nomination_config(
    run: Path, nomination_id: str, *, expected_plan_sha256: str
) -> dict[str, Any]:
    """Restore exactly the chosen metadata into an unprepared fresh-test form."""
    from dataclasses import asdict

    from stateset_agents.finetune_evaluation import EvaluationConfig

    status = nomination_status(run, nomination_id)
    if status["plan_sha256"] != expected_plan_sha256:
        raise ValueError("Nomination differs from the selected receipt; refresh")
    receipt = _read_receipt(run, nomination_id)
    if receipt["plan_sha256"] != expected_plan_sha256:
        raise ValueError("Nomination changed during reopening; refresh")
    plan = read_json(
        run / "evaluations" / status["selection_evaluation_id"] / "plan.json"
    )
    if content_hash(plan) != receipt["plan"]["selection_plan_sha256"]:
        raise ValueError("Selection protocol changed during reopening; refresh")
    values = asdict(EvaluationConfig(**plan["config"]))
    values.update(
        cases="",
        split="test",
        checkpoint=status["checkpoint"],
        nomination_id=nomination_id,
    )
    return {
        "nomination": status,
        "config": values,
        "prepared": False,
        "inference_started": False,
    }


def assert_fresh_test_cases(run: Path, cases: list[dict[str, Any]]) -> None:
    """Reject overlap with bounded retained validation history, including failed jobs.

    IDs and normalized user turns are exact overlap checks; reviewer provenance
    remains an attestation, not proof of semantic or external dataset independence.
    """
    directory = run / "evaluations"
    if directory.is_symlink():
        raise ValueError("Evaluation history cannot be a symlink")
    paths = list(islice(directory.iterdir(), MAX_COMPARISON_RECORDS + 1))
    if len(paths) > MAX_COMPARISON_RECORDS:
        raise ValueError("Cannot establish fresh test separation above 100 records")
    ids = {case["id"] for case in cases}
    turns = {turn.strip().casefold() for case in cases for turn in case["user_turns"]}
    for path in paths:
        if not re.fullmatch(RECORD_ID, path.name):
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Validation history contains an unreadable record")
        plan = read_json(path / "plan.json")
        if plan["config"].get("split", "test") != "validation":
            continue
        document = read_json(path / "cases.json", max_bytes=2 * 1024 * 1024)
        if content_hash(document) != plan["cases_sha256"]:
            raise ValueError(
                "Validation cases changed; fresh test separation is unknown"
            )
        for case in document["cases"]:
            if case["id"] in ids or any(
                turn.strip().casefold() in turns for turn in case["user_turns"]
            ):
                raise ValueError(
                    "Fresh test cases overlap retained validation evidence"
                )
