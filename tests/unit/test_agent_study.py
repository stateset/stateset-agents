"""Offline provenance audit fixtures; synthetic outcomes are not model evidence."""

import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.data.refund_demonstrations import (
    filter_refund_candidates,
    prepare_refund_data,
    reference_refund_demonstration,
)
from stateset_agents.evaluation.agent_runs import content_hash, evaluation_report
from stateset_agents.evaluation.agent_study import (
    ARMS,
    StudyConfig,
    _native_manifest,
    _sft_source,
    audit_study,
    build_plan,
    prepare_study,
)
from stateset_agents.remote.rollout_budget import RolloutAdmissionBudget
from stateset_agents.training.river_progress import summarize_training_activity
from stateset_agents.training.river_refund import EVALUATION_SETTINGS
from tests.unit.river_fakes import traced_refund_outcome


def small_config():
    return StudyConfig(
        train_count=8, validation_count=8, test_count=8, steps=2, sft_epochs=1
    )


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    """Write synthetic or deliberately corrupted evidence, without crash guarantees."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def synthetic_budget(directory, config, run, artifact, admitted):
    if config.rollout_token_budget is None:
        return
    path = directory / "rollout_budget.json"
    ledger = RolloutAdmissionBudget(
        path,
        limit=config.rollout_token_budget,
        trajectory_tokens=1024,
        fingerprint=content_hash(run),
        create=True,
    )
    # Synthetic provenance fixtures are never evidence of provider execution.
    state = ledger.snapshot()
    state.update(
        admitted_trajectories=admitted, reserved_generated_tokens=admitted * 1024
    )
    write_json(path, state)
    artifact["rollout_budget"] = ledger.snapshot()


async def synthetic_study(root, config=None):
    config = config or small_config()
    prepare_study(root, config)
    for seed in config.seeds:
        directory = root / f"seed-{seed}"
        splits = config.splits(seed)
        await prepare_refund_data(
            directory / "data",
            seed=seed,
            train_count=8,
            validation_count=8,
            test_count=8,
        )
        candidates = []
        for case in splits["train"]:
            demo = await reference_refund_demonstration(case)
            candidates.extend([demo] * 8)
        collection = {
            "schema_version": 1,
            "environment": "refund-policy-v2",
            "seed": seed,
            "base_model": config.base_model,
            "run_manifest_hash": content_hash(
                _native_manifest(config, seed, "collection", None)
            ),
            "settings": {**EVALUATION_SETTINGS, "group_size": 8, "temperature": 1.0},
            "checkpoint": {"path": f"river://collection-{seed}"},
            "case_hashes": {r["order_id"]: content_hash(r) for r in splits["train"]},
            "complete": True,
            "group_size": 8,
            "candidates": candidates,
        }
        synthetic_budget(
            directory / "collection",
            config,
            _native_manifest(config, seed, "collection", None),
            collection,
            64,
        )
        write_json(directory / "collection/training_candidates.json", collection)
        write_json(
            directory / "collection/run_manifest.json",
            _native_manifest(config, seed, "collection", None),
        )
        await filter_refund_candidates(
            directory / "data",
            directory / "collection/training_candidates.json",
            directory / "filtered",
        )
        for arm in ARMS:
            source = None
            if arm in ("sft", "rejection_sft"):
                source = f"river://trained-{arm}-{seed}"
                data = (
                    directory / ("data" if arm == "sft" else "filtered") / "train.jsonl"
                )
                hyper = {
                    "seed": seed,
                    "shuffle": True,
                    "lora_r": 16,
                    "num_epochs": 1,
                    "learning_rate": config.sft_learning_rate,
                    "max_length": 4096,
                    "per_device_batch_size": 2,
                    "river_checkpoint": source,
                    "steps": 4,
                }
                write_json(
                    directory / (arm + "_train") / "river_checkpoint.json",
                    {
                        "base_model": config.base_model,
                        "checkpoint": source,
                        "sft_options": {"seed": seed, "shuffle": True},
                        "steps": 4,
                    },
                )
                write_json(
                    directory / (arm + "_train") / "stateset_manifest.json",
                    {
                        "base_model": config.base_model,
                        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                        "dataset_rows": 8,
                        "hyperparameters": hyper,
                    },
                )
            manifest = _native_manifest(config, seed, arm, source)
            write_json(directory / arm / "run_manifest.json", manifest)
            checkpoint = {"path": f"river://evaluation-{arm}-{seed}"}
            report = evaluation_report(
                environment="refund-policy-v2",
                base_model=config.base_model,
                seed=seed,
                cases={r["order_id"]: r for r in splits["test"]},
                settings=EVALUATION_SETTINGS,
                implementation=manifest["implementation"],
                checkpoint=checkpoint,
                case_families={r["order_id"]: r["family"] for r in splits["test"]},
                outcomes=[
                    await traced_refund_outcome(r, successful=arm == "rl")
                    for r in splits["test"]
                ],
            )
            report["run_manifest_hash"] = content_hash(manifest)
            report["selected_validation_step"] = 1 if arm == "rl" else None
            attempt = {
                "schema_version": 1,
                "run_manifest_hash": content_hash(manifest),
                "test_split_hash": content_hash(splits["test"]),
                "selected_checkpoint": report["selected_checkpoint"],
                "selected_validation_step": report["selected_validation_step"],
            }
            write_json(directory / arm / "test_attempt.json", attempt)
            report["test_attempt_hash"] = content_hash(attempt)
            synthetic_budget(
                directory / arm, config, manifest, report, 96 if arm == "rl" else 8
            )
            if arm == "rl":
                write_json(
                    directory / arm / "validation_results.json",
                    [
                        {
                            "step": 0,
                            "checkpoint": {"path": f"river://initial-{seed}"},
                            "metrics": {"reward_mean": 0},
                            "case_hashes": {
                                r["order_id"]: content_hash(r)
                                for r in splits["validation"]
                            },
                            "outcomes": [
                                await traced_refund_outcome(r, successful=i < 4)
                                for i, r in enumerate(splits["validation"])
                            ],
                        },
                        {
                            "step": 1,
                            "checkpoint": checkpoint,
                            "metrics": {"reward_mean": 1},
                            "case_hashes": {
                                r["order_id"]: content_hash(r)
                                for r in splits["validation"]
                            },
                            "outcomes": [
                                await traced_refund_outcome(r, successful=i < 8)
                                for i, r in enumerate(splits["validation"])
                            ],
                        },
                        {
                            "step": 2,
                            "checkpoint": {"path": f"river://later-{seed}"},
                            "metrics": {"reward_mean": 0.5},
                            "case_hashes": {
                                r["order_id"]: content_hash(r)
                                for r in splits["validation"]
                            },
                            "outcomes": [
                                await traced_refund_outcome(r, successful=i < 6)
                                for i, r in enumerate(splits["validation"])
                            ],
                        },
                    ],
                )
                progress = [
                    {
                        "step": n,
                        "metrics": {
                            "loss": 0.1,
                            "train/updated": 1.0,
                            "train/datums": 4.0,
                        },
                    }
                    for n in (1, 2)
                ]
                write_json(directory / arm / "training_metrics.json", progress)
                report["training_activity"] = summarize_training_activity(progress)
                write_json(
                    directory / arm / "training_activity.json",
                    report["training_activity"],
                )
            write_json(directory / arm / "test_results.json", report)
    return config


@pytest.fixture(scope="module")
def study_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("study-template")
    # These fixtures test provenance, not filesystem crash durability. Dedicated
    # budget/data tests exercise durable writes; avoid hundreds of fixture fsyncs.
    with patch("stateset_agents.remote.river_rl.os.fsync"):
        asyncio.run(synthetic_study(root))
    return root


@pytest.fixture(scope="module")
def capped_study_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("capped-study-template")
    with patch("stateset_agents.remote.river_rl.os.fsync"):
        asyncio.run(
            synthetic_study(root, replace(small_config(), rollout_token_budget=200000))
        )
    return root


@pytest.fixture
def study(tmp_path, study_template):
    shutil.copytree(study_template, tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_plan_is_offline_deterministic_and_dependency_ordered(tmp_path):
    plan = prepare_study(tmp_path, small_config())
    assert plan == build_plan(small_config())
    assert len(plan["stages"]) == 54
    assert sum(stage["paid"] for stage in plan["stages"]) == 42
    completed = set()
    for stage in plan["stages"]:
        assert set(stage["depends_on"]) <= completed
        assert stage["argv"][:2] == ["python", "-m"]
        completed.add(stage["id"])
    assert {p.name for p in tmp_path.iterdir()} == {"study_plan.json", ".rl.lock"}
    with pytest.raises(ValueError, match="empty"):
        prepare_study(tmp_path, small_config())


def test_capped_plan_applies_allowance_to_all_native_phases():
    plan = build_plan(replace(small_config(), rollout_token_budget=200000))
    native = [
        s
        for s in plan["stages"]
        if s["argv"][2] == "stateset_agents.training.river_refund"
    ]
    assert len(native) == 30
    assert all(
        s["argv"][s["argv"].index("--rollout-token-budget") + 1] == "200000"
        for s in native
    )
    assert plan["limits"]["global_reserved_rollout_token_cap"] == 6000000
    assert plan["limits"]["minimum_reserved_rollout_tokens_by_run"]["rl"] == 96 * 1024
    assert plan["limits"]["minimum_reserved_rollout_tokens"] == 184 * 6 * 1024
    assert plan["limits"]["dollar_cap"] is None


@pytest.mark.parametrize("patience", [0, 2, 5])
def test_plan_binds_zero_update_patience(patience):
    config = replace(small_config(), zero_update_patience=patience)
    plan = build_plan(config)
    assert plan["config"]["zero_update_patience"] == patience
    for stage in plan["stages"]:
        if stage["argv"][2] == "stateset_agents.training.river_refund":
            argv = stage["argv"]
            assert argv[argv.index("--zero-update-patience") + 1] == str(patience)


@pytest.mark.parametrize("mutation", ["missing", "binding", "insufficient"])
def test_capped_study_requires_durable_budget_evidence(
    tmp_path, capped_study_template, mutation
):
    shutil.copytree(capped_study_template, tmp_path, dirs_exist_ok=True)
    assert asyncio.run(audit_study(tmp_path))["passed"]
    root = tmp_path / "seed-42/rl"
    path = root / "rollout_budget.json"
    if mutation == "missing":
        path.unlink()
    elif mutation == "binding":
        state = read(path)
        state["fingerprint"] = "different"
        write_json(path, state)
    else:
        state = read(path)
        state.update(admitted_trajectories=0, reserved_generated_tokens=0)
        write_json(path, state)
        report = read(root / "test_results.json")
        report["rollout_budget"] = state
        write_json(root / "test_results.json", report)
    result = asyncio.run(audit_study(tmp_path))
    assert not result["passed"]
    assert any(issue.get("arm") == "rl" for issue in result["issues"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": (1, 2, 3)},
        {"seeds": (1, 2, 3, 4, 5, 5)},
        {"seeds": (True, 2, 3, 4, 5, 6)},
        {"train_count": 9},
        {"steps": 0},
        {"rl_learning_rate": float("nan")},
        {"rollout_token_budget": 1023},
        {"rollout_token_budget": 1024},
        {"rollout_token_budget": True},
        {"zero_update_patience": -1},
        {"zero_update_patience": True},
        {"zero_update_patience": 1.5},
    ],
)
def test_invalid_plan_is_rejected(kwargs):
    with pytest.raises(ValueError):
        StudyConfig(**kwargs)


def test_missing_live_evidence_fails_closed(tmp_path):
    prepare_study(tmp_path, small_config())
    report = asyncio.run(audit_study(tmp_path))
    assert not report["passed"] and not report["complete"]
    assert len(report["issues"]) == 6 and report["comparison"] is None


def test_complete_synthetic_fixture_passes_all_provenance_and_comparison_checks(study):
    result = asyncio.run(audit_study(study))
    assert result["passed"], result["issues"]
    assert len(result["evidence"]) == 24
    assert (
        result["comparison"]["comparisons"]["sft"]["inference"]["planned_comparisons"]
        == 3
    )


@pytest.mark.parametrize("missing", [False, True])
def test_audit_requires_planned_implementation_in_reports(study, missing):
    path = study / "seed-42/rl/test_results.json"
    report = read(path)
    if missing:
        del report["implementation"]
    else:
        report["implementation"]["files"]["__init__.py"] = "0" * 64
    write_json(path, report)
    result = asyncio.run(audit_study(study))
    assert not result["passed"]
    assert any(
        "Test protocol mismatch" in issue["reason"] for issue in result["issues"]
    )


def test_audit_rejects_plan_from_different_sources_even_with_recomputed_hash(study):
    path = study / "study_plan.json"
    plan = read(path)
    plan["implementation"]["files"]["__init__.py"] = "0" * 64
    del plan["plan_hash"]
    plan["plan_hash"] = content_hash(plan)
    write_json(path, plan)
    with pytest.raises(ValueError, match="incompatible with this protocol"):
        asyncio.run(audit_study(study))


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("seed", "Training seed"),
        ("dataset", "dataset lineage"),
        ("evaluation_config", "Run configuration"),
        ("case", "reset input"),
        ("test_input", "reset input hashes"),
        ("trace", "Test trace replay failed"),
        ("selected", "selected by validation"),
        ("steps", "Incomplete RL"),
        ("validation", "Incomplete validation"),
        ("validation_cases", "Incomplete validation case coverage"),
        ("validation_hash", "Validation cases differ"),
        ("validation_mean", "reward mean differs"),
        ("validation_trace", "Validation replay failed"),
        ("binding", "not bound"),
        ("collection", "complete best-of-eight"),
        ("collection_binding", "Collection is not bound"),
        ("collection_model", "Collection model"),
    ],
)
def test_audit_rejects_mismatched_training_and_evaluation(study, mutation, expected):
    root = study / "seed-42"
    path = root / "sft_train/stateset_manifest.json"
    if mutation in ("seed", "dataset"):
        value = read(path)
        if mutation == "seed":
            value["hyperparameters"]["seed"] = 99
        else:
            value["dataset_sha256"] = "0" * 64
    elif mutation == "evaluation_config":
        path = root / "sft/run_manifest.json"
        value = read(path)
        value["checkpoint"] = "river://wrong-model"
    elif mutation == "steps":
        path = root / "rl/training_metrics.json"
        value = read(path)[:1]
    elif mutation.startswith("validation"):
        path = root / "rl/validation_results.json"
        value = read(path)
        if mutation == "validation":
            value = value[1:]
        elif mutation == "validation_cases":
            value[1]["outcomes"].pop()
        elif mutation == "validation_trace":
            value[0]["outcomes"][0]["environment_trace"]["steps"][0][
                "observations"
            ] = []
        elif mutation == "validation_hash":
            value[1]["case_hashes"][next(iter(value[1]["case_hashes"]))] = "0" * 64
        else:
            value[1]["metrics"]["reward_mean"] = 100
    elif mutation.startswith("collection"):
        path = root / "collection/training_candidates.json"
        value = read(path)
        if mutation == "collection":
            value["complete"] = False
        elif mutation == "collection_binding":
            value["run_manifest_hash"] = "0" * 64
        else:
            value["base_model"] = "different-model"
    else:
        path = root / "rl/test_results.json"
        value = read(path)
        if mutation == "case":
            value["case_hashes"][next(iter(value["case_hashes"]))] = "0" * 64
        elif mutation == "test_input":
            del value["outcomes"][0]["case_hash"]
        elif mutation == "trace":
            value["outcomes"][0]["environment_trace"]["steps"][0]["observations"] = []
        elif mutation == "selected":
            value["selected_checkpoint"] = {"path": "river://unselected"}
            attempt_path = root / "rl/test_attempt.json"
            attempt = read(attempt_path)
            attempt["selected_checkpoint"] = value["selected_checkpoint"]
            write_json(attempt_path, attempt)
            value["test_attempt_hash"] = content_hash(attempt)
        else:
            value["run_manifest_hash"] = "0" * 64
    write_json(path, value)
    result = asyncio.run(audit_study(study))
    assert not result["passed"]
    assert any(expected in issue["reason"] for issue in result["issues"]), result[
        "issues"
    ]


def test_audit_rejects_truncated_candidates_despite_matching_hashes(study):
    root = study / "seed-42"
    path = root / "collection/training_candidates.json"
    collection = read(path)
    for candidate in collection["candidates"]:
        candidate["truncated"] = "token_limit"
    write_json(path, collection)
    path = root / "filtered/data_manifest.json"
    manifest = read(path)
    manifest["collection_hash"] = content_hash(collection)
    write_json(path, manifest)
    result = asyncio.run(audit_study(study))
    assert not result["passed"]
    assert any("Filtered transcript" in issue["reason"] for issue in result["issues"])


def audit_rejection_source(root):
    """Exercise the real source audit without repeating unrelated evaluation."""
    return asyncio.run(
        _sft_source(
            root,
            small_config(),
            42,
            "rejection_sft",
            small_config().splits(42),
            read(root / "data/data_manifest.json"),
        )
    )


@pytest.mark.parametrize("mutation", ["omit_success", "longer", "family", "order"])
def test_rejection_source_requires_exact_selection_even_with_updated_hashes(
    study, mutation
):
    root = study / "seed-42"
    data_path = root / "filtered/train.jsonl"
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    manifest_path = root / "filtered/data_manifest.json"
    manifest = read(manifest_path)
    if mutation == "omit_success":
        removed = rows.pop()
        manifest["families"][removed["metadata"]["family"]]["selected_cases"] -= 1
        manifest["not_selected"] += 1
    elif mutation == "longer":
        # JSON whitespace changes transcript length without changing actions
        # or observations. Keep shorter successes in the same collection.
        row = rows[0]
        assistant = next(m for m in row["messages"] if m["role"] == "assistant")
        assistant["content"] += " "
        collection_path = root / "collection/training_candidates.json"
        collection = read(collection_path)
        candidate = next(
            c
            for c in collection["candidates"]
            if c["case_id"] == row["metadata"]["case_id"]
        )
        candidate["messages"] = row["messages"]
        write_json(collection_path, collection)
        manifest["collection_hash"] = content_hash(collection)
    elif mutation == "family":
        rows[0]["metadata"]["family"] = "mislabelled"
    else:
        rows.reverse()
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest["artifacts"]["train.jsonl"] = digest
    manifest["selected"] = len(rows)
    write_json(manifest_path, manifest)
    # Keep the downstream training lineage internally consistent as well.
    training_path = root / "rejection_sft_train/stateset_manifest.json"
    training = read(training_path)
    training.update(dataset_sha256=digest, dataset_rows=len(rows))
    write_json(training_path, training)
    with pytest.raises(ValueError, match="deterministic replay selection"):
        audit_rejection_source(root)


@pytest.mark.parametrize("mutation", ["status", "candidate_hash", "omit"])
def test_rejection_audit_must_match_replay_even_after_rehash(study, mutation):
    root = study / "seed-42"
    path = root / "filtered/replay_audit.json"
    audit = read(path)
    if mutation == "status":
        audit[0]["status"] = "rejected"
    elif mutation == "candidate_hash":
        audit[0]["candidate_hash"] = "0" * 64
    else:
        audit.pop()
    write_json(path, audit)
    manifest_path = root / "filtered/data_manifest.json"
    manifest = read(manifest_path)
    manifest["artifacts"]["replay_audit.json"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="rejection audit differs"):
        audit_rejection_source(root)


@pytest.mark.parametrize(
    "field", ["candidates", "rejected", "not_selected", "families"]
)
def test_rejection_coverage_must_be_recomputed_from_collection(study, field):
    root = study / "seed-42"
    path = root / "filtered/data_manifest.json"
    manifest = read(path)
    if field == "families":
        manifest[field][next(iter(manifest[field]))]["available_cases"] += 1
    else:
        manifest[field] += 1
    write_json(path, manifest)
    with pytest.raises(ValueError, match="counts or family coverage"):
        audit_rejection_source(root)


def test_rejection_source_accepts_honest_missing_family_without_teacher_backfill(study):
    root = study / "seed-42"
    collection_path = root / "collection/training_candidates.json"
    collection = read(collection_path)
    case = small_config().splits(42)["train"][0]
    for candidate in collection["candidates"]:
        if candidate["case_id"] == case["order_id"]:
            candidate["truncated"] = "token_limit"
    write_json(collection_path, collection)
    filtered = root / "refiltered"
    manifest = asyncio.run(
        filter_refund_candidates(root / "data", collection_path, filtered)
    )
    assert manifest["selected"] == 7
    assert manifest["families"][case["family"]] == {
        "available_cases": 1,
        "selected_cases": 0,
    }
    shutil.copytree(filtered, root / "filtered", dirs_exist_ok=True)
    lineage_path = root / "rejection_sft_train/stateset_manifest.json"
    lineage = read(lineage_path)
    lineage.update(dataset_sha256=manifest["artifacts"]["train.jsonl"], dataset_rows=7)
    write_json(lineage_path, lineage)
    assert audit_rejection_source(root) == "river://trained-rejection_sft-42"


def test_aliases_cannot_hide_reused_source_training_checkpoint(study):
    first = read(study / "seed-42/sft_train/river_checkpoint.json")["checkpoint"]
    root = study / "seed-43"
    for relative, field in [
        ("sft_train/river_checkpoint.json", "checkpoint"),
        ("sft/run_manifest.json", "checkpoint"),
    ]:
        path = root / relative
        value = read(path)
        value[field] = first
        write_json(path, value)
    path = root / "sft_train/stateset_manifest.json"
    value = read(path)
    value["hyperparameters"]["river_checkpoint"] = first
    write_json(path, value)
    path = root / "sft/test_results.json"
    value = read(path)
    value["run_manifest_hash"] = content_hash(read(root / "sft/run_manifest.json"))
    attempt_path = root / "sft/test_attempt.json"
    attempt = read(attempt_path)
    attempt["run_manifest_hash"] = value["run_manifest_hash"]
    write_json(attempt_path, attempt)
    value["test_attempt_hash"] = content_hash(attempt)
    write_json(path, value)
    result = asyncio.run(audit_study(study))
    assert not result["passed"]
    assert any("checkpoint reused" in i["reason"] for i in result["issues"])


@pytest.mark.parametrize("mutation", ["missing", "checkpoint", "binding"])
def test_audit_requires_consistent_test_attempt(study, mutation):
    path = study / "seed-42/rl/test_attempt.json"
    if mutation == "missing":
        path.unlink()
    else:
        attempt = read(path)
        if mutation == "checkpoint":
            attempt["selected_checkpoint"] = {"path": "river://different"}
        else:
            attempt["run_manifest_hash"] = "different"
        write_json(path, attempt)
    result = asyncio.run(audit_study(study))
    assert not result["passed"]
    assert any(issue.get("arm") == "rl" for issue in result["issues"])


def test_audit_accepts_recovered_commits_only_with_matching_receipts(study):
    from stateset_agents.training.river_progress import RiverTrainingProgress

    root = study / "seed-42/rl"
    write_json(root / "training_metrics.json", read(root / "training_metrics.json")[:1])
    progress = RiverTrainingProgress(
        root, steps=2, run_manifest_hash=content_hash(read(root / "run_manifest.json"))
    )
    progress.reconcile(2)
    activity = summarize_training_activity(progress.records)
    write_json(root / "training_activity.json", activity)
    report = read(root / "test_results.json")
    report["training_activity"] = activity
    write_json(root / "test_results.json", report)
    result = asyncio.run(audit_study(study))
    assert result["passed"]
    evidence = next(
        item
        for item in result["evidence"]
        if item["seed"] == 42 and item["arm"] == "rl"
    )
    assert evidence["training_activity"]["observed_optimizer_updates"] == 1
    assert evidence["training_activity"]["unknown_update_batches"] == 1
    assert evidence["training_progress_hash"] == content_hash(progress.records)
    assert evidence["recovery_receipts_hash"] == content_hash(progress.receipts)
    (root / "recovery_receipts.json").unlink()
    result = asyncio.run(audit_study(study))
    assert not result["passed"]
    assert any(
        "recovery receipt" in issue["reason"].lower() for issue in result["issues"]
    )


def _refresh_activity(root):
    activity = summarize_training_activity(read(root / "training_metrics.json"))
    write_json(root / "training_activity.json", activity)
    report = read(root / "test_results.json")
    report["training_activity"] = activity
    write_json(root / "test_results.json", report)
    return activity


@pytest.mark.parametrize("patience", [0, 1])
def test_audit_enforces_planned_early_stop_even_if_later_batches_update(
    tmp_path, patience
):
    config = replace(small_config(), zero_update_patience=patience)
    with patch("stateset_agents.remote.river_rl.os.fsync"):
        asyncio.run(synthetic_study(tmp_path, config))
    root = tmp_path / "seed-42/rl"
    progress = read(root / "training_metrics.json")
    progress[0]["metrics"].update({"train/updated": 0, "train/datums": 0})
    write_json(root / "training_metrics.json", progress)
    _refresh_activity(root)
    result = asyncio.run(audit_study(tmp_path))
    assert result["passed"] == (patience == 0)
    if patience:
        assert any(
            "zero-update stopping threshold" in issue["reason"]
            for issue in result["issues"]
        )
        assert result["comparison"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_file",
        "missing_report",
        "file_only",
        "report_only",
        "both",
        "boolean_count",
        "float_count",
        "status",
        "changed_metrics",
        "all_skipped",
    ],
)
def test_audit_derives_training_activity_instead_of_trusting_summaries(study, mutation):
    root = study / "seed-42/rl"
    activity = read(root / "training_activity.json")
    report = read(root / "test_results.json")
    if mutation == "missing_file":
        (root / "training_activity.json").unlink()
    elif mutation == "missing_report":
        del report["training_activity"]
        write_json(root / "test_results.json", report)
    elif mutation in ("changed_metrics", "all_skipped"):
        progress = read(root / "training_metrics.json")
        for entry in progress[: 1 if mutation == "changed_metrics" else 2]:
            entry["metrics"].update({"train/updated": 0.0, "train/datums": 0.0})
        write_json(root / "training_metrics.json", progress)
        if mutation == "all_skipped":
            # Even mutually consistent summaries cannot authorize a run that
            # the native producer must stop before held-out evaluation.
            _refresh_activity(root)
    else:
        if mutation == "boolean_count":
            activity["unknown_update_batches"] = False  # False == 0 in Python
        elif mutation == "float_count":
            activity["observed_optimizer_updates"] = 2.0  # 2.0 == 2 in Python
        elif mutation == "status":
            activity["status"] = "unknown"
        else:
            activity["observed_optimizer_updates"] = 200
        if mutation != "report_only":
            write_json(root / "training_activity.json", activity)
        if mutation != "file_only":
            report["training_activity"] = activity
            write_json(root / "test_results.json", report)
    result = asyncio.run(audit_study(study))
    assert not result["passed"] and not result["complete"]
    assert result["comparison"] is None
    assert any(
        issue.get("arm") == "rl" and issue.get("seed") == 42
        for issue in result["issues"]
    )
    assert not any(
        item["arm"] == "rl" and item["seed"] == 42 for item in result["evidence"]
    )


@pytest.mark.parametrize("recovered", [False, True])
def test_audit_preserves_mixed_and_unknown_training_activity(study, recovered):
    from stateset_agents.training.river_progress import RiverTrainingProgress

    root = study / "seed-42/rl"
    if recovered:
        write_json(root / "training_metrics.json", [])
        ledger = RiverTrainingProgress(
            root,
            steps=2,
            run_manifest_hash=content_hash(read(root / "run_manifest.json")),
        )
        ledger.reconcile(2)
    else:
        progress = read(root / "training_metrics.json")
        progress[1]["metrics"].update({"train/updated": 0.0, "train/datums": 0.0})
        write_json(root / "training_metrics.json", progress)
    activity = _refresh_activity(root)
    result = asyncio.run(audit_study(study))
    assert result["passed"]
    entry = next(
        item
        for item in result["evidence"]
        if item["seed"] == 42 and item["arm"] == "rl"
    )
    assert entry["training_activity"] == activity
    assert activity["observed_optimizer_updates"] == (0 if recovered else 1)
    assert activity["unknown_update_batches"] == (2 if recovered else 0)
    assert activity["observed_skipped_batches"] == (0 if recovered else 1)
    assert activity["status"] == ("unknown" if recovered else "updates_observed")


@pytest.mark.parametrize("arm", ["base", "sft", "rejection_sft"])
def test_evaluation_only_reports_cannot_claim_native_rl_activity(study, arm):
    path = study / f"seed-42/{arm}/test_results.json"
    report = read(path)
    report["training_activity"] = read(study / "seed-42/rl/training_activity.json")
    write_json(path, report)
    result = asyncio.run(audit_study(study))
    assert not result["passed"]
    assert any(
        issue.get("arm") == arm and "cannot claim native RL" in issue["reason"]
        for issue in result["issues"]
    )


def test_cli_plan_and_strict_incomplete_audit(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "plan-agent-study",
            "--output",
            str(tmp_path),
            "--zero-update-patience",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert read(tmp_path / "study_plan.json")["config"]["zero_update_patience"] == 3
    result = runner.invoke(
        app,
        ["benchmark", "audit-agent-study", "--study-dir", str(tmp_path), "--strict"],
    )
    assert result.exit_code == 1, result.output
    assert not read(tmp_path / "study_audit.json")["passed"]


def test_packaged_entrypoint_prepares_offline_without_examples_import(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stateset_agents.training.river_refund",
            "--benchmark",
            "refund-policy-v2",
            "--dry-run",
            "--train-count",
            "8",
            "--validation-count",
            "8",
            "--test-count",
            "8",
            "--output",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run_manifest.json").exists()
