"""Offline experiment planning and provenance checks for a four-arm River study.

Plans do not execute commands or authorize spend. Local hashes establish internal
consistency, not independent attestation that a remote service performed work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.refund_policy_environment import (
    refund_policy_benchmark,
)
from stateset_agents.data.refund_demonstrations import (
    _load_data,
    _select_refund_candidates,
    _selection_summary,
    replay_refund_candidate,
)
from stateset_agents.evaluation.agent_runs import (
    compare_study,
    content_hash,
    validate_report,
)
from stateset_agents.evaluation.checkpoint_selection import (
    require_training_progress,
)
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.refund_trace import (
    audit_refund_traces,
    select_replayed_validation_checkpoint,
)
from stateset_agents.remote.river_rl import atomic_json, exclusive_run
from stateset_agents.remote.rollout_budget import RolloutAdmissionBudget
from stateset_agents.training.river_progress import (
    DEFAULT_ZERO_UPDATE_PATIENCE,
    summarize_training_activity,
    zero_update_stop_step,
)
from stateset_agents.training.river_refund import (
    EVALUATION_SETTINGS,
    TRAJECTORY_TOKEN_LIMIT,
    run_manifest,
)

ARMS = ("base", "sft", "rejection_sft", "rl")


@dataclass(frozen=True)
class StudyConfig:
    """Prespecified model, independent seeds, data sizes, and training settings."""

    base_model: str = "Qwen/Qwen3.5-9B"
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46, 47)
    train_count: int = 256
    validation_count: int = 64
    test_count: int = 128
    steps: int = 20
    concurrency: int = 8
    sft_epochs: int = 3
    sft_learning_rate: float = 2e-5
    rl_learning_rate: float = 1e-5
    rollout_token_budget: int | None = None
    zero_update_patience: int = DEFAULT_ZERO_UPDATE_PATIENCE

    def __post_init__(self) -> None:
        zero_update_stop_step([], self.zero_update_patience)
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise ValueError("base_model must be nonempty")
        if not isinstance(self.seeds, (list, tuple)) or any(
            type(s) is not int or s < 0 for s in self.seeds
        ):
            raise ValueError("seeds must be nonnegative integers")
        if len(self.seeds) < 6 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Study requires at least six unique training seeds")
        object.__setattr__(self, "seeds", tuple(sorted(self.seeds)))
        for name in (
            "train_count",
            "validation_count",
            "test_count",
            "steps",
            "concurrency",
            "sft_epochs",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(
            getattr(self, name) % 8
            for name in ("train_count", "validation_count", "test_count")
        ):
            raise ValueError(
                "Split sizes must be multiples of eight for balanced families"
            )
        for name in ("sft_learning_rate", "rl_learning_rate"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.rollout_token_budget is not None and (
            type(self.rollout_token_budget) is not int
            or self.rollout_token_budget < TRAJECTORY_TOKEN_LIMIT
        ):
            raise ValueError("rollout_token_budget must be an integer >= 1024")
        if self.rollout_token_budget is not None:
            minimum = max(self.minimum_rollout_reservations().values())
            if self.rollout_token_budget < minimum:
                raise ValueError(
                    f"rollout_token_budget must be at least {minimum} to complete "
                    "the planned rollouts, before prefetch, failures or recovery"
                )

    def minimum_rollout_reservations(self) -> dict[str, int]:
        """Return nominal output reservations; prefetch and failures add work."""
        trajectories = {
            "base": self.test_count,
            "collection": self.train_count * 8,
            "sft": self.test_count,
            "rejection_sft": self.test_count,
            "rl": self.test_count
            + self.validation_count * (self.steps + 1)
            + 32 * self.steps,
        }
        return {
            name: count * TRAJECTORY_TOKEN_LIMIT for name, count in trajectories.items()
        }

    def splits(self, seed: int) -> dict[str, Any]:
        """Generate the canonical train, validation, and test cases."""
        return {
            name: refund_policy_benchmark(name, getattr(self, name + "_count"), seed)
            for name in ("train", "validation", "test")
        }


def build_plan(config: StudyConfig) -> dict[str, Any]:
    """Build deterministic argv arrays; every path is relative to the study root."""
    cli = ["python", "-m", "stateset_agents.cli"]
    native = ["python", "-m", "stateset_agents.training.river_refund"]
    stages = []
    split_hashes = {}
    for seed in config.seeds:
        root = f"seed-{seed}"
        split_hashes[str(seed)] = {
            name: content_hash(rows) for name, rows in config.splits(seed).items()
        }
        counts = [
            "--train-count",
            str(config.train_count),
            "--validation-count",
            str(config.validation_count),
            "--test-count",
            str(config.test_count),
        ]
        common = [
            "--benchmark",
            "refund-policy-v2",
            "--seed",
            str(seed),
            "--base-model",
            config.base_model,
            "--steps",
            str(config.steps),
            "--concurrency",
            str(config.concurrency),
            "--max-staleness",
            "0",
            "--learning-rate",
            str(config.rl_learning_rate),
            "--zero-update-patience",
            str(config.zero_update_patience),
            *counts,
        ]
        if config.rollout_token_budget is not None:
            common.extend(["--rollout-token-budget", str(config.rollout_token_budget)])

        def stage(
            name: str,
            argv: list[str],
            dependencies: list[str],
            paid: bool,
            run_seed: int = seed,
        ) -> None:
            stages.append(
                {
                    "id": f"{run_seed}/{name}",
                    "argv": argv,
                    "depends_on": [f"{run_seed}/{d}" for d in dependencies],
                    "paid": paid,
                }
            )

        stage(
            "data",
            [
                *cli,
                "benchmark",
                "prepare-refund-data",
                "--seed",
                str(seed),
                *counts,
                "--output",
                f"{root}/data",
            ],
            [],
            False,
        )
        stage(
            "base",
            [*native, *common, "--evaluate-only", "--output", f"{root}/base"],
            ["data"],
            True,
        )
        stage(
            "collection",
            [*native, *common, "--collect-only", "--output", f"{root}/collection"],
            ["data"],
            True,
        )
        stage(
            "filter",
            [
                *cli,
                "benchmark",
                "filter-refund-data",
                "--data-dir",
                f"{root}/data",
                "--candidates",
                f"{root}/collection/training_candidates.json",
                "--output",
                f"{root}/filtered",
            ],
            ["collection"],
            False,
        )
        for arm, data in (("sft", "data"), ("rejection_sft", "filtered")):
            stage(
                arm + "_train",
                [
                    *cli,
                    "train-remote",
                    "--provider",
                    "river",
                    "--dataset",
                    f"{root}/{data}/train.jsonl",
                    "--base-model",
                    config.base_model,
                    "--num-epochs",
                    str(config.sft_epochs),
                    "--learning-rate",
                    str(config.sft_learning_rate),
                    "--lora-r",
                    "16",
                    "--max-length",
                    "4096",
                    "--per-device-batch-size",
                    "2",
                    "--provider-options-json",
                    json.dumps({"seed": seed, "shuffle": True}),
                    "--output-dir",
                    f"{root}/{arm}_train",
                ],
                ["data" if arm == "sft" else "filter"],
                True,
            )
            stage(
                arm,
                [
                    *native,
                    *common,
                    "--evaluate-only",
                    "--checkpoint",
                    f"{root}/{arm}_train",
                    "--output",
                    f"{root}/{arm}",
                ],
                [arm + "_train"],
                True,
            )
        stage("rl", [*native, *common, "--output", f"{root}/rl"], ["data"], True)
    plan = {
        "schema_version": 1,
        "holdout_protocol": "single_attempt_v1",
        "implementation": package_implementation(),
        "environment": "refund-policy-v2",
        "config": {**asdict(config), "seeds": list(config.seeds)},
        "evaluation_settings": EVALUATION_SETTINGS,
        "split_hashes": split_hashes,
        "stages": stages,
        "acceptance": {
            "min_seeds": len(config.seeds),
            "min_gain": 0.03,
            "alpha": 0.05,
            "comparisons": 3,
        },
        "limits": {
            "per_trajectory_generated_tokens": TRAJECTORY_TOKEN_LIMIT,
            "collection_samples_per_case": 8,
            "rl_groups_per_step": 4,
            "rl_group_size": 8,
            "sft_max_length": 4096,
            "global_generated_token_cap": None,
            "minimum_reserved_rollout_tokens_by_run": config.minimum_rollout_reservations(),
            "minimum_reserved_rollout_tokens": len(config.seeds)
            * sum(config.minimum_rollout_reservations().values()),
            "global_reserved_rollout_token_cap": (
                len(config.seeds) * 5 * config.rollout_token_budget
                if config.rollout_token_budget is not None
                else None
            ),
            "dollar_cap": None,
        },
        "execution": "Run argv without a shell, from the study directory, after reviewing paid stages and provider budget. No commands are executed by planning or auditing. Optional rollout admission budgets retain full reservations across errors, prefetch and resume. They do not cap provider billing, input tokens or optimizer work.",
    }
    return {**plan, "plan_hash": content_hash(plan)}


def prepare_study(directory: Path, config: StudyConfig) -> dict[str, Any]:
    """Publish a plan in a fresh directory without contacting a provider."""
    plan = build_plan(config)
    with exclusive_run(directory):
        if any(p.name != ".rl.lock" for p in directory.iterdir()):
            raise ValueError(
                "Study directory must be empty; existing evidence is preserved"
            )
        atomic_json(directory / "study_plan.json", plan)
    return plan


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _native_manifest(
    config: StudyConfig, seed: int, mode: str, checkpoint: str | None
) -> dict[str, Any]:
    args = argparse.Namespace(
        benchmark="refund-policy-v2",
        base_model=config.base_model,
        seed=seed,
        steps=config.steps,
        concurrency=config.concurrency,
        max_staleness=0,
        learning_rate=config.rl_learning_rate,
        zero_update_patience=config.zero_update_patience,
        checkpoint=checkpoint,
        evaluate_only=mode in ARMS[:-1],
        collect_only=mode == "collection",
        rollout_token_budget=config.rollout_token_budget,
    )
    return run_manifest(args, config.splits(seed))


def _audit_budget(
    directory: Path, run: dict[str, Any], artifact: dict[str, Any], minimum: int
) -> None:
    if "rollout_token_budget" not in run:
        return
    snapshot = RolloutAdmissionBudget(
        directory / "rollout_budget.json",
        limit=run["rollout_token_budget"],
        trajectory_tokens=TRAJECTORY_TOKEN_LIMIT,
        fingerprint=content_hash(run),
    ).snapshot()
    _require(
        artifact.get("rollout_budget") == snapshot, "Rollout budget evidence mismatch"
    )
    _require(
        snapshot["admitted_trajectories"] >= minimum,
        "Insufficient rollout reservations",
    )


async def _sft_source(
    directory: Path,
    config: StudyConfig,
    seed: int,
    arm: str,
    splits: dict[str, Any],
    data_manifest: dict[str, Any],
) -> str:
    data_dir = directory / ("data" if arm == "sft" else "filtered")
    manifest = _read(data_dir / "data_manifest.json")
    data = (data_dir / "train.jsonl").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    _require(manifest["artifacts"]["train.jsonl"] == digest, "SFT data bytes changed")
    _require(manifest["seed"] == seed, "SFT data seed mismatch")
    expected_source = "reference_policy" if arm == "sft" else "verified_model_rollout"
    _require(manifest["source"] == expected_source, "SFT data source mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    _require(bool(rows) and manifest["selected"] == len(rows), "SFT row count mismatch")
    train = {row["order_id"]: row for row in splits["train"]}
    seen = set()
    for row in rows:
        case_id = row["metadata"]["case_id"]
        _require(
            row["metadata"]["split"] == "train"
            and row["metadata"]["source"] == expected_source,
            "SFT row provenance mismatch",
        )
        _require(
            case_id in train and case_id not in seen,
            "SFT data contains held-out, unknown or duplicate case",
        )
        seen.add(case_id)
        await replay_refund_candidate(
            train[case_id],
            {
                "case_id": case_id,
                "case_hash": row["metadata"]["case_hash"],
                "messages": row["messages"],
                "truncated": None,
            },
        )
    if arm == "rejection_sft":
        collection = _read(directory / "collection/training_candidates.json")
        collection_run = _read(directory / "collection/run_manifest.json")
        _require(
            collection_run == _native_manifest(config, seed, "collection", None),
            "Collection configuration differs from plan",
        )
        _require(
            collection.get("run_manifest_hash") == content_hash(collection_run),
            "Collection is not bound to its run manifest",
        )
        _require(
            collection["base_model"] == config.base_model
            and collection["seed"] == seed
            and collection["environment"] == "refund-policy-v2"
            and collection["settings"]
            == {**EVALUATION_SETTINGS, "group_size": 8, "temperature": 1.0},
            "Collection model, seed or settings mismatch",
        )
        _require(
            collection["complete"] is True and collection["group_size"] == 8,
            "Study requires complete best-of-eight collection",
        )
        _audit_budget(
            directory / "collection", collection_run, collection, len(train) * 8
        )
        _require(
            collection["case_hashes"] == {k: content_hash(v) for k, v in train.items()},
            "Collection case mismatch",
        )
        _require(
            Counter(c["case_id"] for c in collection["candidates"])
            == dict.fromkeys(train, 8),
            "Collection coverage mismatch",
        )
        _require(
            manifest["collection_hash"] == content_hash(collection)
            and manifest["data_manifest_hash"] == content_hash(data_manifest)
            and manifest["source_checkpoint"] == collection["checkpoint"],
            "Filtered data provenance mismatch",
        )
        candidates = {
            (c["case_id"], content_hash(c["messages"]))
            for c in collection["candidates"]
            if isinstance(c, dict)
            and "messages" in c
            and "truncated" in c
            and c["truncated"] is None
            and c.get("case_hash") == content_hash(train[c["case_id"]])
        }
        _require(
            all(
                (r["metadata"]["case_id"], content_hash(r["messages"])) in candidates
                for r in rows
            ),
            "Filtered transcript is not from the recorded collection",
        )
        selected, replay_audit = await _select_refund_candidates(
            collection["candidates"],
            train,
            {
                row["order_id"]
                for split in ("validation", "test")
                for row in splits[split]
            },
        )
        _require(
            content_hash(rows) == content_hash(selected),
            "Filtered data differs from deterministic replay selection",
        )
        expected_summary = _selection_summary(train, selected, replay_audit)
        _require(
            content_hash({key: manifest.get(key) for key in expected_summary})
            == content_hash(expected_summary),
            "Filtered selection counts or family coverage differ from replay",
        )
        audit_path = data_dir / "replay_audit.json"
        _require(
            manifest["artifacts"].get("replay_audit.json")
            == hashlib.sha256(audit_path.read_bytes()).hexdigest()
            and content_hash(_read(audit_path)) == content_hash(replay_audit),
            "Filtered rejection audit differs from replay",
        )
    trained = directory / (arm + "_train")
    pointer = _read(trained / "river_checkpoint.json")
    lineage = _read(trained / "stateset_manifest.json")
    hyper = lineage["hyperparameters"]
    uri = pointer["checkpoint"]
    _require(
        type(hyper.get("seed")) is int and hyper.get("shuffle") is True,
        "Invalid training seed or shuffle",
    )
    if not isinstance(uri, str) or not uri.startswith("river://"):
        raise ValueError("Missing trained checkpoint URI")
    _require(
        pointer["base_model"] == lineage["base_model"] == config.base_model,
        "Training model mismatch",
    )
    _require(
        lineage["dataset_sha256"] == digest and lineage["dataset_rows"] == len(rows),
        "Training dataset lineage mismatch",
    )
    for key, expected in {
        "seed": seed,
        "shuffle": True,
        "lora_r": 16,
        "num_epochs": config.sft_epochs,
        "learning_rate": config.sft_learning_rate,
        "max_length": 4096,
        "per_device_batch_size": 2,
        "river_checkpoint": uri,
    }.items():
        _require(hyper.get(key) == expected, f"Training {key} differs from plan")
    _require(
        pointer["sft_options"] == {"seed": seed, "shuffle": True},
        "Checkpoint seed mismatch",
    )
    _require(
        hyper["steps"]
        == pointer["steps"]
        == config.sft_epochs * math.ceil(len(rows) / 2),
        "Incomplete SFT optimizer steps",
    )
    return uri


async def audit_study(directory: Path) -> dict[str, Any]:
    """Validate all planned evidence, then run all three corrected comparisons.

    Missing or invalid evidence fails closed and remains listed by seed/arm.
    This is a learning-evidence audit, not a cost or production certification.
    """
    plan = _read(directory / "study_plan.json")
    config = StudyConfig(**plan["config"])
    _require(
        plan == build_plan(config),
        "Study plan changed or is incompatible with this protocol",
    )
    reports: dict[str, list[Any]] = {arm: [] for arm in ARMS}
    issues = []
    sources = []
    evidence = []
    for seed in config.seeds:
        root = directory / f"seed-{seed}"
        try:
            data_manifest, splits = _load_data(root / "data")
            _require(
                data_manifest["seed"] == seed and splits == config.splits(seed),
                "Prepared data differs from planned splits",
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append({"seed": seed, "arm": "data", "reason": str(exc)})
            continue
        for arm in ARMS:
            try:
                source = None
                activity_evidence: dict[str, Any] = {}
                if arm in ("sft", "rejection_sft"):
                    source = await _sft_source(
                        root, config, seed, arm, splits, data_manifest
                    )
                run = _read(root / arm / "run_manifest.json")
                _require(
                    run == _native_manifest(config, seed, arm, source),
                    "Run configuration differs from plan",
                )
                report = _read(root / arm / "test_results.json")
                _require(isinstance(report, dict), "Test report must be an object")
                validate_report(report)
                _require(
                    all(
                        outcome.get("case_hash")
                        == report["case_hashes"][outcome["case_id"]]
                        for outcome in report["outcomes"]
                    ),
                    "Test outcomes require matching reset input hashes",
                )
                _require(
                    report.get("run_manifest_hash") == content_hash(run),
                    "Test report is not bound to this run manifest",
                )
                attempt = _read(root / arm / "test_attempt.json")
                _require(
                    attempt
                    == {
                        "schema_version": 1,
                        "run_manifest_hash": content_hash(run),
                        "test_split_hash": content_hash(splits["test"]),
                        "selected_checkpoint": report["selected_checkpoint"],
                        "selected_validation_step": report["selected_validation_step"],
                    }
                    and report.get("test_attempt_hash") == content_hash(attempt),
                    "Test attempt does not match selected checkpoint and run",
                )
                _require(
                    report["seed"] == seed
                    and report["base_model"] == config.base_model
                    and report["environment"] == plan["environment"]
                    and report.get("implementation") == plan["implementation"]
                    and report["settings"] == EVALUATION_SETTINGS,
                    "Test protocol mismatch",
                )
                _require(
                    report["case_hashes"]
                    == {row["order_id"]: content_hash(row) for row in splits["test"]},
                    "Test cases differ from plan",
                )
                _require(
                    report.get("case_families")
                    == {row["order_id"]: row["family"] for row in splits["test"]},
                    "Test family labels differ from plan",
                )
                trace_audit = await audit_refund_traces(report, splits["test"])
                _require(trace_audit["passed"], "Test trace replay failed")
                if arm == "rl":
                    evaluations = _read(root / arm / "validation_results.json")
                    best = await select_replayed_validation_checkpoint(
                        evaluations,
                        steps=config.steps,
                        cases={row["order_id"]: row for row in splits["validation"]},
                        environment=report["environment"],
                        truncation_reward=EVALUATION_SETTINGS["truncation_reward"],
                    )
                    _require(
                        report["selected_checkpoint"] == best["checkpoint"]
                        and report["selected_validation_step"] == best["step"],
                        "Test checkpoint was not selected by validation",
                    )
                    progress = _read(root / arm / "training_metrics.json")
                    receipts_path = root / arm / "recovery_receipts.json"
                    receipts = _read(receipts_path) if receipts_path.exists() else {}
                    require_training_progress(
                        progress,
                        steps=config.steps,
                        recovery_receipts=receipts,
                        run_manifest_hash=content_hash(run),
                    )
                    activity = summarize_training_activity(progress)
                    _require(
                        zero_update_stop_step(progress, config.zero_update_patience)
                        is None,
                        "RL run crossed the zero-update stopping threshold; test split should remain sealed",
                    )
                    _require(
                        activity["status"] != "no_updates_observed",
                        "RL run reports no optimizer updates; test split should remain sealed",
                    )
                    _require(
                        content_hash(_read(root / arm / "training_activity.json"))
                        == content_hash(activity)
                        and content_hash(report.get("training_activity"))
                        == content_hash(activity),
                        "Training activity differs from recorded batch metrics",
                    )
                    activity_evidence = {
                        "training_activity": activity,
                        "training_progress_hash": content_hash(progress),
                        "recovery_receipts_hash": content_hash(receipts),
                    }
                    source = report["selected_checkpoint"]["path"]
                else:
                    _require(
                        "training_activity" not in report,
                        "Evaluation-only report cannot claim native RL training activity",
                    )
                if source is not None:
                    sources.append(source)
                minimum = config.test_count
                if arm == "rl":
                    minimum += (
                        config.validation_count * (config.steps + 1) + 32 * config.steps
                    )
                _audit_budget(root / arm, run, report, minimum)
                reports[arm].append(report)
                evidence.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "report_hash": content_hash(report),
                        "run_manifest_hash": content_hash(run),
                        "training_checkpoint": source,
                        **activity_evidence,
                    }
                )
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                issues.append({"seed": seed, "arm": arm, "reason": str(exc)})
    if len(sources) != len(set(sources)):
        issues.append({"reason": "Training checkpoint reused across arms or seeds"})
    comparison = (
        compare_study(reports, min_seeds=len(config.seeds)) if not issues else None
    )
    return {
        "schema_version": 1,
        "plan_hash": plan["plan_hash"],
        "complete": not issues,
        "passed": not issues and comparison is not None and comparison["passed"],
        "issues": issues,
        "evidence": evidence,
        "comparison": comparison,
        "scope": "Local provenance and learning evidence only. Hashes do not attest provider execution or prevent coordinated edits. Live recovery and measured cost remain unverified.",
    }
