"""Train and evaluate executed refund actions with River's native RL engine.

Prepare the split files without contacting River:
    python -m stateset_agents.training.river_refund --dry-run --output outputs/refund-seed42

Run a paid experiment (Python >=3.12, pip install -e '.[river]'):
    python -m stateset_agents.training.river_refund --seed 42 --steps 20

Validation chooses the checkpoint. The test split is evaluated only afterwards.
Repeat with several seeds and compare against base/SFT checkpoints using the same
environment and token limits. A dry run is not evidence of a learning improvement.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import os
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.refund_environment import (
    RefundEnvironment,
    refund_benchmark,
)
from stateset_agents.core.environments.refund_policy_environment import (
    FAMILIES as REFUND_POLICY_FAMILIES,
)
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.evaluation.agent_runs import content_hash, evaluation_report
from stateset_agents.evaluation.checkpoint_selection import (
    require_training_progress,
    validate_validation_cases,
)
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.refund_trace import (
    audit_refund_traces,
    select_replayed_validation_checkpoint,
)
from stateset_agents.remote.river_environment import (
    river_environment_factory,
    trajectory_case_identity,
    trajectory_environment_trace,
    trajectory_truncation,
)
from stateset_agents.remote.river_rl import atomic_json, exclusive_run
from stateset_agents.remote.river_runtime import (
    require_model_access,
    require_native_runtime,
)
from stateset_agents.remote.rollout_budget import RolloutAdmissionBudget
from stateset_agents.training.river_progress import (
    DEFAULT_ZERO_UPDATE_PATIENCE,
    RiverTrainingProgress,
    summarize_training_activity,
    zero_update_stop_step,
)
from stateset_agents.utils.async_calls import closing_async_generator, run_sync_owned

logger = logging.getLogger(__name__)

TRAJECTORY_TOKEN_LIMIT = 1024
TRUNCATION_POLICY = {"train": "reward"}
EVALUATION_SETTINGS = {
    "max_turns": 4,
    "max_generated_tokens": TRAJECTORY_TOKEN_LIMIT,
    "max_context_tokens": 8192,
    "max_turn_tokens": 256,
    "temperature": 0,
    "top_p": 1.0,
    "top_k": -1,
    "group_size": 1,
    "thinking": False,
    "truncation_reward": -1.0,
}

BENCHMARKS: dict[
    str,
    tuple[
        type[RefundEnvironment] | type[RefundPolicyEnvironment],
        Callable[[str, int, int], list[dict[str, Any]]],
    ],
] = {
    "refund-v1": (RefundEnvironment, refund_benchmark),
    "refund-policy-v2": (RefundPolicyEnvironment, refund_policy_benchmark),
}


def run_manifest(args: argparse.Namespace, splits: dict[str, Any]) -> dict[str, Any]:
    """Describe the exact experiment before any provider calls."""
    from stateset_agents.remote.river import _checkpoint_from_pointer

    checkpoint = _checkpoint_from_pointer(
        args.checkpoint, base_model=args.base_model, lora_rank=16
    )
    manifest = {
        "schema_version": 1,
        "holdout_protocol": "single_attempt_v1",
        "implementation": package_implementation(),
        "environment": getattr(args, "benchmark", "refund-v1"),
        "base_model": args.base_model,
        "seed": args.seed,
        "steps": args.steps,
        "concurrency": args.concurrency,
        "max_staleness": args.max_staleness,
        "learning_rate": args.learning_rate,
        "checkpoint": checkpoint,
        "evaluate_only": args.evaluate_only,
        "evaluation_settings": EVALUATION_SETTINGS,
        "split_hashes": {name: content_hash(rows) for name, rows in splits.items()},
    }
    if getattr(args, "collect_only", False):
        manifest["collection"] = {"group_size": 8, "temperature": 1.0}
    elif not args.evaluate_only:
        manifest["training_settings"] = {
            "loss": "cispo",
            "optimizer": "adam",
            "advantage": "group_centered",
            "normalize": "token",
            "groups_per_step": 4,
            "group_size": 8,
            "lora_rank": 16,
            "truncation_policy": TRUNCATION_POLICY,
            "zero_update_patience": getattr(
                args, "zero_update_patience", DEFAULT_ZERO_UPDATE_PATIENCE
            ),
        }
    if getattr(args, "rollout_token_budget", None) is not None:
        manifest["rollout_token_budget"] = args.rollout_token_budget
    return manifest


def _validate_run_config(args: argparse.Namespace) -> None:
    """Reject invalid native settings before preparing artifacts or SDK work."""
    if not isinstance(args.base_model, str) or not args.base_model.strip():
        raise ValueError("base_model must be nonempty text")
    benchmark = getattr(args, "benchmark", "refund-v1")
    if not isinstance(benchmark, str) or benchmark not in BENCHMARKS:
        raise ValueError("Unknown refund benchmark")
    for name, minimum in (
        ("steps", 1),
        ("concurrency", 1),
        ("seed", 0),
        ("max_staleness", 0),
    ):
        value = getattr(args, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rate = args.learning_rate
    try:
        valid_rate = type(rate) in (int, float) and math.isfinite(rate) and rate > 0
    except OverflowError:
        valid_rate = False
    if not valid_rate:
        raise ValueError("learning_rate must be finite and positive")
    for name in ("evaluate_only", "collect_only", "dry_run"):
        if type(getattr(args, name, False)) is not bool:
            raise ValueError(f"{name} must be boolean")
    collect = getattr(args, "collect_only", False)
    if collect and (args.evaluate_only or benchmark != "refund-policy-v2"):
        raise ValueError(
            "Collection requires refund-policy-v2 and cannot evaluate test"
        )
    limit = getattr(args, "rollout_token_budget", None)
    if limit is not None and (type(limit) is not int or limit < TRAJECTORY_TOKEN_LIMIT):
        raise ValueError("rollout_token_budget must be an integer >= 1024")
    zero_update_stop_step(
        [], getattr(args, "zero_update_patience", DEFAULT_ZERO_UPDATE_PATIENCE)
    )


def _log_dry_run(args: argparse.Namespace, splits: dict[str, Any]) -> None:
    """Validate cases and report a dry run without SDK inspection or requests."""
    _validate_run_splits(args, splits)
    logger.info(
        "Validated dry run: %s train, %s validation, %s test cases for %s",
        len(splits.get("train", [])),
        len(splits.get("validation", [])),
        len(splits.get("test", [])),
        args.output,
    )


def _validate_run_splits(args: argparse.Namespace, splits: dict[str, Any]) -> None:
    """Require well-formed cases with disjoint identities across known splits."""
    benchmark = getattr(args, "benchmark", "refund-v1")
    collect = getattr(args, "collect_only", False)
    if not isinstance(splits, dict) or set(splits) - {"train", "validation", "test"}:
        raise ValueError("Cases must use only train, validation, and test split names")
    required = (
        {"train"}
        if collect
        else {"test"} if args.evaluate_only else {"train", "validation", "test"}
    )
    if any(
        not isinstance(splits.get(name), list) or not splits[name] for name in required
    ):
        raise ValueError("Missing or empty required cases for the requested run mode")
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        if not isinstance(rows, list):
            raise ValueError(f"{split} cases must be a list")
        for index, row in enumerate(rows):
            try:
                BENCHMARKS[benchmark][0].validate_scenario(row)
            except ValueError as exc:
                raise ValueError(f"Invalid {split} case {index}: {exc}") from exc
            if (
                benchmark == "refund-policy-v2"
                and row.get("family") not in REFUND_POLICY_FAMILIES
            ):
                raise ValueError(
                    f"Invalid {split} case {index}: missing or unknown policy family"
                )
            identity = row["order_id"]
            if identity in seen:
                raise ValueError(
                    f"Duplicate case identity in {seen[identity]} and {split}; splits must be disjoint"
                )
            seen[identity] = split


def prepare_run(args: argparse.Namespace, splits: dict[str, Any]) -> None:
    """Bind a locked directory to one experiment; preserve completed evidence."""
    _validate_run_config(args)
    _validate_run_splits(args, splits)
    manifest = run_manifest(args, splits)
    path = args.output / "run_manifest.json"
    if path.exists():
        if content_hash(json.loads(path.read_text())) != content_hash(manifest):
            raise ValueError(
                "Output directory belongs to a different experiment; use a new --output"
            )
    elif any(p.name != ".rl.lock" for p in args.output.iterdir()):
        raise ValueError("Existing output has no run manifest; use a new --output")
    if not args.dry_run and (args.output / "test_results.json").exists():
        raise ValueError("Test evaluation already completed; use its saved evidence")
    if not args.dry_run and (args.output / "test_attempt.json").exists():
        raise ValueError(
            "Test split is sealed by an earlier attempt; preserve its evidence"
        )
    if not args.dry_run and (args.output / "training_candidates.json").exists():
        raise ValueError(
            "Collection output already exists; filter its saved candidates or use a new --output"
        )
    for split, rows in splits.items():
        destination = args.output / f"{split}.json"
        if destination.exists() and json.loads(destination.read_text()) != rows:
            raise ValueError(f"Saved {split} cases have changed; refusing to resume")
    if getattr(args, "rollout_token_budget", None) is not None:
        RolloutAdmissionBudget(
            args.output / "rollout_budget.json",
            limit=args.rollout_token_budget,
            trajectory_tokens=TRAJECTORY_TOKEN_LIMIT,
            fingerprint=content_hash(manifest),
            create=not path.exists(),
        )
    atomic_json(path, manifest)
    for split, rows in splits.items():
        atomic_json(args.output / f"{split}.json", rows)


def _run_implementation(directory: Path) -> dict[str, Any]:
    """Reject source drift before sampling or opening provider sessions."""
    implementation = package_implementation()
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("implementation") != implementation:
        raise ValueError(
            "Run implementation differs from installed sources; use a new output directory"
        )
    return implementation


def _validated_run_inputs(
    args: argparse.Namespace, splits: dict[str, Any]
) -> tuple[argparse.Namespace, dict[str, Any], dict[str, Any]]:
    """Snapshot caller inputs and match them to the locked, prepared experiment.

    The caller owns the directory lock. Validation never repairs or rewrites
    evidence, and must run before SDK imports or provider resource creation.
    """
    args, splits = copy.deepcopy((args, splits))
    _validate_run_config(args)
    implementation = _run_implementation(args.output)
    expected = run_manifest(args, splits)
    manifest = json.loads((args.output / "run_manifest.json").read_text())
    if content_hash(manifest) != content_hash(expected):
        raise ValueError(
            "Run configuration or cases differ from the prepared experiment"
        )
    _validate_run_splits(args, splits)
    # Resolve once: later changes to a pointer cannot change the model loaded
    # while leaving its recorded identity unchanged.
    args.checkpoint = expected["checkpoint"]
    for split, digest in expected["split_hashes"].items():
        try:
            saved = json.loads((args.output / f"{split}.json").read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Saved {split} cases are missing or unreadable") from exc
        if content_hash(saved) != digest:
            raise ValueError(f"Saved {split} cases differ from the prepared experiment")
    if (args.output / "test_attempt.json").exists():
        raise ValueError(
            "Test split is sealed by an earlier attempt; preserve its evidence"
        )
    if (args.output / "test_results.json").exists():
        raise ValueError("Test evaluation already completed; use its saved evidence")
    if (args.output / "training_candidates.json").exists():
        raise ValueError(
            "Collection output already exists; preserve its saved candidates"
        )
    return args, splits, implementation


async def campaign(
    model: Any,
    eval_session: Any,
    renderer: Any,
    args: argparse.Namespace,
    splits: dict[str, Any],
) -> None:
    """Run native checkpointed training, select on validation, then test once."""
    _validate_run_config(args)
    if getattr(args, "dry_run", False):
        _log_dry_run(args, splits)
        return
    args, splits, implementation = _validated_run_inputs(args, splits)
    from river_client import Checkpoint, rl

    training_activity: dict[str, Any] | None = None
    benchmark = getattr(args, "benchmark", "refund-v1")
    admission = (
        RolloutAdmissionBudget(
            args.output / "rollout_budget.json",
            limit=args.rollout_token_budget,
            trajectory_tokens=TRAJECTORY_TOKEN_LIMIT,
            fingerprint=content_hash(run_manifest(args, splits)),
        )
        if getattr(args, "rollout_token_budget", None) is not None
        else None
    )

    async def reserve_trajectory() -> None:
        if admission is not None:
            # Keep durable filesystem writes off the sampling event loop.
            await asyncio.to_thread(admission.reserve)

    environment = river_environment_factory(
        BENCHMARKS[benchmark][0],
        before_reset=reserve_trajectory if admission else None,
        truncation_reward=EVALUATION_SETTINGS["truncation_reward"],
        record_trace=True,
    )
    budget = rl.Budget(
        **{
            key: value
            for key, value in EVALUATION_SETTINGS.items()
            if key.startswith("max_")
        }
    )

    def evaluation_engine(checkpoint: Any, variant: str) -> Any:
        return rl.RolloutEngine(
            rl.CheckpointSampler(
                eval_session,
                base_model=args.base_model,
                checkpoint=checkpoint,
                tokenizer=renderer.tokenizer,
            ),
            env=environment,
            renderer=renderer,
            budget=budget,
            schedule=rl.Schedule(concurrency=4),
            temperature=EVALUATION_SETTINGS["temperature"],
            top_p=EVALUATION_SETTINGS["top_p"],
            top_k=EVALUATION_SETTINGS["top_k"],
            seed=args.seed,
        )

    async def evaluate_test(checkpoint: Any, validation_step: int | None) -> None:
        attempt_path = args.output / "test_attempt.json"
        if attempt_path.exists():
            raise ValueError("Test split is sealed by an earlier attempt")
        manifest = json.loads((args.output / "run_manifest.json").read_text())
        attempt = {
            "schema_version": 1,
            "run_manifest_hash": content_hash(manifest),
            "test_split_hash": content_hash(splits["test"]),
            "selected_checkpoint": asdict(checkpoint),
            "selected_validation_step": validation_step,
        }
        # Durable before even constructing the test engine. An uncertain attempt
        # consumes the holdout, including failures before a report is published.
        atomic_json(attempt_path, attempt)
        engine = evaluation_engine(checkpoint, "test")
        test_cases = {row["order_id"]: row for row in splits["test"]}
        outcomes = []
        async with closing_async_generator(
            engine.rollout(
                splits["test"],
                group_size=EVALUATION_SETTINGS["group_size"],
                completion=rl.GroupCompletion(mode="wait", min_members=1),
            )
        ) as groups:
            async for group in groups:
                for trajectory in group:
                    outcomes.append(
                        {
                            **trajectory_case_identity(trajectory, cases=test_cases),
                            "environment_trace": trajectory_environment_trace(
                                trajectory
                            ),
                            "reward": trajectory.reward,
                            "success": trajectory.metrics.get("task_success", 0) == 1,
                            "policy_violations": trajectory.metrics.get(
                                "policy_violations", 0
                            ),
                            "tool_calls": trajectory.metrics.get("tool_calls", 0),
                            "generated_tokens": trajectory.generated_tokens,
                            "elapsed_seconds": trajectory.elapsed,
                            "truncated": trajectory_truncation(trajectory),
                        }
                    )
        report = evaluation_report(
            environment=benchmark,
            base_model=args.base_model,
            seed=args.seed,
            cases=test_cases,
            settings=EVALUATION_SETTINGS,
            implementation=implementation,
            checkpoint=asdict(checkpoint),
            outcomes=outcomes,
            case_families=(
                {row["order_id"]: row["family"] for row in splits["test"]}
                if benchmark == "refund-policy-v2"
                else None
            ),
        )
        manifest_path = args.output / "run_manifest.json"
        if manifest_path.exists():
            report["run_manifest_hash"] = content_hash(
                json.loads(manifest_path.read_text())
            )
        report["selected_validation_step"] = validation_step
        if training_activity is not None:
            report["training_activity"] = dict(training_activity)
        report["test_attempt_hash"] = content_hash(attempt)
        if admission is not None:
            report["rollout_budget"] = admission.snapshot()
        trace_audit = await audit_refund_traces(report, splits["test"])
        if not trace_audit["passed"]:
            atomic_json(
                args.output / "test_replay_failure.json",
                {
                    "schema_version": 1,
                    "status": "rejected",
                    "candidate_report": report,
                    "replay_audit": trace_audit,
                },
            )
            raise ValueError(
                "Test trace replay failed; rejected evidence saved to test_replay_failure.json"
            )
        atomic_json(args.output / "test_results.json", report)

    if getattr(args, "collect_only", False):
        if benchmark != "refund-policy-v2":
            raise ValueError(
                "Candidate collection requires --benchmark refund-policy-v2"
            )
        checkpoint = await run_sync_owned(
            model.save_weights, "refund-collection", mode="inference"
        )
        engine = rl.RolloutEngine(
            rl.CheckpointSampler(
                eval_session,
                base_model=args.base_model,
                checkpoint=checkpoint,
                tokenizer=renderer.tokenizer,
            ),
            env=environment,
            renderer=renderer,
            budget=budget,
            schedule=rl.Schedule(concurrency=args.concurrency),
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            seed=args.seed,
        )
        cases = {row["order_id"]: row for row in splits["train"]}
        collection = {
            "schema_version": 1,
            "environment": benchmark,
            "seed": args.seed,
            "base_model": args.base_model,
            "checkpoint": asdict(checkpoint),
            "case_hashes": {key: content_hash(row) for key, row in cases.items()},
            "group_size": 8,
            "temperature": 1.0,
            "settings": {**EVALUATION_SETTINGS, "group_size": 8, "temperature": 1.0},
            "expected_candidates": len(cases) * 8,
            "complete": False,
            "candidates": [],
        }
        destination = args.output / "training_candidates.json"
        manifest_path = args.output / "run_manifest.json"
        if manifest_path.exists():
            collection["run_manifest_hash"] = content_hash(
                json.loads(manifest_path.read_text())
            )
        atomic_json(destination, collection)
        async with closing_async_generator(
            engine.rollout(
                splits["train"],
                group_size=8,
                completion=rl.GroupCompletion(mode="wait", min_members=8),
            )
        ) as groups:
            async for group in groups:
                for trajectory in group:
                    collection["candidates"].append(
                        {
                            **trajectory_case_identity(trajectory, cases=cases),
                            "environment_trace": trajectory_environment_trace(
                                trajectory
                            ),
                            "messages": trajectory.messages,
                            "truncated": trajectory_truncation(trajectory),
                            "reported_reward": trajectory.reward,
                        }
                    )
                atomic_json(destination, collection)
        counts = Counter(candidate["case_id"] for candidate in collection["candidates"])
        if counts != dict.fromkeys(cases, 8):
            raise RuntimeError(
                "Incomplete candidate collection; saved partial candidates remain available for filtering"
            )
        collection["complete"] = True
        if admission is not None:
            collection["rollout_budget"] = admission.snapshot()
        atomic_json(destination, collection)
        return

    if args.evaluate_only:
        checkpoint = await run_sync_owned(
            model.save_weights, "refund-evaluation", mode="inference"
        )
        await evaluate_test(checkpoint, None)
        return

    validation_path = args.output / "validation_results.json"
    evaluations: list[dict[str, Any]] = (
        json.loads(validation_path.read_text()) if validation_path.exists() else []
    )
    validation_cases = {row["order_id"]: row for row in splits["validation"]}
    validation_lock = asyncio.Lock()

    def write_evaluation(result: Any) -> None:
        evidence = {
            "step": result.step,
            "checkpoint": asdict(result.checkpoint),
            "metrics": result.metrics,
            "case_hashes": {
                key: content_hash(row) for key, row in validation_cases.items()
            },
            "outcomes": [
                {
                    **trajectory_case_identity(trajectory, cases=validation_cases),
                    "environment_trace": trajectory_environment_trace(trajectory),
                    "reward": trajectory.reward,
                }
                for trajectory in result.trajectories
            ],
        }
        validate_validation_cases(evidence, cases=validation_cases)
        # Recovery may replay the sink after its prior write reached disk.
        # Keep one current evaluation per step, including when the checkpoint
        # changed after an uncommitted suffix was replayed.
        evaluations[:] = [
            entry for entry in evaluations if entry["step"] != result.step
        ]
        evaluations.append(evidence)
        atomic_json(args.output / "validation_results.json", evaluations)

    async def log_evaluation(result: Any) -> None:
        # River cancels evaluator tasks during shutdown. Own the worker write
        # until it finishes, and serialize callbacks with recovery reconciliation.
        async with validation_lock:
            await run_sync_owned(write_evaluation, result)

    evaluator = rl.Evaluator(
        splits["validation"],
        engine_factory=evaluation_engine,
        every=1,
        group_size=1,
        final_group_size=1,
        sink=log_evaluation,
    )
    trainer = rl.AsyncTrainer(
        engine=rl.RolloutEngine(
            model,
            env=environment,
            renderer=renderer,
            budget=budget,
            schedule=rl.Schedule(concurrency=args.concurrency),
            seed=args.seed,
        ),
        optimizer=rl.Adam(lr=args.learning_rate),
        advantage=rl.GroupCentered(),
        truncation=rl.Truncation(**TRUNCATION_POLICY),
        completion=rl.GroupCompletion(mode="wait"),
        normalize="token",
        loss="cispo",
        groups_per_step=4,
        group_size=8,
        max_staleness=args.max_staleness,
        checkpoint=rl.Checkpointing(
            args.output / "training", weights_every=1, on_signal=("SIGINT", "SIGTERM")
        ),
        evaluator=evaluator,
        run_config={"environment": benchmark, "seed": args.seed},
    )
    manifest_hash = content_hash(
        json.loads((args.output / "run_manifest.json").read_text())
    )
    progress = RiverTrainingProgress(
        args.output, steps=args.steps, run_manifest_hash=manifest_hash
    )

    def require_update_activity() -> None:
        patience = getattr(args, "zero_update_patience", DEFAULT_ZERO_UPDATE_PATIENCE)
        stopped_at = zero_update_stop_step(progress.records, patience)
        if stopped_at is None:
            return
        atomic_json(
            args.output / "training_activity.json",
            summarize_training_activity(progress.records),
        )
        atomic_json(
            args.output / "training_stop.json",
            {
                "schema_version": 1,
                "reason": "consecutive_skipped_optimizer_updates",
                "step": stopped_at,
                "patience": patience,
                "run_manifest_hash": manifest_hash,
                "training_progress_hash": content_hash(progress.records),
            },
        )
        raise ValueError(
            f"Stopped after {patience} consecutive skipped optimizer updates "
            f"at batch {stopped_at}; test split remains sealed. Check reward "
            "variation, rollout diversity, truncation and staleness masks."
        )

    async def after_recovery(completed_batches: int) -> None:
        progress.reconcile(completed_batches)
        async with validation_lock:
            evaluations[:] = [
                entry for entry in evaluations if entry["step"] <= completed_batches
            ]
            await run_sync_owned(atomic_json, validation_path, evaluations)
        require_update_activity()

    async with closing_async_generator(
        trainer.run(splits["train"], steps=args.steps, after_recovery=after_recovery)
    ) as training_steps:
        async for step in training_steps:
            progress.observe(step.n, step.metrics)
            require_update_activity()
            logger.info("Completed training batch %s", step.n)
    require_training_progress(
        progress.records,
        steps=args.steps,
        recovery_receipts=progress.receipts,
        run_manifest_hash=manifest_hash,
    )
    training_activity = summarize_training_activity(progress.records)
    atomic_json(args.output / "training_activity.json", training_activity)
    if training_activity["status"] == "no_updates_observed":
        raise ValueError(
            "No optimizer updates occurred in the completed training batches; "
            "test split remains sealed. Check reward variation, rollout diversity, "
            "truncation and staleness masks."
        )
    best = await select_replayed_validation_checkpoint(
        evaluations,
        steps=args.steps,
        cases=validation_cases,
        environment=benchmark,
        truncation_reward=EVALUATION_SETTINGS["truncation_reward"],
    )
    await evaluate_test(Checkpoint(**best["checkpoint"]), best["step"])


def main() -> None:
    """Prepare deterministic splits, optionally running the paid experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/river-refund"))
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), default="refund-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-staleness", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--zero-update-patience",
        type=int,
        default=DEFAULT_ZERO_UPDATE_PATIENCE,
        help="Stop after this many consecutive observed skipped optimizer updates (default: 5; 0 disables)",
    )
    parser.add_argument(
        "--rollout-token-budget",
        type=int,
        help="Persistent cap on reserved trajectory output tokens across all rollout phases; not a billing cap",
    )
    parser.add_argument(
        "--checkpoint",
        help="Starting river:// URI or StateSet checkpoint pointer directory",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Score base or --checkpoint without optimizer updates",
    )
    mode.add_argument(
        "--collect-only",
        action="store_true",
        help="Sample eight training trajectories per case without optimizer updates",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        _validate_run_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.train_count, args.validation_count, args.test_count) < 1:
        parser.error("train-count, validation-count, and test-count must be positive")
    from stateset_agents.remote.river import _checkpoint_from_pointer

    args.checkpoint = _checkpoint_from_pointer(
        args.checkpoint, base_model=args.base_model, lora_rank=16
    )
    splits = {
        split: BENCHMARKS[args.benchmark][1](split, count, args.seed)
        for split, count in (
            ("train", args.train_count),
            ("validation", args.validation_count),
            ("test", args.test_count),
        )
    }
    with exclusive_run(args.output):
        prepare_run(args, splits)
        execute_run(args, splits)


def execute_run(args: argparse.Namespace, splits: dict[str, Any]) -> None:
    """Run only after validating the locked experiment directory."""
    _validate_run_config(args)
    if getattr(args, "dry_run", False):
        _log_dry_run(args, splits)
        return
    args, splits, _ = _validated_run_inputs(args, splits)
    if getattr(args, "rollout_token_budget", None) is not None:
        RolloutAdmissionBudget(
            args.output / "rollout_budget.json",
            limit=args.rollout_token_budget,
            trajectory_tokens=TRAJECTORY_TOKEN_LIMIT,
            fingerprint=content_hash(run_manifest(args, splits)),
        ).ensure_available()
    require_native_runtime()
    import river_client as river
    from river_client.renderers import get_renderer

    from stateset_agents.remote.river import _checkpoint_from_pointer

    initial_uri = _checkpoint_from_pointer(
        args.checkpoint, base_model=args.base_model, lora_rank=16
    )
    with closing(river.Client(api_key=os.environ["RIVER_API_KEY"])) as client:
        require_model_access(client, args.base_model)
        renderer = get_renderer(args.base_model, thinking=False)
        with (
            client.session(experiment="stateset-refund", role="train") as session,
            client.session(experiment="stateset-refund", role="eval") as evaluation,
        ):
            model = session.create_model(
                base_model=args.base_model,
                tokenizer=renderer.tokenizer,
                lora=river.LoraConfig(rank=16, seed=args.seed),
                checkpoint=(
                    river.Checkpoint(
                        path=initial_uri, step=0, checkpoint_type="inference"
                    )
                    if initial_uri
                    else None
                ),
            )
            asyncio.run(campaign(model, evaluation, renderer, args, splits))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
