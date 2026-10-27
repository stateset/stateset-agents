"""Offline replay of recorded refund actions, including failures and truncations."""

from __future__ import annotations

import copy
from typing import Any

from stateset_agents.core.environments.refund_environment import RefundEnvironment
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)
from stateset_agents.evaluation.agent_runs import content_hash, validate_report
from stateset_agents.evaluation.checkpoint_selection import (
    require_training_progress,
    select_validation_checkpoint,
)
from stateset_agents.evaluation.environment_trace import replay_environment_trace
from stateset_agents.evaluation.implementation import package_implementation

_ENVIRONMENTS = {
    "refund-v1": RefundEnvironment,
    "refund-policy-v2": RefundPolicyEnvironment,
}

VALIDATION_SELECTION_POLICY = "initial_and_unskipped_batches_v1"


async def select_replayed_validation_checkpoint(
    evaluations: Any,
    *,
    steps: int,
    cases: dict[str, Any],
    environment: str,
    truncation_reward: float = -1.0,
    training_progress: Any = None,
    recovery_receipts: dict[str, Any] | None = None,
    run_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Select only after every planned validation reward passes sandbox replay.

    All steps are checked, including checkpoints that would not win selection.
    No test cases or provider calls are used by this selection gate. When native
    training progress is supplied, confirmed skipped batches cannot promote a
    fresh resampling of unchanged weights. Unknown recovery activity remains
    eligible and explicit, rather than being invented as an update or a skip.
    """
    skipped_steps: tuple[int, ...] = ()
    if training_progress is not None:
        require_training_progress(
            training_progress,
            steps=steps,
            recovery_receipts=recovery_receipts,
            run_manifest_hash=run_manifest_hash,
        )
        skipped_steps = tuple(
            entry["step"]
            for entry in training_progress
            if isinstance(entry.get("metrics"), dict)
            and entry["metrics"].get("train/updated") == 0
        )
    selected = select_validation_checkpoint(
        evaluations, steps=steps, cases=cases, skipped_steps=skipped_steps
    )
    for evaluation in evaluations:
        for outcome in evaluation["outcomes"]:
            case_id = outcome["case_id"]
            try:
                replay = await replay_refund_trace(
                    environment,
                    cases[case_id],
                    outcome.get("environment_trace"),
                    truncation_reward=truncation_reward,
                )
                if replay["reward"] != outcome["reward"]:
                    raise ValueError("Validation reward differs from replay")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"Validation replay failed at step {evaluation['step']}, case {case_id}: {exc}"
                ) from exc
    if training_progress is None:
        return selected
    from stateset_agents.evaluation.river_progress import summarize_training_activity

    activity = summarize_training_activity(training_progress[: selected["step"]])
    if selected["step"] == 0:
        activity["status"] = "initial_checkpoint"
    selection = {
        "schema_version": 1,
        "policy": VALIDATION_SELECTION_POLICY,
        "run_manifest_hash": run_manifest_hash,
        "selected_validation_step": selected["step"],
        "selected_checkpoint": copy.deepcopy(selected["checkpoint"]),
        "eligible_validation_steps": [
            step for step in range(steps + 1) if step not in skipped_steps
        ],
        "excluded_skipped_batches": list(skipped_steps),
        "selected_training_activity": activity,
        "validation_results_hash": content_hash(evaluations),
        "training_progress_hash": content_hash(training_progress),
        "recovery_receipts_hash": content_hash(recovery_receipts or {}),
    }
    return {**selected, "selection_activity": selection}


async def replay_refund_trace(
    environment: str,
    case: dict[str, Any],
    trace: Any,
    *,
    truncation_reward: float = -1.0,
) -> dict[str, Any]:
    """Re-execute refund sandbox actions and verify every recorded transition.

    External stops remain declarations, not proof of a token/time-limit trigger.
    No model calls, SDK clients, or real refund services are used.
    """
    if environment not in _ENVIRONMENTS:
        raise ValueError("Unsupported refund trace environment")
    identity = case.get("order_id") if isinstance(case, dict) else None
    if not isinstance(identity, str) or not identity:
        raise ValueError("Missing trace or mismatched trace case identity")
    return await replay_environment_trace(
        _ENVIRONMENTS[environment],
        case,
        trace,
        case_id=identity,
        truncation_reward=truncation_reward,
    )


async def audit_refund_traces(
    report: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check all held-out outcomes against source-matched sandbox trace replay."""
    if not isinstance(report, dict):
        raise ValueError("Trace audit requires a test report object")
    validate_report(report)
    if report["environment"] not in _ENVIRONMENTS:
        raise ValueError("Unsupported refund trace environment")
    implementation = package_implementation()
    if report.get("implementation") != implementation:
        raise ValueError("Trace audit requires the report's exact implementation")
    if not isinstance(cases, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("order_id"), str)
        or not row["order_id"]
        for row in cases
    ):
        raise ValueError("Cases must be a list of identified refund scenarios")
    by_id = {row["order_id"]: row for row in cases}
    if len(by_id) != len(cases) or report["case_hashes"] != {
        key: content_hash(row) for key, row in by_id.items()
    }:
        raise ValueError("Trace audit cases differ from the evaluated cases")
    truncation_reward = report["settings"].get("truncation_reward")
    checked, issues = [], []
    for outcome in report["outcomes"]:
        case_id = outcome["case_id"]
        try:
            replay = await replay_refund_trace(
                report["environment"],
                by_id[case_id],
                outcome.get("environment_trace"),
                truncation_reward=truncation_reward,
            )
            for key in (
                "case_hash",
                "reward",
                "success",
                "policy_violations",
                "tool_calls",
                "truncated",
            ):
                if outcome.get(key) != replay[key]:
                    raise ValueError(f"Reported outcome differs from replay: {key}")
            checked.append(replay)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({"case_id": case_id, "reason": str(exc)})
    return {
        "schema_version": 1,
        "scope": "sandbox_action_consistency",
        "passed": not issues and len(checked) == len(cases),
        "report_hash": content_hash(report),
        "cases_hash": content_hash(cases),
        "implementation": implementation,
        "provider_requests": 0,
        "checked": checked,
        "issues": issues,
        "unverified": [
            "provider_execution",
            "model_identity",
            "generated_tokens",
            "elapsed_seconds",
            "external_truncation_trigger",
        ],
    }
