"""A trace-to-policy workflow with isolated held-out data and mandatory stages."""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stateset_agents.data.curation import curate_conversations
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.training.promotion import artifact_digest


async def _invoke(function: Callable[..., Any], *args: Any) -> Any:
    result = function(*args)
    return await result if inspect.isawaitable(result) else result


async def run_improvement_pipeline(
    records: list[dict[str, Any]],
    *,
    output: Path,
    initial_policy: Any,
    validation_after: str,
    test_after: str,
    supervised_train: Callable[..., Any],
    reinforcement_train: Callable[..., Any],
    evaluate: Callable[..., Any],
    export_policy: Callable[..., Any],
    reload_and_serve: Callable[..., Any],
    minimum_reward: float | None = None,
) -> dict[str, Any]:
    """Run curate → SFT → RL → held-out eval → export → reload/serve validation.

    Training callbacks receive train/validation partitions only. Evaluation loads
    the frozen baseline and candidate artifacts. Callers supply model/backend stages;
    a successful pipeline receipt proves execution, not a production quality gate.
    """
    output = output.resolve()
    bundle = curate_conversations(
        records,
        validation_after=validation_after,
        test_after=test_after,
        minimum_reward=minimum_reward,
    )
    if any(not bundle["splits"][split] for split in ("train", "validation", "test")):
        raise ValueError(
            "Improvement pipeline requires nonempty isolated train/validation/test splits"
        )
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(bundle["manifest"], indent=2, allow_nan=False) + "\n"
    )
    stages: list[dict[str, Any]] = []

    def publish() -> None:
        (output / "pipeline_status.json").write_text(
            json.dumps(
                {"schema_version": 1, "stages": stages}, indent=2, allow_nan=False
            )
            + "\n"
        )

    try:
        train_data, validation = (
            bundle["splits"]["train"],
            bundle["splits"]["validation"],
        )
        baseline_artifact = Path(
            await _invoke(export_policy, initial_policy, output / "baseline_policy")
        )
        if baseline_artifact.resolve() != (output / "baseline_policy").resolve():
            raise ValueError(
                "Baseline exporter must return the requested artifact path"
            )
        baseline_digest = artifact_digest(baseline_artifact)
        baseline_evaluation = await _invoke(
            evaluate, baseline_artifact, copy.deepcopy(bundle["splits"]["test"])
        )
        if (
            not isinstance(baseline_evaluation, dict)
            or artifact_digest(baseline_artifact) != baseline_digest
        ):
            raise ValueError("Baseline evaluation must preserve its exported policy")
        content_hash(baseline_evaluation)
        stages.append(
            {
                "stage": "baseline_evaluation",
                "status": "completed",
                "artifact_sha256": baseline_digest,
            }
        )
        publish()
        policy = await _invoke(
            supervised_train,
            initial_policy,
            copy.deepcopy(train_data),
            copy.deepcopy(validation),
        )
        if policy is None:
            raise ValueError("SFT stage did not return a policy")
        stages.append({"stage": "sft", "status": "completed"})
        publish()
        policy = await _invoke(
            reinforcement_train,
            policy,
            copy.deepcopy(train_data),
            copy.deepcopy(validation),
        )
        if policy is None:
            raise ValueError("RL stage did not return a policy")
        stages.append({"stage": "rl", "status": "completed"})
        publish()
        artifact = Path(await _invoke(export_policy, policy, output / "policy"))
        if artifact.resolve() != (output / "policy").resolve():
            raise ValueError("Policy exporter must return the requested artifact path")
        digest = artifact_digest(artifact)
        evaluation = await _invoke(
            evaluate, artifact, copy.deepcopy(bundle["splits"]["test"])
        )
        if not isinstance(evaluation, dict):
            raise ValueError("Held-out evaluation must return a JSON report")
        evaluation_hash = content_hash(evaluation)
        if artifact_digest(artifact) != digest:
            raise ValueError("Evaluation changed the exported policy")
        stages.append(
            {
                "stage": "evaluation",
                "status": "completed",
                "report_sha256": evaluation_hash,
            }
        )
        publish()
        served = await _invoke(reload_and_serve, artifact)
        if (
            not isinstance(served, dict)
            or served.get("reloaded") is not True
            or served.get("served") is not True
        ):
            raise ValueError("Reload and serving stage did not verify both operations")
        if artifact_digest(artifact) != digest:
            raise ValueError("Serving validation changed the approved artifact")
        if artifact_digest(baseline_artifact) != baseline_digest:
            raise ValueError("Training changed the frozen baseline artifact")
        stages.append({"stage": "reload_and_serve", "status": "completed"})
        report = {
            "schema_version": 1,
            "completed": True,
            "production_approved": False,
            "data_manifest_sha256": content_hash(bundle["manifest"]),
            "artifact_sha256": digest,
            "artifact": str(artifact.relative_to(output)),
            "evaluation": evaluation,
            "baseline_evaluation": baseline_evaluation,
            "baseline_artifact_sha256": baseline_digest,
            "serving": served,
            "stages": stages,
        }
        (output / "pipeline_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        publish()
        return report
    except BaseException as exc:
        stages.append(
            {
                "stage": "pipeline",
                "status": (
                    "cancelled" if type(exc).__name__ == "CancelledError" else "failed"
                ),
                "error_type": type(exc).__name__,
            }
        )
        publish()
        raise
