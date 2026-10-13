"""Experiment tracking with explicit ownership of every external run."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from stateset_agents.utils.async_calls import drain_owned_operation as _drain_operation

from .advanced_training_models import (
    TrainingJob,
    TrainingStatus,
    serialize_training_config,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)
_TERMINAL = {
    TrainingStatus.COMPLETED,
    TrainingStatus.FAILED,
    TrainingStatus.CANCELLED,
    TrainingStatus.TIMED_OUT,
    TrainingStatus.INTERRUPTED,
}


def _validate_metrics(metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        try:
            valid = (
                isinstance(name, str)
                and bool(name.strip())
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(
                "Experiment metrics require names and finite numeric values"
            )


class ExperimentTracker:
    """Keep local records and best-effort projections to job-owned SDK runs."""

    def __init__(self, enable_wandb: bool = True, enable_mlflow: bool = False) -> None:
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.experiments: dict[str, dict[str, Any]] = {}
        self._wandb_runs: dict[str, Any] = {}
        self._mlflow_runs: dict[str, tuple[Any, str]] = {}
        self._operation_locks: dict[str, asyncio.Lock] = {}

    def _lock(self, experiment_id: str) -> asyncio.Lock:
        return self._operation_locks.setdefault(experiment_id, asyncio.Lock())

    async def start_experiment(self, job: TrainingJob) -> str:
        """Create independent runs without blocking the event loop.

        Cancellation drains startup and closes any newly owned run as cancelled
        before returning; the caller may never have received its experiment ID.
        """
        snapshot = TrainingJob(job.job_id, copy.deepcopy(job.config))
        experiment_id = f"{snapshot.config.experiment_name}_{snapshot.job_id}"
        async with self._lock(experiment_id):
            new_experiment = experiment_id not in self.experiments
            try:
                return await _drain_operation(self._start_experiment(snapshot))
            except asyncio.CancelledError:
                if new_experiment and experiment_id in self.experiments:
                    await _drain_operation(
                        self._finish_experiment(
                            experiment_id, status=TrainingStatus.CANCELLED
                        )
                    )
                raise

    async def log_metrics(
        self, experiment_id: str, metrics: dict[str, float], step: int
    ) -> None:
        """Snapshot measurements and serialize updates to this experiment."""
        snapshot = dict(metrics)
        async with self._lock(experiment_id):
            await _drain_operation(self._log_metrics(experiment_id, snapshot, step))

    async def log_artifact(
        self, experiment_id: str, artifact_path: str, artifact_type: str = "model"
    ) -> None:
        """Upload an artifact without stalling unrelated async work."""
        async with self._lock(experiment_id):
            await _drain_operation(
                self._log_artifact(experiment_id, artifact_path, artifact_type)
            )

    async def finish_experiment(
        self,
        experiment_id: str,
        final_metrics: dict[str, float] | None = None,
        *,
        status: TrainingStatus = TrainingStatus.COMPLETED,
    ) -> None:
        """Close owned runs after all prior operations on this experiment finish."""
        snapshot = dict(final_metrics) if final_metrics is not None else None
        async with self._lock(experiment_id):
            await _drain_operation(
                self._finish_experiment(experiment_id, snapshot, status=status)
            )

    async def _attempt(
        self,
        experiment_id: str,
        backend: str,
        operation: str,
        action: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
    ) -> bool:
        try:
            result = await asyncio.to_thread(action)
            if on_success is not None:
                on_success(result)
            return True
        except Exception as exc:  # Optional SDK boundary; preserve measured training.
            self.experiments[experiment_id]["backend_errors"].append(
                {
                    "backend": backend,
                    "operation": operation,
                    "error_type": type(exc).__name__,
                }
            )
            logger.warning(
                "%s %s failed for %s (%s)",
                backend,
                operation,
                experiment_id,
                type(exc).__name__,
            )
            return False

    def _active(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.experiments[experiment_id]
        if "finished_at" in experiment:
            raise ValueError("Experiment is already finished")
        return experiment

    async def _start_experiment(self, job: TrainingJob) -> str:
        """Create independent runs; never select or finish a process-global run."""
        experiment_id = f"{job.config.experiment_name}_{job.job_id}"
        if experiment_id in self.experiments:
            raise ValueError("Experiment has already been started")
        config = copy.deepcopy(serialize_training_config(job.config))
        self.experiments[experiment_id] = {
            "experiment_id": experiment_id,
            "job_id": job.job_id,
            "config": config,
            "started_at": time.time(),
            "status": "running",
            "metrics": {},
            "artifacts": [],
            "backend_errors": [],
        }

        if self.enable_wandb and job.config.enable_wandb:

            def start_wandb() -> Any:
                # Older SDKs must not silently reuse or finish somebody else's run.
                settings = wandb.Settings(reinit="create_new")
                if settings.reinit != "create_new":
                    raise RuntimeError(
                        "Concurrent W&B tracking requires create_new support"
                    )
                run_id = hashlib.sha256(experiment_id.encode()).hexdigest()[:32]
                run = wandb.init(
                    project=job.config.experiment_name,
                    name=job.job_id,
                    id=run_id,
                    resume="never",
                    config=copy.deepcopy(config),
                    settings=settings,
                )
                if run is None or run.id != run_id:
                    raise RuntimeError("W&B did not create an independent run")
                return run

            def own_wandb(run: Any) -> None:
                if any(owned is run for owned in self._wandb_runs.values()):
                    raise RuntimeError("W&B did not create an independent run")
                self._wandb_runs[experiment_id] = run

            await self._attempt(experiment_id, "wandb", "start", start_wandb, own_wandb)

        if self.enable_mlflow and job.config.enable_mlflow:

            def start_mlflow() -> tuple[Any, str]:
                client = mlflow.tracking.MlflowClient()
                experiment = client.get_experiment_by_name(job.config.experiment_name)
                if experiment is None:
                    try:
                        remote_id = client.create_experiment(job.config.experiment_name)
                    except Exception:
                        # Another process may have created the named experiment.
                        experiment = client.get_experiment_by_name(
                            job.config.experiment_name
                        )
                        if experiment is None:
                            raise
                        remote_id = experiment.experiment_id
                else:
                    remote_id = experiment.experiment_id
                run = client.create_run(
                    remote_id,
                    tags={
                        "mlflow.runName": job.job_id,
                        "stateset.job_id": job.job_id,
                    },
                )
                run_id = run.info.run_id
                return client, run_id

            def own_mlflow(result: tuple[Any, str]) -> None:
                client, run_id = result
                if (
                    not isinstance(run_id, str)
                    or not run_id
                    or any(
                        owned_id == run_id for _, owned_id in self._mlflow_runs.values()
                    )
                ):
                    raise RuntimeError("MLflow did not create an independent run")
                self._mlflow_runs[experiment_id] = (client, run_id)

            await self._attempt(
                experiment_id, "mlflow", "start", start_mlflow, own_mlflow
            )
            if experiment_id in self._mlflow_runs:
                client, run_id = self._mlflow_runs[experiment_id]

                def log_parameters() -> None:
                    for name, value in config.items():
                        client.log_param(
                            run_id,
                            name,
                            json.dumps(value, sort_keys=True, allow_nan=False),
                        )

                await self._attempt(experiment_id, "mlflow", "params", log_parameters)
        return experiment_id

    async def _log_metrics(
        self, experiment_id: str, metrics: dict[str, float], step: int
    ) -> None:
        """Log validated measurements only to the specified experiment's runs."""
        experiment = self._active(experiment_id)
        _validate_metrics(metrics)
        if type(step) is not int or step < 0:
            raise ValueError("Metric step must be a nonnegative integer")
        for name, value in metrics.items():
            experiment["metrics"].setdefault(name, []).append(
                {"step": step, "value": value}
            )
        if experiment_id in self._wandb_runs:
            run = self._wandb_runs[experiment_id]
            await self._attempt(
                experiment_id,
                "wandb",
                "metrics",
                lambda: run.log(dict(metrics), step=step),
            )
        if experiment_id in self._mlflow_runs:
            client, run_id = self._mlflow_runs[experiment_id]
            for name, value in metrics.items():
                await self._attempt(
                    experiment_id,
                    "mlflow",
                    "metrics",
                    partial(client.log_metric, run_id, name, value, step=step),
                )

    async def _log_artifact(
        self, experiment_id: str, artifact_path: str, artifact_type: str = "model"
    ) -> None:
        """Attach a file or directory to this experiment's external runs."""
        experiment = self._active(experiment_id)
        path = Path(artifact_path).resolve()
        if not path.is_file() and not path.is_dir():
            raise ValueError(
                "Experiment artifact must be an existing file or directory"
            )
        if not isinstance(artifact_type, str) or not artifact_type.strip():
            raise ValueError("Artifact type must be nonempty text")
        experiment["artifacts"].append(
            {
                "path": str(path),
                "type": artifact_type,
                "logged_at": time.time(),
            }
        )
        if experiment_id in self._wandb_runs:

            def log_wandb() -> None:
                name = hashlib.sha256(experiment_id.encode()).hexdigest()[:16]
                artifact = wandb.Artifact(
                    f"training-{name}-{len(experiment['artifacts'])}",
                    type=artifact_type,
                )
                if path.is_dir():
                    artifact.add_dir(str(path))
                else:
                    artifact.add_file(str(path))
                self._wandb_runs[experiment_id].log_artifact(artifact)

            await self._attempt(experiment_id, "wandb", "artifact", log_wandb)
        if experiment_id in self._mlflow_runs:
            client, run_id = self._mlflow_runs[experiment_id]
            action = client.log_artifacts if path.is_dir() else client.log_artifact
            await self._attempt(
                experiment_id, "mlflow", "artifact", lambda: action(run_id, str(path))
            )

    async def _finish_experiment(
        self,
        experiment_id: str,
        final_metrics: dict[str, float] | None = None,
        *,
        status: TrainingStatus = TrainingStatus.COMPLETED,
    ) -> None:
        """Record an outcome and close owned runs; retry only failed closures."""
        status = TrainingStatus(status)
        if status not in _TERMINAL:
            raise ValueError("An experiment requires a terminal outcome to finish")
        experiment = self.experiments[experiment_id]
        if final_metrics is not None:
            _validate_metrics(final_metrics)
        if "finished_at" in experiment:
            if experiment["status"] != status.value or (
                final_metrics is not None
                and final_metrics != experiment.get("final_metrics")
            ):
                raise ValueError("Experiment outcome has already been recorded")
        else:
            if final_metrics:
                missing = {
                    name: value
                    for name, value in final_metrics.items()
                    if (
                        not experiment["metrics"].get(name)
                        or experiment["metrics"][name][-1]["value"] != value
                    )
                }
                step = max(
                    (
                        point["step"]
                        for history in experiment["metrics"].values()
                        for point in history
                    ),
                    default=0,
                )
                if missing:
                    await self._log_metrics(experiment_id, missing, step)
            experiment.update(finished_at=time.time(), status=status.value)
            if final_metrics is not None:
                experiment["final_metrics"] = dict(final_metrics)
        if experiment_id in self._wandb_runs:
            run = self._wandb_runs[experiment_id]
            await self._attempt(
                experiment_id,
                "wandb",
                "summary",
                lambda: run.summary.update(
                    {
                        **experiment.get("final_metrics", {}),
                        "stateset_status": status.value,
                    }
                ),
            )
            if await self._attempt(
                experiment_id,
                "wandb",
                "finish",
                lambda: run.finish(
                    exit_code=0 if status == TrainingStatus.COMPLETED else 1
                ),
            ):
                self._wandb_runs.pop(experiment_id)
        if experiment_id in self._mlflow_runs:
            client, run_id = self._mlflow_runs[experiment_id]
            remote_status = (
                "FINISHED"
                if status == TrainingStatus.COMPLETED
                else ("FAILED" if status == TrainingStatus.FAILED else "KILLED")
            )
            await self._attempt(
                experiment_id,
                "mlflow",
                "status",
                lambda: client.set_tag(run_id, "stateset.status", status.value),
            )
            if await self._attempt(
                experiment_id,
                "mlflow",
                "finish",
                lambda: client.set_terminated(run_id, status=remote_status),
            ):
                self._mlflow_runs.pop(experiment_id)
