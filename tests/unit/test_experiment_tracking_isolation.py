"""Concurrent jobs must never log or finish through an ambient active run."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.training import experiment_tracking as tracking
from stateset_agents.training.advanced_training_orchestrator import (
    TrainingJob,
    TrainingJobSpec,
    TrainingRunResult,
    TrainingStatus,
    TrainingWorker,
)


def job(name="one"):
    return TrainingJob(
        name,
        TrainingJobSpec(
            "isolation",
            "test",
            {"nested": [1]},
            "unused",
            enable_mlflow=True,
        ),
    )


@pytest.fixture
def sdks(monkeypatch):
    wandb_runs = []

    def init(**kwargs):
        assert kwargs["settings"].reinit == "create_new"
        run = SimpleNamespace(
            id=kwargs["id"],
            log=MagicMock(),
            log_artifact=MagicMock(),
            finish=MagicMock(),
            summary={},
        )
        wandb_runs.append(run)
        return run

    wandb = SimpleNamespace(
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        init=MagicMock(side_effect=init),
        Artifact=MagicMock(side_effect=lambda *args, **kwargs: MagicMock()),
        run=object(),
        log=MagicMock(),
        save=MagicMock(),
        finish=MagicMock(),
    )
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id="experiment"
    )
    run_ids = []

    def create_run(*args, **kwargs):
        run_id = f"run-{len(run_ids) + 1}"
        run_ids.append(run_id)
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    client.create_run.side_effect = create_run
    mlflow = SimpleNamespace(
        tracking=SimpleNamespace(MlflowClient=MagicMock(return_value=client)),
        start_run=MagicMock(),
        log_params=MagicMock(),
        log_metric=MagicMock(),
        log_artifact=MagicMock(),
        end_run=MagicMock(),
    )
    monkeypatch.setattr(tracking, "wandb", wandb, raising=False)
    monkeypatch.setattr(tracking, "mlflow", mlflow, raising=False)
    monkeypatch.setattr(tracking, "WANDB_AVAILABLE", True)
    monkeypatch.setattr(tracking, "MLFLOW_AVAILABLE", True)
    return SimpleNamespace(wandb=wandb, mlflow=mlflow, runs=wandb_runs, client=client)


@pytest.mark.asyncio
async def test_interleaved_jobs_keep_metrics_artifacts_and_closures_isolated(
    sdks, tmp_path
):
    tracker = tracking.ExperimentTracker(True, True)
    ambient = sdks.wandb.run
    first = await tracker.start_experiment(job("one"))
    second = await tracker.start_experiment(job("two"))
    assert (
        sdks.wandb.init.call_args_list[0].kwargs["id"]
        != sdks.wandb.init.call_args_list[1].kwargs["id"]
    )
    assert all(
        call.kwargs["resume"] == "never" for call in sdks.wandb.init.call_args_list
    )
    await tracker.log_metrics(second, {"reward": 0.8}, 20)
    await tracker.log_metrics(first, {"reward": 0.2}, 10)
    file = tmp_path / "checkpoint.bin"
    file.write_bytes(b"model")
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "weights.bin").write_bytes(b"weights")
    await tracker.log_artifact(first, str(file))
    await tracker.log_artifact(second, str(folder))
    sdks.runs[0].log.assert_called_once_with({"reward": 0.2}, step=10)
    sdks.runs[1].log.assert_called_once_with({"reward": 0.8}, step=20)
    assert sdks.client.log_metric.call_args_list[0].args == ("run-2", "reward", 0.8)
    assert sdks.client.log_metric.call_args_list[1].args == ("run-1", "reward", 0.2)
    sdks.client.log_artifact.assert_called_once_with("run-1", str(file))
    sdks.client.log_artifacts.assert_called_once_with("run-2", str(folder))
    assert (
        sdks.runs[0].log_artifact.call_args.args[0]
        is not sdks.runs[1].log_artifact.call_args.args[0]
    )
    await tracker.finish_experiment(
        first, {"reward": 0.2}, status=TrainingStatus.COMPLETED
    )
    sdks.runs[0].finish.assert_called_once_with(exit_code=0)
    sdks.runs[1].finish.assert_not_called()
    sdks.client.set_terminated.assert_called_once_with("run-1", status="FINISHED")
    await tracker.finish_experiment(second, status=TrainingStatus.TIMED_OUT)
    sdks.runs[1].finish.assert_called_once_with(exit_code=1)
    assert sdks.runs[1].summary["stateset_status"] == "timed_out"
    assert sdks.client.set_terminated.call_args.args == ("run-2",)
    assert sdks.client.set_terminated.call_args.kwargs == {"status": "KILLED"}
    assert sdks.wandb.run is ambient
    for api in (
        sdks.wandb.log,
        sdks.wandb.save,
        sdks.wandb.finish,
        sdks.mlflow.start_run,
        sdks.mlflow.log_params,
        sdks.mlflow.log_metric,
        sdks.mlflow.log_artifact,
        sdks.mlflow.end_run,
    ):
        api.assert_not_called()


@pytest.mark.asyncio
async def test_per_job_opt_out_prevents_all_external_run_creation(sdks):
    tracker = tracking.ExperimentTracker(True, True)
    task = job()
    task.config.enable_wandb = task.config.enable_mlflow = False
    identifier = await tracker.start_experiment(task)
    await tracker.log_metrics(identifier, {"reward": 0.5}, 1)
    await tracker.finish_experiment(identifier, status=TrainingStatus.FAILED)
    sdks.wandb.init.assert_not_called()
    sdks.mlflow.tracking.MlflowClient.assert_not_called()
    assert tracker.experiments[identifier]["status"] == "failed"


@pytest.mark.asyncio
async def test_configuration_is_serializable_and_does_not_share_job_mutations(sdks):
    tracker = tracking.ExperimentTracker(True, True)
    task = job()
    identifier = await tracker.start_experiment(task)
    task.config.model_config["nested"].append(2)
    assert tracker.experiments[identifier]["config"]["model_config"] == {"nested": [1]}
    assert sdks.wandb.init.call_args.kwargs["config"]["model_config"] == {"nested": [1]}
    json.dumps(tracker.experiments[identifier])


@pytest.mark.asyncio
async def test_old_wandb_settings_do_not_reuse_or_finish_an_existing_run(sdks):
    sdks.wandb.Settings = lambda **kwargs: SimpleNamespace(reinit=True)
    tracker = tracking.ExperimentTracker(True, False)
    identifier = await tracker.start_experiment(job())
    await tracker.log_metrics(identifier, {"loss": 0.2}, 1)
    await tracker.finish_experiment(identifier, status=TrainingStatus.FAILED)
    sdks.wandb.init.assert_not_called()
    sdks.wandb.finish.assert_not_called()
    assert tracker.experiments[identifier]["backend_errors"][0]["operation"] == "start"


@pytest.mark.asyncio
async def test_failed_backend_start_cannot_fall_back_to_another_jobs_run(sdks, caplog):
    tracker = tracking.ExperimentTracker(True, False)
    await tracker.start_experiment(job("one"))
    sdks.wandb.init.side_effect = RuntimeError("private-api-token")
    second = await tracker.start_experiment(job("two"))
    await tracker.log_metrics(second, {"reward": 9.0}, 1)
    await tracker.finish_experiment(second, status=TrainingStatus.FAILED)
    sdks.runs[0].log.assert_not_called()
    sdks.runs[0].finish.assert_not_called()
    assert "private-api-token" not in repr(tracker.experiments[second])
    assert "private-api-token" not in caplog.text


@pytest.mark.asyncio
async def test_repeated_finish_retries_only_failed_backend_closure(sdks):
    tracker = tracking.ExperimentTracker(True, True)
    identifier = await tracker.start_experiment(job())
    sdks.runs[0].finish.side_effect = [RuntimeError("offline"), None]
    await tracker.finish_experiment(identifier, status=TrainingStatus.CANCELLED)
    timestamp = tracker.experiments[identifier]["finished_at"]
    await tracker.finish_experiment(identifier, status=TrainingStatus.CANCELLED)
    await tracker.finish_experiment(identifier, status=TrainingStatus.CANCELLED)
    assert sdks.runs[0].finish.call_count == 2
    sdks.client.set_terminated.assert_called_once_with("run-1", status="KILLED")
    assert tracker.experiments[identifier]["finished_at"] == timestamp
    with pytest.raises(ValueError, match="already been recorded"):
        await tracker.finish_experiment(identifier, status=TrainingStatus.COMPLETED)


@pytest.mark.asyncio
async def test_partial_mlflow_initialization_still_closes_its_owned_run(sdks):
    sdks.client.log_param.side_effect = RuntimeError("offline")
    tracker = tracking.ExperimentTracker(False, True)
    identifier = await tracker.start_experiment(job())
    await tracker.finish_experiment(identifier, status=TrainingStatus.FAILED)
    sdks.client.set_terminated.assert_called_once_with("run-1", status="FAILED")
    assert tracker.experiments[identifier]["backend_errors"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metrics,step",
    [
        ({"loss": float("nan")}, 1),
        ({"loss": True}, 1),
        ({"": 1.0}, 1),
        ({"loss": 10**400}, 1),
        ({"loss": 0.1}, -1),
        ({"loss": 0.1}, True),
    ],
)
async def test_invalid_measurements_have_no_tracking_side_effects(sdks, metrics, step):
    tracker = tracking.ExperimentTracker(True, True)
    identifier = await tracker.start_experiment(job())
    with pytest.raises(ValueError):
        await tracker.log_metrics(identifier, metrics, step)
    assert tracker.experiments[identifier]["metrics"] == {}
    sdks.runs[0].log.assert_not_called()
    sdks.client.log_metric.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_start_and_updates_after_finish_are_rejected(sdks, tmp_path):
    tracker = tracking.ExperimentTracker(True, True)
    task = job()
    identifier = await tracker.start_experiment(task)
    with pytest.raises(ValueError, match="already been started"):
        await tracker.start_experiment(task)
    with pytest.raises(ValueError, match="terminal"):
        await tracker.finish_experiment(identifier, status=TrainingStatus.RUNNING)
    with pytest.raises(ValueError, match="existing"):
        await tracker.log_artifact(identifier, str(tmp_path / "missing"))
    await tracker.finish_experiment(identifier)
    with pytest.raises(ValueError, match="finished"):
        await tracker.log_metrics(identifier, {"loss": 0.1}, 1)
    assert sdks.wandb.init.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled", "timed_out"])
async def test_worker_passes_real_terminal_outcome_to_both_backends(
    sdks, tmp_path, outcome
):
    tracker = tracking.ExperimentTracker(True, True)
    task = job()
    artifact = tmp_path / "checkpoint"
    artifact.write_bytes(b"weights")

    async def runner(task):
        if outcome == "failed":
            raise RuntimeError("training failed")
        if outcome == "cancelled":
            raise asyncio.CancelledError
        if outcome == "timed_out":
            await asyncio.Event().wait()
        return TrainingRunResult(artifact, {"loss": 0.1}, 1, 1)

    if outcome == "timed_out":
        task.config.max_runtime = 0.1
    worker = TrainingWorker("worker", runner)
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await worker.execute_job(task, tracker)
    else:
        assert await worker.execute_job(task, tracker) is (outcome == "completed")
    experiment = next(iter(tracker.experiments.values()))
    assert experiment["status"] == outcome == task.status.value
    sdks.runs[0].finish.assert_called_once_with(
        exit_code=0 if outcome == "completed" else 1
    )
    remote_status = {
        "completed": "FINISHED",
        "failed": "FAILED",
        "cancelled": "KILLED",
        "timed_out": "KILLED",
    }[outcome]
    sdks.client.set_terminated.assert_called_once_with("run-1", status=remote_status)


@pytest.mark.asyncio
async def test_cancellation_during_artifact_logging_finishes_as_cancelled(
    sdks, tmp_path
):
    tracker = tracking.ExperimentTracker(True, True)
    task = job()
    artifact = tmp_path / "checkpoint"
    artifact.write_bytes(b"weights")

    async def cancel(*args):
        task.status = TrainingStatus.CANCELLED

    tracker.log_artifact = cancel
    worker = TrainingWorker(
        "worker",
        AsyncMock(return_value=TrainingRunResult(artifact, {"loss": 0.1}, 1, 1)),
    )
    assert not await worker.execute_job(task, tracker)
    assert next(iter(tracker.experiments.values()))["status"] == "cancelled"
    sdks.runs[0].finish.assert_called_once_with(exit_code=1)


class SDKGate:
    """Hold a synchronous SDK call until the test releases its worker thread."""

    def __init__(self, delegate=None):
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.thread_id = None
        self.delegate = delegate

    def __call__(self, *args, **kwargs):
        self.thread_id = threading.get_ident()
        self.loop.call_soon_threadsafe(self.entered.set)
        try:
            if not self.release.wait(8):
                raise TimeoutError("Test did not release SDK call")
            if self.delegate is not None:
                return self.delegate(*args, **kwargs)
        finally:
            self.finished.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["wandb", "mlflow"])
@pytest.mark.parametrize("operation", ["start", "metrics", "artifact", "finish"])
async def test_slow_sdk_operations_leave_other_experiments_responsive(
    sdks, tmp_path, backend, operation
):
    tracker = tracking.ExperimentTracker(backend == "wandb", backend == "mlflow")
    first = await tracker.start_experiment(job("first"))
    second = await tracker.start_experiment(job("second"))
    artifact = tmp_path / "weights"
    artifact.write_bytes(b"weights")
    if backend == "wandb":
        target = {
            "start": sdks.wandb.init,
            "metrics": sdks.runs[0].log,
            "artifact": sdks.runs[0].log_artifact,
            "finish": sdks.runs[0].finish,
        }[operation]
    else:
        target = {
            "start": sdks.client.create_run,
            "metrics": sdks.client.log_metric,
            "artifact": sdks.client.log_artifact,
            "finish": sdks.client.set_terminated,
        }[operation]
    delegate = target.side_effect
    gate = SDKGate(delegate)

    def intercept(*args, **kwargs):
        # The fake MLflow client is shared, while the real contract uses run IDs.
        if backend == "mlflow" and operation != "start" and args[0] != "run-1":
            return delegate(*args, **kwargs) if delegate else None
        return gate(*args, **kwargs)

    target.side_effect = intercept
    operation_call = {
        "start": lambda: tracker.start_experiment(job("third")),
        "metrics": lambda: tracker.log_metrics(first, {"loss": 0.1}, 1),
        "artifact": lambda: tracker.log_artifact(first, str(artifact)),
        "finish": lambda: tracker.finish_experiment(first),
    }[operation]
    task = asyncio.create_task(operation_call())
    try:
        await asyncio.wait_for(gate.entered.wait(), 2)
        assert gate.thread_id != threading.get_ident()
        assert not task.done()
        await asyncio.wait_for(tracker.log_metrics(second, {"loss": 0.2}, 2), 2)
        assert tracker.experiments[second]["metrics"]["loss"][-1]["value"] == 0.2
        assert not gate.finished.is_set()
    finally:
        gate.release.set()
        await task
    for identifier in tracker.experiments:
        await tracker.finish_experiment(identifier)
    assert all(not value["backend_errors"] for value in tracker.experiments.values())


@pytest.mark.asyncio
async def test_cancelled_start_drains_creation_and_closes_run_before_return(sdks):
    tracker = tracking.ExperimentTracker(True, True)
    gate = SDKGate(sdks.wandb.init.side_effect)
    sdks.wandb.init.side_effect = gate
    task = asyncio.create_task(tracker.start_experiment(job()))
    try:
        await asyncio.wait_for(gate.entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # Repeated cancellation cannot detach SDK startup.
        await asyncio.sleep(0)
        assert not task.done()
        assert not sdks.runs
    finally:
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert gate.finished.is_set()
    assert next(iter(tracker.experiments.values()))["status"] == "cancelled"
    sdks.runs[0].finish.assert_called_once_with(exit_code=1)
    sdks.client.set_terminated.assert_called_once_with("run-1", status="KILLED")
    assert not tracker._wandb_runs and not tracker._mlflow_runs


@pytest.mark.asyncio
async def test_cancelled_upload_holds_order_and_snapshots_queued_measurements(sdks):
    tracker = tracking.ExperimentTracker(True, False)
    identifier = await tracker.start_experiment(job())
    gate = SDKGate()
    sdks.runs[0].log.side_effect = gate
    first = asyncio.create_task(tracker.log_metrics(identifier, {"loss": 1.0}, 1))
    others = []
    try:
        await asyncio.wait_for(gate.entered.wait(), 2)
        values = {"loss": 0.5}
        others.append(asyncio.create_task(tracker.log_metrics(identifier, values, 2)))
        await asyncio.sleep(0)  # The queued call takes its snapshot before waiting.
        values["loss"] = 99
        others.append(asyncio.create_task(tracker.finish_experiment(identifier)))
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        sdks.runs[0].finish.assert_not_called()
    finally:
        gate.release.set()
        await asyncio.gather(first, *others, return_exceptions=True)
    assert first.cancelled()
    assert all(task.exception() is None for task in others)
    assert tracker.experiments[identifier]["metrics"]["loss"] == [
        {"step": 1, "value": 1.0},
        {"step": 2, "value": 0.5},
    ]
    assert tracker.experiments[identifier]["status"] == "completed"
    sdks.runs[0].finish.assert_called_once_with(exit_code=0)


@pytest.mark.asyncio
async def test_blocked_tracking_does_not_delay_another_workers_runtime_limit(sdks):
    tracker = tracking.ExperimentTracker(True, False)
    identifier = await tracker.start_experiment(job("blocked"))
    gate = SDKGate()
    sdks.runs[0].log.side_effect = gate
    upload = asyncio.create_task(tracker.log_metrics(identifier, {"loss": 0.1}, 1))
    limited = job("limited")
    limited.config.enable_wandb = limited.config.enable_mlflow = False
    limited.config.max_runtime = 0.05

    async def runner(task):
        await asyncio.Event().wait()

    try:
        await asyncio.wait_for(gate.entered.wait(), 2)
        worker = TrainingWorker("worker", runner)
        assert not await asyncio.wait_for(worker.execute_job(limited, tracker), 2)
        assert limited.status == TrainingStatus.TIMED_OUT
        assert not gate.finished.is_set()
    finally:
        gate.release.set()
        await upload
        await tracker.finish_experiment(identifier)


@pytest.mark.asyncio
async def test_cancelled_finish_preserves_outcome_and_waits_for_sdk_closure(sdks):
    tracker = tracking.ExperimentTracker(True, False)
    identifier = await tracker.start_experiment(job())
    gate = SDKGate()
    sdks.runs[0].finish.side_effect = gate
    close = asyncio.create_task(tracker.finish_experiment(identifier))
    try:
        await asyncio.wait_for(gate.entered.wait(), 2)
        close.cancel()
        await asyncio.sleep(0)
        close.cancel()
        await asyncio.sleep(0)
        assert not close.done()
        assert identifier in tracker._wandb_runs
        assert tracker.experiments[identifier]["status"] == "completed"
    finally:
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await close
    assert gate.finished.is_set()
    assert identifier not in tracker._wandb_runs
    await tracker.finish_experiment(identifier)
    sdks.runs[0].finish.assert_called_once_with(exit_code=0)


@pytest.mark.asyncio
async def test_cancelled_duplicate_start_cannot_close_the_original_run(sdks):
    tracker = tracking.ExperimentTracker(True, False)
    gate = SDKGate(sdks.wandb.init.side_effect)
    sdks.wandb.init.side_effect = gate
    original = asyncio.create_task(tracker.start_experiment(job()))
    duplicate = None
    try:
        await asyncio.wait_for(gate.entered.wait(), 2)
        duplicate = asyncio.create_task(tracker.start_experiment(job()))
        await asyncio.sleep(0)
        duplicate.cancel()
        with pytest.raises(asyncio.CancelledError):
            await duplicate
    finally:
        gate.release.set()
        identifier = await original
    assert tracker.experiments[identifier]["status"] == "running"
    sdks.runs[0].finish.assert_not_called()
    await tracker.finish_experiment(identifier)
