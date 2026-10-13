"""Optional real-SDK contracts; all tracking stays on the local filesystem."""

import importlib.util
import os
import subprocess
import sys
from importlib.metadata import version

import pytest
from packaging.version import Version

pytestmark = pytest.mark.integration


def run_contract(script, tmp_path):
    environment = dict(os.environ)
    environment.update(
        WANDB_MODE="offline",
        WANDB_RUN_ID="ambient-environment-id",
        WANDB_API_KEY="0" * 40,
        WANDB_DIR=str(tmp_path),
        WANDB_SILENT="true",
        MLFLOW_TRACKING_URI=f"sqlite:///{tmp_path / 'tracking.db'}",
    )
    # Keep subprocess imports independent of the temporary artifact directory.
    environment["PYTHONPATH"] = os.getcwd()
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("with_ambient", [False, True])
def test_real_wandb_offline_runs_do_not_share_active_run(tmp_path, with_ambient):
    if importlib.util.find_spec("wandb") is None:
        pytest.skip("W&B is optional")
    if Version(version("wandb")) < Version("0.19.10"):
        pytest.skip("Concurrent W&B tracking requires 0.19.10+")
    run_contract(
        """
import asyncio
from pathlib import Path
import wandb
from stateset_agents.training.experiment_tracking import ExperimentTracker
from stateset_agents.training.advanced_training_models import TrainingJob, TrainingJobSpec, TrainingStatus
async def main():
    ambient = wandb.init(project='stateset-sdk-contract', name='ambient', mode='offline') if AMBIENT_ENABLED else None
    tracker = ExperimentTracker(enable_wandb=True, enable_mlflow=False)
    try:
        first, second = await asyncio.gather(
            tracker.start_experiment(TrainingJob('first', TrainingJobSpec('stateset-sdk-contract', 'test', {}, 'unused'))),
            tracker.start_experiment(TrainingJob('second', TrainingJobSpec('stateset-sdk-contract', 'test', {}, 'unused'))),
        )
        assert wandb.run is ambient
        assert tracker._wandb_runs[first].id != tracker._wandb_runs[second].id
        assert tracker._wandb_runs[first].settings.mode == 'offline'
        await tracker.log_metrics(first, {'loss': 0.2}, 1)
        await tracker.log_metrics(second, {'loss': 0.8}, 1)
        artifact = Path('model'); artifact.mkdir(); (artifact / 'weights').write_bytes(b'weights')
        await tracker.log_artifact(second, str(artifact))
        await tracker.finish_experiment(first, {'loss': 0.2}, status=TrainingStatus.COMPLETED)
        await tracker.log_metrics(second, {'loss': 0.7}, 2)
        await tracker.finish_experiment(second, status=TrainingStatus.TIMED_OUT)
        assert wandb.run is ambient
        if ambient is not None:
            ambient.log({'still_active': 1})
        assert all(not item['backend_errors'] for item in tracker.experiments.values()), tracker.experiments
    finally:
        for run in list(tracker._wandb_runs.values()):
            run.finish(exit_code=1)
        if ambient is not None:
            ambient.finish()
        wandb.teardown()
asyncio.run(main())
""".replace("AMBIENT_ENABLED", str(with_ambient)),
        tmp_path,
    )


def test_real_mlflow_local_runs_keep_metrics_and_statuses_separate(tmp_path):
    if importlib.util.find_spec("mlflow") is None:
        pytest.skip("MLflow is optional")
    run_contract(
        """
import asyncio
from pathlib import Path
from mlflow.tracking import MlflowClient
from stateset_agents.training.experiment_tracking import ExperimentTracker
from stateset_agents.training.advanced_training_models import TrainingJob, TrainingJobSpec, TrainingStatus
async def main():
    client = MlflowClient()
    client.create_experiment('sdk-contract', artifact_location=Path('artifacts').resolve().as_uri())
    tracker = ExperimentTracker(enable_wandb=False, enable_mlflow=True)
    first = await tracker.start_experiment(TrainingJob('first', TrainingJobSpec('sdk-contract', 'test', {}, 'unused', enable_mlflow=True)))
    second = await tracker.start_experiment(TrainingJob('second', TrainingJobSpec('sdk-contract', 'test', {}, 'unused', enable_mlflow=True)))
    one, two = tracker._mlflow_runs[first][1], tracker._mlflow_runs[second][1]
    await tracker.log_metrics(first, {'loss': 0.2}, 1)
    await tracker.log_metrics(second, {'loss': 0.8}, 2)
    model = Path('weights'); model.write_bytes(b'weights')
    await tracker.log_artifact(second, str(model))
    await tracker.finish_experiment(first, {'loss': 0.2}, status=TrainingStatus.COMPLETED)
    await tracker.finish_experiment(second, status=TrainingStatus.TIMED_OUT)
    assert client.get_run(one).data.metrics['loss'] == 0.2
    assert client.get_run(two).data.metrics['loss'] == 0.8
    assert client.get_run(one).info.status == 'FINISHED'
    assert client.get_run(two).info.status == 'KILLED'
    assert client.get_run(two).data.tags['stateset.status'] == 'timed_out'
    assert all(not item['backend_errors'] for item in tracker.experiments.values())
asyncio.run(main())
""",
        tmp_path,
    )
