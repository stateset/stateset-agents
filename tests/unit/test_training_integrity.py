"""Training completion must reflect real data, updates, and saved output."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from stateset_agents.training.advanced_training_orchestrator import (
    TrainingJob,
    TrainingJobSpec,
    TrainingRunResult,
    TrainingStatus,
    TrainingWorker,
)
from stateset_agents.training.offline_grpo_trainer import (
    OfflineGRPOConfig,
    OfflineGRPOTrainer,
)


def job():
    return TrainingJob("integrity", TrainingJobSpec("test", "test", {}, "unused"))


@pytest.mark.asyncio
async def test_runner_performs_real_update_and_publishes_measured_artifact(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0)
    path = tmp_path / "weights.pt"

    async def runner(job):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.ones(4, 1)
        initial = float(((model(x) - 1) ** 2).mean().detach())
        for _ in range(5):
            optimizer.zero_grad()
            loss = ((model(x) - 1) ** 2).mean()
            loss.backward()
            optimizer.step()
        final = float(((model(x) - 1) ** 2).mean().detach())
        torch.save(model.state_dict(), path)
        return TrainingRunResult(path, {"initial_loss": initial, "loss": final}, 5, 1)

    task = job()
    tracker = AsyncMock()
    worker = TrainingWorker("cpu", runner)
    assert await worker.execute_job(task, tracker)
    assert task.status is TrainingStatus.COMPLETED
    assert task.metrics["loss"][0] < task.metrics["initial_loss"][0]
    saved = torch.load(path, weights_only=True)
    assert torch.equal(saved["weight"], model.weight)
    assert task.checkpoint_path == str(path) and task.current_step == 5
    tracker.finish_experiment.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["missing", "empty", "nan", "zero_steps", "wrong_result"]
)
async def test_invalid_runner_output_cannot_complete_job(tmp_path, kind):
    path = tmp_path / "weights.pt"
    if kind != "missing":
        path.write_bytes(b"" if kind == "empty" else b"test-artifact")
    result = TrainingRunResult(
        path,
        {"loss": float("nan") if kind == "nan" else 0.5},
        0 if kind == "zero_steps" else 1,
        1,
    )
    runner = AsyncMock(return_value=True if kind == "wrong_result" else result)
    task = job()
    assert not await TrainingWorker("test", runner).execute_job(task, None)
    assert task.status is TrainingStatus.FAILED and task.last_error
    assert not task.metrics and task.checkpoint_path is None
    runner.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_failure_is_not_blindly_retried():
    runner = AsyncMock(side_effect=TimeoutError("optimizer result uncertain"))
    task = job()
    assert not await TrainingWorker("test", runner).execute_job(task, None)
    runner.assert_awaited_once()
    assert task.last_error == "optimizer result uncertain"


@pytest.mark.asyncio
async def test_runner_cancellation_is_preserved():
    task = job()
    worker = TrainingWorker("test", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await worker.execute_job(task, None)
    assert task.status is TrainingStatus.CANCELLED
    assert worker.current_job is None


@pytest.mark.asyncio
async def test_cancellation_during_tracking_cannot_be_overwritten(tmp_path):
    task = job()
    path = tmp_path / "checkpoint"
    path.write_bytes(b"test-artifact")
    tracker = AsyncMock()

    async def cancel(*args, **kwargs):
        task.status = TrainingStatus.CANCELLED

    tracker.finish_experiment.side_effect = cancel
    worker = TrainingWorker(
        "test", AsyncMock(return_value=TrainingRunResult(path, {"loss": 0.1}, 1, 1))
    )
    assert not await worker.execute_job(task, tracker)
    assert task.status is TrainingStatus.CANCELLED


@pytest.fixture
def trainer():
    config = OfflineGRPOConfig(
        state_dim=2,
        action_dim=2,
        value_hidden_size=4,
        value_num_layers=2,
        pretrain_value_batch_size=8,
        pretrain_value_epochs=1,
    )
    return OfflineGRPOTrainer(config, device="cpu")


def transitions():
    return {
        "states": np.ones((2, 2)),
        "actions": np.ones((2, 2)),
        "next_states": np.zeros((2, 2)),
        "rewards": np.ones(2),
        "dones": np.ones(2),
    }


@pytest.mark.asyncio
async def test_missing_features_and_conversion_errors_never_train(trainer):
    with pytest.raises(ValueError, match="real state/action"):
        await trainer.pretrain_value_functions(["raw conversation"], num_steps=1)

    def broken():
        raise ValueError("Embedding cache required")

    with pytest.raises(ValueError, match="Embedding cache required"):
        await trainer.pretrain_value_functions(
            SimpleNamespace(to_offline_rl_format=broken), num_steps=1
        )
    assert trainer._offline_learner is None and not trainer.value_pretrained


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("states", []),
        ("states", [[float("nan"), 0], [0, 0]]),
        ("actions", [[1], [2]]),
        ("rewards", [1]),
        ("dones", [0, 0.5]),
        ("next_states", [["x", "y"], ["z", "a"]]),
    ],
)
async def test_invalid_transition_features_fail_before_optimizer(trainer, field, value):
    data = transitions()
    data[field] = value
    with pytest.raises(ValueError):
        await trainer.pretrain_value_functions(
            SimpleNamespace(to_offline_rl_format=lambda: data), num_steps=1
        )
    assert trainer._offline_learner is None and not trainer.value_pretrained


@pytest.mark.asyncio
async def test_small_real_feature_dataset_updates_value_network(trainer):
    before = [p.detach().clone() for p in trainer.value_net.parameters()]
    # Exercise the real IQL learner and value optimizer on actual supplied features.
    result = await trainer.pretrain_value_functions(
        SimpleNamespace(to_offline_rl_format=transitions)
    )
    assert result["num_steps"] == 1 and trainer.value_pretrained
    assert result["final_loss"] > 0
    assert "offline_q_loss" in result
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.value_net.parameters(), strict=True)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("steps", [0, -1, True])
async def test_invalid_step_count_does_not_mark_pretrained(trainer, steps):
    with pytest.raises(ValueError, match="num_steps"):
        await trainer.pretrain_value_functions(
            SimpleNamespace(to_offline_rl_format=transitions), num_steps=steps
        )
    assert trainer._offline_learner is None and not trainer.value_pretrained


@pytest.mark.asyncio
async def test_nonfinite_learner_metric_never_marks_pretrained(trainer):
    trainer._offline_learner = SimpleNamespace(
        train_step=lambda *a: {"loss": float("nan")}
    )
    before = [p.detach().clone() for p in trainer.value_net.parameters()]
    with pytest.raises(ValueError, match="finite measured"):
        await trainer.pretrain_value_functions(
            SimpleNamespace(to_offline_rl_format=transitions), num_steps=1
        )
    assert not trainer.value_pretrained
    assert all(
        torch.equal(old, new)
        for old, new in zip(before, trainer.value_net.parameters(), strict=True)
    )
