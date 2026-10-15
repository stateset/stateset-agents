"""Single-turn failures must propagate without committing pending gradients."""

import asyncio
import copy
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from stateset_agents.training import single_turn_trainer as module


@pytest.fixture
def lifecycle_trainer(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    environment = SimpleNamespace(
        reset=AsyncMock(return_value={"prompt": "hello"}),
        step=AsyncMock(return_value={"state": {"prompt": "hello"}, "reward": 1.0}),
    )
    callback = SimpleNamespace(
        fail_on_error=True,
        on_training_start=AsyncMock(),
        on_episode_end=AsyncMock(),
        on_training_end=AsyncMock(),
    )
    tracker = SimpleNamespace(log=MagicMock(), finish_run=MagicMock())
    trainer = module.SingleTurnGRPOTrainer(
        SimpleNamespace(model=model),
        environment,
        config=SimpleNamespace(
            num_episodes=1,
            max_steps_per_episode=1,
            num_generations=2,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            bf16=False,
            beta=0.0,
        ),
        callbacks=[callback],
        wandb_logger=tracker,
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda _: 1
    )
    trainer._setup_scheduler = MagicMock()
    trainer._get_episode_scenario = MagicMock(return_value={})
    trainer._resolve_task_id = lambda _: "task"
    trainer._maybe_handle_task_switch = MagicMock()
    trainer._generate_trajectory_group = AsyncMock(
        return_value=(SimpleNamespace(rewards=[0.0, 1.0]), "answer")
    )
    trainer.continual_manager = SimpleNamespace(
        sample_replay_groups=MagicMock(return_value=[]),
        add_trajectory_groups=MagicMock(),
        get_effective_beta=lambda beta: beta,
        compute_ewc_penalty=MagicMock(return_value=None),
        on_task_end=MagicMock(),
        reference_model=None,
        buffer=SimpleNamespace(size=0),
    )
    trainer._current_task_id = "task"
    loss = MagicMock(
        side_effect=lambda **kwargs: {"total_loss": model(torch.ones(1, 1)).sum()}
    )
    monkeypatch.setattr(module, "compute_grpo_loss", loss)
    monkeypatch.setattr(module, "compute_enhanced_grpo_loss", loss)
    return trainer, callback, loss


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize(
    "error_type",
    [asyncio.CancelledError, KeyboardInterrupt, RuntimeError, AssertionError],
)
async def test_failed_episode_discards_pending_work(
    lifecycle_trainer, committed, error_type
):
    trainer, callback, _ = lifecycle_trainer
    if committed:
        trainer.agent.model(torch.ones(1, 1)).sum().backward()
        trainer._apply_optimizer_step(torch, False, 1.0)
        trainer._grad_accum_step = 2
    before = trainer.agent.model.weight.detach().clone()
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    scheduler = copy.deepcopy(trainer.lr_scheduler.state_dict())
    error = error_type("stop")
    callback.on_episode_end.side_effect = error
    with pytest.raises(error_type) as caught:
        await trainer.train()
    assert caught.value is error
    assert torch.equal(before, trainer.agent.model.weight)
    assert trainer.agent.model.weight.grad is None
    assert trainer.optimizer.state_dict() == optimizer
    assert trainer.lr_scheduler.state_dict() == scheduler
    assert trainer.global_step == int(committed)
    assert trainer._grad_accum_step == 0
    trainer.continual_manager.on_task_end.assert_not_called()
    callback.on_training_end.assert_not_awaited()
    trainer.wandb_logger.finish_run.assert_called_once()
    summary = trainer.wandb_logger.finish_run.call_args.args[0]
    assert summary["errored"] is True
    assert summary["cancelled"] == isinstance(
        error, (asyncio.CancelledError, KeyboardInterrupt)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        "setup",
        "start",
        "reset",
        "rollout",
        "loss",
        "enhanced_loss",
        "step",
        "log",
        "optimizer",
        "continual",
        "end",
    ],
)
async def test_stage_errors_propagate_and_close_tracking(lifecycle_trainer, stage):
    trainer, callback, loss = lifecycle_trainer
    error = RuntimeError(stage)
    if stage == "enhanced_loss":
        trainer.config.beta = 0.1
    operations = {
        "setup": trainer._setup_scheduler,
        "start": callback.on_training_start,
        "reset": trainer.environment.reset,
        "rollout": trainer._generate_trajectory_group,
        "loss": loss,
        "enhanced_loss": loss,
        "step": trainer.environment.step,
        "log": trainer.wandb_logger.log,
        "continual": trainer.continual_manager.on_task_end,
        "end": callback.on_training_end,
    }
    if stage == "optimizer":
        trainer._apply_optimizer_step = MagicMock(side_effect=error)
    else:
        operations[stage].side_effect = error
    with pytest.raises(RuntimeError) as caught:
        await trainer.train()
    assert caught.value is error
    assert trainer.agent.model.weight.grad is None
    assert trainer._grad_accum_step == 0
    trainer.wandb_logger.finish_run.assert_called_once()
    assert trainer.wandb_logger.finish_run.call_args.args[0]["errored"] is True
    if stage != "end":
        callback.on_training_end.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["reset", "step"])
async def test_internal_environment_type_error_is_not_retried(
    lifecycle_trainer, method
):
    trainer, callback, _ = lifecycle_trainer
    error = TypeError("inside environment after a side effect")
    operation = getattr(trainer.environment, method)
    operation.side_effect = error
    with pytest.raises(TypeError) as caught:
        await trainer.train()
    assert caught.value is error
    operation.assert_awaited_once()
    callback.on_training_end.assert_not_awaited()
    assert trainer.agent.model.weight.grad is None


@pytest.mark.asyncio
async def test_legacy_environment_signatures_still_run_once(lifecycle_trainer):
    trainer, callback, _ = lifecycle_trainer
    calls = []

    async def reset():
        calls.append("reset")
        return {"prompt": "hello"}

    async def step(response):
        calls.append(response)
        return {"reward": 1.0, "done": True}

    trainer.environment = SimpleNamespace(reset=reset, step=step)
    before = trainer.agent.model.weight.detach().clone()
    assert await trainer.train() is trainer.agent
    assert calls == ["reset", "answer"]
    assert trainer.global_step == 1 and trainer._grad_accum_step == 0
    assert not torch.equal(before, trainer.agent.model.weight)
    callback.on_training_end.assert_awaited_once()
    trainer.continual_manager.on_task_end.assert_called_once()
    assert trainer.wandb_logger.finish_run.call_args.args[0]["errored"] is False


@pytest.mark.asyncio
async def test_actual_task_cancellation_during_environment_step(lifecycle_trainer):
    trainer, callback, _ = lifecycle_trainer
    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def step(*args):
        entered.set()
        await blocked.wait()

    trainer.environment.step.side_effect = step
    before = trainer.agent.model.weight.detach().clone()
    task = asyncio.create_task(trainer.train())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert trainer.agent.model.weight.grad is not None
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    assert trainer.agent.model.weight.grad is None
    assert torch.equal(before, trainer.agent.model.weight)
    assert trainer.global_step == 0 and trainer._grad_accum_step == 0
    callback.on_training_end.assert_not_awaited()
    assert trainer.wandb_logger.finish_run.call_args.args[0]["cancelled"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", [False, True])
async def test_tracking_failure_preserves_original_error(lifecycle_trainer, primary):
    trainer, callback, _ = lifecycle_trainer
    original = LookupError("training failed")
    cleanup = RuntimeError("tracking failed")
    if primary:
        callback.on_episode_end.side_effect = original
    trainer.wandb_logger.finish_run.side_effect = cleanup
    with pytest.raises(LookupError if primary else RuntimeError) as caught:
        await trainer.train()
    assert caught.value is (original if primary else cleanup)


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_tracking_cleanup(lifecycle_trainer):
    trainer, callback, _ = lifecycle_trainer
    callback.on_episode_end.side_effect = asyncio.CancelledError("stop")
    entered = asyncio.Event()
    release, closed = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def finish(summary):
        assert summary["cancelled"] and summary["errored"]
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        closed.set()

    trainer.wandb_logger.finish_run.side_effect = finish
    task = asyncio.create_task(trainer.train())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not closed.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    assert closed.is_set()
    callback.on_training_end.assert_not_awaited()


@pytest.mark.asyncio
async def test_logging_only_tracker_remains_supported(lifecycle_trainer):
    trainer, callback, _ = lifecycle_trainer
    trainer.wandb_logger = SimpleNamespace(log=MagicMock())
    assert await trainer.train() is trainer.agent
    trainer.wandb_logger.log.assert_called_once()
    callback.on_training_end.assert_awaited_once()


@pytest.mark.asyncio
async def test_packaged_wandb_logger_interface_is_supported_without_network(
    lifecycle_trainer,
):
    from stateset_agents.utils.wandb_logger import WandBLogger

    trainer, callback, _ = lifecycle_trainer
    tracker = WandBLogger(enabled=False)
    tracker.log_metrics = MagicMock(wraps=tracker.log_metrics)
    tracker.finish_run = MagicMock(wraps=tracker.finish_run)
    trainer.wandb_logger = tracker
    assert await trainer.train() is trainer.agent
    tracker.log_metrics.assert_called_once()
    assert tracker.log_metrics.call_args.kwargs == {"step": 0}
    tracker.finish_run.assert_called_once()
    assert tracker.finish_run.call_args.args[0]["final_step"] == 1
    callback.on_training_end.assert_awaited_once()


@pytest.mark.asyncio
async def test_uninspectable_environment_receives_canonical_call_once(
    lifecycle_trainer,
):
    trainer, callback, _ = lifecycle_trainer
    error = TypeError("environment side effect failed")

    class Reset:
        __signature__ = object()

        def __init__(self):
            self.calls = []

        async def __call__(self, scenario):
            self.calls.append(scenario)
            raise error

    reset = Reset()
    trainer.environment.reset = reset
    with pytest.raises(TypeError) as caught:
        await trainer.train()
    assert caught.value is error
    assert reset.calls == [{}]
    callback.on_training_end.assert_not_awaited()
    trainer.environment.step.assert_not_awaited()
