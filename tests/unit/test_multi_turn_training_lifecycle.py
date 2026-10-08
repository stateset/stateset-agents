"""Trainer failure cleanup must never perform success-only model updates."""

import asyncio
import copy
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer


@pytest.fixture
def lifecycle_trainer():
    model = torch.nn.Linear(1, 1, bias=False)
    agent = SimpleNamespace(model=model)
    config = SimpleNamespace(
        num_episodes=1,
        gradient_accumulation_steps=2,
        resume_from_checkpoint=None,
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        max_grad_norm=1.0,
        early_stopping=False,
    )
    callback = SimpleNamespace(
        fail_on_error=True,
        on_training_start=AsyncMock(),
        on_episode_end=AsyncMock(),
        on_training_end=AsyncMock(),
    )
    trainer = MultiTurnGRPOTrainer(
        agent, object(), config=config, callbacks=[callback], wandb_logger=MagicMock()
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda _: 1
    )
    trainer._setup_scheduler = MagicMock()
    trainer._get_training_scenarios = lambda: [{"id": "one"}, {"id": "two"}]
    trainer._get_eval_scenarios = lambda: []
    trainer._resolve_task_id = lambda _: "task"
    trainer._maybe_handle_task_switch = lambda _: None
    group = SimpleNamespace(trajectories=[SimpleNamespace(total_reward=0.5)])
    trainer.generate_trajectories = AsyncMock(return_value=[group])
    trainer._maybe_mix_replay = lambda groups, _: (groups, [])
    trainer.continual_manager = SimpleNamespace(
        on_task_end=MagicMock(), reference_model=None
    )
    trainer._current_task_id = "task"
    trainer.save_checkpoint = AsyncMock()

    async def accumulate(groups):
        loss = model(torch.ones(1, 1)).sum()
        loss.backward()
        trainer._grad_accum_step += 1
        return {"total_loss": float(loss.detach())}

    trainer.training_step = accumulate
    return trainer, callback


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        asyncio.CancelledError("stop"),
        KeyboardInterrupt("stop"),
        RuntimeError("failed"),
        AssertionError("unexpected"),
    ],
)
async def test_unsuccessful_episode_discards_gradients_without_finalizing(
    lifecycle_trainer, error, committed
):
    trainer, callback = lifecycle_trainer
    if committed:
        trainer.agent.model(torch.ones(1, 1)).sum().backward()
        trainer._apply_optimizer_step(torch)
        trainer._grad_accum_step = 2
    before = trainer.agent.model.weight.detach().clone()
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    scheduler = copy.deepcopy(trainer.lr_scheduler.state_dict())
    callback.on_episode_end.side_effect = error
    with pytest.raises(type(error)) as caught:
        await trainer.train()
    assert caught.value is error
    assert torch.equal(trainer.agent.model.weight, before)
    assert trainer.agent.model.weight.grad is None
    assert trainer.optimizer.state_dict() == optimizer
    assert trainer.lr_scheduler.state_dict() == scheduler
    assert trainer.global_step == int(committed)
    assert trainer._grad_accum_step == 0
    trainer.continual_manager.on_task_end.assert_not_called()
    trainer.save_checkpoint.assert_not_awaited()
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
    ["setup", "start_callback", "optimizer", "continual", "checkpoint", "end_callback"],
)
async def test_failures_during_setup_or_finalization_still_close_tracking(
    lifecycle_trainer, stage
):
    trainer, callback = lifecycle_trainer
    error = OSError(stage)
    operation = {
        "setup": trainer._setup_scheduler,
        "start_callback": callback.on_training_start,
        "continual": trainer.continual_manager.on_task_end,
        "checkpoint": trainer.save_checkpoint,
        "end_callback": callback.on_training_end,
    }.get(stage)
    if stage == "optimizer":
        trainer._apply_optimizer_step = MagicMock(side_effect=error)
    else:
        operation.side_effect = error
    with pytest.raises(OSError) as caught:
        await trainer.train()
    assert caught.value is error
    assert trainer.agent.model.weight.grad is None
    assert trainer._grad_accum_step == 0
    trainer.wandb_logger.finish_run.assert_called_once()
    assert trainer.wandb_logger.finish_run.call_args.args[0]["errored"] is True
    if stage != "end_callback":
        callback.on_training_end.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("early_stop", [False, True])
async def test_success_flushes_once_and_checkpoint_has_no_pending_window(
    lifecycle_trainer, early_stop
):
    trainer, callback = lifecycle_trainer
    if early_stop:
        trainer.config.num_episodes = 2
        trainer.config.early_stopping = True
        trainer.config.patience = 0
    saved = []

    async def save():
        saved.append((trainer.global_step, trainer._grad_accum_step))

    trainer.save_checkpoint.side_effect = save
    before = trainer.agent.model.weight.detach().clone()
    assert await trainer.train() is trainer.agent
    assert not torch.equal(trainer.agent.model.weight, before)
    assert trainer.global_step == 1
    assert saved == [(1, 0)]
    callback.on_training_end.assert_awaited_once()
    trainer.continual_manager.on_task_end.assert_called_once()
    assert trainer.wandb_logger.finish_run.call_args.args[0]["errored"] is False

    # An already-complete resume must not reapply optimizer momentum merely
    # because the last completed run flushed a partial accumulation window.
    trainer.config.num_episodes = 1
    trainer.config.resume_from_checkpoint = "saved"
    trainer.load_checkpoint = lambda _: True
    completed_weights = trainer.agent.model.weight.detach().clone()
    assert await trainer.train() is trainer.agent
    assert trainer.global_step == 1
    assert torch.equal(trainer.agent.model.weight, completed_weights)
    assert saved == [(1, 0), (1, 0)]


@pytest.mark.asyncio
async def test_tracking_failure_does_not_replace_original_training_error(
    lifecycle_trainer,
):
    trainer, callback = lifecycle_trainer
    original = LookupError("training failure")
    callback.on_episode_end.side_effect = original
    trainer.wandb_logger.finish_run.side_effect = RuntimeError("tracking failure")
    with pytest.raises(LookupError) as caught:
        await trainer.train()
    assert caught.value is original
    trainer.save_checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_training_owns_tracking_cleanup_through_repeated_cancellation(
    lifecycle_trainer,
):
    trainer, callback = lifecycle_trainer
    callback.on_episode_end.side_effect = asyncio.CancelledError("stop")
    entered = asyncio.Event()
    release, closed = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def finish(summary):
        assert summary["cancelled"] and summary["errored"]
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "tracking cleanup was not released"
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
    trainer.save_checkpoint.assert_not_awaited()
