"""Committed updates at every lifecycle boundary share one reporting cadence."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from stateset_agents.training.diagnostics import DiagnosticsMonitor
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from tests.unit import test_multi_turn_training_lifecycle as lifecycle


@pytest.fixture(name="lifecycle_trainer")
def _lifecycle_trainer():
    return lifecycle.lifecycle_trainer.__wrapped__()


def _observe(trainer):
    receipts, saves, generated = [], [], []
    trainer.config.logging_steps = trainer.config.eval_steps = (
        trainer.config.save_steps
    ) = 1
    trainer.callbacks.append(
        SimpleNamespace(
            fail_on_error=True,
            on_optimizer_step=lambda step, metrics: receipts.append(dict(metrics)),
        )
    )
    trainer.evaluate = AsyncMock(return_value={"eval_reward": 0.5})

    async def save():
        saves.append(
            (trainer.global_step, trainer._next_episode, trainer._current_task_id)
        )

    trainer.save_checkpoint.side_effect = save
    groups = trainer.generate_trajectories.return_value

    async def generate(scenarios):
        generated.append(scenarios[0].get("task_id", "task"))
        return groups

    trainer.generate_trajectories.side_effect = generate
    return receipts, saves, generated


def _tasks(trainer):
    trainer.config.num_episodes = 2
    trainer._get_training_scenarios = lambda: [{"task_id": "A"}, {"task_id": "B"}]
    trainer._resolve_task_id = lambda scenario: scenario["task_id"]
    trainer._maybe_handle_task_switch = (
        MultiTurnGRPOTrainer._maybe_handle_task_switch.__get__(trainer)
    )
    trainer._current_task_id = None


@pytest.mark.asyncio
async def test_final_flush_runs_cadence_without_fabricating_batch_metrics(
    lifecycle_trainer,
):
    trainer, callback = lifecycle_trainer
    receipts, saves, generated = _observe(trainer)
    await trainer.train()
    assert generated == ["task"]
    assert saves == [(1, 1, "task")]
    assert trainer.evaluate.await_count == 1
    assert callback.on_episode_end.await_count == 1
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["update_source"] == "final_flush"
    assert receipt["optimizer_step_start"] == 0 and receipt["optimizer_step_end"] == 1
    assert receipt["optimizer_updates"] == receipt["accumulation_flush_updates"] == 1
    assert receipt["trajectory_metrics"] == {}
    assert "total_loss" not in receipt and "total_reward" not in receipt
    assert trainer.wandb_logger.log_training_step.call_args.kwargs["losses"] == {
        "optimizer_updates": 1,
        "accumulation_flush_updates": 1,
        "final_flush_updates": 1,
    }


@pytest.mark.asyncio
async def test_batch_and_final_flush_have_distinct_update_receipts(lifecycle_trainer):
    trainer, _ = lifecycle_trainer
    trainer.config.num_episodes = 3
    trainer._get_training_scenarios = lambda: [{}] * 3
    receipts, saves, _ = _observe(trainer)
    accumulate = trainer.training_step

    async def step(groups):
        metrics = await accumulate(groups)
        applied = False
        if trainer._grad_accum_step % 2 == 0:
            applied = trainer._apply_optimizer_step(torch)
        return {**metrics, "optimizer_step": applied, "optimizer_updates": int(applied)}

    trainer.training_step = step
    await trainer.train()
    assert [r["update_source"] for r in receipts] == ["batch", "final_flush"]
    assert [(r["optimizer_step_start"], r["optimizer_step_end"]) for r in receipts] == [
        (0, 1),
        (1, 2),
    ]
    assert saves == [(1, 2, "task"), (2, 3, "task")]
    assert trainer.evaluate.await_count == 2


@pytest.mark.asyncio
async def test_task_boundary_reports_before_next_rollout(lifecycle_trainer):
    trainer, _ = lifecycle_trainer
    _tasks(trainer)
    receipts, saves, generated = _observe(trainer)

    async def evaluate(scenarios):
        assert generated == (["A"] if trainer.global_step == 1 else ["A", "B"])
        return {"eval_reward": 0.5}

    trainer.evaluate.side_effect = evaluate
    await trainer.train()
    assert [r["update_source"] for r in receipts] == ["task_boundary", "final_flush"]
    assert [r["update_task_id"] for r in receipts] == ["A", "B"]
    assert saves == [(1, 1, "B"), (2, 2, "B")]
    assert [
        c.kwargs["task_id"]
        for c in trainer.continual_manager.on_task_end.call_args_list
    ] == ["A", "B"]


@pytest.mark.asyncio
async def test_boundary_evaluation_can_stop_before_entering_untrained_task(
    lifecycle_trainer,
):
    trainer, _ = lifecycle_trainer
    _tasks(trainer)
    receipts, saves, generated = _observe(trainer)
    trainer.config.early_stopping = True
    trainer.config.patience = 1

    async def evaluate(scenarios):
        trainer.steps_without_improvement = 1
        return {"eval_reward": 0.5}

    trainer.evaluate.side_effect = evaluate
    await trainer.train()
    assert generated == ["A"]
    assert len(receipts) == 1 and receipts[0]["update_source"] == "task_boundary"
    assert trainer._current_task_id == "A" and trainer._next_episode == 1
    assert saves[-1] == (1, 1, "A")
    trainer.continual_manager.on_task_end.assert_called_once()


@pytest.mark.asyncio
async def test_flush_receipts_preserve_signal_history_and_isolate_callback_mutations(
    lifecycle_trainer,
):
    trainer, _ = lifecycle_trainer
    trainer.config.zero_signal_patience = 3
    monitor = DiagnosticsMonitor(trainer.config)
    monitor("step_end", {"step": 1, "nonzero_advantage_fraction": 0.0})
    original = monitor.state_dict()

    def mutate(step, metrics):
        metrics["optimizer_updates"] = 99
        metrics["trajectory_metrics"]["fake_reward"] = 99

    trainer.callbacks = [monitor, SimpleNamespace(on_optimizer_step=mutate)]
    trainer.global_step = 1
    trainer.config.logging_steps = 1
    receipt = trainer._optimizer_update_receipt(0, "final_flush")
    await trainer._report_optimizer_update(receipt, [], [])
    assert monitor.state_dict() == original
    assert receipt["optimizer_updates"] == 1 and receipt["trajectory_metrics"] == {}
    assert (
        trainer.wandb_logger.log_training_step.call_args.kwargs["losses"][
            "optimizer_updates"
        ]
        == 1
    )


@pytest.mark.asyncio
async def test_cpu_amp_skipped_final_flush_emits_no_update_event(lifecycle_trainer):
    trainer, callback = lifecycle_trainer
    receipts, saves, _ = _observe(trainer)
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128)

    async def overflow(groups):
        loss = trainer.agent.model(torch.ones(1, 1)).sum()
        trainer.scaler.scale(loss).backward()
        trainer.agent.model.weight.grad.fill_(float("inf"))
        trainer._grad_accum_step += 1
        return {
            "total_loss": float(loss.detach()),
            "optimizer_step": False,
            "optimizer_updates": 0,
        }

    trainer.training_step = overflow
    before = trainer.agent.model.weight.detach().clone()
    await trainer.train()
    torch.testing.assert_close(trainer.agent.model.weight, before, rtol=0, atol=0)
    assert trainer.global_step == 0 and not receipts
    assert saves == [(0, 1, "task")]
    trainer.evaluate.assert_not_awaited()
    trainer.wandb_logger.log_training_step.assert_not_called()
    callback.on_training_end.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["task_boundary", "final_flush"])
@pytest.mark.parametrize("stage", ["callback", "log", "evaluate", "save"])
async def test_flush_reporting_failure_does_not_repeat_committed_update(
    lifecycle_trainer, source, stage
):
    trainer, callback = lifecycle_trainer
    if source == "task_boundary":
        _tasks(trainer)
    _observe(trainer)
    error = OSError("injected reporting failure")
    if stage == "callback":
        trainer.callbacks.append(
            SimpleNamespace(
                fail_on_error=True, on_optimizer_step=AsyncMock(side_effect=error)
            )
        )
    elif stage == "log":
        trainer.wandb_logger.log_training_step.side_effect = error
    elif stage == "evaluate":
        trainer.evaluate.side_effect = error
    else:
        trainer.save_checkpoint.side_effect = error
    with pytest.raises(OSError) as caught:
        await trainer.train()
    assert caught.value is error
    assert trainer.global_step == 1 and trainer._grad_accum_step == 0
    callback.on_training_end.assert_not_awaited()
    assert trainer.wandb_logger.finish_run.call_args.args[0]["errored"] is True


def test_receipt_uses_actual_step_delta_and_does_not_repeat_completed_updates(
    lifecycle_trainer,
):
    trainer, _ = lifecycle_trainer
    trainer.global_step = 3
    receipt = trainer._optimizer_update_receipt(
        0, "batch", {"optimizer_updates": 1, "total_loss": 0.5}
    )
    assert receipt["optimizer_updates"] == 3
    assert trainer._step_event_due(2, receipt)
    assert trainer._optimizer_update_receipt(3, "final_flush") is None
