"""Native GRPO schedules advance over the planned optimizer updates."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from stateset_agents.training import multi_turn_trainer as multi
from stateset_agents.training import single_turn_trainer as single
from stateset_agents.training.config import (
    TrainingConfig,
    estimate_grpo_optimizer_steps,
)


class TinyModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.weight.fill_(0.75)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path / "pytorch_model.bin")


def make_trainer(monkeypatch, path, kind, inner, schedule, warmup):
    model = TinyModel()
    module = single if kind == "single" else multi
    trainer_type = (
        single.SingleTurnGRPOTrainer if kind == "single" else multi.MultiTurnGRPOTrainer
    )
    trainer = trainer_type(
        SimpleNamespace(model=model, tokenizer=None),
        SimpleNamespace(
            reset=AsyncMock(return_value={"prompt": "hello"}),
            step=AsyncMock(return_value={"done": False, "reward": 1.0}),
        ),
        config=SimpleNamespace(
            output_dir=str(path),
            num_episodes=5,
            max_steps_per_episode=2,
            num_gradient_updates=inner,
            gradient_accumulation_steps=2,
            max_grad_norm=100.0,
            lr_scheduler_type=schedule,
            warmup_ratio=warmup,
            logging_steps=1000,
            eval_steps=1000,
            save_steps=1000,
        ),
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    group = SimpleNamespace(rewards=[0.0, 1.0], trajectories=[])

    def loss(*args, **kwargs):
        return {"total_loss": model.weight.square().sum(), "path": "token"}

    monkeypatch.setattr(
        module,
        "compute_token_old_logprobs",
        lambda *args: [model.weight.detach().clone()],
    )
    if kind == "single":
        monkeypatch.setattr(module, "compute_grpo_loss", loss)
        trainer._generate_trajectory_group = AsyncMock(return_value=(group, "answer"))
    else:
        trainer.compute_grpo_loss = loss
        trainer._get_training_scenarios = lambda: [{} for _ in range(5)]
        trainer._get_eval_scenarios = lambda: []
        trainer.generate_trajectories = AsyncMock(return_value=[group])
    rates = []
    trainer.optimizer.register_step_pre_hook(
        lambda optimizer, args, kwargs: rates.append(optimizer.param_groups[0]["lr"])
    )
    return trainer, rates


def schedule_rate(step, horizon, warmup, schedule):
    warmup_steps = int(horizon * warmup)
    if step < warmup_steps:
        return 0.1 * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, horizon - warmup_steps)
    factor = (
        1 - progress if schedule == "linear" else (1 + math.cos(math.pi * progress)) / 2
    )
    return 0.1 * max(0.0, factor)


def assert_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_equal(first, second)
    else:
        assert left == right


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("inner", [1, 3])
@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize("warmup", [0.0, 0.25])
async def test_schedule_covers_actual_token_updates(
    monkeypatch, tmp_path, kind, inner, schedule, warmup
):
    trainer, rates = make_trainer(monkeypatch, tmp_path, kind, inner, schedule, warmup)
    batches = 10 if kind == "single" else 5
    horizon = batches * inner if inner > 1 else math.ceil(batches / 2)
    await trainer.train()
    assert rates == pytest.approx(
        [schedule_rate(step, horizon, warmup, schedule) for step in range(horizon)]
    )
    assert trainer.global_step == trainer.lr_scheduler.last_epoch == horizon
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("inner", [1, 3])
@pytest.mark.parametrize("schedule", ["linear", "cosine"])
async def test_resumed_schedule_matches_uninterrupted_run(
    monkeypatch, tmp_path, kind, inner, schedule
):
    baseline, baseline_rates = make_trainer(
        monkeypatch, tmp_path / "baseline", kind, inner, schedule, 0.25
    )
    await baseline.train()
    interrupted, first_rates = make_trainer(
        monkeypatch, tmp_path / "interrupted", kind, inner, schedule, 0.25
    )

    async def stop(episode, metrics):
        if episode == 0:
            await interrupted.save_checkpoint(checkpoint_name="resume")
            raise RuntimeError("planned interruption")

    interrupted.callbacks = [SimpleNamespace(fail_on_error=True, on_episode_end=stop)]
    with pytest.raises(RuntimeError, match="planned interruption"):
        await interrupted.train()
    resumed, remaining_rates = make_trainer(
        monkeypatch, tmp_path / "resumed", kind, inner, schedule, 0.25
    )
    resumed.config.resume_from_checkpoint = str(tmp_path / "interrupted" / "resume")
    await resumed.train()
    assert first_rates + remaining_rates == pytest.approx(baseline_rates)
    assert_equal(baseline.agent.model.state_dict(), resumed.agent.model.state_dict())
    assert_equal(baseline.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_equal(baseline.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
    assert baseline.global_step == resumed.global_step
    assert resumed._grad_accum_step == 0


@pytest.mark.parametrize(
    "batches,accumulation,inner,expected",
    [
        (5, 2, 1, 3),
        (5, 2, 3, 15),
        (5, 8, 3, 15),
        (0, 2, 3, 0),
        (0, 2, 1, 0),
        (10**20 + 1, 2, 1, 5 * 10**19 + 1),
    ],
)
def test_budget_and_config_estimates_agree(batches, accumulation, inner, expected):
    config = TrainingConfig(
        num_episodes=batches,
        gradient_accumulation_steps=accumulation,
        num_gradient_updates=inner,
        warmup_ratio=0.2,
    )
    assert (
        estimate_grpo_optimizer_steps(
            batches,
            gradient_accumulation_steps=accumulation,
            num_gradient_updates=inner,
        )
        == expected
    )
    assert config.get_total_steps() == expected
    assert config.get_total_steps(0) == 0
    assert config.get_warmup_steps() == int(expected * 0.2)
