"""Final partial GRPO windows must match the mean of their actual batches."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer
from stateset_agents.training.trainer_utils import (
    backward_training_loss,
    safe_optimizer_step,
)


class TinyModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.weight.fill_(0.75)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path / "pytorch_model.bin")


def make_trainer(path, trainer_type, *, amp, max_norm):
    model = TinyModel()
    value = trainer_type(
        SimpleNamespace(model=model, tokenizer=None),
        object(),
        config=SimpleNamespace(
            output_dir=str(path),
            num_episodes=1,
            gradient_accumulation_steps=4,
            max_grad_norm=max_norm,
            fp16=amp,
        ),
    )
    value.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    value.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        value.optimizer, lambda step: 0.9**step
    )
    value.scaler = (
        torch.amp.GradScaler("cpu", init_scale=64, growth_interval=2) if amp else None
    )
    value._setup_scheduler = MagicMock()
    value._get_training_scenarios = lambda: [{}]
    value._get_eval_scenarios = lambda: []
    return value


def batch_loss(model, values):
    return model(torch.tensor([values], dtype=torch.float64)).square().mean()


def accumulate(value, values):
    loss = (
        batch_loss(value.agent.model, values) / value.config.gradient_accumulation_steps
    )
    backward_training_loss(value, loss, torch, scaler=value.scaler)
    value._grad_accum_step += 1
    if value._grad_accum_step % value.config.gradient_accumulation_steps == 0:
        if isinstance(value, SingleTurnGRPOTrainer):
            value._apply_optimizer_step(
                torch, value.scaler is not None, value.config.max_grad_norm
            )
        else:
            value._apply_optimizer_step(torch)


def close(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            close(first, second)
    else:
        assert left == right


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trainer_type",
    [SingleTurnGRPOTrainer, MultiTurnGRPOTrainer],
    ids=["single", "multi"],
)
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("pending", [1, 2, 3])
@pytest.mark.parametrize("max_norm", [10.0, 0.15])
async def test_completed_resume_flush_matches_direct_partial_mean(
    tmp_path, trainer_type, amp, pending, max_norm
):
    original = make_trainer(tmp_path, trainer_type, amp=amp, max_norm=max_norm)
    # Warm optimizer momentum, scheduler, and AMP growth state with a full window.
    for values in ([0.1, 0.5], [0.2, 0.4], [0.3, 0.7], [0.4, 0.2]):
        accumulate(original, values)
    assert original.global_step == 1
    reference = copy.deepcopy(original.agent.model)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    optimizer.load_state_dict(copy.deepcopy(original.optimizer.state_dict()))
    scheduler.load_state_dict(copy.deepcopy(original.lr_scheduler.state_dict()))
    scaler = (
        torch.amp.GradScaler("cpu", init_scale=64, growth_interval=2) if amp else None
    )
    if amp:
        scaler.load_state_dict(original.scaler.state_dict())
    batches = [[0.3, 0.8], [0.2, 0.6], [0.9, 0.1]][:pending]
    for values in batches:
        accumulate(original, values)
    original.current_epoch = 0
    await original.save_checkpoint(checkpoint_name="pending")
    restored = make_trainer(
        tmp_path / "restored", trainer_type, amp=amp, max_norm=max_norm
    )
    restored.config.resume_from_checkpoint = str(tmp_path / "pending")
    # Actual checkpoint recovery and the trainer's final flush, with no episodes left.
    assert await restored.train() is restored.agent

    # Independent reference: one mean loss, without accumulation correction.
    loss = sum(batch_loss(reference, values) for values in batches) / pending
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(
        reference.parameters(), max_norm, error_if_nonfinite=True
    )
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    close(reference.state_dict(), restored.agent.model.state_dict())
    close(optimizer.state_dict(), restored.optimizer.state_dict())
    close(scheduler.state_dict(), restored.lr_scheduler.state_dict())
    if amp:
        close(scaler.state_dict(), restored.scaler.state_dict())
    assert restored.global_step == 2
    assert restored._grad_accum_step == 0
    assert all(
        parameter.grad is None for parameter in restored.agent.model.parameters()
    )

    # A second resume must not repeat the flushed update (including momentum).
    await restored.save_checkpoint(checkpoint_name="completed")
    completed = make_trainer(
        tmp_path / "completed-run", trainer_type, amp=amp, max_norm=max_norm
    )
    completed.config.resume_from_checkpoint = str(tmp_path / "restored" / "completed")
    assert await completed.train() is completed.agent
    close(restored.agent.model.state_dict(), completed.agent.model.state_dict())
    close(restored.optimizer.state_dict(), completed.optimizer.state_dict())
    close(restored.lr_scheduler.state_dict(), completed.lr_scheduler.state_dict())
    if amp:
        close(restored.scaler.state_dict(), completed.scaler.state_dict())
    assert completed.global_step == 2
    assert completed._grad_accum_step == 0
    assert all(
        parameter.grad is None for parameter in completed.agent.model.parameters()
    )


@pytest.mark.parametrize("correction", [0, -1, float("nan"), float("inf")])
def test_invalid_correction_cannot_mutate_optimizer(tmp_path, correction):
    value = make_trainer(tmp_path, SingleTurnGRPOTrainer, amp=False, max_norm=1)
    accumulate(value, [0.2, 0.7])
    weights = copy.deepcopy(value.agent.model.state_dict())
    optimizer = copy.deepcopy(value.optimizer.state_dict())
    with pytest.raises(ValueError, match="gradient_scale"):
        safe_optimizer_step(value, torch, max_grad_norm=1, gradient_scale=correction)
    close(weights, value.agent.model.state_dict())
    close(optimizer, value.optimizer.state_dict())
    assert value.global_step == 0 and value._grad_accum_step == 0
    assert value.agent.model.weight.grad is None


def test_overflow_from_correction_is_not_an_amp_skip(tmp_path):
    value = make_trainer(tmp_path, SingleTurnGRPOTrainer, amp=True, max_norm=1)
    accumulate(value, [0.2, 0.7])
    weights = copy.deepcopy(value.agent.model.state_dict())
    scaler = copy.deepcopy(value.scaler.state_dict())
    with pytest.raises(RuntimeError, match="non-finite"):
        safe_optimizer_step(
            value, torch, max_grad_norm=1, scaler=value.scaler, gradient_scale=1e308
        )
    close(weights, value.agent.model.state_dict())
    close(scaler, value.scaler.state_dict())
    assert value.global_step == 0 and value._grad_accum_step == 0
    assert value.agent.model.weight.grad is None
