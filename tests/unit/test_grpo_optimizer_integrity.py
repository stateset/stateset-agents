"""Real optimizer/AMP state must advance only for valid, committed updates."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer
from stateset_agents.training.trainer_utils import backward_training_loss


def make_trainer(trainer_type):
    model = torch.nn.Linear(2, 1, bias=False)
    value = trainer_type(
        SimpleNamespace(model=model, sync_rollout_backend=MagicMock(return_value=True)),
        object(),
        config=SimpleNamespace(max_grad_norm=1.0, gradient_accumulation_steps=2),
    )
    value.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    value.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        value.optimizer, lambda step: 0.9**step
    )
    return value


@pytest.fixture(
    params=[SingleTurnGRPOTrainer, MultiTurnGRPOTrainer], ids=["single", "multi"]
)
def trainer(request):
    return make_trainer(request.param)


def apply(trainer):
    if isinstance(trainer, SingleTurnGRPOTrainer):
        return trainer._apply_optimizer_step(
            torch, trainer.scaler is not None, trainer.config.max_grad_norm
        )
    return trainer._apply_optimizer_step(torch)


def backward(trainer, *, multiplier=1.0):
    loss = trainer.agent.model(torch.ones(1, 2)).sum() * multiplier
    backward_training_loss(trainer, loss, torch, scaler=trainer.scaler)
    trainer._grad_accum_step += 1


def equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def snapshot(trainer):
    return copy.deepcopy(
        (
            trainer.agent.model.state_dict(),
            trainer.optimizer.state_dict(),
            trainer.lr_scheduler.state_dict(),
            trainer.global_step,
        )
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_gradients_cannot_mutate_optimizer_or_weights(trainer, bad):
    backward(trainer)
    assert apply(trainer)
    before = snapshot(trainer)
    trainer.agent.sync_rollout_backend.reset_mock()
    backward(trainer)
    trainer.agent.model.weight.grad[0, 0] = bad
    with pytest.raises(ValueError, match="gradients must be finite"):
        apply(trainer)
    equal(before, snapshot(trainer))
    assert trainer.agent.model.weight.grad is None
    assert trainer._grad_accum_step == 0
    trainer.agent.sync_rollout_backend.assert_not_called()


@pytest.mark.parametrize("bad", ["nan", "infinite", "vector", "detached", "float"])
def test_invalid_loss_discards_pending_accumulation(trainer, bad):
    backward(trainer)
    before = snapshot(trainer)
    loss = {
        "nan": lambda: trainer.agent.model.weight.sum() * float("nan"),
        "infinite": lambda: trainer.agent.model.weight.sum() * float("inf"),
        "vector": lambda: trainer.agent.model.weight,
        "detached": lambda: trainer.agent.model.weight.sum().detach(),
        "float": lambda: 1.0,
    }[bad]()
    with pytest.raises(ValueError, match="finite differentiable scalar"):
        backward_training_loss(trainer, loss, torch)
    equal(before, snapshot(trainer))
    assert trainer.agent.model.weight.grad is None and trainer._grad_accum_step == 0


@pytest.mark.parametrize("max_norm", [float("nan"), float("inf"), -1.0])
def test_invalid_clipping_configuration_cannot_apply_update(trainer, max_norm):
    backward(trainer)
    before = snapshot(trainer)
    trainer.config.max_grad_norm = max_norm
    with pytest.raises(ValueError, match="max_grad_norm"):
        apply(trainer)
    equal(before, snapshot(trainer))
    assert trainer.agent.model.weight.grad is None


def test_empty_gradient_window_is_not_an_optimizer_step(trainer):
    backward(trainer)
    apply(trainer)
    before = snapshot(trainer)
    trainer.agent.sync_rollout_backend.reset_mock()
    assert apply(trainer) is False
    equal(before, snapshot(trainer))
    trainer.agent.sync_rollout_backend.assert_not_called()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_real_amp_overflow_skips_update_then_recovers(trainer, bad):
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=2)
    backward(trainer)
    assert apply(trainer)
    before = snapshot(trainer)
    scale = trainer.scaler.get_scale()
    trainer.agent.sync_rollout_backend.reset_mock()
    backward(trainer)
    trainer.agent.model.weight.grad[0, 0] = bad
    assert apply(trainer) is False
    equal(before, snapshot(trainer))
    assert trainer.scaler.get_scale() == scale / 2
    assert trainer.agent.model.weight.grad is None and trainer._grad_accum_step == 0
    trainer.agent.sync_rollout_backend.assert_not_called()
    backward(trainer)
    assert apply(trainer) is True
    assert trainer.global_step == before[-1] + 1
    trainer.agent.sync_rollout_backend.assert_called_once()


@pytest.mark.parametrize("amp", [False, True])
def test_finite_gradients_with_overflowing_norm_fail_before_step(trainer, amp):
    if amp:
        trainer.scaler = torch.amp.GradScaler("cpu", init_scale=1, growth_interval=2)
    backward(trainer)
    trainer.agent.model.weight.grad.fill_(1e30)
    before = snapshot(trainer)
    scaler_before = copy.deepcopy(trainer.scaler.state_dict()) if amp else None
    with pytest.raises(RuntimeError, match="non-finite"):
        apply(trainer)
    equal(before, snapshot(trainer))
    if amp:
        equal(scaler_before, trainer.scaler.state_dict())
    assert trainer.agent.model.weight.grad is None
    # Rejected clipping must not strand GradScaler in an already-unscaled phase.
    backward(trainer)
    assert apply(trainer) is True


def test_scheduler_failure_keeps_committed_step_count(trainer):
    backward(trainer)
    before = trainer.agent.model.weight.detach().clone()
    trainer.lr_scheduler.step = MagicMock(side_effect=RuntimeError("scheduler failed"))
    with pytest.raises(RuntimeError, match="scheduler failed"):
        apply(trainer)
    assert not torch.equal(before, trainer.agent.model.weight)
    assert trainer.global_step == 1
    assert trainer.agent.model.weight.grad is None


class OverflowBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("inf"))


@pytest.mark.asyncio
@pytest.mark.parametrize("inner", [1, 3])
async def test_multi_turn_metrics_count_only_committed_inner_updates(
    monkeypatch, inner
):
    from stateset_agents.training import multi_turn_trainer as module

    value = make_trainer(MultiTurnGRPOTrainer)
    value.config.num_gradient_updates = inner
    value.config.gradient_accumulation_steps = 1
    value.scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=2)
    calls = 0

    def loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = value.agent.model(torch.ones(1, 2)).sum()
        if calls != 2:
            result = OverflowBackward.apply(result)
        return {"total_loss": result}

    value.compute_grpo_loss = loss
    monkeypatch.setattr(
        module, "compute_token_old_logprobs", lambda *args: [torch.zeros(1)]
    )
    metrics = await value.training_step([])
    expected = int(inner == 3)
    assert metrics["optimizer_step"] is bool(expected)
    assert metrics["optimizer_updates"] == expected
    assert metrics["global_step"] == value.global_step == expected
    assert value.lr_scheduler.last_epoch == expected
    assert value.agent.sync_rollout_backend.call_count == expected
    assert value.agent.model.weight.grad is None


@pytest.mark.asyncio
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("inner", [1, 3])
async def test_single_turn_loop_rejects_or_skips_invalid_backward(
    monkeypatch, amp, inner
):
    from unittest.mock import AsyncMock

    from stateset_agents.training import single_turn_trainer as module

    value = make_trainer(SingleTurnGRPOTrainer)
    value.config.num_episodes = 1
    value.config.max_steps_per_episode = 1
    value.config.num_gradient_updates = inner
    value.config.gradient_accumulation_steps = 1
    value.config.fp16 = amp
    if amp:
        value.scaler = torch.amp.GradScaler("cpu", init_scale=128)
    value.environment = SimpleNamespace(
        reset=AsyncMock(return_value={"prompt": "hello"}),
        step=AsyncMock(return_value={"done": True, "reward": 1.0}),
    )
    value._setup_scheduler = MagicMock()
    value._generate_trajectory_group = AsyncMock(
        return_value=(SimpleNamespace(rewards=[0, 1]), "answer")
    )
    callback = SimpleNamespace(on_training_end=AsyncMock())
    value.callbacks = [callback]

    def loss(*args, **kwargs):
        result = value.agent.model(torch.ones(1, 2)).sum()
        return {"total_loss": OverflowBackward.apply(result), "path": "token"}

    monkeypatch.setattr(module, "compute_grpo_loss", loss)
    monkeypatch.setattr(
        module, "compute_token_old_logprobs", lambda *args: [torch.zeros(1)]
    )
    before = snapshot(value)
    if amp:
        assert await value.train() is value.agent
        callback.on_training_end.assert_awaited_once()
        assert value.scaler.get_scale() == 128 / 2**inner
    else:
        with pytest.raises(ValueError, match="gradients must be finite"):
            await value.train()
        callback.on_training_end.assert_not_awaited()
    equal(before, snapshot(value))
    assert value._grad_accum_step == 0 and value.agent.model.weight.grad is None
    value.agent.sync_rollout_backend.assert_not_called()


@pytest.mark.parametrize("amp", [False, True])
def test_checks_every_optimizer_parameter_group(trainer, amp):
    extra = torch.nn.Parameter(torch.ones(1))
    trainer.agent.model.register_parameter("extra", extra)
    trainer.optimizer.add_param_group({"params": [extra]})
    if amp:
        trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128)
    backward(trainer)
    extra.grad = torch.full_like(extra, float("inf"))
    before = snapshot(trainer)
    if amp:
        assert apply(trainer) is False
        assert trainer.scaler.get_scale() == 64
    else:
        with pytest.raises(ValueError, match="gradients must be finite"):
            apply(trainer)
    equal(before, snapshot(trainer))
    assert all(parameter.grad is None for parameter in trainer.agent.model.parameters())
    trainer.agent.sync_rollout_backend.assert_not_called()
