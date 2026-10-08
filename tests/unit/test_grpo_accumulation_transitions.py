"""Switching rollout loss paths must keep accumulated and inner updates separate."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from stateset_agents.training import multi_turn_trainer as multi
from stateset_agents.training import single_turn_trainer as single
from stateset_agents.training.checkpoint_state import _capture_accumulation


def components(amp):
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0.75)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    scaler = (
        torch.amp.GradScaler("cpu", init_scale=64, growth_interval=2) if amp else None
    )
    return model, optimizer, scheduler, scaler


def objective(model, target, ewc):
    result = (model.weight - target).square().sum()
    if ewc:
        result = result + 0.2 * model.weight.square().sum()
    return result


def reference_update(parts, targets, max_norm, ewc):
    model, optimizer, scheduler, scaler = parts
    loss = sum(objective(model, target, ewc) for target in targets) / len(targets)
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, error_if_nonfinite=True
    )
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


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
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("ewc", [False, True])
@pytest.mark.parametrize("max_norm", [0.15, 10.0])
async def test_sequence_token_sequence_matches_separate_reference_updates(
    monkeypatch, tmp_path, kind, amp, ewc, max_norm
):
    model, optimizer, scheduler, scaler = components(amp)
    module = single if kind == "single" else multi
    trainer_type = (
        single.SingleTurnGRPOTrainer if kind == "single" else multi.MultiTurnGRPOTrainer
    )
    trainer = trainer_type(
        SimpleNamespace(model=model, sync_rollout_backend=MagicMock(return_value=True)),
        SimpleNamespace(
            reset=AsyncMock(return_value={"prompt": "hello"}),
            step=AsyncMock(return_value={"done": True, "reward": 1.0}),
        ),
        config=SimpleNamespace(
            output_dir=str(tmp_path),
            num_episodes=4,
            max_steps_per_episode=1,
            num_gradient_updates=2,
            gradient_accumulation_steps=4,
            max_grad_norm=max_norm,
            fp16=amp,
        ),
    )
    trainer.optimizer, trainer.lr_scheduler, trainer.scaler = (
        optimizer,
        scheduler,
        scaler,
    )
    trainer._setup_scheduler = MagicMock()
    trainer._resolve_task_id = lambda _: None
    trainer._maybe_handle_task_switch = MagicMock()
    if ewc:
        trainer.continual_manager = SimpleNamespace(
            reference_model=None,
            sample_replay_groups=lambda _: [],
            add_trajectory_groups=lambda *args, **kwargs: None,
            get_effective_beta=lambda beta: beta,
            compute_ewc_penalty=lambda agent: 0.2 * agent.model.weight.square().sum(),
            buffer=SimpleNamespace(size=0),
        )
    groups = [
        SimpleNamespace(path=path, target=target, rewards=[0, 1])
        for path, target in [
            ("sequence", 0.1),
            ("sequence", -0.3),
            ("token", 1.2),
            ("sequence", -0.6),
        ]
    ]
    snapshots = []
    token_forwards = []

    def old_policy(batch, *args):
        if batch[0].path != "token":
            return None
        result = [model.weight.detach().clone()]
        snapshots.append(result)
        return result

    def loss(trajectory_groups, **kwargs):
        group = trajectory_groups[0]
        old = kwargs.get("old_logprobs")
        if old is not None:
            assert old is snapshots[0]
            token_forwards.append(model.weight.detach().clone())
        return {"total_loss": objective(model, group.target, False), "path": group.path}

    monkeypatch.setattr(module, "compute_token_old_logprobs", old_policy)
    if kind == "single":
        monkeypatch.setattr(module, "compute_grpo_loss", loss)
        trainer._generate_trajectory_group = AsyncMock(
            side_effect=[(group, "answer") for group in groups]
        )
        await trainer.train()
    else:
        trainer.compute_grpo_loss = loss
        first = await trainer.training_step([groups[0]])
        second = await trainer.training_step([groups[1]])
        assert first["optimizer_updates"] == second["optimizer_updates"] == 0
        inner = await trainer.training_step([groups[2]])
        assert inner["optimizer_updates"] == 3
        assert inner["accumulation_flush_updates"] == 1
        assert inner["inner_updates"] == 2
        assert inner["grad_accum_step"] == 0
        assert _capture_accumulation(trainer, torch)["gradients"] == {}
        last = await trainer.training_step([groups[3]])
        assert last["grad_accum_step"] == 1
        assert last["optimizer_updates"] == 0
        trainer._flush_accumulation(torch)

    # The old policy is captured BEFORE closing accumulated work. Every inner
    # forward is rebuilt AFTER that commit, so no stale autograd graph is used.
    reference = components(amp)
    assert_equal(snapshots[0][0], reference[0].weight)
    reference_update(reference, [0.1, -0.3], max_norm, ewc)
    assert_equal(token_forwards[0], reference[0].weight)
    reference_update(reference, [1.2], max_norm, ewc)
    assert_equal(token_forwards[1], reference[0].weight)
    reference_update(reference, [1.2], max_norm, ewc)
    reference_update(reference, [-0.6], max_norm, ewc)
    for actual, expected in zip(
        (model, optimizer, scheduler, scaler), reference, strict=True
    ):
        if actual is not None:
            assert_equal(actual.state_dict(), expected.state_dict())
    assert trainer.global_step == 4
    assert trainer.agent.sync_rollout_backend.call_count == 4
    assert trainer._grad_accum_step == 0
    assert model.weight.grad is None
    # Checkpoint capture sees no stale pending window after the transition.
    assert _capture_accumulation(trainer, torch)["gradients"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("amp", [False, True])
async def test_invalid_pending_window_cannot_contaminate_inner_updates(
    monkeypatch, amp
):
    model, optimizer, scheduler, scaler = components(amp)
    trainer = multi.MultiTurnGRPOTrainer(
        SimpleNamespace(model=model, sync_rollout_backend=MagicMock()),
        object(),
        config=SimpleNamespace(
            num_gradient_updates=2, gradient_accumulation_steps=4, max_grad_norm=10.0
        ),
    )
    trainer.optimizer, trainer.lr_scheduler, trainer.scaler = (
        optimizer,
        scheduler,
        scaler,
    )
    pending_loss = objective(model, 0.1, False) / 4
    if scaler is None:
        pending_loss.backward()
    else:
        scaler.scale(pending_loss).backward()
    model.weight.grad.fill_(float("inf"))
    trainer._grad_accum_step = 1
    old = [model.weight.detach().clone()]
    monkeypatch.setattr(multi, "compute_token_old_logprobs", lambda *args: old)

    def loss(*args, **kwargs):
        assert kwargs["old_logprobs"] is old
        assert model.weight.grad is None
        return {"total_loss": objective(model, 1.2, False)}

    trainer.compute_grpo_loss = MagicMock(side_effect=loss)
    if not amp:
        with pytest.raises(ValueError, match="gradients must be finite"):
            await trainer.training_step([])
        trainer.compute_grpo_loss.assert_not_called()
        assert_equal(model.weight, old[0])
        assert trainer.global_step == scheduler.last_epoch == 0
        assert not optimizer.state
        trainer.agent.sync_rollout_backend.assert_not_called()
    else:
        metrics = await trainer.training_step([])
        assert metrics["accumulation_flush_updates"] == 0
        assert metrics["optimizer_updates"] == metrics["inner_updates"] == 2
        reference = components(True)
        reference[3].load_state_dict({**reference[3].state_dict(), "scale": 32.0})
        for _ in range(2):
            reference_update(reference, [1.2], 10.0, False)
        for actual, expected in zip(
            (model, optimizer, scheduler, scaler), reference, strict=True
        ):
            assert_equal(actual.state_dict(), expected.state_dict())
        assert trainer.global_step == trainer.agent.sync_rollout_backend.call_count == 2
    assert trainer._grad_accum_step == 0
    assert model.weight.grad is None
