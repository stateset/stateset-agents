"""Process topology and wrapping contracts for Accelerate training."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from accelerate import Accelerator

from stateset_agents.training import distributed
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.distributed import DistributedConfig, DistributedTrainer
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer


def trainer_with_accelerator(accelerator, *, world_size=1):
    trainer = object.__new__(DistributedTrainer)
    trainer.distributed_config = DistributedConfig(
        strategy="accelerate", world_size=world_size
    )
    trainer.config = SimpleNamespace(bf16=False, fp16=False, report_to="none")
    trainer.accelerator = accelerator
    trainer.agent = SimpleNamespace(model=torch.nn.Linear(1, 1))
    trainer.optimizer = torch.optim.SGD(trainer.agent.model.parameters(), lr=0.1)
    trainer.lr_scheduler = None
    return trainer


def test_accelerate_topology_comes_from_runtime(monkeypatch):
    accelerator = SimpleNamespace(
        is_main_process=False,
        num_processes=4,
        process_index=2,
        local_process_index=0,
    )
    factory = MagicMock(return_value=accelerator)
    monkeypatch.setattr(distributed, "Accelerator", factory)
    trainer = trainer_with_accelerator(None, world_size=1)

    trainer._init_distributed()

    assert trainer.distributed_config.world_size == 4
    assert trainer.distributed_config.rank == 2
    assert trainer.distributed_config.local_rank == 0
    assert trainer.is_main_process is False
    factory.assert_called_once()


@pytest.mark.parametrize("override, expected", [(None, 4), (2, 2)])
def test_accumulation_schedule_matches_base_trainer_and_accelerate(
    monkeypatch, override, expected
):
    factory = MagicMock(
        return_value=SimpleNamespace(
            is_main_process=True,
            num_processes=1,
            process_index=0,
            local_process_index=0,
        )
    )
    monkeypatch.setattr(distributed, "Accelerator", factory)
    monkeypatch.setattr(
        MultiTurnGRPOTrainer, "__init__", lambda *_args, **_kwargs: None
    )
    training_config = TrainingConfig(gradient_accumulation_steps=4)

    trainer = DistributedTrainer(
        agent=SimpleNamespace(),
        environment=SimpleNamespace(),
        config=training_config,
        distributed_config=DistributedConfig(
            strategy="accelerate", gradient_accumulation_steps=override
        ),
    )

    assert trainer.config.gradient_accumulation_steps == expected
    assert trainer.distributed_config.gradient_accumulation_steps == expected
    assert training_config.gradient_accumulation_steps == 4
    assert factory.call_args.kwargs["gradient_accumulation_steps"] == expected


def test_invalid_accumulation_count_fails_before_runtime_setup(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(distributed, "Accelerator", factory)

    with pytest.raises(ValueError, match="positive integer"):
        DistributedTrainer(
            agent=SimpleNamespace(),
            environment=SimpleNamespace(),
            distributed_config=DistributedConfig(
                strategy="accelerate", gradient_accumulation_steps=0
            ),
        )

    factory.assert_not_called()


def test_accelerate_rejects_conflicting_inner_update_schedule(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(distributed, "Accelerator", factory)

    with pytest.raises(ValueError, match="multiple inner optimizer updates"):
        DistributedTrainer(
            agent=SimpleNamespace(),
            environment=SimpleNamespace(),
            config=TrainingConfig(
                gradient_accumulation_steps=2, num_gradient_updates=2
            ),
            distributed_config=DistributedConfig(strategy="accelerate"),
        )

    factory.assert_not_called()


@pytest.mark.asyncio
async def test_accelerate_prepares_even_single_process(monkeypatch):
    monkeypatch.setattr(MultiTurnGRPOTrainer, "initialize", AsyncMock())
    trainer = trainer_with_accelerator(SimpleNamespace(), world_size=1)
    trainer._wrap_model_accelerate = MagicMock()

    await trainer.initialize()

    trainer._wrap_model_accelerate.assert_called_once_with()


def test_accelerate_prepare_omits_missing_scheduler():
    prepared_model = object()
    prepared_optimizer = object()
    accelerator = SimpleNamespace(
        prepare=MagicMock(return_value=(prepared_model, prepared_optimizer))
    )
    trainer = trainer_with_accelerator(accelerator)

    trainer._wrap_model_accelerate()

    assert trainer.agent.model is prepared_model
    assert trainer.optimizer is prepared_optimizer
    assert trainer.lr_scheduler is None
    assert len(accelerator.prepare.call_args.args) == 2


def test_accelerate_metric_sync_uses_accelerator_device_and_reduce():
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=MagicMock(side_effect=lambda tensor, reduction: tensor),
    )
    trainer = trainer_with_accelerator(accelerator, world_size=2)

    synced = trainer._sync_metrics(
        {"loss": 0.25, "optimizer_step": True, "label": "ok"}
    )

    assert synced == {"loss": 0.25, "optimizer_step": True, "label": "ok"}
    assert accelerator.reduce.call_count == 2
    assert all(
        call.kwargs["reduction"] == "mean" for call in accelerator.reduce.call_args_list
    )


def test_manual_metric_sync_uses_model_parameter_device(monkeypatch):
    trainer = trainer_with_accelerator(None, world_size=2)

    def sum_across_workers(tensor, *, op):
        assert op == distributed.dist.ReduceOp.SUM
        tensor.mul_(2)

    monkeypatch.setattr(distributed.dist, "all_reduce", sum_across_workers)

    assert trainer._sync_metrics({"loss": 0.25, "optimizer_step": True}) == {
        "loss": 0.25,
        "optimizer_step": True,
    }


def test_real_accelerate_prepares_cpu_model_and_optimizer():
    accelerator = Accelerator(cpu=True)
    trainer = trainer_with_accelerator(accelerator)

    trainer._wrap_model_accelerate()

    assert isinstance(trainer.agent.model, torch.nn.Module)
    assert next(trainer.agent.model.parameters()).device.type == "cpu"
    assert trainer.optimizer is not None


@pytest.mark.asyncio
async def test_real_accelerate_accumulates_before_optimizer_step():
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    trainer = trainer_with_accelerator(accelerator)
    trainer.config = TrainingConfig(
        gradient_accumulation_steps=2,
        bf16=False,
        fp16=False,
        rollout_sync=False,
    )
    trainer.continual_manager = None
    trainer.reference_model = None
    trainer.scaler = None
    trainer._grad_accum_step = 0
    trainer.global_step = 0
    trainer.compute_grpo_loss = lambda _groups: {
        "total_loss": trainer.agent.model(torch.ones(1, 1)).square().mean()
    }
    trainer._wrap_model_accelerate()
    before = trainer.agent.model.weight.detach().clone()

    first = await trainer.training_step([])
    after_first = trainer.agent.model.weight.detach().clone()
    second = await trainer.training_step([])

    assert first["optimizer_step"] is False
    assert second["optimizer_step"] is True
    assert trainer.global_step == 1
    assert torch.equal(before, after_first)
    assert not torch.equal(after_first, trainer.agent.model.weight.detach())
