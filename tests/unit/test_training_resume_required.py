"""Explicit resume requests must not silently become new training runs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer


@pytest.fixture(
    params=[SingleTurnGRPOTrainer, MultiTurnGRPOTrainer], ids=["single", "multi"]
)
def resume_trainer(request, tmp_path):
    callback = SimpleNamespace(
        fail_on_error=True,
        on_training_start=AsyncMock(),
        on_training_end=AsyncMock(),
    )
    environment = SimpleNamespace(reset=AsyncMock(), step=AsyncMock())
    model = torch.nn.Linear(2, 1)
    trainer = request.param(
        SimpleNamespace(model=model, tokenizer=None),
        environment,
        config=SimpleNamespace(
            num_episodes=2,
            gradient_accumulation_steps=1,
            resume_from_checkpoint=str(tmp_path / "checkpoint"),
        ),
        callbacks=[callback],
    )
    trainer._setup_scheduler = MagicMock()
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.generate_trajectories = AsyncMock()
    trainer.save_checkpoint = AsyncMock()
    trainer._get_training_scenarios = MagicMock(return_value=[{}, {}])
    trainer._get_eval_scenarios = MagicMock(return_value=[])
    return trainer, callback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["missing", "empty", "invalid_state", "blank", "false", "none", "truthy"]
)
async def test_failed_resume_stops_before_start_callbacks_or_rollouts(
    resume_trainer, tmp_path, failure
):
    trainer, callback = resume_trainer
    path = tmp_path / "checkpoint"
    if failure in ("empty", "invalid_state"):
        path.mkdir()
        if failure == "invalid_state":
            torch.save(["not training state"], path / "training_state.pt")
    elif failure == "blank":
        trainer.config.resume_from_checkpoint = "  "
    elif failure in ("false", "none", "truthy"):
        trainer.load_checkpoint = MagicMock(
            return_value={"false": False, "none": None, "truthy": 1}[failure]
        )
    weights = trainer.agent.model.weight.detach().clone()
    with pytest.raises(ValueError, match="resume_from_checkpoint|instead of resuming"):
        await trainer.train()
    assert torch.equal(weights, trainer.agent.model.weight)
    assert trainer.global_step == 0
    callback.on_training_start.assert_not_awaited()
    callback.on_training_end.assert_not_awaited()
    trainer.environment.reset.assert_not_awaited()
    trainer.environment.step.assert_not_awaited()
    trainer.generate_trajectories.assert_not_awaited()
    trainer.save_checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_loader_exception_is_preserved_without_fresh_training(resume_trainer):
    trainer, callback = resume_trainer
    error = RuntimeError("optimizer restore failed")
    trainer.load_checkpoint = MagicMock(side_effect=error)
    with pytest.raises(RuntimeError) as caught:
        await trainer.train()
    assert caught.value is error
    callback.on_training_start.assert_not_awaited()
    trainer.environment.reset.assert_not_awaited()
    trainer.generate_trajectories.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_completed_resume_skips_already_finished_episodes(
    resume_trainer,
):
    trainer, callback = resume_trainer

    def restore(path):
        trainer.current_epoch = 1
        trainer.global_step = 7
        return True

    trainer.load_checkpoint = MagicMock(side_effect=restore)
    assert await trainer.train() is trainer.agent
    assert trainer.global_step == 7
    callback.on_training_start.assert_awaited_once()
    callback.on_training_end.assert_awaited_once()
    trainer.environment.reset.assert_not_awaited()
    trainer.generate_trajectories.assert_not_awaited()
    trainer.load_checkpoint.assert_called_once_with(
        trainer.config.resume_from_checkpoint
    )
