"""A real native restore publishes recovered weights once, after full commit."""

from unittest.mock import MagicMock

import pytest

from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer
from tests.unit.test_multi_turn_checkpoint_accumulation import accumulate, assert_equal
from tests.unit.test_multi_turn_checkpoint_accumulation import trainer as make_trainer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trainer_type",
    [SingleTurnGRPOTrainer, MultiTurnGRPOTrainer],
    ids=["single", "multi"],
)
@pytest.mark.parametrize("loaded", [True, False])
async def test_successful_load_syncs_rollout_backend(tmp_path, trainer_type, loaded):
    original = make_trainer(tmp_path / "original", trainer_type=trainer_type)
    accumulate(original, [1, 2])
    accumulate(original, [2, 3])
    await original.save_checkpoint(checkpoint_name="saved")
    target = make_trainer(tmp_path / "target", trainer_type=trainer_type)
    sync = MagicMock(return_value=True)
    target.agent.sync_rollout_backend = sync
    path = tmp_path / "original" / "saved" if loaded else tmp_path / "missing"
    assert target.load_checkpoint(path) is loaded
    assert sync.call_count == (1 if loaded else 0)
    if loaded:
        assert_equal(target.agent.model.state_dict(), original.agent.model.state_dict())
        assert_equal(target.optimizer.state_dict(), original.optimizer.state_dict())
        assert not target._runtime_checkpoint_poisoned
    else:
        assert target.global_step == 0
