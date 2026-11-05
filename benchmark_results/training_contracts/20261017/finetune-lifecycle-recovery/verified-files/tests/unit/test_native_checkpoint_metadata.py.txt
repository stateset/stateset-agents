"""Partial metadata without weights cannot change a native learner's status."""

from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from stateset_agents.training.single_turn_trainer import (  # noqa: E402
    SingleTurnGRPOTrainer,
)
from tests.unit.test_multi_turn_checkpoint_accumulation import (  # noqa: E402
    accumulate,
    assert_equal,
)
from tests.unit.test_multi_turn_checkpoint_accumulation import (  # noqa: E402
    trainer as make_native,
)
from tests.unit.test_native_checkpoint_safety import snapshot  # noqa: E402


@pytest.mark.parametrize(
    "trainer_type",
    [SingleTurnGRPOTrainer, MultiTurnGRPOTrainer],
    ids=["single", "multi"],
)
@pytest.mark.parametrize(
    "initial_poison", [False, True], ids=["healthy", "quarantined"]
)
def test_partial_native_metadata_without_weights_preserves_learner_and_quarantine(
    tmp_path, trainer_type, initial_poison
):
    target = make_native(tmp_path / "target", trainer_type=trainer_type)
    # Preserve meaningful optimizer slots and pending gradients, rather than
    # merely proving a fresh zero-step learner stays empty.
    accumulate(target, [1, 2])
    accumulate(target, [2, 3])
    accumulate(target, [3, 2])
    target._runtime_checkpoint_poisoned = initial_poison
    sync = target.agent.sync_rollout_backend = Mock(return_value=True)
    tokenizer = target.agent.tokenizer
    before = snapshot(target)
    path = tmp_path / "partial"
    path.mkdir()
    torch.save({"global_step": 7, "current_epoch": 42}, path / "training_state.pt")

    assert target.load_checkpoint(path) is False
    assert_equal(snapshot(target), before)
    assert target._runtime_checkpoint_poisoned is initial_poison
    assert target.agent.tokenizer is tokenizer
    sync.assert_not_called()
