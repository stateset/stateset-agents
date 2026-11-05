"""``spine.grpo_update.GRPOUpdateMixin``: one GRPO update for both trainers.

The numerical equivalence gate is the step goldens (multi-turn and
single-turn, K=1/K=2 and a partial accumulation window). These tests pin the
sharing itself and the paths the goldens do not drive.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from stateset_agents.training.single_turn_trainer import (  # noqa: E402
    SingleTurnGRPOTrainer,
)
from stateset_agents.training.spine import GRPOUpdateMixin  # noqa: E402
from tests.unit.test_grpo_inner_updates import (  # noqa: E402
    _config,
    _group,
    _tiny_model,
)


def test_both_trainers_run_the_same_update() -> None:
    for cls in (MultiTurnGRPOTrainer, SingleTurnGRPOTrainer):
        assert issubclass(cls, GRPOUpdateMixin)
        for name in ("_grpo_update", "_inner_update_steps", "_update_metrics"):
            assert getattr(cls, name) is getattr(GRPOUpdateMixin, name), (cls, name)


def _single_turn(num_gradient_updates: int, optimizer: bool) -> SingleTurnGRPOTrainer:
    model = _tiny_model()
    trainer = SingleTurnGRPOTrainer(
        agent=SimpleNamespace(model=model, tokenizer=None, initialize=AsyncMock()),
        environment=MagicMock(),
        reward_fn=MagicMock(),
        config=_config(num_gradient_updates=num_gradient_updates),
    )
    trainer.optimizer = (
        torch.optim.SGD(model.parameters(), lr=0.5) if optimizer else None
    )
    return trainer


@pytest.mark.parametrize("num_gradient_updates", [1, 3])
def test_single_turn_without_an_optimizer_only_evaluates(num_gradient_updates):
    trainer = _single_turn(num_gradient_updates, optimizer=False)
    loss_dict, metrics = trainer._grpo_update([_group()], backward=False)
    assert loss_dict["path"] == "token"
    assert metrics["optimizer_updates"] == 0
    assert all(p.grad is None for p in trainer.agent.model.parameters())


def test_single_turn_inner_passes_step_once_each() -> None:
    trainer = _single_turn(3, optimizer=True)
    steps = []
    original = trainer.optimizer.step
    trainer.optimizer.step = lambda *a, **k: (steps.append(1), original())[1]
    loss_dict, metrics = trainer._grpo_update([_group()])
    assert len(steps) == 3
    assert metrics["inner_updates"] == 3 and metrics["optimizer_updates"] == 3
    assert metrics["ratio_mean_last"] != pytest.approx(1.0, abs=1e-4)
