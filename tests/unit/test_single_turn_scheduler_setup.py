"""SingleTurn's scheduler setup fails loudly, like MultiTurn's.

It used to log a failed scheduler build at DEBUG and train at a constant
learning rate, so a warmup/decay request could silently do nothing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training import single_turn_trainer as module  # noqa: E402


def _trainer(scheduler: str = "cosine") -> module.SingleTurnGRPOTrainer:
    model = torch.nn.Linear(1, 1, bias=False)
    trainer = module.SingleTurnGRPOTrainer(
        SimpleNamespace(model=model),
        SimpleNamespace(reset=AsyncMock(), step=AsyncMock()),
        config=SimpleNamespace(
            num_episodes=1,
            warmup_ratio=0.1,
            lr_scheduler_type=scheduler,
            bf16=False,
            beta=0.0,
        ),
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return trainer


def test_a_failed_scheduler_build_propagates(monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise ValueError("bad warmup")

    monkeypatch.setattr(module, "get_cosine_schedule_with_warmup", broken)
    with pytest.raises(ValueError, match="bad warmup"):
        _trainer()._setup_scheduler(10)


def test_missing_scheduler_helpers_raise(monkeypatch) -> None:
    monkeypatch.setattr(module, "get_linear_schedule_with_warmup", None)
    with pytest.raises(ImportError, match="scheduler helpers"):
        _trainer("linear")._setup_scheduler(10)


@pytest.mark.parametrize(
    ("scheduler", "built"), [("cosine", True), ("linear", True), ("constant", False)]
)
def test_supported_schedulers_still_build(scheduler: str, built: bool) -> None:
    trainer = _trainer(scheduler)
    trainer._setup_scheduler(10)
    assert (trainer.lr_scheduler is not None) is built
