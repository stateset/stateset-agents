"""MultiTurnGRPOTrainer can skip (and diagnose) a non-finite update on request."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer


def _trainer(**config: Any) -> MultiTurnGRPOTrainer:
    model = torch.nn.Linear(2, 2)
    trainer = MultiTurnGRPOTrainer.__new__(MultiTurnGRPOTrainer)
    trainer.config = TrainingConfig(report_to="none", **config)
    trainer.agent = SimpleNamespace(model=model)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.lr_scheduler = None
    trainer.scaler = None
    trainer.global_step = 0
    trainer._last_loss_scalars = {"kl_penalty": 12.5, "ratio_mean": 1.0}
    return trainer


def _poison(trainer: MultiTurnGRPOTrainer) -> None:
    for parameter in trainer.agent.model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))


def _finite(trainer: MultiTurnGRPOTrainer) -> None:
    for parameter in trainer.agent.model.parameters():
        parameter.grad = torch.ones_like(parameter)


def test_default_policy_still_raises() -> None:
    trainer = _trainer()
    _poison(trainer)
    with pytest.raises(ValueError, match="finite"):
        trainer._apply_optimizer_step(torch)


def test_skip_policy_discards_update_and_logs_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = _trainer(nonfinite_grad_policy="skip")
    before = [p.detach().clone() for p in trainer.agent.model.parameters()]
    _poison(trainer)

    with caplog.at_level(logging.WARNING):
        applied = trainer._apply_optimizer_step(torch)

    assert applied is False
    after = list(trainer.agent.model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert all(p.grad is None for p in after)
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "weight" in message and "kl_penalty" in message
    assert trainer.nonfinite_skips == 1


def test_skip_policy_aborts_after_too_many_consecutive_skips() -> None:
    trainer = _trainer(nonfinite_grad_policy="skip", max_consecutive_nonfinite_skips=2)
    for _ in range(2):
        _poison(trainer)
        assert trainer._apply_optimizer_step(torch) is False
    _poison(trainer)
    with pytest.raises(ValueError, match="consecutive"):
        trainer._apply_optimizer_step(torch)


def test_a_finite_step_resets_the_consecutive_count() -> None:
    trainer = _trainer(nonfinite_grad_policy="skip", max_consecutive_nonfinite_skips=1)
    _poison(trainer)
    assert trainer._apply_optimizer_step(torch) is False
    _finite(trainer)
    assert trainer._apply_optimizer_step(torch) is True
    _poison(trainer)
    assert trainer._apply_optimizer_step(torch) is False
    assert trainer.nonfinite_skips == 2


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonfinite_grad_policy"):
        TrainingConfig(nonfinite_grad_policy="ignore")
