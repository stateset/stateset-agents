"""Trainers build AdamW from the configured betas and weight decay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.config import TrainingConfig  # noqa: E402
from stateset_agents.training.gspo_config import GSPOConfig  # noqa: E402
from stateset_agents.training.gspo_trainer import GSPOTrainer  # noqa: E402
from stateset_agents.training.single_turn_trainer import (  # noqa: E402
    SingleTurnGRPOTrainer,
)


def test_single_turn_optimizer_uses_configured_betas():
    holder = SimpleNamespace(
        agent=SimpleNamespace(model=torch.nn.Linear(2, 2)),
        config=TrainingConfig(adam_beta1=0.8, adam_beta2=0.95),
        _get_torch_module=lambda: torch,
    )
    SingleTurnGRPOTrainer._setup_optimizer(holder)
    for group in holder.optimizer.param_groups:
        assert group["betas"] == (0.8, 0.95)


def test_single_turn_optimizer_default_beta2_is_config_default():
    holder = SimpleNamespace(
        agent=SimpleNamespace(model=torch.nn.Linear(2, 2)),
        config=TrainingConfig(),
        _get_torch_module=lambda: torch,
    )
    SingleTurnGRPOTrainer._setup_optimizer(holder)
    assert holder.optimizer.param_groups[0]["betas"] == (0.9, 0.99)


def test_gspo_optimizer_uses_configured_betas_and_weight_decay():
    trainer = GSPOTrainer(
        config=GSPOConfig(
            model_name="gpt2", adam_beta1=0.85, adam_beta2=0.97, weight_decay=0.123
        ),
        model=torch.nn.Linear(2, 2),
        tokenizer=None,
        agent=None,
        environment=None,
        reward_model=None,
        ref_model=None,
    )
    group = trainer.optimizer.param_groups[0]
    assert group["betas"] == (0.85, 0.97)
    assert group["weight_decay"] == pytest.approx(0.123)


def test_gspo_validate_flags_fields_the_trainer_ignores():
    warnings = GSPOConfig(
        model_name="gpt2", num_iterations=3, reward_clip=1.0, lr_scheduler_type="linear"
    ).validate()
    ignored = [w for w in warnings if "ignored by the GSPO trainer" in w]
    assert len(ignored) == 1
    for name in ("num_iterations", "reward_clip"):
        assert name in ignored[0]
    assert "mini_batch_size" not in ignored[0]
    # The spine LR schedule honours lr_scheduler_type (test_lr_schedule_budget).
    assert "lr_scheduler_type" not in ignored[0]


def test_gspo_validate_quiet_for_defaults():
    warnings = GSPOConfig(model_name="gpt2").validate()
    assert not [w for w in warnings if "ignored by the GSPO trainer" in w]
