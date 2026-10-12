"""``TrainingConfig.disable_dropout`` zeroes every dropout module in the policy.

With dropout active, separately scored old/reference log-probs and the
learner's forward pass use different masks, so the first PPO-style ratio is
noise rather than exactly 1 (TRL's ``disable_dropout``).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training.config import TrainingConfig  # noqa: E402
from stateset_agents.training.trainer_utils import (  # noqa: E402
    disable_dropout_in_model,
)


def _dropout_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(n_embd=16, n_layer=1, n_head=2, vocab_size=50, n_positions=16)
    )


def _dropouts(model):
    return [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]


def test_helper_zeroes_every_dropout_module():
    model = _dropout_model()
    assert any(p > 0 for p in _dropouts(model))
    assert disable_dropout_in_model(model) == len(_dropouts(model))
    assert all(p == 0 for p in _dropouts(model))
    model.train()
    ids = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(model(ids).logits, model(ids).logits)


def test_field_is_keyword_only_and_off_by_default():
    field = {f.name: f for f in dataclasses.fields(TrainingConfig)}["disable_dropout"]
    assert field.kw_only and field.default is False
    assert TrainingConfig(disable_dropout=True).disable_dropout is True
    assert TrainingConfig.from_dict(TrainingConfig(disable_dropout=True).to_dict())
    assert (
        TrainingConfig.from_dict(
            TrainingConfig(disable_dropout=True).to_dict()
        ).disable_dropout
        is True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_multi_turn_trainer_applies_the_setting(enabled):
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    model = _dropout_model()
    agent = SimpleNamespace(model=model, tokenizer=None, initialize=AsyncMock())
    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=MagicMock(),
        reward_fn=MagicMock(),
        config=TrainingConfig(disable_dropout=enabled, report_to="none"),
    )
    await trainer.initialize()
    assert all(p == 0 for p in _dropouts(model)) is enabled


def test_ppo_trainer_applies_the_setting():
    from stateset_agents.training.ppo_trainer import PPOConfig, PPOTrainer

    model = _dropout_model()
    PPOTrainer(
        PPOConfig(disable_dropout=True, beta=0.0),
        model,
        tokenizer=MagicMock(),
        reward_fn=lambda p, c: 0.0,
    )
    assert all(p == 0 for p in _dropouts(model))
