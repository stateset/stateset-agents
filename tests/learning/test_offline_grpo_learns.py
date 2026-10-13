"""OfflineGRPOTrainer: value pre-training learns, the policy must too.

The trainer advertises "offline value estimates as advantage baseline during
GRPO" for a language policy. Its value/Q pre-training is checked on the logged
two-action bandit; its policy side is checked on the digit task: logged
replies made of digits carry reward 1, other replies reward 0, so training on
the log must raise the policy's probability of emitting a digit.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from tests.learning._offline_toy import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    bandit_dataset,
    heldout_states,
    mean_gap,
)
from tests.learning._toy import (  # noqa: E402
    TRAIN_PROMPTS,
    digit_fraction,
    make_model,
    make_tokenizer,
    seeds,
    target_mass,
)

pytestmark = pytest.mark.learning


class _LoggedDataset:
    """Minimal ConversationDataset stand-in: logged replies + their features."""

    def __init__(self, seed: int, sign: float = 1.0) -> None:
        self._data = bandit_dataset(seed, sign=sign)
        n = len(self._data["states"]) // 2
        # Rewarded (GOOD) rows are digit replies, the others are letters.
        self.responses = ["4821"] * n + ["abcd"] * n
        assert all(
            digit_fraction(r) == float(abs(rw) > 0)
            for r, rw in zip(self.responses, self._data["rewards"], strict=True)
        )

    def __len__(self) -> int:
        return len(self.responses)

    def to_offline_rl_format(self) -> dict[str, Any]:
        # Like ConversationDataset: the logged context before each reply and
        # the reply itself, aligned with the transition rows.
        contexts = [
            [{"role": "user", "content": TRAIN_PROMPTS[i % len(TRAIN_PROMPTS)]}]
            for i in range(len(self.responses))
        ]
        return {
            **{k: v.copy() for k, v in self._data.items()},
            "contexts": contexts,
            "responses": list(self.responses),
        }


def _trainer(seed: int, model=None, tokenizer=None):
    from stateset_agents.experimental.offline_rl.offline_grpo import (
        OfflineGRPOConfig,
        OfflineGRPOTrainer,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    config = OfflineGRPOConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        value_hidden_size=64,
        value_num_layers=2,
        value_learning_rate=1e-3,
        pretrain_value_batch_size=32,
        pretrain_value_epochs=40,
        batch_size=16,
        learning_rate=1e-2,
        bf16=False,
        use_lora=False,
    )
    return OfflineGRPOTrainer(config, model=model, tokenizer=tokenizer, device="cpu")


@pytest.mark.parametrize("seed", seeds())
def test_offline_grpo_value_pretraining_prefers_rewarded_action(seed: int) -> None:
    trainer = _trainer(seed)
    asyncio.run(trainer.pretrain_value_functions(_LoggedDataset(seed)))
    # Q(s, a) regresses r + 0.99 V(s') with done=1, i.e. the logged reward.
    gap = mean_gap(trainer.get_q_estimate, heldout_states(seed))
    assert gap > 0.5, gap


@pytest.mark.parametrize("seed", seeds())
def test_offline_grpo_value_pretraining_negated_reward(seed: int) -> None:
    trainer = _trainer(seed)
    asyncio.run(trainer.pretrain_value_functions(_LoggedDataset(seed, sign=-1.0)))
    assert mean_gap(trainer.get_q_estimate, heldout_states(seed)) < -0.5


@pytest.mark.parametrize("seed", seeds())
def test_offline_grpo_policy_learns_from_logged_rewards(seed: int) -> None:
    tokenizer = make_tokenizer()
    model = make_model(seed)
    before_params = [p.detach().clone() for p in model.parameters()]
    before = target_mass(model, tokenizer)
    trainer = _trainer(seed, model=model, tokenizer=tokenizer)
    asyncio.run(trainer.train(num_epochs=3, dataset=_LoggedDataset(seed)))
    after = target_mass(model, tokenizer)
    changed = any(
        not torch.equal(a, b)
        for a, b in zip(before_params, model.parameters(), strict=True)
    )
    assert changed, "policy parameters were never updated"
    assert after > before + 0.04, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_offline_grpo_policy_negated_reward(seed: int) -> None:
    tokenizer = make_tokenizer()
    model = make_model(seed)
    before = target_mass(model, tokenizer)
    trainer = _trainer(seed, model=model, tokenizer=tokenizer)
    asyncio.run(trainer.train(num_epochs=3, dataset=_LoggedDataset(seed, sign=-1.0)))
    assert target_mass(model, tokenizer) < before - 0.01
