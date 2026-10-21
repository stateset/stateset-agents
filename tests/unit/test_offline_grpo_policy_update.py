"""IQL policy extraction updates logged assistant tokens with detached weights.

Value pre-training is a separate phase; zero advantages give unit cloning
weights rather than silently skipping the optimizer.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.data.conversation_dataset import (  # noqa: E402
    ConversationDataset,
    ConversationDatasetConfig,
    ConversationTurnData,
    TrajectoryData,
)
from stateset_agents.experimental.offline_rl.offline_grpo import (  # noqa: E402
    OfflineGRPOConfig,
    OfflineGRPOTrainer,
)
from tests._tiny_tokenizer import tiny_tokenizer  # noqa: E402

STATE_DIM = 4
ACTION_DIM = 2


def _model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=16,
            n_layer=1,
            n_head=2,
            vocab_size=103,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def _trainer(model=None, tokenizer=None, **overrides):
    config = OfflineGRPOConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        value_hidden_size=8,
        value_num_layers=1,
        learning_rate=overrides.pop("learning_rate", 1e-2),
        gradient_accumulation_steps=1,
        bf16=False,
        use_lora=False,
        **overrides,
    )
    return OfflineGRPOTrainer(config, model=model, tokenizer=tokenizer, device="cpu")


def _batch(n=4, rewards=None):
    return {
        "states": torch.randn(n, STATE_DIM),
        "actions": torch.randn(n, ACTION_DIM),
        "rewards": torch.tensor(rewards or [1.0, 0.0, 1.0, 0.0][:n]),
        "contexts": [[{"role": "user", "content": "hi"}]] * n,
        "responses": ["123", "abc", "456", "def"][:n],
    }


def _params(model):
    return [p.detach().clone() for p in model.parameters()]


def _changed(before, model):
    return any(
        not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
    )


def _prepared(model, tok):
    trainer = _trainer(model, tok)
    data = _batch()
    data["next_states"] = torch.zeros_like(data["states"])
    data["dones"] = torch.ones(len(data["states"]))
    asyncio.run(
        trainer.pretrain_value_functions(
            SimpleNamespace(to_offline_rl_format=lambda: data), num_steps=1
        )
    )
    return trainer, data


def _policy_batch(trainer, data):
    rows = trainer._logged_policy_rows(data)
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full(
        (len(rows), width), trainer.tokenizer.pad_token_id, dtype=torch.long
    )
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"])
        labels[i, :n] = torch.tensor(row["labels"])
        attention[i, :n] = 1
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": attention,
        "states": data["states"],
        "actions": data["actions"],
    }


def test_train_step_updates_the_policy_with_logged_text():
    model, tok = _model(), tiny_tokenizer(103)
    trainer, data = _prepared(model, tok)
    before = _params(model)
    metrics = asyncio.run(trainer.train_step(_policy_batch(trainer, data)))
    assert metrics["optimizer_updates"] == 1.0
    assert _changed(before, model)
    assert trainer.policy_optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)


def test_zero_optimizer_learning_rate_leaves_policy_unchanged():
    model, tok = _model(), tiny_tokenizer(103)
    trainer, data = _prepared(model, tok)
    trainer._setup_policy_optimizer()
    for group in trainer.policy_optimizer.param_groups:
        group["lr"] = 0.0
    before = _params(model)
    asyncio.run(trainer.train_step(_policy_batch(trainer, data)))
    assert not _changed(before, model)


def test_zero_iql_advantages_are_unit_weight_behavior_cloning(monkeypatch):
    model, tok = _model(), tiny_tokenizer(103)
    trainer, data = _prepared(model, tok)
    for network in (
        trainer._offline_learner.value_net,
        trainer._offline_learner.q1,
        trainer._offline_learner.q2,
    ):
        monkeypatch.setattr(
            network, "forward", lambda *args: torch.zeros(len(args[0]), 1)
        )
    before = _params(model)
    metrics = asyncio.run(trainer.train_step(_policy_batch(trainer, data)))
    assert metrics["policy_weight_mean"] == 1.0
    assert metrics["optimizer_updates"] == 1.0
    assert _changed(before, model)


def test_policy_extraction_requires_an_initialized_policy():
    trainer = _trainer()
    with pytest.raises(ValueError, match="initialized language model"):
        asyncio.run(trainer.train_step(_batch()))
    assert trainer.training_step == 0


def test_policy_loss_scores_response_tokens_only_and_detaches_iql():
    model, tok = _model(), tiny_tokenizer(103)
    trainer, data = _prepared(model, tok)
    batch = _policy_batch(trainer, data)
    with torch.no_grad():
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].transpose(1, 2),
            batch["labels"][:, 1:],
            ignore_index=-100,
            reduction="none",
        )
        mask = batch["labels"][:, 1:] != -100
        expected = ((losses * mask).sum(1) / mask.sum(1)).mean()
    networks = (
        trainer._offline_learner.value_net,
        trainer._offline_learner.q1,
        trainer._offline_learner.q2,
    )
    for network in networks:
        network.zero_grad(set_to_none=True)
    out = asyncio.run(trainer.train_step(batch))
    assert out["nll"] == pytest.approx(float(expected), rel=1e-5)
    assert all(p.grad is None for network in networks for p in network.parameters())


def test_conversation_dataset_exposes_logged_text():
    class _Cache:
        def get_or_compute(self, key, text):
            return torch.tensor([float(len(text))])

    turns = [
        ConversationTurnData(role="user", content="u1"),
        ConversationTurnData(role="assistant", content="a1"),
        ConversationTurnData(role="user", content="u2"),
        ConversationTurnData(role="assistant", content="a2"),
    ]
    dataset = ConversationDataset(
        [TrajectoryData("t", turns, 1.0, [0.0, 0.0, 0.0, 1.0])],
        config=ConversationDatasetConfig(
            normalize_rewards=False, compute_returns=False
        ),
        embedding_cache=_Cache(),
    )
    data = dataset.to_offline_rl_format()
    assert data["responses"] == ["a1", "a2"]
    assert data["contexts"][0] == [{"role": "user", "content": "u1"}]
    assert [m["content"] for m in data["contexts"][1]] == ["u1", "a1", "u2"]


def test_mismatched_text_lengths_are_rejected():
    import numpy as np

    trainer = _trainer()
    data = {
        "states": np.zeros((2, STATE_DIM)),
        "actions": np.zeros((2, ACTION_DIM)),
        "rewards": np.zeros(2),
        "next_states": np.zeros((2, STATE_DIM)),
        "dones": np.ones(2),
        "contexts": [[]],
        "responses": ["a", "b"],
    }
    with pytest.raises(ValueError, match="one entry per transition"):
        trainer._validate_offline_data(data)
