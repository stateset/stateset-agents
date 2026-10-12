"""OfflineGRPOTrainer updates its policy from logged responses.

Each logged response's length-normalised log-likelihood (response tokens only)
is weighted by its detached hybrid advantage; ``learning_rate`` drives a
dedicated policy optimizer, and a batch with no advantage signal leaves the
policy untouched.
"""

from __future__ import annotations

import asyncio

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


def test_train_step_updates_the_policy_with_logged_text():
    model, tok = _model(), tiny_tokenizer(103)
    trainer = _trainer(model, tok)
    before = _params(model)
    metrics = asyncio.run(trainer.train_step(_batch()))
    assert metrics["policy_updated"] == 1.0
    assert _changed(before, model)
    assert trainer.policy_optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)


def test_zero_learning_rate_leaves_policy_unchanged():
    model, tok = _model(), tiny_tokenizer(103)
    trainer = _trainer(model, tok, learning_rate=0.0)
    before = _params(model)
    asyncio.run(trainer.train_step(_batch()))
    assert not _changed(before, model)


def test_zero_advantages_skip_the_policy_step(monkeypatch):
    model, tok = _model(), tiny_tokenizer(103)
    trainer = _trainer(model, tok)
    monkeypatch.setattr(
        trainer, "_compute_hybrid_advantage", lambda r, v, q: torch.zeros_like(v)
    )
    before = _params(model)
    metrics = asyncio.run(trainer.train_step(_batch()))
    assert metrics["policy_updated"] == 0.0
    assert not _changed(before, model)
    assert trainer.policy_optimizer is None


def test_value_only_training_without_a_policy_is_unchanged():
    trainer = _trainer()
    metrics = asyncio.run(trainer.train_step(_batch()))
    assert "policy_loss" not in metrics


def test_policy_loss_scores_response_tokens_only():
    model, tok = _model(), tiny_tokenizer(103)
    trainer = _trainer(model, tok, learning_rate=0.0)
    contexts = [[{"role": "user", "content": "hi"}]]
    adv = torch.tensor([1.0])
    out = trainer._policy_step(contexts, ["12"], adv)

    prompt = tok("User: hi\nAssistant: ", add_special_tokens=False)["input_ids"]
    response = tok("12", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = torch.tensor([prompt + response])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1]
    lp = torch.log_softmax(logits, -1).gather(-1, ids[0, 1:, None]).squeeze(-1)
    expected = -lp[len(prompt) - 1 :].mean()
    assert out["policy_loss"] == pytest.approx(float(expected), rel=1e-5)


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
