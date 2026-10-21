"""GRPO advantage baselines and the enhanced sequence path's KL/entropy terms.

* ``baseline_type`` survives ``advantage_normalization``: normalisation scales
  by the group std and must not re-centre on the group mean.
* ``global_mean`` falls back to the group mean before any global statistics
  exist, and still feeds those statistics.
* The enhanced (KL) sequence path uses the same advantage estimator as the
  plain path (reward clipping, baselines, normalisation switch) and adds the
  entropy bonus.
* Its KL term is masked on the shifted axis: logits at ``t`` predict ``t+1``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.config import TrainingConfig  # noqa: E402
from stateset_agents.training.loss_computation import (  # noqa: E402
    _legacy_group_advantages,
    compute_enhanced_grpo_loss,
    compute_grpo_loss,
)

VOCAB = 8


class _CharTokenizer:
    """Character-level tokenizer with offsets and no chat template."""

    chat_template = None

    def __call__(self, text, return_offsets_mapping=False, **_kwargs):
        ids = torch.tensor([[ord(c) % VOCAB for c in text]])
        out = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor(
                [[(i, i + 1) for i in range(len(text))]]
            )
        return out


class _Model(torch.nn.Module):
    """Uniform logits except a peaked distribution at position ``hot``."""

    def __init__(self, hot: int | None = None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(VOCAB))
        self.hot = hot

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, labels=None):
        length = input_ids.shape[1]
        logits = self.weight.expand(1, length, VOCAB).clone()
        if self.hot is not None:
            pos = self.hot % length
            peak = torch.zeros(VOCAB)
            peak[0] = 6.0
            logits[0, pos] = logits[0, pos] + peak
        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1], input_ids[0, 1:].clamp(min=0)
        )
        return SimpleNamespace(loss=loss, logits=logits)


def _trajectory(reward: float, reply: str = "ok"):
    return SimpleNamespace(
        turns=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": reply},
        ],
        total_reward=reward,
        metadata={},
    )


def _config(**kwargs):
    base = {
        "clip_ratio": 0.0,
        "entropy_coef": 0.0,
        "max_prompt_length": 64,
        "max_completion_length": 64,
        "bf16": False,
        "fp16": False,
    }
    base.update(kwargs)
    return TrainingConfig(allow_sequence_fallback=True, **base)


TEXT = "User: hi\nAssistant: ok\n"
FIRST_ASSISTANT_CHAR = TEXT.index("Assistant")


# --------------------------------------------------------------------------
# 1. baselines survive normalisation
# --------------------------------------------------------------------------


def test_group_median_baseline_survives_normalization():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0])
    adv = _legacy_group_advantages(
        rewards, _config(baseline_type="group_median", advantage_normalization=True)
    )
    std = rewards.std(correction=0)
    expected = (rewards - rewards.median()) / (std + 1e-8)
    assert torch.allclose(adv, expected, atol=1e-6)


def test_global_mean_baseline_survives_normalization():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0])
    adv = _legacy_group_advantages(
        rewards,
        _config(baseline_type="global_mean", advantage_normalization=True),
        global_reward_mean=100.0,
    )
    std = rewards.std(correction=0)
    assert torch.allclose(adv, (rewards - 100.0) / (std + 1e-8), atol=1e-6)
    assert bool((adv < 0).all())


def test_group_mean_normalization_unchanged():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0])
    centred = rewards - rewards.mean()
    expected = centred / (centred.std(correction=0) + 1e-8)
    adv = _legacy_group_advantages(
        rewards, _config(baseline_type="group_mean", advantage_normalization=True)
    )
    assert torch.allclose(adv, expected, atol=1e-6)


def _sequence_advantages(loss_fn, groups, **kwargs):
    seen = []
    import stateset_agents.training.loss_computation as lc

    original = lc._compute_group_policy_loss

    def spy(group, advantages, *args, **kw):
        seen.append(advantages.detach().clone())
        return original(group, advantages, *args, **kw)

    lc._compute_group_policy_loss = spy
    try:
        loss_fn(groups, **kwargs)
    finally:
        lc._compute_group_policy_loss = original
    return seen


def test_sequence_path_global_mean_survives_normalization():
    agent = SimpleNamespace(model=_Model(), tokenizer=_CharTokenizer())
    group = SimpleNamespace(trajectories=[_trajectory(r) for r in (1, 2, 3, 10)])
    updates = []
    seen = _sequence_advantages(
        lambda groups, **kw: compute_grpo_loss(groups, **kw),
        [group],
        config=_config(baseline_type="global_mean"),
        agent=agent,
        global_reward_mean=100.0,
        global_reward_count=50,
        update_global_stats=lambda m, n: updates.append((m, n)),
    )
    assert bool((seen[0] < 0).all())
    assert updates == [(4.0, 4)]


def test_global_mean_falls_back_to_group_mean_without_stats():
    agent = SimpleNamespace(model=_Model(), tokenizer=_CharTokenizer())
    group = SimpleNamespace(trajectories=[_trajectory(r) for r in (1.0, 3.0)])
    updates = []
    seen = _sequence_advantages(
        lambda groups, **kw: compute_grpo_loss(groups, **kw),
        [group],
        config=_config(baseline_type="global_mean", advantage_normalization=False),
        agent=agent,
        global_reward_mean=0.0,
        global_reward_count=0,
        update_global_stats=lambda m, n: updates.append((m, n)),
    )
    # Group-mean fallback, not a zero baseline ([1, 3]).
    assert torch.allclose(seen[0], torch.tensor([-1.0, 1.0]))
    # The statistics still learn from this batch.
    assert updates == [(2.0, 2)]


# --------------------------------------------------------------------------
# 2. enhanced path shares the advantage estimator and entropy bonus
# --------------------------------------------------------------------------


def _enhanced(config, rewards, *, beta=0.0, ref=None, model=None):
    agent = SimpleNamespace(model=model or _Model(), tokenizer=_CharTokenizer())
    group = SimpleNamespace(trajectories=[_trajectory(r) for r in rewards])
    return compute_enhanced_grpo_loss(
        [group], beta=beta, config=config, agent=agent, reference_model=ref
    )


def test_enhanced_path_clips_rewards():
    config = _config(reward_clip=1.0, advantage_normalization=False)
    out = _enhanced(config, [5.0, 0.0], beta=0.04, ref=_Model())
    # Clipped to [1, 0] -> advantages [0.5, -0.5].
    assert out["advantage_std"] == pytest.approx(0.5)


def test_enhanced_path_respects_normalization_switch():
    config = _config(advantage_normalization=False)
    out = _enhanced(config, [3.0, 1.0])
    assert out["advantage_std"] == pytest.approx(1.0)


def test_enhanced_path_global_mean_baseline():
    config = _config(baseline_type="global_mean", advantage_normalization=False)
    agent = SimpleNamespace(model=_Model(), tokenizer=_CharTokenizer())
    group = SimpleNamespace(trajectories=[_trajectory(r) for r in (1.0, 3.0)])
    out = compute_enhanced_grpo_loss(
        [group],
        beta=0.0,
        config=config,
        agent=agent,
        global_reward_mean=10.0,
        global_reward_count=5,
        update_global_stats=lambda m, n: None,
    )
    assert out["mean_advantage"] == pytest.approx(-8.0)


def test_enhanced_path_adds_entropy_bonus():
    # Equal rewards: zero advantages, so only the entropy bonus remains.
    without = _enhanced(_config(entropy_coef=0.0), [1.0, 1.0])
    with_bonus = _enhanced(_config(entropy_coef=0.5), [1.0, 1.0])
    assert float(without["total_loss"]) == pytest.approx(0.0, abs=1e-7)
    expected = -0.5 * torch.log(torch.tensor(float(VOCAB)))
    assert float(with_bonus["total_loss"]) == pytest.approx(float(expected), rel=1e-4)
    assert with_bonus["entropy"] == pytest.approx(float(-expected / 0.5), rel=1e-4)


# --------------------------------------------------------------------------
# 3. KL mask alignment
# --------------------------------------------------------------------------


def test_enhanced_kl_counts_position_predicting_first_assistant_token():
    # Logits at FIRST_ASSISTANT_CHAR - 1 predict the first assistant token.
    out = _enhanced(
        _config(),
        [1.0, 1.0],
        beta=0.1,
        ref=_Model(),
        model=_Model(hot=FIRST_ASSISTANT_CHAR - 1),
    )
    assert float(out["kl_penalty"]) > 1e-3


def test_enhanced_kl_ignores_final_position():
    # The last position predicts past the end of the sequence.
    out = _enhanced(_config(), [1.0, 1.0], beta=0.1, ref=_Model(), model=_Model(hot=-1))
    assert float(out["kl_penalty"]) == pytest.approx(0.0, abs=1e-7)
