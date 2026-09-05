"""GRPO's batched per-token path: one padded forward pass over the stored
token ids of a whole trajectory group, per-token ratios, any token-level
objective preset, and the sequence-level fallback for trajectories that carry
no token data."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.trajectory import (  # noqa: E402
    ConversationTurn,
    MultiTurnTrajectory,
    TrajectoryGroup,
)
from stateset_agents.training import loss_computation as lc  # noqa: E402
from stateset_agents.training import objectives as O  # noqa: E402
from stateset_agents.training import rl_losses  # noqa: E402

VOCAB = 200


def _tiny_model(seed: int = 0):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=VOCAB,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def _agent(model=None):
    model = model or _tiny_model()
    return SimpleNamespace(model=model, tokenizer=None)


def _turn(prompt_ids, token_ids):
    return ConversationTurn(
        role="assistant",
        content="x",
        metadata={
            "prompt_token_ids": list(prompt_ids),
            "token_ids": list(token_ids),
            "sampler_log_probs": [-1.0] * len(token_ids),
        },
    )


def _traj(reward, prompt_len, resp_len, seed, n_assistant=1):
    g = torch.Generator().manual_seed(seed)
    turns = []
    for k in range(n_assistant):
        turns.append(ConversationTurn(role="user", content=f"q{k}"))
        p = torch.randint(0, VOCAB, (prompt_len + 2 * k,), generator=g).tolist()
        r = torch.randint(0, VOCAB, (resp_len + k,), generator=g).tolist()
        turns.append(_turn(p, r))
    return MultiTurnTrajectory(turns=turns, total_reward=reward)


def _group(rewards=(1.0, 0.0, 0.5), n_assistant=1):
    trajs = [
        _traj(r, prompt_len=4 + i, resp_len=5 + i, seed=10 + i, n_assistant=n_assistant)
        for i, r in enumerate(rewards)
    ]
    return TrajectoryGroup(scenario_id="s", trajectories=trajs)


def _config(**over):
    base = {
        "max_prompt_length": 64,
        "max_completion_length": 64,
        "clip_ratio": 0.2,
        "seq_clip_ratio": 3e-4,
        "entropy_coef": 0.0,
        "advantage_normalization": True,
        "baseline_type": "group_mean",
        "reward_clip": None,
        "generation_batch_size": 4,
        "objective": None,
        "objective_overrides": None,
        "bf16": False,
        "fp16": False,
    }
    base.update(over)
    return SimpleNamespace(**base)


def _run(groups, config, agent, beta=0.0, reference_model=None):
    if beta > 0 or reference_model is not None:
        return lc.compute_enhanced_grpo_loss(
            groups, beta, config, agent, reference_model=reference_model
        )
    return lc.compute_grpo_loss(groups, config, agent, 0.0, 0, lambda m, n: None)


def _reference_loss(group, config, model, objective):
    """Manual: pad rows, one forward, gather, policy_loss with logp_old=None."""
    rows, adv_index = [], []
    for ti, traj in enumerate(group.trajectories):
        for t in traj.turns:
            if t.role == "assistant":
                rows.append((t.metadata["prompt_token_ids"], t.metadata["token_ids"]))
                adv_index.append(ti)
    width = max(len(p) + len(r) for p, r in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    attn = torch.zeros(len(rows), width, dtype=torch.long)
    resp = torch.zeros(len(rows), width)
    for i, (p, r) in enumerate(rows):
        seq = p + r
        ids[i, : len(seq)] = torch.tensor(seq)
        attn[i, : len(seq)] = 1
        resp[i, len(p) : len(seq)] = 1
    logits = model(input_ids=ids, attention_mask=attn).logits
    lp, mask = rl_losses.gather_token_logprobs(logits, ids, resp)
    rewards = torch.tensor([t.total_reward for t in group.trajectories])
    adv = O.compute_advantages(
        rewards, torch.zeros(len(rewards), dtype=torch.long), objective
    )
    adv_rows = adv[torch.tensor(adv_index)]
    return O.policy_loss(
        logp_cur=lp, mask=mask, advantages=adv_rows, objective=objective
    ).loss


def test_token_path_is_used_and_matches_manual_batched_objective():
    model = _tiny_model()
    group = _group()
    out = _run([group], _config(), _agent(model))
    assert out["path"] == "token"
    assert torch.isfinite(out["total_loss"])
    want = _reference_loss(group, _config(), _tiny_model(), O.OBJECTIVES["grpo"])
    torch.testing.assert_close(
        out["total_loss"].detach(), want.detach(), atol=1e-5, rtol=1e-5
    )


def test_token_path_gradient_reaches_model():
    model = _tiny_model()
    out = _run([_group()], _config(), _agent(model))
    out["total_loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


def test_token_path_one_forward_call_per_group(monkeypatch):
    model = _tiny_model()
    calls = []
    orig = model.forward

    def spy(*a, **kw):
        calls.append(kw.get("input_ids", a[0] if a else None).shape)
        return orig(*a, **kw)

    monkeypatch.setattr(model, "forward", spy)
    _run([_group(), _group(rewards=(0.2, 0.9))], _config(), _agent(model))
    assert len(calls) == 2  # one padded forward per group, not per trajectory


def test_token_path_respects_generation_batch_size_chunks(monkeypatch):
    model = _tiny_model()
    calls = []
    orig = model.forward

    def spy(*a, **kw):
        calls.append(kw.get("input_ids", a[0] if a else None).shape[0])
        return orig(*a, **kw)

    monkeypatch.setattr(model, "forward", spy)
    out = _run(
        [_group(rewards=(1.0, 0.0, 0.5, 0.25, 0.75))],
        _config(generation_batch_size=2),
        _agent(model),
    )
    assert calls == [2, 2, 1]
    want = _reference_loss(
        _group(rewards=(1.0, 0.0, 0.5, 0.25, 0.75)),
        _config(),
        _tiny_model(),
        O.OBJECTIVES["grpo"],
    )
    torch.testing.assert_close(
        out["total_loss"].detach(), want.detach(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("name", ["dapo", "cispo", "rloo", "dr_grpo", "bnpo"])
def test_token_level_presets_work_on_grpo_path(name):
    model = _tiny_model()
    group = _group()
    out = _run([group], _config(objective=name), _agent(model))
    assert out["path"] == "token" and out["objective"] == name
    obj = O.OBJECTIVES[name]
    if obj.aggregate == "seq_sum_const":
        obj = obj.with_(max_completion_length=64)
    want = _reference_loss(group, _config(), _tiny_model(), obj)
    torch.testing.assert_close(
        out["total_loss"].detach(), want.detach(), atol=1e-5, rtol=1e-5
    )


def test_multi_turn_trajectory_yields_one_row_per_assistant_turn():
    model = _tiny_model()
    group = _group(n_assistant=2)
    out = _run([group], _config(), _agent(model))
    assert out["path"] == "token" and out["num_rows"] == 6
    want = _reference_loss(group, _config(), _tiny_model(), O.OBJECTIVES["grpo"])
    torch.testing.assert_close(
        out["total_loss"].detach(), want.detach(), atol=1e-5, rtol=1e-5
    )


def test_group_without_token_data_falls_back_to_sequence_path():
    group = _group()
    del group.trajectories[1].turns[1].metadata["token_ids"]
    model = _tiny_model()
    agent = SimpleNamespace(model=model, tokenizer=None)
    # The sequence path needs a tokenizer; it must be selected, and fail
    # loudly for THAT reason rather than silently using the token path.
    with pytest.raises((AttributeError, TypeError, ValueError)):
        _run([group], _config(), agent)


def test_enhanced_path_uses_k3_token_kl_against_reference_model():
    model = _tiny_model()
    ref = _tiny_model(seed=1)
    out = _run([_group()], _config(), _agent(model), beta=0.05, reference_model=ref)
    assert out["path"] == "token"
    assert out["kl_penalty"] >= 0
    same = _run([_group()], _config(), _agent(model), beta=0.05, reference_model=model)
    assert float(same["kl_penalty"]) == pytest.approx(0.0, abs=1e-6)


def test_entropy_bonus_contributes_gradient_on_token_path():
    model = _tiny_model()
    out = _run(
        [_group(rewards=(0.5, 0.5, 0.5))], _config(entropy_coef=0.1), _agent(model)
    )
    # zero advantages: only the entropy bonus can produce a gradient
    out["total_loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert out["entropy"] > 0


def test_leave_one_out_baseline_on_token_path():
    model = _tiny_model()
    group = _group()
    out = _run(
        [group],
        _config(baseline_type="leave_one_out", advantage_normalization=False),
        _agent(model),
    )
    want = _reference_loss(group, _config(), _tiny_model(), O.OBJECTIVES["rloo"])
    torch.testing.assert_close(
        out["total_loss"].detach(), want.detach(), atol=1e-5, rtol=1e-5
    )
