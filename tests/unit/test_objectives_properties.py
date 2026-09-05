"""Invariants every objective must satisfy, checked with Hypothesis."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from stateset_agents.training import objectives as O  # noqa: E402

SETTINGS = settings(max_examples=40, deadline=None)
PRESETS = sorted(O.OBJECTIVES)


def _batch(seed: int, n: int = 4, t: int = 5):
    g = torch.Generator().manual_seed(seed)
    logp_cur = (-torch.rand(n, t, generator=g) * 3).requires_grad_(True)
    logp_old = logp_cur.detach() + 0.05 * torch.randn(n, t, generator=g)
    logp_ref = logp_cur.detach() + 0.1 * torch.randn(n, t, generator=g)
    mask = torch.ones(n, t)
    mask[0, 3:] = 0
    group_ids = torch.arange(n) % 2
    return logp_cur, logp_old, logp_ref, mask, group_ids


def _ready(name: str):
    obj = O.OBJECTIVES[name]
    if obj.aggregate == "seq_sum_const":
        obj = obj.with_(max_completion_length=5)
    return obj


def _call(obj, logp_cur, logp_old, mask, group_ids, advantages, **kw):
    if obj.ratio == "group_expectation":
        logp_old = (logp_old * mask).sum(-1)
    return O.policy_loss(
        logp_cur=logp_cur,
        mask=mask,
        advantages=advantages,
        objective=obj,
        logp_old=logp_old,
        group_ids=group_ids,
        **kw,
    )


@pytest.mark.parametrize("name", PRESETS)
@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_zero_advantage_gives_zero_gradient(name, seed):
    obj = _ready(name)
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    res = _call(obj, logp_cur, logp_old, mask, group_ids, torch.zeros(4))
    res.loss.backward()
    assert torch.allclose(logp_cur.grad, torch.zeros_like(logp_cur.grad))


@pytest.mark.parametrize(
    "name",
    [
        n
        for n in PRESETS
        if O.OBJECTIVES[n].clip == "clipped"
        # GEPO's coefficient is normalised by the group's sampler expectation,
        # so a uniform shift of logp_old does not move every sample out of
        # the trust region; its clipping is covered by the reference parity.
        and O.OBJECTIVES[n].ratio != "group_expectation"
    ],
)
@SETTINGS
@given(seed=st.integers(0, 10_000), drift=st.floats(0.5, 2.0))
def test_out_of_region_sample_has_zero_gradient(name, seed, drift):
    """Ratio far above 1+clip_high with A>0 selects the clipped branch: no grad."""
    obj = _ready(name)
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    logp_old = logp_cur.detach() - drift  # ratio = exp(drift) >> 1 + clip_high
    adv = torch.ones(4)
    res = _call(obj, logp_cur, logp_old, mask, group_ids, adv)
    res.loss.backward()
    assert torch.allclose(logp_cur.grad, torch.zeros_like(logp_cur.grad), atol=1e-7)
    assert res.metrics["clip_fraction"] == pytest.approx(1.0)


@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_cispo_gradient_is_capped_weight_times_score(seed):
    obj = O.OBJECTIVES["cispo"]
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    adv = torch.randn(4, generator=torch.Generator().manual_seed(seed))
    res = _call(obj, logp_cur, logp_old, mask, group_ids, adv)
    res.loss.backward()
    weight = torch.clamp(torch.exp(logp_cur.detach() - logp_old), max=obj.is_cap)
    expected = -(weight * adv.unsqueeze(-1) * mask) / mask.sum()
    torch.testing.assert_close(logp_cur.grad, expected, atol=1e-6, rtol=1e-5)


@SETTINGS
@given(seed=st.integers(0, 10_000), coef=st.floats(0.01, 1.0))
def test_k3_kl_is_nonnegative_and_zero_at_equality(seed, coef):
    obj = O.OBJECTIVES["grpo"].with_(kl_coef=coef)
    logp_cur, logp_old, logp_ref, mask, group_ids = _batch(seed)
    res = _call(
        obj, logp_cur, logp_old, mask, group_ids, torch.zeros(4), logp_ref=logp_ref
    )
    assert res.metrics["kl"] >= 0
    same = _call(
        obj,
        logp_cur,
        logp_old,
        mask,
        group_ids,
        torch.zeros(4),
        logp_ref=logp_cur.detach(),
    )
    assert same.metrics["kl"] == pytest.approx(0.0, abs=1e-7)


@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_token_mean_equals_seq_mean_for_equal_lengths(seed):
    logp_cur, logp_old, _, _, group_ids = _batch(seed)
    mask = torch.ones(4, 5)
    adv = torch.randn(4, generator=torch.Generator().manual_seed(seed))
    a = _call(O.OBJECTIVES["bnpo"], logp_cur, logp_old, mask, group_ids, adv)
    b = _call(
        O.OBJECTIVES["grpo"].with_(kl="none"), logp_cur, logp_old, mask, group_ids, adv
    )
    torch.testing.assert_close(a.loss, b.loss)


@SETTINGS
@given(seed=st.integers(0, 10_000), length=st.integers(5, 64))
def test_seq_sum_const_is_token_mean_rescaled(seed, length):
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    adv = torch.randn(4, generator=torch.Generator().manual_seed(seed))
    dr = O.OBJECTIVES["dr_grpo"].with_(max_completion_length=length)
    bn = O.OBJECTIVES["bnpo"].with_(advantage="group_mean")
    a = _call(dr, logp_cur, logp_old, mask, group_ids, adv)
    b = _call(bn, logp_cur, logp_old, mask, group_ids, adv)
    scale = mask.sum() / (4 * length)
    torch.testing.assert_close(a.loss, b.loss * scale)


@SETTINGS
@given(rewards=st.lists(st.floats(-5, 5), min_size=2, max_size=8))
def test_leave_one_out_advantages_sum_to_zero(rewards):
    obj = O.OBJECTIVES["rloo"]
    r = torch.tensor(rewards)
    adv = O.compute_advantages(r, torch.zeros(len(rewards), dtype=torch.long), obj)
    assert adv.sum().item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("name", PRESETS)
@SETTINGS
@given(seed=st.integers(0, 10_000), scale=st.floats(10.0, 1000.0))
def test_no_preset_produces_non_finite_loss(name, seed, scale):
    obj = _ready(name)
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    logp_old = logp_old - scale  # log-ratio up to +1000
    adv = torch.randn(4, generator=torch.Generator().manual_seed(seed))
    res = _call(obj, logp_cur, logp_old, mask, group_ids, adv)
    assert torch.isfinite(res.loss)
    res.loss.backward()
    assert torch.isfinite(logp_cur.grad).all()
