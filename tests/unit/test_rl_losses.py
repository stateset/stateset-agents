import math

import pytest

torch = pytest.importorskip("torch")
from stateset_agents.training import rl_losses as L


def _naive_gather(logits, ids, mask):
    lp = torch.log_softmax(logits[:, :-1], -1)
    out = torch.zeros(ids.shape[0], ids.shape[1] - 1)
    for b in range(ids.shape[0]):
        for t in range(ids.shape[1] - 1):
            out[b, t] = lp[b, t, ids[b, t + 1]] * mask[b, t + 1]
    return out, mask[:, 1:]


def test_gather_matches_naive_loop():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 7, generator=g)
    ids = torch.randint(0, 7, (2, 5), generator=g)
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]], dtype=torch.float32)
    got, got_mask = L.gather_token_logprobs(logits, ids, mask)
    want, want_mask = _naive_gather(logits, ids, mask)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_mask, want_mask)


def test_gather_dtype_kwarg_controls_softmax_precision():
    logits = torch.randn(2, 5, 7, dtype=torch.bfloat16)
    ids = torch.randint(0, 7, (2, 5))
    mask = torch.ones(2, 5)
    fp32, _ = L.gather_token_logprobs(logits, ids, mask)
    bf16, _ = L.gather_token_logprobs(logits, ids, mask, dtype=torch.bfloat16)
    assert fp32.dtype == torch.float32
    assert bf16.dtype == torch.bfloat16
    torch.testing.assert_close(bf16.float(), fp32, rtol=1e-1, atol=1e-1)


def test_masked_mean_token_and_seq():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]])
    m = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    assert L.masked_mean(x, m, mode="token").item() == pytest.approx(10 / 4)
    assert L.masked_mean(x, m, mode="seq").item() == pytest.approx((2.0 + 4.0) / 2)


def test_masked_mean_empty_mask_is_zero_not_nan():
    x = torch.ones(2, 3)
    m = torch.zeros(2, 3)
    assert L.masked_mean(x, m).item() == 0.0


def test_group_advantages_matches_manual():
    r = torch.tensor([1.0, 2.0, 3.0, 6.0])
    a = L.group_advantages(r)
    want = (r - r.mean()) / (r.std(correction=0) + 1e-8)
    torch.testing.assert_close(a, want)
    assert a.mean().abs().item() < 1e-6


def test_group_advantages_single_sample_is_zero_not_nan():
    a = L.group_advantages(torch.tensor([0.7]))
    assert a.shape == (1,) and a.item() == 0.0 and torch.isfinite(a).all()


def test_group_advantages_constant_rewards_zero():
    a = L.group_advantages(torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(a, torch.zeros(3))


def test_group_advantages_unnormalized():
    r = torch.tensor([0.0, 2.0])
    torch.testing.assert_close(
        L.group_advantages(r, normalize=False), torch.tensor([-1.0, 1.0])
    )


def test_clipped_surrogate_zero_advantage_zero_grad():
    logp = torch.zeros(3, requires_grad=True)
    ratio = torch.exp(logp - torch.tensor([0.1, -0.1, 0.0]))
    loss = L.clipped_surrogate(ratio, torch.zeros(3), clip_low=0.2, clip_high=0.2).sum()
    loss.backward()
    assert torch.equal(logp.grad, torch.zeros(3))


def test_clipped_surrogate_out_of_region_has_zero_grad_inside_has_grad():
    # ratio 1.5 with A>0 is above 1+clip_high -> clipped branch wins -> no grad
    logp = torch.tensor([0.0, 0.0], requires_grad=True)
    old = torch.tensor([-0.405465, 0.0])  # exp(0.405)=1.5 ; exp(0)=1.0
    ratio = torch.exp(logp - old)
    loss = L.clipped_surrogate(
        ratio, torch.tensor([1.0, 1.0]), clip_low=0.2, clip_high=0.2
    ).sum()
    loss.backward()
    assert logp.grad[0].item() == 0.0
    assert logp.grad[1].item() != 0.0


def test_sequence_ratio_length_normalised():
    cur = torch.tensor([[0.0, -1.0, -1.0]])
    old = torch.tensor([[0.0, -2.0, -2.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    torch.testing.assert_close(
        L.sequence_ratio(cur, old, mask), torch.tensor([torch.e])
    )


def test_clip_fraction():
    ratio = torch.tensor([1.0, 1.5, 0.5, 1.1])
    assert L.clip_fraction(ratio, clip_low=0.2, clip_high=0.2) == pytest.approx(0.5)


def test_k3_kl_nonnegative_and_zero_at_equality():
    cur = torch.tensor([[-1.0, -2.0]])
    ref = torch.tensor([[-1.5, -1.0]])
    assert L.k3_kl(cur, ref).item() >= 0
    assert L.k3_kl(cur, cur).item() == 0.0


def test_k3_kl_gradient_pulls_toward_ref():
    ref = torch.tensor([[-1.0, -1.0]])
    cur = torch.tensor([[-2.0, -0.5]], requires_grad=True)
    before = L.k3_kl(cur, ref)
    before.backward()
    with torch.no_grad():
        cur2 = cur - 0.1 * cur.grad
    after = L.k3_kl(cur2, ref)
    assert after.item() < before.item()


def test_k3_kl_respects_mask():
    cur = torch.tensor([[0.0, -5.0]])
    ref = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert L.k3_kl(cur, ref, mask).item() == 0.0


def test_safe_exp_ratio_is_finite_for_huge_log_ratios():
    log_ratio = torch.tensor([50.0, -50.0, 0.0])
    ratio = L.safe_exp_ratio(log_ratio)
    assert torch.isfinite(ratio).all()
    assert ratio[2].item() == pytest.approx(1.0)
    assert ratio[0].item() == pytest.approx(math.exp(20.0))
    assert ratio[1].item() == pytest.approx(math.exp(-20.0))


def test_safe_exp_ratio_is_transparent_inside_the_clamp():
    log_ratio = torch.tensor([0.5, -0.5])
    assert torch.allclose(L.safe_exp_ratio(log_ratio), torch.exp(log_ratio))


def test_safe_exp_ratio_custom_clamp():
    log_ratio = torch.tensor([100.0])
    assert L.safe_exp_ratio(log_ratio, clamp=30.0).item() == pytest.approx(
        math.exp(30.0)
    )


def test_safe_exp_ratio_keeps_gradients_inside_the_clamp():
    log_ratio = torch.tensor([0.25], requires_grad=True)
    L.safe_exp_ratio(log_ratio).backward()
    assert log_ratio.grad is not None
    assert log_ratio.grad.item() == pytest.approx(math.exp(0.25))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, None),
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp16", torch.float16),
        ("FP16", torch.float16),
    ],
)
def test_resolve_logprob_dtype(name, expected):
    assert L.resolve_logprob_dtype(name) is expected


def test_resolve_logprob_dtype_rejects_unknown_names():
    with pytest.raises(ValueError, match="logprob_dtype"):
        L.resolve_logprob_dtype("int8")
