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
    torch.testing.assert_close(L.group_advantages(r, normalize=False), torch.tensor([-1.0, 1.0]))


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
    loss = L.clipped_surrogate(ratio, torch.tensor([1.0, 1.0]), clip_low=0.2, clip_high=0.2).sum()
    loss.backward()
    assert logp.grad[0].item() == 0.0
    assert logp.grad[1].item() != 0.0


def test_sequence_ratio_length_normalised():
    cur = torch.tensor([[0.0, -1.0, -1.0]])
    old = torch.tensor([[0.0, -2.0, -2.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    torch.testing.assert_close(L.sequence_ratio(cur, old, mask), torch.tensor([torch.e]))


def test_clip_fraction():
    ratio = torch.tensor([1.0, 1.5, 0.5, 1.1])
    assert L.clip_fraction(ratio, clip_low=0.2, clip_high=0.2) == pytest.approx(0.5)


def test_k3_kl_nonnegative_and_zero_at_equality():
    cur = torch.tensor([[-1.0, -2.0]]); ref = torch.tensor([[-1.5, -1.0]])
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
    cur = torch.tensor([[0.0, -5.0]]); ref = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert L.k3_kl(cur, ref, mask).item() == 0.0
