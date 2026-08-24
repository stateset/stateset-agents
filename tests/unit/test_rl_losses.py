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
