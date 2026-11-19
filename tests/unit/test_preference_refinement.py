"""Check the preference trainer against finite conditional policies and analytic gradients."""

import copy
import itertools
import math
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from stateset_agents.core.enhanced.advanced_rl_algorithms import DPOConfig, DPOTrainer
from stateset_agents.training.preference_loss import (
    completion_terms,
    preference_pair_loss,
)


class Tokens:
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [int(token) for token in text.split()]


class Conditional(torch.nn.Module):
    """Token 3 is the prompt. A single live parameter moves action one's logit."""

    def __init__(self):
        super().__init__()
        self.tilt = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.register_buffer(
            "one", torch.tensor([1.0, 2.0, 1.0, 1.0], dtype=torch.float64)
        )

    def forward(self, input_ids):
        one = self.one[input_ids]
        off = torch.full_like(one, -math.inf)
        return SimpleNamespace(
            logits=torch.stack((off, one.log() + self.tilt, (3 - one).log(), off), -1)
        )


def make(model, **config):
    return DPOTrainer(
        SimpleNamespace(model=model, tokenizer=Tokens()), DPOConfig(**config), "cpu"
    )


def record(c, r):
    return {
        "prompt": "3",
        "chosen": " ".join(map(str, c)),
        "rejected": " ".join(map(str, r)),
    }


# Complete-response probabilities in ninths and log-score slopes in thirds.
# These are independent of the implementation's teacher-forced softmax.
TABLE = {
    (1,): ((3, 2), (6, 1)),
    (2,): ((6, -1), (3, -2)),
    (1, 1): ((2, 3), (2, 3)),
    (1, 2): ((1, 0), (4, 0)),
    (2, 1): ((2, 1), (2, -1)),
    (2, 2): ((4, -2), (1, -4)),
}


def oracle(c, r, head):
    pc, dc = TABLE[c][head]
    pr, dr = TABLE[r][head]
    qc, qr = TABLE[c][0][0], TABLE[r][0][0]
    odds = Fraction(pc * qr, pr * qc)
    slope = Fraction(dr - dc, 3) / (1 + odds)
    return math.log1p(1 / float(odds)), slope


@pytest.mark.asyncio
@pytest.mark.parametrize("head", [0, 1])
@pytest.mark.parametrize("keep", [1, 2])
async def test_every_bounded_batch_against_independent_rational_table(head, keep):
    model = Conditional()
    t = make(
        model, beta=1.0, learning_rate=0.0, max_length=1 + keep, max_grad_norm=10.0
    )
    if head:
        model.one.copy_(3 - model.one)
    for c, r, reverse in itertools.product(TABLE, TABLE, [False, True]):
        pairs = [(c, r), (r, c) if reverse else (c, r)]
        expected = [oracle(a[:keep], b[:keep], head) for a, b in pairs]
        result = await t.train_step([record(a, b) for a, b in pairs])
        assert result["dpo_loss"] == pytest.approx(
            sum(v for v, _ in expected) / 2, abs=3e-15, rel=1e-14
        )
        assert model.tilt.grad.item() == pytest.approx(
            float(sum(g for _, g in expected) / 2), abs=2e-15, rel=1e-14
        )
        assert t.reference_model.tilt.grad is None


@pytest.mark.asyncio
async def test_two_adamw_steps_with_fixed_reference_match_closed_form():
    model = Conditional()
    t = make(model, beta=0.7, learning_rate=0.025, max_grad_norm=0.13)
    w = 0.0
    m, v = 0.0, 0.0
    for step, pairs in enumerate(
        [
            [((1, 1), (2,)), ((1,), (2, 1)), ((1, 1), (1, 1))],
            [((2, 2), (1,)), ((1, 2), (1, 1))],
        ],
        1,
    ):

        def path_score(path, tilt):
            history = 3
            total, slope = 0.0, 0.0
            for action in path:
                one = 2 if history == 1 else 1
                p = one * math.exp(tilt) / (one * math.exp(tilt) + 3 - one)
                total += math.log(p if action == 1 else 1 - p)
                slope += (action == 1) - p
                history = action
            return total, slope

        gradient, loss = 0.0, 0.0
        for c, r in pairs:
            pc, dc = path_score(c, w)
            pr, dr = path_score(r, w)
            margin = 0.7 * ((pc - pr) - (path_score(c, 0)[0] - path_score(r, 0)[0]))
            loss += math.log1p(math.exp(-margin)) / len(pairs)
            gradient += (-0.7 / (1 + math.exp(margin))) * (dc - dr) / len(pairs)
        gradient *= min(1.0, 0.13 / (abs(gradient) + 1e-6))
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient**2
        w = w * (1 - 0.025 * 0.01) - 0.025 * (m / (1 - 0.9**step)) / (
            math.sqrt(v / (1 - 0.999**step)) + 1e-8
        )
        metrics = await t.train_step([record(a, b) for a, b in pairs])
        assert metrics["dpo_loss"] == pytest.approx(loss, abs=1e-13)
        assert model.tilt.item() == pytest.approx(w, abs=1e-13)
        assert model.tilt.grad.item() == pytest.approx(gradient, abs=1e-13)
        state = t.optimizer.state[model.tilt]
        assert state["exp_avg"].item() == pytest.approx(m, abs=1e-14)
        assert state["exp_avg_sq"].item() == pytest.approx(v, abs=1e-14)
        assert state["step"].item() == step
        assert t.reference_model.tilt.item() == 0


class Table(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(5, 5, dtype=dtype))

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.weight[input_ids])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dtype, prefix",
    [(torch.float16, -40000.0), (torch.float32, -1e8), (torch.float64, -1e308)],
)
async def test_shared_rare_action_keeps_suffix_loss_and_gradients(dtype, prefix):
    model = Table(dtype)
    with torch.no_grad():
        model.weight[3, 1] = prefix
    t = make(model, beta=1.0, learning_rate=0.001)
    with torch.no_grad():
        model.weight[1, 2] = 2.0
    result = await t.train_step([record((1, 2), (1, 4))])
    assert result["dpo_loss"] == pytest.approx(math.log1p(math.exp(-2)), abs=1e-14)
    expected = torch.zeros_like(model.weight)
    expected[1, 4] = 1 / (1 + math.exp(2))
    expected[1, 2] = -expected[1, 4]
    torch.testing.assert_close(
        model.weight.grad,
        expected,
        atol=1e-9 if dtype == torch.float64 else 2e-4,
        rtol=1e-3,
    )


@pytest.mark.asyncio
async def test_normalizer_survives_an_unrepresentable_selected_log_probability():
    model = Table(torch.float64)
    with torch.no_grad():
        model.weight[1].fill_(1.5e308)
        model.weight[1, 2] = -1.5e308
    t = make(model, beta=1.0, learning_rate=0.0)
    with torch.no_grad():
        model.weight[1, 0] = 0.0
    # The selected numerator and maximum are identical across policies. Only
    # three live maxima and four reference maxima contribute to this ratio.
    result = await t.train_step([record((1, 2), (4,))])
    assert result["dpo_loss"] == pytest.approx(math.log1p(3 / 4), abs=1e-14)
    assert model.weight.grad[1, 2].item() == pytest.approx(-3 / 7)


def test_tiny_probability_tail_retains_its_log_derivative():
    z = torch.tensor(
        [[[0.0, -60.0], [0.0, 0.0]]], dtype=torch.float64, requires_grad=True
    )
    terms = completion_terms(z, torch.tensor([[1, 0]]), 1)
    score = terms.sum()
    assert score.item() == pytest.approx(-math.log1p(math.exp(-60)), rel=1e-14, abs=0.0)
    score.backward()
    assert z.grad[0, 0, 0].item() == pytest.approx(math.exp(-60), rel=1e-14, abs=0.0)


def test_extreme_beta_preserves_the_representable_derivative():
    beta = 1e300
    c = torch.tensor([[800 / beta]], dtype=torch.float64, requires_grad=True)
    empty = c[:0]
    loss = preference_pair_loss(c, empty, empty, empty, beta)
    loss.backward()
    assert loss.item() == 0
    assert c.grad.item() == pytest.approx(
        -math.exp(math.log(beta) - 800), abs=0, rel=2e-13
    )


def test_scaled_margin_survives_overflow_in_partial_sum():
    beta = 1e-308
    chosen = torch.full((2,), 1.5e308, dtype=torch.float64, requires_grad=True)
    rejected = chosen[:1].detach().clone().requires_grad_()
    empty = chosen[:0]
    assert not torch.isfinite(chosen.sum())
    loss = preference_pair_loss(chosen, rejected, empty, empty, beta)
    expected = 1.5e308 * beta
    assert loss.item() == pytest.approx(math.log1p(math.exp(-expected)), abs=2e-15)
    loss.backward()
    slope = -beta / (1 + math.exp(expected))
    assert chosen.grad.tolist() == pytest.approx([slope, slope], rel=2e-14, abs=5e-324)
    assert rejected.grad.item() == pytest.approx(-slope, rel=2e-14, abs=5e-324)


@pytest.mark.asyncio
async def test_failed_later_pair_discards_partial_gradient_and_retry_matches_clean_run():
    t = make(Conditional(), beta=0.3, learning_rate=0.01)
    good = record((2,), (1, 1))
    await t.train_step([good])
    control = copy.deepcopy(t)
    model = t.agent.model
    # The rejection follows a valid pair that already ran backward.
    with pytest.raises(ValueError, match="positive probability"):
        await t.train_step([good, record((0,), (1,))])
    assert model.training and not t.reference_model.training
    assert model.tilt.grad is None
    torch.testing.assert_close(model.tilt, control.agent.model.tilt, rtol=0, atol=0)
    torch.testing.assert_close(
        t.optimizer.state_dict(), control.optimizer.state_dict(), rtol=0, atol=0
    )
    assert await t.train_step([]) == {"dpo_loss": 0.0}
    assert await t.train_step([good]) == await control.train_step([good])
    torch.testing.assert_close(model.tilt, control.agent.model.tilt, rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype,lr", [(torch.float16, 0.0002), (torch.bfloat16, 0.001)])
async def test_low_precision_published_weights_and_adamw_moments_match_fp32_master(
    dtype, lr
):
    model = Table(dtype)
    with torch.no_grad():
        model.weight[3, 1] = 1
    t = make(model, learning_rate=lr, beta=1.0, max_grad_norm=10.0)
    master = torch.nn.Parameter(model.weight.detach().float())
    opt = torch.optim.AdamW([master], lr=lr)
    pair = record((1,), (2,))
    for step in range(9):
        if step == 6:
            with torch.no_grad():
                model.weight[3, 2] = 0.125
                master[3, 2] = 0.125
        # Use the actual published policy and the initial frozen reference.
        published = master.detach().to(dtype)
        gap = published[3, 1].item() - published[3, 2].item() - 1.0
        g = 1 / (1 + math.exp(gap))
        grad = torch.zeros_like(published)
        grad[3, 1], grad[3, 2] = -g, g
        master.grad = grad.float()
        opt.step()
        loss = (await t.train_step([pair]))["dpo_loss"]
        assert loss == pytest.approx(math.log1p(math.exp(-gap)), abs=1e-14)
        assert model.weight.isfinite().all()
        torch.testing.assert_close(
            model.weight, master.detach().to(dtype), rtol=0, atol=0
        )
        actual = t.optimizer.state[t.optimizer.param_groups[0]["params"][0]]
        for key, expected in opt.state[master].items():
            torch.testing.assert_close(actual[key], expected, rtol=2e-6, atol=1e-9)
        torch.testing.assert_close(
            actual["master_weight"], master, rtol=1e-6, atol=1e-9
        )
    assert model.weight[3, 1].item() > 1
    assert t.reference_model.weight[3, 1].item() == 1


@pytest.mark.asyncio
async def test_gibbs_policy_is_stationary_on_calibrated_preference_pairs():
    model = Table(torch.float64)
    with torch.no_grad():
        model.weight[3] = torch.arange(1, 6, dtype=torch.float64).log()
    t = make(model, beta=1.0, learning_rate=0.0)
    with torch.no_grad():
        model.weight[3] += torch.arange(5, dtype=torch.float64) * math.log(2)
    batch = []
    expected = 0.0
    # A connected tree suffices even when most pairs are never observed.
    for c, r, wins in [(1, 0, 2), (2, 0, 4), (3, 2, 2), (4, 3, 2)]:
        batch.extend([record((c,), (r,))] * wins + [record((r,), (c,))])
        expected += wins * math.log1p(1 / wins) + math.log1p(wins)
    score = (await t.train_step(batch))["dpo_loss"]
    assert score == pytest.approx(expected / len(batch), abs=2e-15)
    assert model.weight.grad.abs().max().item() < 1e-15
    with torch.no_grad():
        model.weight[3, 3] += 0.5
    assert (await t.train_step(batch))["dpo_loss"] > score


@pytest.mark.asyncio
async def test_disconnected_comparisons_leave_component_mass_undetermined():
    model = Table(torch.float64)
    t = make(model, beta=1.0, learning_rate=0.0)
    # Comparisons stay within {0, 1} and {2, 3, 4}. Each edge has equal wins/losses.
    batch = [
        record((chosen,), (rejected,))
        for a, b in [(0, 1), (2, 3), (3, 4)]
        for chosen, rejected in [(a, b), (b, a)]
    ]
    reference = t.reference_model.weight[3].softmax(-1)
    before = (await t.train_step(batch))["dpo_loss"]
    with torch.no_grad():
        model.weight[3, :2] += math.log(4)
    after = (await t.train_step(batch))["dpo_loss"]
    policy = model.weight[3].detach().softmax(-1)
    assert before == after == pytest.approx(math.log(2))
    assert model.weight.grad.abs().max().item() < 1e-15
    assert reference[:2].sum().item() == pytest.approx(2 / 5)
    assert policy[:2].sum().item() == pytest.approx(8 / 11)
    # With constant zero reward the regularized return drops by this forward KL.
    kl = (policy * (policy.log() - reference.log())).sum().item()
    assert kl == pytest.approx((8 * math.log(20 / 11) + 3 * math.log(5 / 11)) / 11)
    assert kl > 0.0


@pytest.mark.asyncio
async def test_real_causal_transformer_all_gradients_match_conditional_likelihood(
    monkeypatch,
):
    from transformers import GPT2Config, GPT2LMHeadModel

    from stateset_agents.training import preference_loss

    # Force standard backward recomputation over more than one scoring block.
    monkeypatch.setattr(preference_loss, "_SCORE_CHUNK_ELEMENTS", 8)
    torch.manual_seed(4)
    config = GPT2Config(
        vocab_size=8,
        n_positions=12,
        n_embd=8,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.5,
        attn_pdrop=0.3,
        embd_pdrop=0.2,
    )
    model = GPT2LMHeadModel(config).double().train()
    model.transformer.h[0].attn.eval()
    modes = [module.training for module in model.modules()]
    t = make(model, beta=0.4, learning_rate=0.0, max_grad_norm=1e10)
    with torch.no_grad():
        model.transformer.wte.weight[2, 0] += 0.35
    oracle_model = copy.deepcopy(model).eval()
    candidates = [
        ((1, 2, 3, 4), (1, 4, 3)),
        ((2, 1, 4), (3,)),
        ((1,), (1, 4)),
        ((2,), (2,)),
    ]
    reference = t.reference_model

    def score(network, path):
        ids = torch.tensor([[3, *path]])
        logp = network(ids).logits[0, :-1].log_softmax(-1)
        return logp[torch.arange(len(path)), ids[0, 1:]].sum()

    losses = []
    for c, r in candidates:
        delta = (score(oracle_model, c) - score(oracle_model, r)) - (
            score(reference, c) - score(reference, r)
        )
        losses.append(torch.nn.functional.softplus(-0.4 * delta))
    expected = torch.stack(losses).mean()
    expected.backward()
    actual = await t.train_step([record(c, r) for c, r in candidates])
    assert actual["dpo_loss"] == pytest.approx(expected.item(), abs=2e-13)
    for p, q in zip(model.parameters(), oracle_model.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-9, atol=2e-12)
    assert [module.training for module in model.modules()] == modes
    assert all(p.grad is None for p in reference.parameters())


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_half_optimizer_restore_preserves_hidden_progress_and_accepts_legacy_moments(
    legacy,
):
    pair = record((1,), (2,))
    model = Table(torch.float16)
    with torch.no_grad():
        model.weight[3, 1] = 1
    original = make(model, learning_rate=0.0002, beta=1.0)
    await original.train_step([pair])
    saved = copy.deepcopy(original.optimizer.state_dict())
    state = next(iter(saved["state"].values()))
    assert state["master_weight"][3, 1].item() != model.weight[3, 1].item()
    if legacy:
        del state["master_weight"]
        state["exp_avg"] = state["exp_avg"].half()
        state["exp_avg_sq"] = state["exp_avg_sq"].half()
        # Both runs resume this same older snapshot at its stored precision.
        original.optimizer.load_state_dict(copy.deepcopy(saved))
    resumed = make(Table(torch.float16), learning_rate=0.0002, beta=1.0)
    resumed.agent.model.load_state_dict(original.agent.model.state_dict())
    resumed.reference_model.load_state_dict(original.reference_model.state_dict())
    resumed.optimizer.load_state_dict(saved)
    for _ in range(5):
        assert await resumed.train_step([pair]) == await original.train_step([pair])
    torch.testing.assert_close(
        original.agent.model.state_dict(),
        resumed.agent.model.state_dict(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original.optimizer.state_dict(), resumed.optimizer.state_dict(), rtol=0, atol=0
    )
