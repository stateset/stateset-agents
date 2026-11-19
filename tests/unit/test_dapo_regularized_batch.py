"""Compare streamed DAPO updates with an unpadded categorical objective."""

import copy
import math
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer


class TablePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(
            torch.tensor([[0.8, -0.5, 0.4], [0.7, -1.2, 0.0], [-0.8, 0.2, 1.1]])
        )

    def save_pretrained(self, directory):
        torch.save(self.state_dict(), Path(directory) / "pytorch_model.bin")

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.scores[input_ids])


# Three prompt groups, including an empty response group. Groups and actions
# have different lengths so the three denominators can be distinguished.
GROUPS = [[([0, 1, 2], 0.7), ([2, 0], -0.5)], [([1, 2, 1, 0], -0.4)], [([1], 0.9)]]


def reference_loss(
    policy, reference, old, mode, ratio_kind, kl_kind, beta, entropy_coef, corrected
):
    rows, sequence_kl, entropies = [], [], []
    for group in GROUPS:
        for ids, advantage in group:
            tokens, deltas = [], []
            chosen = []
            for state, action in zip(ids[:-1], ids[1:], strict=True):
                p = policy.scores[state].log_softmax(0)
                q = reference.scores[state].detach().log_softmax(0)
                old_prob = old.scores[state].detach().softmax(0)[action]
                chosen.append((p, q, action, p[action].exp() / old_prob))
            if chosen:
                geometric = math.prod(item[3] for item in chosen) ** (1 / len(chosen))
            for p, q, action, ratio in chosen:
                if ratio_kind == "sequence":
                    ratio = geometric
                elif ratio_kind == "sequence_token":
                    # The specified local score derivative is the frozen sequence weight.
                    ratio = geometric.detach() * (1 + (p[action] - p[action].detach()))
                term = -torch.minimum(
                    ratio * advantage, ratio.clamp(0.8, 1.28) * advantage
                )
                delta = q[action] - p[action]
                if kl_kind == "k3_token":
                    term = term + beta * (delta.expm1() - delta) * (
                        ratio if corrected else 1
                    )
                if kl_kind == "external":
                    term = term + beta * (p.exp() * (p - q)).sum()
                tokens.append(term)
                deltas.append(delta)
                entropies.append(-(p.exp() * p).sum())
            if deltas:
                d = torch.stack(deltas).mean()
                sequence_kl.append(d.expm1() - d)
            rows.append(tokens)

    total = sum(
        sum(row) / max(len(row), 1) if mode == "seq_mean" else sum(row) for row in rows
    )
    denominator = {
        "seq_mean": len(rows),
        "token_mean": len(entropies),
        "seq_sum_const": 5 * len(rows),
    }[mode]
    loss = total / denominator - entropy_coef * torch.stack(entropies).mean()
    if kl_kind == "k3_sequence":
        loss = loss + beta * sum(sequence_kl) / len(rows)
    return loss


def fixed_samples(old):
    samples = []
    for group in GROUPS:
        responses, frozen = [], []
        width = max(len(ids) for ids, _ in group)
        for ids, _ in group:
            row = torch.tensor(ids)
            mask = torch.ones_like(row)
            mask[0] = 0
            responses.append(
                {
                    "input_ids": row,
                    "attention_mask": torch.ones_like(row),
                    "response_mask": mask,
                    "sequence_length": len(row) - 1,
                }
            )
            logp = old.scores.detach()[row[:-1]].log_softmax(-1)
            chosen = logp.gather(1, row[1:, None]).flatten()
            frozen.append(torch.nn.functional.pad(chosen, (0, width - len(ids))))
        samples.append(
            {
                "responses": responses,
                "advantages": torch.tensor([a for _, a in group]),
                "rewards": [a for _, a in group],
                "accuracy": 0.5,
                "old_token_log_probs": torch.stack(frozen),
            }
        )

    return samples


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["seq_mean", "token_mean", "seq_sum_const"])
@pytest.mark.parametrize("ratio_kind", ["token", "sequence", "sequence_token"])
@pytest.mark.parametrize(
    "kl_kind,corrected",
    [
        ("none", False),
        ("k3_token", False),
        ("k3_token", True),
        ("external", False),
        ("k3_sequence", False),
    ],
)
@pytest.mark.parametrize("eta", [0.0, 0.7])
async def test_regularized_updates_match_global_batch(
    monkeypatch, mode, ratio_kind, kl_kind, corrected, eta
):
    model = TablePolicy()
    old = copy.deepcopy(model)
    beta = 0.0 if kl_kind == "none" else 0.6
    cfg = DAPOConfig(
        num_gradient_updates=2,
        learning_rate=0.04,
        max_grad_norm=100,
        max_completion_length=5,
        use_vllm=False,
        objective_overrides={
            "aggregate": mode,
            "ratio": ratio_kind,
            "kl": kl_kind,
            "kl_coef": beta,
            "entropy_coef": eta,
            "kl_bias_correction": corrected,
        },
    )
    trainer = DAPOTrainer(cfg, model, None, lambda p, r: 0.0)
    # Move learner off both the previous sampling policy and the initial reference.
    with torch.no_grad():
        model.scores.add_(
            torch.tensor([[0.03, -0.08, 0.24], [-0.25, 0.37, 0.08], [0.2, -0.13, 0.01]])
        )
    expected = copy.deepcopy(model)
    oracle = torch.optim.AdamW(
        expected.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    samples = fixed_samples(old)

    async def collect(prompts, target_batch_size):
        return samples, 0.0

    monkeypatch.setattr(trainer, "collect_samples_with_dynamic_sampling", collect)
    losses = []
    gradients = []
    for _ in range(2):
        oracle.zero_grad()
        loss = reference_loss(
            expected, old, old, mode, ratio_kind, kl_kind, beta, eta, corrected
        )
        losses.append(float(loss.detach()))
        loss.backward()
        gradients.append(expected.scores.grad.clone())
        oracle.step()
    captured = []
    trainer.optimizer.register_step_pre_hook(
        lambda *args: captured.append(model.scores.grad.clone())
    )
    metrics = await trainer.train_step(["p", "q", "empty"])
    assert metrics["policy_loss"] == pytest.approx(sum(losses) / 2, abs=3e-6)
    for got, want in zip(captured, gradients, strict=True):
        torch.testing.assert_close(got, want, rtol=3e-5, atol=1e-6)
    torch.testing.assert_close(model.scores, expected.scores, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(
        trainer.optimizer.state[model.scores],
        oracle.state[expected.scores],
        rtol=5e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("epsilon", [1e-3, 1e-6, 1e-9])
def test_near_equal_categorical_kl_and_gradient(epsilon):
    from stateset_agents.training.causal_likelihood import categorical_regularizers

    policy = torch.tensor(
        [[[0.8, -0.5, 0.4], [float("nan")] * 3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    ref = policy.detach().clone()
    ref[0, 0, 0] += epsilon
    ref.requires_grad_()
    with localcontext() as ctx:
        ctx.prec = 80

        def normalized(logits):
            weights = [Decimal.from_float(v).exp() for v in logits]
            return [w / sum(weights) for w in weights]

        p = normalized(policy[0, 0].tolist())
        q = normalized(ref[0, 0].tolist())
        d = [(a / b).ln() for a, b in zip(p, q, strict=True)]
        kl = sum(a * b for a, b in zip(p, d, strict=True))
        slope = [float(a * (b - kl)) for a, b in zip(p, d, strict=True)]
    _, got = categorical_regularizers(
        policy, ref, torch.tensor([[1, 0]]), want_entropy=False
    )
    assert got[0, 0].item() == pytest.approx(float(kl), rel=1e-6, abs=0)
    got.sum().backward()
    assert policy.grad[0, 0].tolist() == pytest.approx(slope, rel=1e-6, abs=0)
    assert policy.grad[0, 1].count_nonzero() == 0
    assert ref.grad is None


def test_sparse_support_chunking_and_dominant_entropy():
    from stateset_agents.training.causal_likelihood import categorical_regularizers

    z = torch.tensor(
        [[[0.0, -80.0, -math.inf], [-math.inf, 0.0, 0.0], [math.nan] * 3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    ref = torch.tensor(
        [[[0.0, -80.0, 0.0], [-2000.0, 0.0, 0.0], [math.inf] * 3]], dtype=z.dtype
    )
    mask = torch.tensor([[1, 1, 0]])
    entropy, kl = categorical_regularizers(
        z, ref, mask, want_entropy=True, chunk_elements=3
    )
    assert entropy[0, 0].item() == pytest.approx(81 * math.exp(-80), rel=1e-12, abs=0)
    assert kl[0, 0].item() == pytest.approx(math.log(2), abs=1e-14)
    assert entropy[0, 1].item() == pytest.approx(math.log(2))
    assert entropy[0, 2] == kl[0, 2] == 0
    ent_gradient = torch.autograd.grad(entropy.sum(), z, retain_graph=True)[0]
    assert ent_gradient[0, 0, :2].tolist() == pytest.approx(
        [-80 * math.exp(-80), 80 * math.exp(-80)], rel=1e-12, abs=0
    )
    (entropy + kl).sum().backward()
    assert z.grad.isfinite().all()
    assert z.grad[0, 2].count_nonzero() == 0
    ref[0, 0, 1] = -math.inf
    with pytest.raises(ValueError, match="reference support"):
        categorical_regularizers(z, ref, mask, want_entropy=False)


@pytest.mark.asyncio
async def test_late_invalid_reference_discards_partial_group_gradients(monkeypatch):
    model = TablePolicy()
    cfg = DAPOConfig(
        use_vllm=False, objective_overrides={"kl": "external", "kl_coef": 0.6}
    )
    ref = copy.deepcopy(model)
    with torch.no_grad():
        ref.scores[1, 2] = -math.inf
    trainer = DAPOTrainer(cfg, model, None, lambda p, r: 0.0, ref_model=ref)
    samples = fixed_samples(model)
    # The first group has only prefix 2, the next includes the invalid prefix 1.
    samples[0]["responses"] = samples[0]["responses"][1:]
    samples[0]["advantages"] = samples[0]["advantages"][1:]
    samples[0]["old_token_log_probs"] = samples[0]["old_token_log_probs"][1:, :1]

    async def collect(prompts, target_batch_size):
        return samples, 0.0

    monkeypatch.setattr(trainer, "collect_samples_with_dynamic_sampling", collect)
    original = model.scores.detach().clone()
    with pytest.raises(ValueError, match="reference support"):
        await trainer.train_step(["p", "q"])
    assert model.scores.grad is None
    assert trainer.global_step == trainer.scheduler.last_epoch == 0
    assert not trainer.optimizer.state
    torch.testing.assert_close(model.scores, original, rtol=0, atol=0)


@pytest.mark.asyncio
async def test_checkpoint_restores_frozen_reference_and_next_update(
    tmp_path, monkeypatch
):
    model = TablePolicy()
    cfg = DAPOConfig(
        use_vllm=False,
        learning_rate=0.04,
        objective_overrides={"kl": "external", "kl_coef": 0.6, "entropy_coef": 0.5},
    )
    trainer = DAPOTrainer(cfg, model, None, lambda p, r: 0.0)
    samples = fixed_samples(model)

    async def collect(prompts, target_batch_size):
        return samples, 0.0

    monkeypatch.setattr(trainer, "collect_samples_with_dynamic_sampling", collect)
    await trainer.train_step(["p", "q"])
    trainer.save_checkpoint(str(tmp_path / "ckpt"))
    restored_model = TablePolicy()
    with torch.no_grad():
        restored_model.scores.add_(1.5)
    restored = DAPOTrainer(cfg, restored_model, None, lambda p, r: 0.0)
    restored.load_runtime_checkpoint(str(tmp_path / "ckpt"))
    monkeypatch.setattr(restored, "collect_samples_with_dynamic_sampling", collect)
    assert not torch.equal(restored.model.scores, restored.ref_model.scores)
    torch.testing.assert_close(
        restored.ref_model.scores, trainer.ref_model.scores, rtol=0, atol=0
    )
    original_metrics = await trainer.train_step(["p", "q"])
    restored_metrics = await restored.train_step(["p", "q"])
    for key in ("policy_loss", "learning_rate", "global_step"):
        assert original_metrics[key] == restored_metrics[key]
    torch.testing.assert_close(
        restored.model.state_dict(), trainer.model.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored.optimizer.state_dict(), trainer.optimizer.state_dict(), rtol=0, atol=0
    )


def test_population_weights_use_actual_observations_including_an_empty_batch():
    from stateset_agents.training import objectives

    current = torch.zeros(2, 3, requires_grad=True)

    def loss(mask, population):
        return objectives.policy_loss(
            logp_cur=current,
            mask=mask,
            advantages=torch.ones(2),
            objective=objectives.OBJECTIVES["dapo"],
            batch_population=population,
        ).loss

    empty = loss(torch.zeros_like(current), (2, 0))
    empty.backward()
    assert empty == 0 and current.grad.count_nonzero() == 0
    with pytest.raises(ValueError, match="cover"):
        loss(torch.ones_like(current), (2, 4))
    with pytest.raises(ValueError, match="binary"):
        loss(torch.full_like(current, 0.5), (2, 3))


@pytest.mark.parametrize(("beta", "eta"), [(1.0, 0.0), (0.0, 0.7), (0.3, 0.7)])
def test_categorical_penalty_excess_is_kl_to_tilted_reference(beta, eta):
    from stateset_agents.training.causal_likelihood import categorical_regularizers

    ref = torch.tensor([[[-1.0, 0.5, -0.2]]], dtype=torch.float64)
    mask = torch.ones((1, 1), dtype=torch.bool)
    optimum = (beta / (beta + eta) * ref).requires_grad_()
    logits = torch.tensor([[[0.3, -0.6, 0.1]]], dtype=ref.dtype, requires_grad=True)

    def penalty(z):
        entropy, divergence = categorical_regularizers(z, ref, mask, want_entropy=True)
        return (beta * divergence - eta * entropy).sum()

    best = penalty(optimum)
    best.backward()
    torch.testing.assert_close(
        optimum.grad, torch.zeros_like(optimum), rtol=0, atol=2e-16
    )
    p = logits.log_softmax(-1)
    g = optimum.detach().log_softmax(-1)
    predicted = (beta + eta) * (p.exp() * (p - g)).sum()
    torch.testing.assert_close(
        penalty(logits) - best.detach(), predicted, rtol=2e-14, atol=1e-15
    )


def test_reference_snapshot_is_independent_with_shared_input_parameters():
    learner = TablePolicy()
    supplied = TablePolicy()
    supplied.scores = learner.scores
    initial = learner.scores.detach().clone()
    cfg = DAPOConfig(
        use_vllm=False, objective_overrides={"kl": "k3_token", "kl_coef": 0.2}
    )
    trainer = DAPOTrainer(cfg, learner, None, lambda p, r: 0.0, ref_model=supplied)
    assert learner.scores.requires_grad and not trainer.ref_model.scores.requires_grad
    with torch.no_grad():
        supplied.scores[0, 0].add_(0.5)
    torch.testing.assert_close(trainer.ref_model.scores, initial, atol=0, rtol=0)
