"""Check VAPO's two optimizers against an independently enumerated rollout."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer  # noqa: E402


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.hidden = torch.nn.Embedding(4, 3)
        self.actor = torch.nn.Linear(3, 4)
        self.dropout = torch.nn.Dropout(0.55)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        h = self.dropout(self.hidden(input_ids))
        return SimpleNamespace(logits=self.actor(h), hidden_states=(h,))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")


# The first two groups contain different numbers of tokens and verified actions.
# The third group has no response tokens but still owns a row in sequence means.
DATA = {
    "short": [([0, 2, 1], 0.8), ([3, 1], -0.4)],
    "long": [([1, 0, 3, 2], -0.9)],
    "empty": [([0], 0.0)],
}


def make_trainer(
    mode="token_mean", ratio="token", kl="none", eta=0.35, model_dtype=None, **kwargs
):
    torch.manual_seed(61)
    model = Policy()
    if model_dtype is not None:
        model.to(model_dtype)
    reference = copy.deepcopy(model)
    with torch.no_grad():
        reference.actor.bias.add_(torch.tensor([0.1, 0.2, -0.5, 0.3]))
    cfg = VAPOConfig(
        value_warmup_steps=0,
        num_episodes=4,
        num_epochs=3,
        num_gradient_updates=2,
        group_size=2,
        per_device_train_batch_size=4,
        max_completion_length=5,
        gamma=0.75,
        lambda_critic=0.65,
        lambda_policy_alpha=0.5,
        value_clip=0.02,
        positive_lm_weight=0.35,
        value_loss_coef=0.4,
        value_num_layers=1,
        old_logprobs_source="sampler",
        warmup_ratio=0.0,
        actor_learning_rate=0.06,
        critic_learning_rate=0.08,
        weight_decay=0.013,
        max_grad_norm=0.19,
        entropy_coef=eta,
        objective_overrides={
            "aggregate": mode,
            "ratio": ratio,
            "kl": kl,
            "kl_coef": 0.4 if kl != "none" else 0.0,
            "kl_bias_correction": kl == "k3_token",
        },
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    trainer = VAPOTrainer(
        cfg,
        model,
        None,
        lambda prompt, response: DATA[prompt][int(response)][1],
        lambda prompt, response: prompt == "long"
        or prompt == "short"
        and response == "0",
        ref_model=reference,
    )
    return trainer, reference


def distribution(model, token):
    return model.actor(model.hidden.weight[token]).log_softmax(-1)


def forward_gae(residuals, factor, start):
    return sum(factor**offset * value for offset, value in enumerate(residuals[start:]))


def prepare(trainer, reference):
    rows, generated = [], {}
    cfg = trainer.config
    head = trainer.value_head.network[-1]
    with torch.no_grad():
        for prompt, group in DATA.items():
            generated[prompt] = []
            group_rows = []
            for i, (ids, reward) in enumerate(group):
                values = [
                    float(
                        head(
                            trainer.model.hidden.weight[s].to(head.weight.dtype)
                        ).item()
                    )
                    for s in ids[:-1]
                ]
                old = [
                    float(distribution(reference, s)[a])
                    for s, a in zip(ids[:-1], ids[1:], strict=True)
                ]
                length = len(values)
                lam = 1 - 1 / (1 + cfg.lambda_policy_alpha * length)
                residual = [
                    (reward if t + 1 == length else cfg.gamma * values[t + 1]) - v
                    for t, v in enumerate(values)
                ]

                group_rows.append(
                    {
                        "ids": ids,
                        "old": old,
                        "values": values,
                        "adv": [
                            forward_gae(residual, cfg.gamma * lam, t)
                            for t in range(length)
                        ],
                        "target": [
                            v + forward_gae(residual, cfg.gamma * cfg.lambda_critic, t)
                            for t, v in enumerate(values)
                        ],
                        "positive": trainer.verifier_fn(prompt, str(i)),
                    }
                )
                generated[prompt].append(
                    {
                        "input_ids": torch.tensor(ids),
                        "attention_mask": torch.ones(len(ids)),
                        "response_mask": torch.tensor([0] + [1] * length),
                        "response": str(i),
                        "sampler_log_probs": torch.tensor(old, dtype=torch.float64),
                        "sequence_length": length,
                    }
                )
            advantages = [a for r in group_rows for a in r["adv"]]
            if len(advantages) > 1:
                avg = sum(advantages) / len(advantages)
                sd = math.sqrt(
                    sum((a - avg) ** 2 for a in advantages) / (len(advantages) - 1)
                )
                for row in group_rows:
                    row["adv"] = [(a - avg) / (sd + 1e-8) for a in row["adv"]]
            rows.extend(group_rows)
    return rows, generated


def global_losses(model, head, ref, rows, cfg, mode, ratio_kind, kind, eta):
    row_terms, values, imitation, entropy, seq_kl = [], [], [], [], []
    for row in rows:
        ids = row["ids"]
        scores = [distribution(model, s) for s in ids[:-1]]
        old = row["old"]
        ratios = [
            p[a].exp() / math.exp(q)
            for p, a, q in zip(scores, ids[1:], old, strict=True)
        ]
        terms, deltas = [], []
        for j, (p, s, a, r) in enumerate(
            zip(scores, ids[:-1], ids[1:], ratios, strict=True)
        ):
            if ratio_kind != "token":
                geometric = math.prod(ratios) ** (1 / len(ratios))
                r = (
                    geometric
                    if ratio_kind == "sequence"
                    else geometric.detach() * (1 + (p[a] - p[a].detach()))
                )
            adv = row["adv"][j]
            term = -torch.minimum(
                r * adv, r.clamp(1 - cfg.clip_eps_low, 1 + cfg.clip_eps_high) * adv
            )
            q = distribution(ref, s).detach()
            d = q[a] - p[a]
            if kind == "external":
                term = term + 0.4 * (p.exp() * (p - q)).sum()
            if kind == "k3_token":
                term = term + 0.4 * (d.expm1() - d) * r
            terms.append(term)
            deltas.append(d)
            entropy.append(-(p.exp() * p).sum())
            if row["positive"]:
                imitation.append(-p[a])
            v = head(model.hidden.weight[s]).squeeze()
            target = row["target"][j]
            # Independent critic representation. The constant floor is the
            # squared distance of the target outside the rollout clip interval.
            radius = max(0, abs(target - row["values"][j]) - cfg.value_clip)
            values.append(torch.maximum((v - target).square(), v.new_tensor(radius**2)))
        if deltas:
            delta = torch.stack(deltas).mean()
            seq_kl.append(delta.expm1() - delta)
        row_terms.append(terms)
    den = {
        "token_mean": len(values),
        "seq_mean": len(rows),
        "seq_sum_const": len(rows) * cfg.max_completion_length,
    }[mode]
    policy = (
        sum(
            sum(ts) / max(1, len(ts)) if mode == "seq_mean" else sum(ts)
            for ts in row_terms
        )
        / den
    )
    if kind == "k3_sequence":
        policy = policy + 0.4 * sum(seq_kl) / len(rows)
    policy = policy - eta * torch.stack(entropy).mean()
    return (
        policy,
        torch.stack(values).mean(),
        torch.stack(imitation).mean() if imitation else policy.new_zeros(()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["seq_mean", "token_mean", "seq_sum_const"])
@pytest.mark.parametrize("ratio", ["token", "sequence", "sequence_token"])
@pytest.mark.parametrize("kind", ["none", "k3_token", "k3_sequence", "external"])
async def test_two_joint_updates_match_unpadded_returns_gradients_and_moments(
    mode, ratio, kind, monkeypatch
):
    trainer, ref = make_trainer(mode, ratio, kind)
    cfg = trainer.config
    rows, generated = prepare(trainer, ref)

    async def generate(prompt):
        return generated[prompt]

    monkeypatch.setattr(trainer, "generate_group_responses", generate)
    actor = copy.deepcopy(trainer.model)
    value = copy.deepcopy(trainer.value_head)
    oracle_opts = [
        torch.optim.AdamW(
            m.parameters(),
            lr=lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.weight_decay,
        )
        for m, lr in [
            (actor, cfg.actor_learning_rate),
            (value, cfg.critic_learning_rate),
        ]
    ]
    gradients = [[], []]

    def record(index, network):
        def capture(*args):
            gradients[index].append([p.grad.clone() for p in network.parameters()])

        return capture

    trainer.actor_optimizer.register_step_pre_hook(record(0, trainer.model))
    trainer.critic_optimizer.register_step_pre_hook(record(1, trainer.value_head))
    actual = await trainer.train_step(list(DATA))
    for step in range(2):
        for opt in oracle_opts:
            opt.zero_grad()
        policy_loss, value_loss, imitation_loss = global_losses(
            actor,
            value.network[-1],
            ref,
            rows,
            cfg,
            mode,
            ratio,
            kind,
            cfg.entropy_coef,
        )
        (
            policy_loss
            + cfg.value_loss_coef * value_loss
            + cfg.positive_lm_weight * imitation_loss
        ).backward()
        for idx, (m, opt, lr) in enumerate(
            zip(
                (actor, value),
                oracle_opts,
                (cfg.actor_learning_rate, cfg.critic_learning_rate),
                strict=True,
            )
        ):
            params = list(m.parameters())
            norm = math.sqrt(sum(p.grad.double().square().sum().item() for p in params))
            factor = min(1, cfg.max_grad_norm / (norm + 1e-6))
            for a, b in zip(params, gradients[idx][step], strict=True):
                a.grad.mul_(factor)
                torch.testing.assert_close(a.grad, b, atol=3e-6, rtol=1e-4)
            opt.param_groups[0]["lr"] = (
                lr
                * (
                    1
                    + math.cos(
                        math.pi * step / (cfg.num_episodes * cfg.num_gradient_updates)
                    )
                )
                / 2
            )
            opt.step()
    for key, want in zip(
        ("policy_loss", "value_loss", "positive_lm_loss"),
        (policy_loss, value_loss, imitation_loss),
        strict=True,
    ):
        assert actual[key] == pytest.approx(want.item(), abs=3e-6)
    assert (
        trainer.actor_scheduler.last_epoch == trainer.critic_scheduler.last_epoch == 2
    )
    for oracle, live, o, t in zip(
        (actor, value),
        (trainer.model, trainer.value_head),
        oracle_opts,
        (trainer.actor_optimizer, trainer.critic_optimizer),
        strict=True,
    ):
        for x, y in zip(oracle.parameters(), live.parameters(), strict=True):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=5e-6)
            torch.testing.assert_close(o.state[x], t.state[y], rtol=1e-4, atol=1e-6)


@pytest.mark.asyncio
async def test_warmup_does_not_build_policy_gradients_or_advance_actor_clock(
    monkeypatch,
):
    t, ref = make_trainer(value_warmup_steps=1)
    _, generated = prepare(t, ref)

    async def generate(prompt):
        return generated["short"]

    monkeypatch.setattr(t, "generate_group_responses", generate)
    policy = copy.deepcopy(t.model.state_dict())
    head = copy.deepcopy(t.value_head.state_dict())
    await t.train_step(["short"])
    assert t.value_warmup_complete and t.value_warmup_updates == 1
    assert t.actor_scheduler.last_epoch == t.critic_scheduler.last_epoch == 0
    assert not t.actor_optimizer.state
    assert all(p.grad is None for p in t.model.parameters())
    torch.testing.assert_close(t.model.state_dict(), policy, atol=0, rtol=0)
    assert any(
        not torch.equal(t.value_head.state_dict()[k], v) for k, v in head.items()
    )


@pytest.mark.asyncio
async def test_checkpoint_replays_the_reference_shared_backbone_and_two_optimizers(
    tmp_path, monkeypatch
):
    t, ref = make_trainer(kl="external")
    _, generated = prepare(t, ref)

    async def generate(prompt):
        return generated[prompt]

    monkeypatch.setattr(t, "generate_group_responses", generate)
    await t.train_step(list(DATA))
    t.save_checkpoint(str(tmp_path / "saved"))
    other, _ = make_trainer(kl="external")
    with torch.no_grad():
        other.ref_model.actor.bias.add_(0.5)
    other.load_checkpoint(str(tmp_path / "saved"))
    monkeypatch.setattr(other, "generate_group_responses", generate)
    first = await t.train_step(list(DATA))
    replay = await other.train_step(list(DATA))
    for k in ("policy_loss", "value_loss", "positive_lm_loss", "actor_lr", "critic_lr"):
        assert replay[k] == first[k]
    for key in (
        "model",
        "ref_model",
        "value_head",
        "actor_optimizer",
        "critic_optimizer",
    ):
        torch.testing.assert_close(
            getattr(other, key).state_dict(),
            getattr(t, key).state_dict(),
            atol=0,
            rtol=0,
        )


@pytest.mark.asyncio
async def test_undefined_late_reference_clears_both_pending_gradients(monkeypatch):
    t, ref = make_trainer(kl="external")
    _, generated = prepare(t, ref)
    orig = t.ref_model.forward

    def invalid(input_ids, attention_mask):
        out = orig(input_ids, attention_mask)
        if int(input_ids[0, 0]) == 1:
            out.logits[..., 1] = -math.inf
        return out

    monkeypatch.setattr(t.ref_model, "forward", invalid)

    async def generate(prompt):
        return generated[prompt]

    monkeypatch.setattr(t, "generate_group_responses", generate)
    before = [copy.deepcopy(m.state_dict()) for m in (t.model, t.value_head)]
    with pytest.raises(ValueError, match="reference support"):
        await t.train_step(list(DATA))
    for model, saved in zip((t.model, t.value_head), before, strict=True):
        torch.testing.assert_close(model.state_dict(), saved, rtol=0, atol=0)
        assert all(p.grad is None for p in model.parameters())
    assert not t.critic_optimizer.state and not t.actor_optimizer.state
    assert t.actor_scheduler.last_epoch == t.critic_scheduler.last_epoch == 0


@pytest.mark.asyncio
async def test_partial_optimizer_commit_requires_complete_reload(monkeypatch, tmp_path):
    trainer, ref = make_trainer()
    _, generated = prepare(trainer, ref)

    async def generate(prompt):
        return generated[prompt]

    def fail(*args):
        raise RuntimeError("critic step interrupted")

    monkeypatch.setattr(trainer, "generate_group_responses", generate)
    checkpoint = str(tmp_path / "coherent")
    trainer.save_checkpoint(checkpoint)
    handle = trainer.critic_optimizer.register_step_pre_hook(fail)
    with pytest.raises(RuntimeError, match="critic step interrupted"):
        await trainer.train_step(list(DATA))
    assert trainer.actor_optimizer.state and not trainer.critic_optimizer.state
    handle.remove()
    with pytest.raises(RuntimeError, match="quarantined"):
        await trainer.train_step(list(DATA))
    with pytest.raises(RuntimeError, match="quarantined"):
        await trainer.warmup_value_network(list(DATA))
    with pytest.raises(RuntimeError, match="quarantined"):
        trainer.save_checkpoint(str(tmp_path / "partial"))
    trainer.load_checkpoint(checkpoint)
    assert not trainer.actor_optimizer.state and not trainer.critic_optimizer.state
    await trainer.train_step(list(DATA))
    assert trainer.global_step == 1
    assert (
        trainer.actor_scheduler.last_epoch == trainer.critic_scheduler.last_epoch == 2
    )


def test_clipped_critic_matches_the_floor_and_valid_subgradients():
    from stateset_agents.training.rl_losses import clipped_value_error

    for old in (-0.25, 0.5):
        for radius in (0.0, 0.25, 1.0):
            lower, upper = old - radius, old + radius
            for target in (lower - 2, lower, old, upper, upper + 2):
                d = abs(target - max(lower, min(upper, target)))
                for v in (
                    lower - 4,
                    lower,
                    upper,
                    upper + 4,
                    target,
                    target + d,
                    target - d,
                    target + d / 2,
                ):
                    x = torch.tensor(v, dtype=torch.float64, requires_grad=True)
                    t = x.new_tensor(target).requires_grad_()
                    o = x.new_tensor(old).requires_grad_()
                    loss = clipped_value_error(x, o, t, radius)
                    loss.backward()
                    assert loss.item() == pytest.approx(max((v - target) ** 2, d**2))
                    assert t.grad is None and o.grad is None
                    slope = x.grad.item()
                    if abs(v - target) < d:
                        assert slope == 0
                    elif abs(v - target) > d:
                        assert slope == pytest.approx(2 * (v - target))
                    for w in (lower - 5, t.item(), upper + 5, v - 1e-4, v + 1e-4):
                        assert (
                            loss.item() + slope * (w - v)
                            <= max((w - target) ** 2, d**2) + 1e-12
                        )


@pytest.mark.asyncio
async def test_warmup_and_joint_preserve_reward_above_float32_integer_precision(
    monkeypatch,
):
    t, ref = make_trainer(value_warmup_steps=1)
    _, generated = prepare(t, ref)

    async def generate(prompt):
        return generated["short"][1:]

    monkeypatch.setattr(t, "generate_group_responses", generate)
    t.reward_fn = lambda p, r: 16777217.0
    observed = []
    from stateset_agents.training import rl_losses

    original = rl_losses.critic_value_loss

    def record(v, old, returns, mask, clip, **kwargs):
        observed.extend(returns[mask.bool()].tolist())
        return original(v, old, returns, mask, clip, **kwargs)

    monkeypatch.setattr(rl_losses, "critic_value_loss", record)
    await t.train_step(["p"])
    await t.train_step(["p"])
    assert observed == [16777217.0] * 3


@pytest.mark.asyncio
async def test_public_entry_finishes_warmup_and_runs_requested_inner_updates(
    tmp_path, monkeypatch
):
    from stateset_agents.training.vapo_entrypoints import train_with_vapo
    from stateset_agents.training.vapo_trainer import VAPOModelManager

    template, ref = make_trainer(
        value_warmup_steps=1, num_episodes=1, num_epochs=11, num_gradient_updates=3
    )
    cfg = template.config
    _, generated = prepare(template, ref)

    async def generate(self, prompt):
        return generated[prompt]

    monkeypatch.setattr(VAPOTrainer, "generate_group_responses", generate)
    monkeypatch.setattr(
        VAPOModelManager, "load_model_and_tokenizer", lambda self: (Policy(), None)
    )
    recorded = []
    original = VAPOTrainer.save_checkpoint

    def save(self, directory):
        recorded.append(self)
        original(self, directory)

    monkeypatch.setattr(VAPOTrainer, "save_checkpoint", save)
    await train_with_vapo(
        "local-policy",
        lambda p, r: DATA[p][int(r)][1],
        ["short"],
        cfg,
        output_dir=str(tmp_path),
    )
    live = recorded[-1]
    assert live.global_step == 1 and live.value_warmup_updates == 1
    assert live.actor_scheduler.last_epoch == live.critic_scheduler.last_epoch == 3
    assert all(int(s["step"]) == 3 for s in live.actor_optimizer.state.values())
    assert all(int(s["step"]) == 4 for s in live.critic_optimizer.state.values())
    for scheduler in (live.actor_scheduler, live.critic_scheduler):
        assert scheduler.get_last_lr() == [0.0]
        assert scheduler.lr_lambdas[0](7) == 0.0
