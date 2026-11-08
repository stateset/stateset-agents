"""Full-model keyed edits must improve a nonlinear path without model mutation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    BudgetExhausted,
    KeyedEditConfig,
    LowRankDelta,
    WeightSpaceSession,
)


class NonlinearLM(torch.nn.Module):
    """An early write influences later positions through a nonlinear accumulator."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=4)
        self.embed = torch.nn.Embedding(4, 4)
        self.write = torch.nn.Linear(4, 4, bias=False)
        self.readout = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(4))
            self.write.weight.zero_()
            self.readout.weight.copy_(torch.eye(4) * 2)
        self.calls = 0
        self.fail_on = None
        self.repeat = False
        self.detach = False

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("injected failure")
        x = self.embed(input_ids)
        y = self.write(x)
        if self.repeat:
            y = self.write(x)
        z = (x + y).tanh().cumsum(dim=1)
        logits = self.readout(z)
        return SimpleNamespace(logits=logits.detach() if self.detach else logits)


def session(budget=30):
    return WeightSpaceSession(
        NonlinearLM().eval(),
        lambda s: ["abcd".index(c) for c in s],
        budget=budget,
    )


def optimize(s):
    return s.optimize_keyed_edit(
        "write",
        torch.eye(4)[:1],
        target_pairs=[("ac", "d"), ("ca", "d")],
        locality_prompts=["bc", "cb"],
        protect=torch.eye(4)[1:3],
        locality=torch.eye(4)[1:3],
        config=KeyedEditConfig(steps=10, learning_rate=0.2),
    )


@torch.no_grad()
def score(model, strings):
    rows = []
    for text in strings:
        ids = torch.tensor([["abcd".index(c) for c in text]])
        rows.append(model(ids, torch.ones_like(ids), False, 0).logits[0].clone())
    return rows


def test_nonlinear_early_edit_improves_targets_preserves_references_and_state():
    s = session()
    state = {name: p.clone() for name, p in s.model.named_parameters()}
    flags, grads = [], []
    for i, p in enumerate(s.model.parameters()):
        p.requires_grad_(i % 2 == 0)
        p.grad = torch.full_like(p, 7)
        flags.append(p.requires_grad)
        grads.append(p.grad)
    before = score(s.model, ["acd", "cad", "bc", "cb"])
    rng = torch.get_rng_state().clone()
    calls_before = s.model.calls
    result = optimize(s)
    assert torch.equal(torch.get_rng_state(), rng)
    assert s.model.calls - calls_before == result.model_forwards == 44
    assert result.backward_calls == 40 and s.budget.used == 11
    assert result.selected_step > 0
    assert (
        result.history[result.selected_step].target_nll
        < result.history[0].target_nll / 3
    )
    assert result.history[result.selected_step].locality_kl == pytest.approx(
        0, abs=1e-12
    )
    assert result.history[result.selected_step].objective == min(
        h.objective for h in result.history
    )
    assert s.active_edit_ids == [] and not s.model.write._forward_hooks
    for (name, p), flag, grad in zip(
        s.model.named_parameters(), flags, grads, strict=True
    ):
        assert torch.equal(p, state[name]) and p.requires_grad == flag
        assert p.grad is grad and bool((p.grad == 7).all())
    # Independent dense application checks the actual whole-model effect.
    with torch.no_grad():
        s.model.write.weight.add_(result.delta.B @ result.delta.A)
    after = score(s.model, ["acd", "cad", "bc", "cb"])
    assert all(int(row[1].argmax()) == 3 for row in after[:2])
    assert all(torch.equal(a, b) for a, b in zip(before[2:], after[2:], strict=True))
    oracle = sum(float(-row[1].double().log_softmax(0)[3]) for row in after[:2]) / 2
    assert oracle == pytest.approx(
        result.history[result.selected_step].target_nll, abs=1e-6
    )
    assert result.protected_span_error == 0


def test_gradients_match_independent_dense_weight_finite_differences(monkeypatch):
    s = session()
    original = s.model.write.weight.detach().clone()
    base = score(s.model, ["ab"])[0].double().log_softmax(-1)
    seen = []
    adam_step = torch.optim.Adam.step

    def record_step(optimizer, *args, **kwargs):
        parameter = optimizer.param_groups[0]["params"][0]
        seen.append((parameter.detach().clone()[:, 0], parameter.grad.clone()[:, 0]))
        return adam_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    s.optimize_keyed_edit(
        "write",
        torch.eye(4)[:1],
        target_pairs=[("ac", "d")],
        locality_prompts=["ab"],
        config=KeyedEditConfig(steps=2, learning_rate=0.1, kl_weight=3, l2_weight=0.05),
    )

    def objective(vector):
        with torch.no_grad():
            s.model.write.weight.copy_(original)
            s.model.write.weight[:, 0].add_(vector)
        row, other = score(s.model, ["acd", "ab"])
        nll = -row[1].double().log_softmax(0)[3]
        kl = (base.exp() * (base - other.double().log_softmax(-1))).sum(-1).mean()
        return float(nll + 3 * kl + 0.05 * vector.double().square().sum())

    eps = 0.001
    assert len(seen) == 2
    for vector, gradient in seen:
        numeric = torch.tensor(
            [
                (
                    objective(vector + torch.eye(4)[i] * eps)
                    - objective(vector - torch.eye(4)[i] * eps)
                )
                / (2 * eps)
                for i in range(4)
            ]
        )
        assert torch.allclose(gradient, numeric, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("failure", ["forward", "repeated", "detached", "nonfinite"])
def test_failure_restores_flags_gradients_hooks_weights_and_prior_edit(failure):
    s = session()
    prior = s.apply("write", LowRankDelta(torch.ones(4, 1) * 0.01, torch.eye(4)[:1]))
    original = s.weight("write")
    flags = [p.requires_grad for p in s.model.parameters()]
    if failure == "forward":
        s.model.fail_on = 6
    elif failure == "repeated":
        s.model.repeat = True
    elif failure == "detached":
        s.model.detach = True
    else:
        with torch.no_grad():
            s.model.readout.weight[0, 0] = float("nan")
    with pytest.raises((ConfigurationError, RuntimeError)):
        optimize(s)
    assert s.active_edit_ids == [prior] and torch.equal(s.weight("write"), original)
    assert [p.requires_grad for p in s.model.parameters()] == flags
    assert not s.model.write._forward_hooks
    assert s.edit_designs[-1]["status"] == "failed"
    assert s.edit_designs[-1]["model_forwards"] == s.model.calls
    assert s.budget.used > 0


@pytest.mark.parametrize(
    "bad",
    [
        "budget",
        "forward_cap",
        "logit_cap",
        "reference_cap",
        "context_cap",
        "tokens_cap",
        "training",
        "dtype",
        "key_conflict",
        "inference",
    ],
)
def test_preflight_rejects_without_forward_or_mutation(bad):
    s = session(budget=0 if bad == "budget" else 30)
    config = KeyedEditConfig(steps=2)
    changes = {
        "forward_cap": {"max_model_forwards": 1},
        "logit_cap": {"max_logit_elements": 1},
        "reference_cap": {"max_reference_elements": 1},
        "context_cap": {"max_context_tokens": 1},
        "tokens_cap": {"max_total_tokens": 1},
    }
    config = replace(config, **changes.get(bad, {}))
    if bad == "training":
        s.model.train()
    if bad == "dtype":
        s.model.half()
    protected = torch.eye(4) if bad == "key_conflict" else None
    with (
        torch.inference_mode(bad == "inference"),
        pytest.raises((ConfigurationError, BudgetExhausted)),
    ):
        s.optimize_keyed_edit(
            "write",
            torch.eye(4)[:1],
            target_pairs=[("ac", "d")],
            locality_prompts=["bc"],
            protect=protected,
            config=config,
        )
    assert s.model.calls == s.budget.used == 0
    assert not s.model.write._forward_hooks and not s.active_edit_ids


def test_no_grad_context_is_supported_and_output_norm_is_bounded():
    s = session()
    with torch.no_grad():
        result = s.optimize_keyed_edit(
            "write",
            torch.eye(4)[:1],
            target_pairs=[("ac", "d")],
            locality_prompts=[],
            config=KeyedEditConfig(steps=3, kl_weight=0, max_output_norm=0.05),
        )
    assert result.selected_step > 0
    assert float(result.delta.B.norm()) <= 0.050001
    assert all(h.output_norm <= 0.050001 for h in result.history)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": True},
        {"learning_rate": float("nan")},
        {"kl_weight": -1},
        {"max_update_norm": 0},
        {"max_reference_elements": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ConfigurationError):
        KeyedEditConfig(**kwargs)


def test_kl_penalty_reduces_independently_measured_collateral_drift():
    measured = []
    for strength in (0, 100):
        s = session()
        before = score(s.model, ["ab"])[0].double().log_softmax(-1)
        result = s.optimize_keyed_edit(
            "write",
            torch.eye(4)[:1],
            target_pairs=[("ac", "d")],
            locality_prompts=["ab"],
            config=KeyedEditConfig(steps=15, learning_rate=0.1, kl_weight=strength),
        )
        with torch.no_grad():
            s.model.write.weight.add_(result.delta.B @ result.delta.A)
        after = score(s.model, ["ab"])[0].double().log_softmax(-1)
        kl = float((before.exp() * (before - after)).sum(-1).mean())
        assert kl == pytest.approx(
            result.history[result.selected_step].locality_kl, abs=1e-8
        )
        measured.append(kl)
    assert measured[0] > 0.01
    assert measured[1] < measured[0] / 5


def test_two_distinct_subject_keys_share_one_learned_output():
    s = session()
    result = s.optimize_keyed_edit(
        "write",
        torch.eye(4)[:2],
        target_pairs=[("ac", "d"), ("bc", "d")],
        locality_prompts=["cc"],
        protect=torch.eye(4)[2:3],
        config=KeyedEditConfig(steps=10, learning_rate=0.2),
    )
    assert result.delta.rank == 1
    assert torch.allclose(torch.eye(4)[:2] @ result.delta.A.T, torch.ones(2, 1))
    assert (
        result.history[result.selected_step].target_nll < result.history[0].target_nll
    )
    assert torch.equal(result.delta.dense()[:, 2], torch.zeros(4))


def test_design_audit_exports_without_raw_design_strings(tmp_path):
    import json

    pytest.importorskip("peft")
    s = session()
    s.encode = lambda text: {
        "private target question": [0, 2],
        "private answer": [3],
        "private preservation question": [1, 2],
    }[text]
    result = s.optimize_keyed_edit(
        "write",
        torch.eye(4)[:1],
        target_pairs=[("private target question", "private answer")],
        locality_prompts=["private preservation question"],
        config=KeyedEditConfig(steps=1),
    )
    s.apply("write", result.delta)
    out = s.export(tmp_path / "adapter")
    raw = (out / "weight_space_manifest.json").read_text()
    assert "private target question" not in raw and "private answer" not in raw
    assert "private preservation question" not in raw
    record = json.loads(raw)["edit_designs"][0]
    assert (
        record["status"] == "completed"
        and record["cases_sha256"] == result.cases_sha256
    )
    assert record["model_forwards"] == 4 and record["backward_calls"] == 2
