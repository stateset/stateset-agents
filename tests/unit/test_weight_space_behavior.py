"""Full-token capture and whole-model trial acceptance/rollback contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    BehaviorLimits,
    BudgetExhausted,
    LowRankDelta,
    WeightSpaceSession,
)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=7)
        self.embed = torch.nn.Embedding(7, 7)
        self.readout = torch.nn.Linear(7, 7, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(7))
            self.readout.weight.copy_(torch.eye(7))
        self.calls = 0
        self.failure = None

    def get_output_embeddings(self):
        return self.readout

    def forward(
        self, input_ids, attention_mask=None, logits_to_keep=0, use_cache=False
    ):
        self.calls += 1
        if self.failure == "exception" and self.calls == 2:
            raise RuntimeError("intentional candidate failure")
        x = self.embed(input_ids)
        logits = self.readout(x[:, -logits_to_keep:] if logits_to_keep else x)
        if self.failure == "nan" and self.calls == 2:
            logits = logits * float("nan")
        return SimpleNamespace(logits=logits)


def make_session(budget=6, batch_size=2, **kwargs):
    return WeightSpaceSession(
        TinyLM().eval(),
        lambda text: [ord(c) % 7 for c in text],
        budget=budget,
        batch_size=batch_size,
        **kwargs,
    )


def delta(size=1.0):
    b, a = torch.zeros(7, 1), torch.zeros(1, 7)
    b[0, 0], a[0, 6] = size, 1  # At 'a', favor next token 'b'.
    return LowRankDelta(b, a)


def limits(**kwargs):
    return BehaviorLimits(
        **{
            "max_mean_kl": 10.0,
            "max_token_kl": 10.0,
            "max_nll_increase": 10.0,
            **kwargs,
        }
    )


def test_all_token_capture_omits_padding_and_preserves_order_and_dtype():
    s = make_session(batch_size=2)
    prompts = ["abcd", "x", "yz"]
    got = s.capture_many(["readout"], prompts, read="input", position="all")["readout"]
    expected = s.model.embed(torch.tensor([t for p in prompts for t in s.encode(p)]))
    assert torch.equal(got, expected)
    assert got.shape == (7, 7)
    assert s.budget.used == 1
    assert not s.model.readout._forward_pre_hooks


def test_all_token_capture_keeps_low_precision_inputs():
    s = make_session()
    s.model.to(torch.bfloat16)
    values = s.capture_many(["readout"], ["ab", "c"], position="all", read="input")[
        "readout"
    ]
    assert values.dtype == torch.bfloat16
    assert torch.equal(values, s.model.embed(torch.tensor(s.encode("abc"))))


def test_capture_rejects_missing_or_repeated_modules_and_removes_hooks():
    s = make_session()
    s.model.unused = torch.nn.Linear(7, 7)
    with pytest.raises(ConfigurationError, match="not invoked"):
        s.capture_many(["unused"], ["abc"], position="all", read="input")
    assert not s.model.unused._forward_pre_hooks
    original_forward = s.model.forward

    def repeated(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        s.model.readout(torch.zeros(1, 3, 7))
        return result

    s.model.forward = repeated
    with pytest.raises(ConfigurationError, match="one full-sequence"):
        s.capture_many(["readout"], ["abc"], position="all", read="input")
    assert not s.model.readout._forward_pre_hooks


def test_capture_output_head_keeps_all_tokens_even_with_last_or_mean_selection():
    s = make_session()
    for position in ("last", "mean", "all"):
        got = s.capture("readout", ["abcd", "x"], position=position)
        assert got.shape == ((5, 7) if position == "all" else (2, 7))


def test_all_token_capture_limit_fails_without_spending_or_installing_hooks():
    s = make_session()
    with pytest.raises(ConfigurationError, match="max_rows"):
        s.capture_many(
            ["readout"], ["abcd", "xy"], read="input", position="all", max_rows=5
        )
    assert s.budget.used == s.model.calls == 0
    assert not s.model.readout._forward_pre_hooks


def test_trial_metrics_match_independent_probability_and_loss_calculations():
    s = make_session()
    original = s.weight("readout")
    prompts = ["abc", "za"]
    result = s.trial_edit(
        "readout",
        delta(),
        locality_prompts=prompts,
        target_pairs=[("a", "b")],
        limits=limits(min_target_gain=0.1),
    )
    assert result.accepted
    kl, agree, base_losses, edited_losses = [], [], [], []
    for prompt in prompts:
        ids = s.encode(prompt)
        for i, token in enumerate(ids):
            base_logits, edit_logits = (
                original[:, token].double(),
                s.weight("readout")[:, token].double(),
            )
            p, q = torch.softmax(base_logits, 0), torch.softmax(edit_logits, 0)
            kl.append(float(torch.sum(p * torch.log(p / q))))
            agree.append(int(p.argmax() == q.argmax()))
            if i + 1 < len(ids):
                base_losses.append(float(-torch.log(p[ids[i + 1]])))
                edited_losses.append(float(-torch.log(q[ids[i + 1]])))
    m = result.metrics
    assert m.locality_tokens == 5 and m.observed_tokens == 3
    assert m.mean_kl == pytest.approx(sum(kl) / 5, abs=1e-12)
    assert m.max_token_kl == pytest.approx(max(kl), abs=1e-12)
    assert m.top1_agreement == sum(agree) / 5
    assert m.base_nll == pytest.approx(sum(base_losses) / 3)
    assert m.edited_nll == pytest.approx(sum(edited_losses) / 3)
    assert m.mean_target_gain > 0.1
    assert s.budget.used == 2
    assert s.active_edit_ids == [result.edit_id]
    assert s.trials[0]["accepted"] is True
    s.revert()
    assert torch.equal(s.weight("readout"), original)


@pytest.mark.parametrize(
    "caps, violation",
    [
        ({"max_mean_kl": 0}, "mean_kl"),
        ({"max_token_kl": 0}, "max_token_kl"),
        ({"max_nll_increase": 0}, "nll_increase"),
        ({"min_top1_agreement": 1}, "top1_agreement"),
        ({"min_target_gain": 100}, "mean_target_gain"),
        ({"min_each_target_gain": 100}, "each_target_gain"),
    ],
)
def test_rejection_restores_prior_stack_exactly(caps, violation):
    s = make_session()
    existing = s.apply("readout", delta(0.01))
    before = s.weight("readout")
    result = s.trial_edit(
        "readout",
        delta(2.0),
        locality_prompts=["az", "qa"],
        target_pairs=[("a", "b")],
        limits=limits(**caps),
    )
    assert not result.accepted and violation in result.violations
    assert s.active_edit_ids == [existing]
    assert torch.equal(s.weight("readout"), before)
    assert s.budget.used == 2


def test_single_token_kl_spike_is_not_hidden_by_mean_or_observed_loss():
    s = make_session()
    trial = s.trial_edit(
        "readout",
        delta(),
        locality_prompts=["zzzzza"],
        limits=limits(max_mean_kl=1, max_token_kl=0.01),
    )
    assert trial.metrics.mean_kl < 1
    assert trial.metrics.nll_increase == 0
    assert trial.violations == ("max_token_kl",)


def test_other_session_operations_cannot_interleave_with_trial():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    s = make_session()
    entered, release, other_started = Event(), Event(), Event()
    forward = s.model.forward
    original = s.weight("readout")

    def wait_on_candidate(*args, **kwargs):
        if s.model.calls == 1:
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test synchronization timeout")
        return forward(*args, **kwargs)

    s.model.forward = wait_on_candidate

    def other_edit():
        other_started.set()
        return s.apply("readout", delta(0.1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        trial = pool.submit(
            s.trial_edit,
            "readout",
            delta(),
            locality_prompts=["abc"],
            limits=limits(max_mean_kl=0),
        )
        try:
            assert entered.wait(5)
            other = pool.submit(other_edit)
            assert other_started.wait(5)
            assert not other.done()
        finally:
            release.set()
        assert not trial.result(timeout=5).accepted
        other_id = other.result(timeout=5)
    assert s.active_edit_ids == [other_id]
    assert torch.equal(s.weight("readout"), original + delta(0.1).dense())


@pytest.mark.parametrize("failure", ["exception", "nan"])
def test_candidate_forward_failure_rolls_back_and_records_failure(failure):
    s = make_session()
    s.model.failure = failure
    before = s.weight("readout")
    with pytest.raises((RuntimeError, ConfigurationError)):
        s.trial_edit("readout", delta(), locality_prompts=["abc"], limits=limits())
    assert s.active_edit_ids == []
    assert torch.equal(s.weight("readout"), before)
    assert s.trials[-1]["reverted"] is True
    assert s.budget.used == 2


def test_trial_budget_and_resource_caps_fail_before_forward_or_mutation():
    s = make_session(budget=1)
    with pytest.raises(BudgetExhausted, match="two"):
        s.trial_edit("readout", delta(), locality_prompts=["abc"], limits=limits())
    s = make_session()
    with pytest.raises(ConfigurationError, match="max_logit_elements"):
        s.trial_edit(
            "readout",
            delta(),
            locality_prompts=["abc"],
            limits=limits(),
            max_logit_elements=20,
        )
    assert s.budget.used == s.model.calls == 0


def test_invalid_trial_proposal_fails_before_baseline_and_valid_proposal_is_owned():
    s = make_session()
    invalid = delta(float("nan"))
    with pytest.raises(ConfigurationError, match="finite"):
        s.trial_edit("readout", invalid, locality_prompts=["abc"], limits=limits())
    assert s.model.calls == s.budget.used == 0
    proposal = delta()
    original_forward = s.model.forward

    def mutate_caller_factors(*args, **kwargs):
        proposal.B.zero_()
        return original_forward(*args, **kwargs)

    s.model.forward = mutate_caller_factors
    result = s.trial_edit(
        "readout",
        proposal,
        locality_prompts=["abc"],
        target_pairs=[("a", "b")],
        limits=limits(min_target_gain=0.1),
    )
    assert result.accepted and result.metrics.mean_target_gain > 0.1


@pytest.mark.parametrize(
    "budget,batch_size", [(-1, 1), (1.5, 1), (True, 1), (1, 0), (1, 2.5)]
)
def test_invalid_session_budget_or_batch_size_is_rejected(budget, batch_size):
    with pytest.raises(ConfigurationError):
        make_session(budget=budget, batch_size=batch_size)


def test_trial_requires_eval_mode_and_data_for_requested_gates():
    s = make_session()
    s.model.train()
    with pytest.raises(ConfigurationError, match="eval mode"):
        s.trial_edit("readout", delta(), locality_prompts=["abc"], limits=limits())
    s.model.eval()
    with pytest.raises(ConfigurationError, match="target_pairs"):
        s.trial_edit(
            "readout",
            delta(),
            locality_prompts=["abc"],
            limits=limits(min_target_gain=0),
        )
    with pytest.raises(ConfigurationError, match="observed next token"):
        s.trial_edit("readout", delta(), locality_prompts=["a"], limits=limits())
    assert s.model.calls == 0


def test_trial_acceptance_record_is_in_export_manifest(tmp_path):
    pytest.importorskip("peft")
    s = make_session()
    trial = s.trial_edit("readout", delta(), locality_prompts=["abc"], limits=limits())
    out = s.export(tmp_path / "adapter")
    manifest = json.loads((out / "weight_space_manifest.json").read_text())
    assert manifest["behavior_trials"][0]["cases_sha256"] == trial.cases_sha256
    assert manifest["behavior_trials"][0]["metrics"]["locality_tokens"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_mean_kl": -1},
        {"max_token_kl": float("nan")},
        {"min_top1_agreement": float("nan")},
        {"min_target_gain": float("inf")},
    ],
)
def test_limits_reject_invalid_numbers(kwargs):
    with pytest.raises(ConfigurationError):
        limits(**kwargs)


def test_fixed_reference_stops_small_edits_accumulating_drift():
    s = make_session(budget=20)
    original = s.weight("readout")
    cap = limits(max_mean_kl=0.01)
    reference = s.add_behavior_reference(locality_prompts=["aa"], limits=cap)
    accepted = []
    for _ in range(10):
        previous = s.weight("readout")
        trial = s.trial_edit("readout", delta(0.2), locality_prompts=["aa"], limits=cap)
        # The incremental gate alone would accept every proposal.
        assert trial.metrics.mean_kl < cap.max_mean_kl
        check = trial.reference_checks[0]
        assert check.reference_id == reference.reference_id
        candidate = previous + delta(0.2).dense()
        p = original[:, 6].double().softmax(0)
        q = candidate[:, 6].double().softmax(0)
        expected = float((p * (p.log() - q.log())).sum())
        assert check.metrics.mean_kl == pytest.approx(expected, abs=1e-12)
        if not trial.accepted:
            assert check.violations == ("mean_kl",)
            assert trial.violations == (f"reference:{reference.reference_id}:mean_kl",)
            assert torch.equal(s.weight("readout"), previous)
            break
        accepted.append(trial.edit_id)
    else:
        pytest.fail("cumulative drift was never stopped")
    assert accepted and s.active_edit_ids == accepted
    assert s.budget.used == 1 + 2 * (
        len(accepted) + 1
    )  # identical plans share forwards
    assert reference.edit_ids == ()


def test_reference_preserves_earlier_target_when_new_target_improves():
    s = make_session(budget=10)
    first = s.apply("readout", delta())
    previous = s.weight("readout")
    reference = s.add_behavior_reference(
        locality_prompts=["zz"],
        target_pairs=[("a", "b")],
        limits=limits(min_each_target_gain=-0.01),
    )
    proposal = delta()
    proposal.B.zero_()
    proposal.B[1] = 3  # Improve 'c' after 'a', competing with the existing 'b'.
    trial = s.trial_edit(
        "readout",
        proposal,
        locality_prompts=["zz"],
        target_pairs=[("a", "c")],
        limits=limits(min_target_gain=0.1),
    )
    assert trial.metrics.mean_target_gain > 0.1
    assert trial.reference_checks[0].metrics.target_gains[0] < -0.01
    assert trial.violations == (f"reference:{reference.reference_id}:each_target_gain",)
    assert not trial.accepted
    assert reference.edit_ids == (first,)
    assert s.active_edit_ids == [first] and torch.equal(s.weight("readout"), previous)
    assert s.budget.used == 4  # capture, incremental before/after, distinct reference


def test_references_with_same_cases_keep_distinct_baselines_and_share_forward():
    s = make_session(budget=10)
    base = s.add_behavior_reference(locality_prompts=["aa"], limits=limits())
    s.apply("readout", delta(0.2))
    current = s.add_behavior_reference(locality_prompts=["aa"], limits=limits())
    trial = s.trial_edit(
        "readout", delta(0.2), locality_prompts=["aa"], limits=limits()
    )
    checks = {c.reference_id: c for c in trial.reference_checks}
    assert (
        checks[base.reference_id].metrics.mean_kl
        > checks[current.reference_id].metrics.mean_kl
    )
    assert checks[current.reference_id].metrics == trial.metrics
    assert s.budget.used == 4


def test_reference_token_ids_are_frozen_even_when_encoder_changes():
    s = make_session(budget=5)
    reference = s.add_behavior_reference(
        locality_prompts=["aa"], limits=limits(max_mean_kl=0)
    )
    s.encode = lambda text: [1, 1]
    trial = s.trial_edit(
        "readout", delta(), locality_prompts=["aa"], limits=limits(max_mean_kl=0)
    )
    assert trial.metrics.mean_kl == 0
    assert trial.reference_checks[0].metrics.mean_kl > 0
    assert trial.cases_sha256 != reference.cases_sha256
    assert not trial.accepted and s.budget.used == 4


def test_reference_budget_preflight_and_failure_restore_previous_stack():
    s = make_session(budget=3)
    reference = s.add_behavior_reference(locality_prompts=["aa"], limits=limits())
    before = s.weight("readout")
    with pytest.raises(BudgetExhausted, match="3 total"):
        s.trial_edit("readout", delta(), locality_prompts=["ab"], limits=limits())
    assert s.budget.used == s.model.calls == 1
    assert torch.equal(s.weight("readout"), before)
    s.budget.limit = 4
    forward = s.model.forward

    def fail_reference(*args, **kwargs):
        if s.model.calls == 3:
            raise RuntimeError("reference measurement failed")
        return forward(*args, **kwargs)

    s.model.forward = fail_reference
    with pytest.raises(RuntimeError, match="reference measurement failed"):
        s.trial_edit("readout", delta(), locality_prompts=["ab"], limits=limits())
    assert s.budget.used == 4 and s.active_edit_ids == []
    assert torch.equal(s.weight("readout"), before)
    assert s.behavior_references == (reference,)
    assert s.trials[-1]["reference_ids"] == [reference.reference_id]
    assert s.trials[-1]["reverted"]


def test_reference_memory_cap_and_explicit_release():
    from dataclasses import FrozenInstanceError, replace

    s = make_session(reference_logit_limit=14)
    ref = s.add_behavior_reference(locality_prompts=["aa"], limits=limits())
    assert ref.retained_logit_elements == 14
    with pytest.raises(FrozenInstanceError):
        ref.limits = limits(max_mean_kl=0)
    with pytest.raises(ConfigurationError, match="reference_logit_limit"):
        s.add_behavior_reference(locality_prompts=["aa"], limits=limits())
    assert s.budget.used == 1
    with pytest.raises(ConfigurationError, match="not active"):
        make_session().release_behavior_reference(ref)
    with pytest.raises(ConfigurationError, match="not active"):
        s.release_behavior_reference(replace(ref))
    assert s.behavior_references == (ref,)
    s.release_behavior_reference(ref)
    assert s.behavior_references == ()
    with pytest.raises(ConfigurationError, match="not active"):
        s.release_behavior_reference(ref)
    s.add_behavior_reference(locality_prompts=["ab"], limits=limits())
    assert s.budget.used == 2


def test_reference_runtime_cap_when_model_does_not_advertise_vocabulary():
    s = make_session(reference_logit_limit=13)
    del s.model.config.vocab_size
    with pytest.raises(ConfigurationError, match="reference_logit_limit"):
        s.add_behavior_reference(locality_prompts=["aa"], limits=limits())
    assert not s.behavior_references and s.budget.used == 1


@pytest.mark.parametrize("cap", [-1, 1.5, True])
def test_invalid_reference_memory_cap(cap):
    with pytest.raises(ConfigurationError, match="reference_logit_limit"):
        make_session(reference_logit_limit=cap)


def test_reference_case_plan_rechecks_padding_cap_if_batch_size_changes():
    s = make_session(batch_size=1)
    s.add_behavior_reference(
        locality_prompts=["a", "aaaa"], limits=limits(), max_logit_elements=35
    )
    s.batch_size = 2
    with pytest.raises(ConfigurationError, match="max_logit_elements"):
        s.trial_edit("readout", delta(), locality_prompts=["ab"], limits=limits())
    assert s.budget.used == 1 and not s.active_edit_ids


def test_trial_export_retains_released_reference_provenance(tmp_path):
    pytest.importorskip("peft")
    s = make_session()
    ref = s.add_behavior_reference(locality_prompts=["aa"], limits=limits())
    trial = s.trial_edit("readout", delta(), locality_prompts=["aa"], limits=limits())
    s.release_behavior_reference(ref)
    out = s.export(tmp_path / "adapter")
    manifest = json.loads((out / "weight_space_manifest.json").read_text())
    record = manifest["behavior_references"][0]
    assert record["reference_id"] == ref.reference_id
    assert record["cases_sha256"] == ref.cases_sha256
    assert record["active"] is False and record["edit_ids"] == []
    check = manifest["behavior_trials"][0]["reference_checks"][0]
    assert check["reference_id"] == ref.reference_id
    assert check["metrics"]["mean_kl"] == trial.reference_checks[0].metrics.mean_kl
