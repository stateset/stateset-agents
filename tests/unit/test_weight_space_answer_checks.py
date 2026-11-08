"""Generated answers participate in atomic acceptance, rollback and accounting."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    AnswerCase,
    AnswerLimits,
    BehaviorLimits,
    BudgetExhausted,
    LowRankDelta,
    WeightSpaceSession,
    answer_occurrences,
)


class AnswerLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=7, eos_token_id=5)
        self.embed = torch.nn.Embedding(7, 7)
        self.readout = torch.nn.Linear(7, 7, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(7))
            self.readout.weight.copy_(torch.eye(7))
            self.readout.weight[5, 0] = 2  # b -> EOS
        self.calls = 0
        self.fail_generation = False

    def forward(
        self, input_ids, attention_mask=None, logits_to_keep=0, use_cache=False
    ):
        self.calls += 1
        if (
            self.fail_generation
            and input_ids.shape[1] == 1
            and self.readout.weight[0, 6] > 1
        ):
            raise RuntimeError("candidate generation failed")
        x = self.embed(input_ids)
        return SimpleNamespace(
            logits=self.readout(x[:, -logits_to_keep:] if logits_to_keep else x)
        )


def make_session(budget=10):
    return WeightSpaceSession(
        AnswerLM().eval(),
        lambda text: [ord(c) % 7 for c in text],
        budget=budget,
        batch_size=1,
        decode=lambda ids: " ".join("bcdefga"[i] for i in ids if i != 5),
    )


def proposal(*entries):
    d = torch.zeros(7, 7)
    for row, col, value in entries or ((0, 6, 2),):
        d[row, col] = value
    return LowRankDelta(d, torch.eye(7))


def run(s, edit=None, cases=None, limits=None, **kwargs):
    return s.trial_edit(
        "readout",
        edit or proposal(),
        locality_prompts=["dd"],
        target_pairs=[("a", "b")],
        limits=BehaviorLimits(10, 10, 10, min_target_gain=0.1),
        answer_cases=(
            cases
            if cases is not None
            else [AnswerCase("a", ("b",)), AnswerCase("c", ("c",), "locality")]
        ),
        answer_limits=limits or AnswerLimits(max_new_tokens=3),
        **kwargs,
    )


def test_probability_gain_without_correct_generation_is_rejected():
    s = make_session()
    before = s.weight("readout")
    trial = run(s, proposal((0, 6, 0.5)))
    assert trial.metrics.mean_target_gain > 0.1
    assert trial.violations == ("generation:target_accuracy",)
    assert trial.answer_check.metrics.target_correct == 0
    assert s.active_edit_ids == [] and torch.equal(s.weight("readout"), before)
    assert s.budget.used == 4


def test_pretrained_factory_provides_the_answer_decoder(monkeypatch):
    transformers = pytest.importorskip("transformers")

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(c) % 7 for c in text]}

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return " ".join("bcdefga"[i] for i in ids if i != 5)

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **kw: AnswerLM(),
    )
    s = WeightSpaceSession.from_pretrained("fake", budget=4, batch_size=1)
    assert run(s).accepted
    assert all(not p.requires_grad for p in s.model.parameters())


def test_all_gates_pass_and_actual_forward_counts_include_early_eos():
    s = make_session()
    trial = run(s)
    assert trial.accepted and s.active_edit_ids == [trial.edit_id]
    check = trial.answer_check
    assert check.violations == () and check.eos_token_ids == (5,)
    assert check.metrics.target_accuracy == check.metrics.locality_retention == 1
    assert check.metrics.baseline_locality_correct == 1
    assert check.metrics.max_target_answer_occurrences == 1
    assert [r["model_forwards"] for r in s.generations] == [6, 5]
    assert s.model.calls == 15  # four teacher forwards plus eleven generated tokens
    assert s.budget.used == 4


@pytest.mark.parametrize("failure", ["repetition", "locality", "empty_baseline"])
def test_answer_gates_restore_prior_stack_exactly(failure):
    s = make_session()
    prior = s.apply("readout", proposal((0, 6, 0.01)))
    before = s.weight("readout")
    edit = (
        proposal((0, 6, 2), (0, 0, 2))
        if failure == "repetition"
        else proposal((0, 6, 2), (6, 1, 2)) if failure == "locality" else proposal()
    )
    cases = [
        AnswerCase("a", ("b",)),
        AnswerCase("c", ("wrong" if failure == "empty_baseline" else "c",), "locality"),
    ]
    trial = run(s, edit, cases)
    expected = {
        "repetition": "answer_repetition",
        "locality": "locality_retention",
        "empty_baseline": "baseline_locality_correct",
    }[failure]
    assert trial.violations == (f"generation:{expected}",)
    assert s.active_edit_ids == [prior]
    assert torch.equal(s.weight("readout"), before)


def test_reference_checks_still_apply_when_generation_passes():
    s = make_session()
    s.add_behavior_reference(locality_prompts=["aa"], limits=BehaviorLimits(0, 10, 10))
    trial = run(s)
    assert not trial.accepted and trial.answer_check.violations == ()
    assert trial.violations[0].startswith("reference:")
    assert s.budget.used == 6


def test_generation_failure_rolls_back_and_is_audited():
    s = make_session()
    s.model.fail_generation = True
    before = s.weight("readout")
    with pytest.raises(RuntimeError, match="candidate generation"):
        run(s)
    assert torch.equal(s.weight("readout"), before) and not s.active_edit_ids
    assert s.trials[-1]["reverted"] and s.trials[-1]["answer_cases_sha256"]
    assert s.generations[-1]["status"] == "failed"
    assert s.budget.used == 4


@pytest.mark.parametrize("failure", ["exception", "nonstring", "oversized"])
def test_candidate_decoding_failure_restores_weights(failure):
    s = make_session()
    before = s.weight("readout")
    original_decode = s.decode

    def decode(ids):
        if s.model.readout.weight[0, 6] > 1:
            if failure == "exception":
                raise RuntimeError("decoder failed")
            return None if failure == "nonstring" else "b" * 101
        return original_decode(ids)

    s.decode = decode
    with pytest.raises((RuntimeError, ConfigurationError)):
        run(s, limits=AnswerLimits(max_new_tokens=3, max_decoded_chars=100))
    assert torch.equal(s.weight("readout"), before) and not s.active_edit_ids
    assert s.trials[-1]["reverted"]


def test_encoder_decoder_and_eos_policy_are_frozen_before_measurement():
    s = make_session()
    forward = s.model.forward

    def changing(*args, **kwargs):
        s.encode = lambda _: [4, 4]
        s.decode = lambda _: "wrong"
        s.model.config.eos_token_id = 4
        return forward(*args, **kwargs)

    s.model.forward = changing
    trial = run(s)
    assert trial.accepted and trial.answer_check.eos_token_ids == (5,)
    assert s.generations[0]["cases_sha256"] == s.generations[1]["cases_sha256"]
    assert s.generations[1]["finish_reasons"][0] == "eos"


def test_multiple_eos_ids_are_a_canonical_set_without_changing_stopping():
    s = make_session()
    s.model.config.eos_token_id = [5, 4, 5]
    trial = run(s)
    assert trial.accepted and trial.answer_check.eos_token_ids == (4, 5)
    assert s.generations[-1]["finish_reasons"] == ["eos", "length"]


def test_full_generation_and_reference_budget_reserved_before_any_forward():
    s = make_session(budget=3)
    with pytest.raises(BudgetExhausted, match="4 total"):
        run(s)
    assert s.model.calls == s.budget.used == 0
    s = make_session(budget=5)
    s.add_behavior_reference(locality_prompts=["aa"], limits=BehaviorLimits(10, 10, 10))
    with pytest.raises(BudgetExhausted, match="5 total"):
        run(s)
    assert s.budget.used == 1


@pytest.mark.parametrize(
    "bad",
    ["decoder", "duplicate", "no_target", "no_locality", "token_cap", "missing_limits"],
)
def test_invalid_generation_gate_fails_before_spending(bad):
    s = make_session()
    cases = [AnswerCase("a", ("b",)), AnswerCase("c", ("c",), "locality")]
    caps = AnswerLimits(max_new_tokens=3)
    if bad == "decoder":
        s.decode = None
    elif bad == "duplicate":
        cases.append(cases[0])
    elif bad == "no_target":
        cases = cases[1:]
    elif bad == "no_locality":
        cases = cases[:1]
    elif bad == "token_cap":
        caps = AnswerLimits(max_new_tokens=3, max_total_tokens=1)
    with pytest.raises(ConfigurationError):
        if bad == "missing_limits":
            s.trial_edit(
                "readout",
                proposal(),
                locality_prompts=["dd"],
                limits=BehaviorLimits(10, 10, 10),
                answer_cases=cases,
            )
        else:
            run(s, cases=cases, limits=caps)
    assert s.model.calls == s.budget.used == 0 and not s.active_edit_ids


def test_declared_partial_target_accuracy_is_obeyed():
    s = make_session()
    cases = [
        AnswerCase("a", ("b",)),
        AnswerCase("d", ("b",)),
        AnswerCase("c", ("c",), "locality"),
    ]
    trial = run(
        s, cases=cases, limits=AnswerLimits(min_target_accuracy=0.5, max_new_tokens=3)
    )
    assert trial.accepted and trial.answer_check.metrics.target_accuracy == 0.5


def test_empty_baseline_requires_an_explicit_opt_out():
    s = make_session()
    trial = run(
        s,
        cases=[AnswerCase("a", ("b",))],
        limits=AnswerLimits(max_new_tokens=3, min_baseline_locality_correct=0),
    )
    assert trial.accepted
    assert trial.answer_check.metrics.baseline_locality_correct == 0
    assert trial.answer_check.metrics.locality_retention == 1.0


def test_baseline_decoder_failure_never_applies_the_proposal():
    s = make_session()
    before = s.weight("readout")

    def decode(_):
        raise RuntimeError("baseline decoding failed")

    s.decode = decode
    with pytest.raises(RuntimeError, match="baseline decoding failed"):
        run(s)
    assert torch.equal(s.weight("readout"), before)
    assert not s.active_edit_ids and not s.trials
    assert s.budget.used == 2


def test_independent_answer_oracle_matches_gate_with_nested_aliases():
    from dataclasses import asdict

    from benchmarks.weight_space.verify_answer_gate_trial import answer_check

    s = make_session()
    s.decode = lambda ids: (
        "Charles Dickens, Dickens"
        if ids[0] == 0
        else "William Shakespeare" if ids[0] == 1 else "wrong"
    )
    cases = [
        AnswerCase("a", ("Dickens", "Charles Dickens")),
        AnswerCase("c", ("Shakespeare", "William Shakespeare"), "locality"),
    ]
    trial = run(s, cases=cases)
    before = [
        {"text": "wrong", "token_ids": [6, 6, 6]},
        {"text": "William Shakespeare", "token_ids": [1, 1, 1]},
    ]
    after = [{"text": "Charles Dickens, Dickens", "token_ids": [0, 5]}, before[1]]
    expected = answer_check(
        [asdict(c) for c in cases],
        [[6], [1]],
        before,
        after,
        asdict(AnswerLimits(max_new_tokens=3)),
        [5],
    )
    assert json.loads(json.dumps(asdict(trial.answer_check))) == expected
    assert trial.accepted and expected["metrics"]["max_target_answer_occurrences"] == 2


def test_answer_evidence_is_exported_without_raw_prompts_or_answers(tmp_path):
    pytest.importorskip("peft")
    s = make_session()
    s.encode = lambda text: {
        "private target": [6],
        "private locality": [1],
        "dd": [2, 2],
        "a": [6],
        "b": [0],
    }[text]
    s.decode = lambda ids: (
        "private answer"
        if ids[0] == 0
        else "private preserved" if ids[0] == 1 else "wrong"
    )
    trial = run(
        s,
        cases=[
            AnswerCase("private target", ("private answer",)),
            AnswerCase("private locality", ("private preserved",), "locality"),
        ],
    )
    assert trial.accepted
    out = s.export(tmp_path / "adapter")
    raw = (out / "weight_space_manifest.json").read_text()
    for secret in (
        "private target",
        "private locality",
        "private answer",
        "private preserved",
    ):
        assert secret not in raw
    record = json.loads(raw)["behavior_trials"][0]["answer_check"]
    assert record["metrics"]["target_correct"] == 1
    assert record["cases_sha256"] == trial.answer_check.cases_sha256


@pytest.mark.parametrize(
    "text,aliases,count",
    [
        ("Charles Dickens and Dickens", ("Dickens", "Charles Dickens"), 2),
        ("ＡＭＢＥＲ ambergris amber", ("amber",), 2),
        ("New\nYork, new york", ("New York",), 2),
        ("C++ C++", ("C++",), 2),
    ],
)
def test_repetition_alias_normalization_and_overlap(text, aliases, count):
    assert answer_occurrences(text, aliases) == count


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_target_accuracy": float("nan")},
        {"min_locality_retention": 2},
        {"min_baseline_locality_correct": -1},
        {"max_answer_occurrences": 0},
        {"max_new_tokens": True},
        {"use_cache": 1},
        {"max_decoded_chars": 0},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(ConfigurationError):
        AnswerLimits(**kwargs)


def test_case_owns_aliases_and_rejects_a_string_as_an_alias_sequence():
    aliases = ["b"]
    case = AnswerCase("a", aliases)
    aliases[0] = "bad"
    assert case.answers == ("b",)
    with pytest.raises(ConfigurationError):
        AnswerCase("a", "bad")
