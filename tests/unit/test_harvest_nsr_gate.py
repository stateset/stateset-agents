"""
Tests for the NSR gate in the harvest path.

A prompt spec may carry an ``nsr`` key — a /v1/decisions request body.
When present, a sample only survives the harvest if the verdict it asserts
agrees with the verified NSR decision. Like the judge gate, an unavailable
verifier REJECTS the sample (fail-closed): a broken gate must not silently
become a pass, or a dead NSR endpoint harvests unverified rows.
"""

import pytest

import stateset_agents.training.harvest as h
from stateset_agents.training.sft import normalize_eval_prompts


class FakeNSRReward:
    """Mimics NSRVerifierReward.compute_reward with a canned score."""

    def __init__(self, score, mode="verified", raise_error=None):
        self.score = score
        self.mode = mode
        self.raise_error = raise_error
        self.calls = []

    async def compute_reward(self, turns, context=None):
        if self.raise_error is not None:
            raise self.raise_error
        self.calls.append((turns, context))

        class R:
            pass

        r = R()
        r.score = self.score
        r.metadata = {"mode": self.mode}
        return r


NSR_SPEC = {
    "prompt": "Refund order #A1?",
    "nsr": {"query": "Can order #A1 be refunded?", "action": "issue_refund"},
}


class TestNSRGate:
    def _gate(self, monkeypatch, reward):
        monkeypatch.setattr(h, "_create_nsr_reward", lambda: reward)

    def test_agreeing_sample_passes(self, monkeypatch):
        self._gate(monkeypatch, FakeNSRReward(score=1.0))
        assert h.sample_passes(NSR_SPEC, "Approved: return passed inspection.")

    def test_disagreeing_sample_rejects(self, monkeypatch):
        self._gate(monkeypatch, FakeNSRReward(score=0.0))
        assert not h.sample_passes(NSR_SPEC, "Approved: return passed inspection.")

    def test_verifier_error_rejects_fail_closed(self, monkeypatch):
        self._gate(
            monkeypatch, FakeNSRReward(score=1.0, raise_error=RuntimeError("down"))
        )
        assert not h.sample_passes(NSR_SPEC, "Approved.")

    def test_nsr_error_mode_rejects_even_with_neutral_score(self, monkeypatch):
        # NSRVerifierReward returns error_score with mode="nsr_error" when the
        # API is unreachable; the harvest gate must treat that as a reject, not
        # trust the neutral score.
        self._gate(monkeypatch, FakeNSRReward(score=1.0, mode="nsr_error"))
        assert not h.sample_passes(NSR_SPEC, "Approved.")

    def test_gate_receives_the_spec_request_as_context(self, monkeypatch):
        reward = FakeNSRReward(score=1.0)
        self._gate(monkeypatch, reward)
        h.sample_passes(NSR_SPEC, "Approved.")
        ((turns, context),) = reward.calls
        assert context["nsr_request"] == NSR_SPEC["nsr"]
        assert turns[0].role == "user"
        assert turns[0].content == "Refund order #A1?"
        assert turns[1].role == "assistant"
        assert turns[1].content == "Approved."

    def test_substring_failure_short_circuits_before_nsr(self, monkeypatch):
        def exploding_factory():
            raise AssertionError("NSR must not be consulted")

        monkeypatch.setattr(h, "_create_nsr_reward", exploding_factory)
        spec = {**NSR_SPEC, "expect": ["refund id"]}
        assert not h.sample_passes(spec, "Approved.")

    def test_spec_without_nsr_never_builds_the_gate(self, monkeypatch):
        def exploding_factory():
            raise AssertionError("NSR must not be consulted")

        monkeypatch.setattr(h, "_create_nsr_reward", exploding_factory)
        assert h.sample_passes({"prompt": "p", "expect": ["ok"]}, "ok then")


class TestSpecValidation:
    def test_nsr_key_is_accepted(self):
        specs = normalize_eval_prompts([NSR_SPEC])
        assert specs[0]["nsr"] == NSR_SPEC["nsr"]

    def test_nsr_must_be_an_object(self):
        with pytest.raises(ValueError, match="nsr"):
            normalize_eval_prompts([{"prompt": "p", "nsr": "not-a-dict"}])
