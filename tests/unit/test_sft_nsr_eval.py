"""
Tests for the NSR eval gate in sft.py — the independent-eval side of the
NSR integration. Eval specs carrying an ``nsr`` key gain a per-row
``nsr_verified`` extra, and an unverified row fails the eval gate.

Unlike the judge (which degrades when unavailable, by design), NSR
verification is fail-closed even in eval: ``nsr_verified: false`` for an
unreachable verifier, and the gate fails — an eval that silently skips its
strongest check would overstate the model.
"""

import stateset_agents.training.harvest as h
from stateset_agents.training.sft import build_eval_extras, eval_gate_failures

NSR_SPEC = {
    "prompt": "Refund order #A1?",
    "nsr": {"query": "Can order #A1 be refunded?"},
}


class TestBuildEvalExtras:
    def test_adds_nsr_verified_true_when_gate_passes(self, monkeypatch):
        monkeypatch.setattr(h, "nsr_gate_passes", lambda spec, sample: True)
        extras = build_eval_extras([NSR_SPEC], ["Approved."])
        assert extras[0]["nsr_verified"] is True

    def test_adds_nsr_verified_false_when_gate_rejects(self, monkeypatch):
        monkeypatch.setattr(h, "nsr_gate_passes", lambda spec, sample: False)
        extras = build_eval_extras([NSR_SPEC], ["Approved."])
        assert extras[0]["nsr_verified"] is False

    def test_specs_without_nsr_gain_no_flag(self, monkeypatch):
        def exploding_gate(spec, sample):
            raise AssertionError("NSR must not be consulted")

        monkeypatch.setattr(h, "nsr_gate_passes", exploding_gate)
        extras = build_eval_extras([{"prompt": "p", "expect": ["ok"]}], ["ok then"])
        assert "nsr_verified" not in extras[0]


class TestEvalGateFailures:
    def test_unverified_row_fails_the_gate(self):
        failures = eval_gate_failures([NSR_SPEC], [{"nsr_verified": False}])
        assert len(failures) == 1
        assert "nsr" in failures[0].lower()

    def test_verified_row_passes_the_gate(self):
        assert eval_gate_failures([NSR_SPEC], [{"nsr_verified": True}]) == []

    def test_rows_without_the_flag_are_unaffected(self):
        assert eval_gate_failures([{"prompt": "p"}], [{}]) == []
