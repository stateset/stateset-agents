"""
Tests for the NSR request builders — helpers that produce correctly-shaped
/v1/decisions bodies. The shapes were established by driving a live
nsr-server: facts wrap their predicate ({"predicate": {...}, "confidence"}),
rules use "if"/"then" condition lists, and variables are "?x" strings.
"""

import pytest

from stateset_agents.rewards.nsr_verifier import (
    decision_request,
    fact,
    predicate,
    rule,
)


class TestPredicate:
    def test_basic_shape(self):
        assert predicate("return_received", "A1") == {
            "name": "return_received",
            "args": ["A1"],
        }

    def test_negated(self):
        p = predicate("fraud_flagged", "?o", negated=True)
        assert p["negated"] is True


class TestFact:
    def test_wraps_predicate(self):
        f = fact("return_received", "A1")
        assert f == {"predicate": {"name": "return_received", "args": ["A1"]}}

    def test_confidence_and_source(self):
        f = fact("inspection_passed", "A1", confidence=0.9, source="wms")
        assert f["confidence"] == 0.9
        assert f["source"] == "wms"

    def test_rejects_variable_args(self):
        with pytest.raises(ValueError, match="grounded"):
            fact("return_received", "?o")


class TestRule:
    def test_if_then_shape(self):
        r = rule(
            "refund_ok",
            effect="permit",
            when=[predicate("return_received", "?o")],
            then=[predicate("permit_refund", "?o")],
        )
        assert r["name"] == "refund_ok"
        assert r["effect"] == "permit"
        assert r["if"] == [{"name": "return_received", "args": ["?o"]}]
        assert r["then"] == [{"name": "permit_refund", "args": ["?o"]}]

    def test_effect_is_validated(self):
        with pytest.raises(ValueError, match="effect"):
            rule("r", effect="allow", when=[predicate("a", "?x")], then=[predicate("b", "?x")])


class TestDecisionRequest:
    def test_full_request_shape(self):
        req = decision_request(
            "Can order A1 be refunded?",
            action="issue_refund",
            goal=predicate("permit_refund", "A1"),
            rules=[
                rule(
                    "refund_ok",
                    effect="permit",
                    when=[predicate("return_received", "?o")],
                    then=[predicate("permit_refund", "?o")],
                )
            ],
            facts=[fact("return_received", "A1")],
            external_ref="rl-episode-1",
            hydrate_org_context=False,
        )
        assert req["query"] == "Can order A1 be refunded?"
        assert req["action"] == "issue_refund"
        assert req["authorization_goal"] == {"name": "permit_refund", "args": ["A1"]}
        assert req["facts"][0]["predicate"]["name"] == "return_received"
        assert req["external_ref"] == "rl-episode-1"
        assert req["hydrate_org_context"] is False

    def test_omits_unset_fields(self):
        req = decision_request("q?")
        assert req == {"query": "q?"}

    def test_goal_must_be_grounded(self):
        with pytest.raises(ValueError, match="grounded"):
            decision_request("q?", goal=predicate("permit_refund", "?o"))
