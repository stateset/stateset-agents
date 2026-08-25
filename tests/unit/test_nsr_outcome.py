"""
Tests for the NSR outcome reporter — closing the loop backwards.

Every decision made during RL rollouts has a known outcome by episode end;
posting it back (POST /v1/decisions/{id}/outcome or /outcome-by-ref) feeds
NSR's calibration curve and conjecture mining. Reporting is best-effort:
a dead endpoint must never crash a training run.
"""

import pytest

from stateset_agents.core.reward_base import RewardResult
from stateset_agents.rewards.nsr_verifier import NSROutcomeReporter


class FakePoster:
    def __init__(self, raise_error=None):
        self.calls: list[tuple[str, dict]] = []
        self.raise_error = raise_error

    async def __call__(self, path: str, payload: dict) -> dict:
        if self.raise_error is not None:
            raise self.raise_error
        self.calls.append((path, payload))
        return {"recorded": True}


class TestNSROutcomeReporter:
    async def test_record_by_decision_id(self):
        poster = FakePoster()
        reporter = NSROutcomeReporter(poster=poster)
        assert await reporter.record(decision_id="dec_9", outcome="honored") is True
        path, payload = poster.calls[0]
        assert path == "/v1/decisions/dec_9/outcome"
        assert payload == {"outcome": "honored"}

    async def test_record_by_external_ref(self):
        poster = FakePoster()
        reporter = NSROutcomeReporter(poster=poster)
        assert (
            await reporter.record(external_ref="episode-42", outcome="reversed") is True
        )
        path, payload = poster.calls[0]
        assert path == "/v1/decisions/outcome-by-ref"
        assert payload == {"external_ref": "episode-42", "outcome": "reversed"}

    async def test_decision_id_wins_when_both_given(self):
        poster = FakePoster()
        reporter = NSROutcomeReporter(poster=poster)
        await reporter.record(
            decision_id="dec_1", external_ref="ref-1", outcome="honored"
        )
        assert poster.calls[0][0] == "/v1/decisions/dec_1/outcome"

    async def test_invalid_outcome_raises_without_posting(self):
        poster = FakePoster()
        reporter = NSROutcomeReporter(poster=poster)
        with pytest.raises(ValueError, match="outcome"):
            await reporter.record(decision_id="dec_1", outcome="great_success")
        assert poster.calls == []

    async def test_missing_target_raises_without_posting(self):
        poster = FakePoster()
        reporter = NSROutcomeReporter(poster=poster)
        with pytest.raises(ValueError, match="decision_id or external_ref"):
            await reporter.record(outcome="honored")
        assert poster.calls == []

    async def test_transport_error_is_best_effort_false(self):
        poster = FakePoster(raise_error=RuntimeError("connection refused"))
        reporter = NSROutcomeReporter(poster=poster)
        assert await reporter.record(decision_id="dec_1", outcome="honored") is False

    async def test_record_from_reward_uses_metadata_decision_id(self):
        poster = FakePoster()
        reporter = NSROutcomeReporter(poster=poster)
        result = RewardResult(
            score=1.0, metadata={"mode": "verified", "decision_id": "dec_7"}
        )
        assert await reporter.record_from_reward(result, outcome="honored") is True
        assert poster.calls[0][0] == "/v1/decisions/dec_7/outcome"

    async def test_record_from_reward_without_decision_id_is_false(self):
        poster = FakePoster()
        reporter = NSROutcomeReporter(poster=poster)
        result = RewardResult(score=0.0, metadata={"mode": "nsr_error"})
        assert await reporter.record_from_reward(result, outcome="honored") is False
        assert poster.calls == []
