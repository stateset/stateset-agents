"""
Tests for the NSR verifier reward — RLVR-style reward backed by the
StateSet NSR decision API (POST /v1/decisions).

The reward asks NSR to decide the scenario's authorization question and
scores the policy model on whether its verdict agrees with the verified
decision. Transport is injectable so tests never hit the network.
"""

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.nsr_verifier import (
    NSRVerifierConfig,
    NSRVerifierReward,
    extract_verdict,
)


def turns(user: str, assistant: str | None) -> list[ConversationTurn]:
    out = [ConversationTurn(role="user", content=user)]
    if assistant is not None:
        out.append(ConversationTurn(role="assistant", content=assistant))
    return out


class FakeNSRClient:
    """Records the request payload and returns a canned decision response."""

    def __init__(self, decision: str = "approved", confidence: float = 0.95):
        self.decision = decision
        self.confidence = confidence
        self.calls: list[dict] = []
        self.raise_error: Exception | None = None

    async def __call__(self, payload: dict) -> dict:
        if self.raise_error is not None:
            raise self.raise_error
        self.calls.append(payload)
        return {
            "decision_id": "dec_123",
            "decision": self.decision,
            "confidence": self.confidence,
            "rationale": "test",
            "plain_explanation": "test",
        }


CONTEXT = {
    "nsr_request": {
        "query": "Can order #A1 be refunded?",
        "action": "issue_refund",
    }
}


# ===========================
# Verdict extraction
# ===========================


class TestExtractVerdict:
    def test_json_decision_field(self):
        assert extract_verdict('{"decision": "denied", "reason": "x"}') == "denied"

    def test_plain_approve_keyword(self):
        assert extract_verdict("I approve this refund request.") == "approved"

    def test_plain_deny_keyword(self):
        assert extract_verdict("The refund is denied per policy.") == "denied"

    def test_refusal_keywords(self):
        assert extract_verdict("I must refuse; escalating to a human.") == "refused"

    def test_ambiguous_both_verdicts_is_none(self):
        assert extract_verdict("I could approve it or deny it.") is None

    def test_no_verdict_is_none(self):
        assert extract_verdict("The weather is nice today.") is None


# ===========================
# Reward semantics
# ===========================


class TestNSRVerifierReward:
    async def test_agreement_with_approved_scores_one(self):
        client = FakeNSRClient(decision="approved")
        reward = NSRVerifierReward(client=client)
        result = await reward.compute_reward(
            turns("Refund order #A1?", "Approved: the return passed inspection."),
            CONTEXT,
        )
        assert result.score == 1.0
        assert result.metadata["nsr_decision"] == "approved"
        assert result.metadata["model_verdict"] == "approved"
        assert result.metadata["mode"] == "verified"
        assert client.calls[0]["query"] == "Can order #A1 be refunded?"
        assert client.calls[0]["action"] == "issue_refund"

    async def test_disagreement_scores_zero(self):
        client = FakeNSRClient(decision="denied")
        reward = NSRVerifierReward(client=client)
        result = await reward.compute_reward(
            turns("Refund order #A1?", "Approved, refund issued."), CONTEXT
        )
        assert result.score == 0.0
        assert result.metadata["mode"] == "verified"

    async def test_model_refusal_matches_nsr_refused(self):
        client = FakeNSRClient(decision="refused")
        reward = NSRVerifierReward(client=client)
        result = await reward.compute_reward(
            turns("Refund order #A1?", "I can't verify this; escalating."), CONTEXT
        )
        assert result.score == 1.0

    async def test_asserting_verdict_when_nsr_refused_scores_zero(self):
        client = FakeNSRClient(decision="refused")
        reward = NSRVerifierReward(client=client)
        result = await reward.compute_reward(
            turns("Refund order #A1?", "Approved, go ahead."), CONTEXT
        )
        assert result.score == 0.0

    async def test_no_assistant_response_scores_zero_without_calling_nsr(self):
        client = FakeNSRClient()
        reward = NSRVerifierReward(client=client)
        result = await reward.compute_reward(turns("Refund order #A1?", None), CONTEXT)
        assert result.score == 0.0
        assert client.calls == []
        assert result.metadata["mode"] == "no_response"

    async def test_unparseable_verdict_scores_zero_without_calling_nsr(self):
        client = FakeNSRClient()
        reward = NSRVerifierReward(client=client)
        result = await reward.compute_reward(
            turns("Refund order #A1?", "Interesting question!"), CONTEXT
        )
        assert result.score == 0.0
        assert client.calls == []
        assert result.metadata["mode"] == "unparseable_verdict"

    async def test_missing_nsr_request_falls_back_to_user_query(self):
        client = FakeNSRClient(decision="denied")
        reward = NSRVerifierReward(client=client)
        result = await reward.compute_reward(
            turns("Refund order #A1?", "Denied per policy."), context=None
        )
        assert result.score == 1.0
        assert client.calls[0]["query"] == "Refund order #A1?"

    async def test_transport_error_returns_neutral_error_score(self):
        client = FakeNSRClient()
        client.raise_error = RuntimeError("connection refused")
        reward = NSRVerifierReward(client=client)
        result = await reward.compute_reward(
            turns("Refund order #A1?", "Approved."), CONTEXT
        )
        assert result.score == 0.5
        assert result.metadata["mode"] == "nsr_error"

    async def test_error_score_is_configurable_fail_closed(self):
        client = FakeNSRClient()
        client.raise_error = RuntimeError("connection refused")
        reward = NSRVerifierReward(
            config=NSRVerifierConfig(error_score=0.0), client=client
        )
        result = await reward.compute_reward(
            turns("Refund order #A1?", "Approved."), CONTEXT
        )
        assert result.score == 0.0

    async def test_external_ref_from_context_is_forwarded(self):
        client = FakeNSRClient()
        reward = NSRVerifierReward(client=client)
        ctx = {**CONTEXT, "external_ref": "episode-42"}
        await reward.compute_reward(turns("Refund order #A1?", "Approved."), ctx)
        assert client.calls[0]["external_ref"] == "episode-42"


# ===========================
# Default HTTP transport
# ===========================


class TestDefaultHttpTransport:
    async def test_posts_to_v1_decisions_with_auth_headers(self, monkeypatch):
        from contextlib import asynccontextmanager

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            async def json(self):
                return {"decision": "approved", "decision_id": "dec_1"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class FakeSession:
            def post(self, url, json=None, headers=None, timeout=None):
                captured.update(url=url, json=json, headers=headers)
                return FakeResponse()

        class FakePool:
            @asynccontextmanager
            async def acquire(self):
                yield FakeSession()

        async def fake_get_http_pool():
            return FakePool()

        import stateset_agents.core.async_pool as async_pool

        monkeypatch.setattr(async_pool, "get_http_pool", fake_get_http_pool)

        cfg = NSRVerifierConfig(
            api_url="http://localhost:8080/", api_key="sk-1", org_id="org-9"
        )
        reward = NSRVerifierReward(config=cfg)
        result = await reward.compute_reward(
            turns("Refund order #A1?", "Approved."), CONTEXT
        )
        assert result.score == 1.0
        assert captured["url"] == "http://localhost:8080/v1/decisions"
        assert captured["headers"]["Authorization"] == "Bearer sk-1"
        assert captured["headers"]["X-Org-ID"] == "org-9"
        assert captured["json"]["query"] == "Can order #A1 be refunded?"


# ===========================
# Config from environment
# ===========================


class TestNSRVerifierConfig:
    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("NSR_API_URL", "http://localhost:8080")
        monkeypatch.setenv("STATESET_NSR_API_KEY", "sk-test")
        monkeypatch.setenv("NSR_ORG_ID", "org-7")
        cfg = NSRVerifierConfig.from_env()
        assert cfg.api_url == "http://localhost:8080"
        assert cfg.api_key == "sk-test"
        assert cfg.org_id == "org-7"

    def test_defaults_without_env(self, monkeypatch):
        for var in ("NSR_API_URL", "STATESET_NSR_API_KEY", "NSR_API_KEY", "NSR_ORG_ID"):
            monkeypatch.delenv(var, raising=False)
        cfg = NSRVerifierConfig.from_env()
        assert cfg.api_url == "https://api.nsr.stateset.com"
        assert cfg.api_key is None
        assert cfg.error_score == 0.5
