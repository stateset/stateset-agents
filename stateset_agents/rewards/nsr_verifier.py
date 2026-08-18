"""
NSR verifier reward — verifiable reward backed by the StateSet NSR
decision API (POST /v1/decisions).

Unlike LLM-as-judge rewards, NSR is a deterministic symbolic verifier:
every decision it returns carries a machine-checkable proof, so the policy
cannot sweet-talk the reward. The reward asks NSR to decide the scenario's
authorization question and scores the model on agreement with the verified
verdict:

- model verdict matches NSR's ``approved``/``denied`` decision -> 1.0
- model refuses/escalates when NSR ``refused`` (unprovable)     -> 1.0
- any disagreement, or asserting a verdict NSR refused          -> 0.0
- no response / no extractable verdict                          -> 0.0 (NSR not called)
- NSR unreachable -> ``config.error_score`` (0.5 neutral by default;
  set 0.0 for fail-closed gating, e.g. harvest curation)

Scenario context supplies the decision request via ``context["nsr_request"]``
(the /v1/decisions body: query, action, facts, rules, ...); without it the
last user turn is used as the query.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Awaitable

from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)

NSRClient = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
NSRPoster = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

_APPROVE_RE = re.compile(r"\bapprov\w*\b", re.IGNORECASE)
_DENY_RE = re.compile(r"\bden(y|ies|ied|ial)\w*\b", re.IGNORECASE)
_REFUSE_RE = re.compile(r"\b(refus\w*|escalat\w*|can(?:no|')t verify)\b", re.IGNORECASE)


def extract_verdict(text: str) -> str | None:
    """Extract the model's decision verdict from its response text.

    Tries a JSON ``{"decision": ...}`` field first, then keyword scanning.
    Returns ``"approved"``, ``"denied"``, ``"refused"``, or ``None`` when no
    verdict (or an ambiguous mix of verdicts) is found.
    """
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("decision"), str):
            verdict = data["decision"].strip().lower()
            if verdict in ("approved", "denied", "refused"):
                return verdict
    except (json.JSONDecodeError, ValueError):
        pass

    found = set()
    if _APPROVE_RE.search(text):
        found.add("approved")
    if _DENY_RE.search(text):
        found.add("denied")
    if _REFUSE_RE.search(text):
        found.add("refused")
    if len(found) == 1:
        return found.pop()
    return None


@dataclass
class NSRVerifierConfig:
    """Connection + scoring config; ``from_env`` follows the judge-key
    conventions in ``rewards/llm_judge.py`` (NSR_API_URL, STATESET_NSR_API_KEY
    or NSR_API_KEY, NSR_ORG_ID)."""

    api_url: str = "https://api.nsr.stateset.com"
    api_key: str | None = None
    org_id: str | None = None
    timeout_seconds: float = 30.0
    error_score: float = 0.5
    mode: str = "safe"

    @classmethod
    def from_env(cls) -> NSRVerifierConfig:
        return cls(
            api_url=os.getenv("NSR_API_URL", cls.api_url),
            api_key=os.getenv("STATESET_NSR_API_KEY") or os.getenv("NSR_API_KEY"),
            org_id=os.getenv("NSR_ORG_ID"),
        )


class NSRVerifierReward(RewardFunction):
    """RewardFunction that scores agreement with a verified NSR decision.

    Example::

        reward = NSRVerifierReward()  # config from env
        trainer = MultiTurnGRPOTrainer(agent, env, reward_fn=reward, ...)

    A custom ``client`` (async ``payload -> decision response`` callable) can
    be injected for testing or a local nsr-server sidecar.
    """

    def __init__(
        self,
        config: NSRVerifierConfig | None = None,
        client: NSRClient | None = None,
        weight: float = 1.0,
        name: str = "NSRVerifierReward",
    ):
        super().__init__(weight=weight, reward_type=RewardType.IMMEDIATE, name=name)
        self.config = config or NSRVerifierConfig.from_env()
        self._client = client

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        response_text = _last_content(turns, role="assistant")
        if not response_text:
            return self._result(0.0, mode="no_response")

        model_verdict = extract_verdict(response_text)
        if model_verdict is None:
            return self._result(0.0, mode="unparseable_verdict")

        payload = self._build_payload(turns, context)
        try:
            decision = await self._get_client()(payload)
        except Exception as exc:  # network-dependent verifier: degrade per config
            logger.warning("NSR verifier call failed: %s", exc)
            return self._result(self.config.error_score, mode="nsr_error", error=str(exc))

        nsr_decision = str(decision.get("decision", "")).lower()
        score = 1.0 if model_verdict == nsr_decision else 0.0
        return self._result(
            score,
            mode="verified",
            nsr_decision=nsr_decision,
            model_verdict=model_verdict,
            decision_id=decision.get("decision_id"),
            nsr_confidence=decision.get("confidence"),
        )

    def _build_payload(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if context and isinstance(context.get("nsr_request"), dict):
            payload.update(context["nsr_request"])
        if "query" not in payload:
            payload["query"] = _last_content(turns, role="user") or ""
        if context and context.get("external_ref") and "external_ref" not in payload:
            payload["external_ref"] = context["external_ref"]
        payload.setdefault("mode", self.config.mode)
        return payload

    def _result(self, score: float, mode: str, **metadata: Any) -> RewardResult:
        return RewardResult(
            score=score,
            components={"nsr_verifier": score},
            metadata={"mode": mode, **{k: v for k, v in metadata.items() if v is not None}},
        )

    def _get_client(self) -> NSRClient:
        if self._client is None:
            self._client = self._make_http_client()
        return self._client

    def _make_http_client(self) -> NSRClient:
        return make_nsr_client(self.config)


def make_nsr_poster(cfg: NSRVerifierConfig) -> NSRPoster:
    """Generic NSR transport: POST {api_url}{path} via the shared pool."""

    async def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        from stateset_agents.core.async_pool import get_http_pool

        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        if cfg.org_id:
            headers["X-Org-ID"] = cfg.org_id

        pool = await get_http_pool()
        async with pool.acquire() as session:
            async with session.post(
                f"{cfg.api_url.rstrip('/')}{path}",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=cfg.timeout_seconds),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    return post


def make_nsr_client(cfg: NSRVerifierConfig) -> NSRClient:
    """Default decision transport: POST {api_url}/v1/decisions."""
    poster = make_nsr_poster(cfg)

    async def call(payload: dict[str, Any]) -> dict[str, Any]:
        return await poster("/v1/decisions", payload)

    return call


VALID_OUTCOMES = frozenset({"honored", "reversed", "overridden", "escalated"})


class NSROutcomeReporter:
    """Post real-world episode outcomes back to NSR — the write side of the
    calibration loop. Every decision made during RL rollouts has a known
    outcome by episode end; recording it feeds NSR's calibration curve and
    conjecture mining.

    ``record`` validates inputs eagerly (bad outcome values are caller bugs)
    but treats transport failures as best-effort ``False`` — a dead endpoint
    must never crash a training run.
    """

    def __init__(
        self,
        config: NSRVerifierConfig | None = None,
        poster: NSRPoster | None = None,
    ):
        self.config = config or NSRVerifierConfig.from_env()
        self._poster = poster

    async def record(
        self,
        decision_id: str | None = None,
        external_ref: str | None = None,
        outcome: str = "honored",
    ) -> bool:
        """Record ``outcome`` (honored|reversed|overridden|escalated) against
        a decision, by ``decision_id`` when known, else by ``external_ref``."""
        outcome = outcome.strip().lower()
        if outcome not in VALID_OUTCOMES:
            raise ValueError(
                f"outcome must be one of {sorted(VALID_OUTCOMES)}, got '{outcome}'"
            )
        if decision_id:
            path = f"/v1/decisions/{decision_id}/outcome"
            payload: dict[str, Any] = {"outcome": outcome}
        elif external_ref:
            path = "/v1/decisions/outcome-by-ref"
            payload = {"external_ref": external_ref, "outcome": outcome}
        else:
            raise ValueError("record needs a decision_id or external_ref")

        try:
            await self._get_poster()(path, payload)
            return True
        except Exception as exc:
            logger.warning("NSR outcome report failed (best-effort): %s", exc)
            return False

    async def record_from_reward(self, result: RewardResult, outcome: str) -> bool:
        """Record an outcome for the decision an ``NSRVerifierReward`` result
        was verified against; ``False`` when the result carries no decision
        (unverified modes never reached NSR)."""
        decision_id = result.metadata.get("decision_id")
        if not decision_id:
            return False
        return await self.record(decision_id=str(decision_id), outcome=outcome)

    def _get_poster(self) -> NSRPoster:
        if self._poster is None:
            self._poster = make_nsr_poster(self.config)
        return self._poster


def _last_content(turns: list[ConversationTurn], role: str) -> str:
    for turn in reversed(turns):
        if turn.role == role and isinstance(turn.content, str) and turn.content:
            return turn.content
    return ""


__all__ = [
    "NSROutcomeReporter",
    "NSRVerifierConfig",
    "NSRVerifierReward",
    "VALID_OUTCOMES",
    "extract_verdict",
    "make_nsr_client",
    "make_nsr_poster",
]
