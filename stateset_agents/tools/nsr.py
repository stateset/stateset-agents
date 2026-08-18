"""
NSR rollout tools — expose the StateSet NSR verifier to tool-calling agents.

``create_nsr_tools`` returns two FunctionDefinitions a policy can call
mid-episode (register them on a ToolAgent / FunctionCallingMixin):

- ``nsr_decide``: one verified decision for a single action.
- ``nsr_verify_plan``: one decision per plan step; ``plan_verdict`` is
  "approved" only when every step approves, evaluation stops at the first
  blocking step, and an empty plan raises — a plan that was never parsed
  must never come back approved (mirrors the NSR MCP server's semantics).

Paired with ``rewards.nsr_verifier.NSRVerifierReward``, this trains policies
that consult the verifier before acting.
"""

from __future__ import annotations

from typing import Any

from stateset_agents.core.function_calling import FunctionDefinition
from stateset_agents.rewards.nsr_verifier import (
    NSRClient,
    NSRVerifierConfig,
    make_nsr_client,
)

_DECIDE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The decision question, e.g. 'Can order #A1 be refunded?'",
        },
        "action": {
            "type": "string",
            "description": "Explicit action being authorized, e.g. 'issue_refund'.",
        },
    },
    "required": ["query"],
}

_VERIFY_PLAN_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "description": "Plan steps in execution order.",
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    "required": ["steps"],
}


def _summarize(decision: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields a policy needs; drop the heavyweight proof payload."""
    return {
        key: decision.get(key)
        for key in (
            "decision_id",
            "decision",
            "confidence",
            "plain_explanation",
        )
        if decision.get(key) is not None
    }


def create_nsr_tools(
    config: NSRVerifierConfig | None = None,
    client: NSRClient | None = None,
) -> list[FunctionDefinition]:
    """Build the nsr_decide / nsr_verify_plan tool definitions.

    ``client`` (async ``payload -> decision response``) is injectable for
    tests or a local nsr-server sidecar; by default an HTTP client is built
    from ``config`` (or the NSR_* environment variables).
    """
    nsr = client or make_nsr_client(config or NSRVerifierConfig.from_env())

    async def nsr_decide(query: str, action: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if action:
            payload["action"] = action
        return _summarize(await nsr(payload))

    async def nsr_verify_plan(steps: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(steps, list) or not steps:
            raise ValueError(
                "'steps' must be a non-empty array of {query, action} — "
                "an empty plan cannot be verified and is never approved"
            )
        for i, step in enumerate(steps):
            if not isinstance(step, dict) or not str(step.get("query", "")).strip():
                raise ValueError(
                    f"plan step {i}: 'query' is required and must be non-empty"
                )

        results: list[dict[str, Any]] = []
        for i, step in enumerate(steps):
            payload: dict[str, Any] = {"query": step["query"]}
            if step.get("action"):
                payload["action"] = step["action"]
            summary = {"step": i, **_summarize(await nsr(payload))}
            results.append(summary)
            if summary.get("decision") != "approved":
                return {
                    "plan_verdict": summary.get("decision") or "refused",
                    "blocking_step": i,
                    "steps": results,
                }
        return {"plan_verdict": "approved", "steps": results}

    return [
        FunctionDefinition(
            name="nsr_decide",
            description=(
                "Make one verified decision via StateSet NSR. Returns the "
                "verdict (approved|denied|refused) with a proof-backed "
                "explanation. Consult it before taking an accountable action."
            ),
            parameters=_DECIDE_PARAMETERS,
            handler=nsr_decide,
        ),
        FunctionDefinition(
            name="nsr_verify_plan",
            description=(
                "Verify a multi-step plan via StateSet NSR before executing "
                "it. Execute only on plan_verdict='approved'; a single "
                "denied/refused step blocks the whole plan."
            ),
            parameters=_VERIFY_PLAN_PARAMETERS,
            handler=nsr_verify_plan,
        ),
    ]


__all__ = ["create_nsr_tools"]
