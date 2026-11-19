"""Run scoped reasoning assessment offline with a scripted proposal backend.

Run: python examples/enhanced_reasoning.py

The stub deliberately proposes an incorrect window first. The policy checker
evaluates actual supplied policy data; model confidence never authorizes stopping.
Replace the separate planner with a configured model to use the same contract.
"""

from __future__ import annotations

import asyncio
import json
import logging

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.enhanced import (
    ModelReasoningProvider,
    ReasoningAssessment,
    ReasoningEngine,
    ReasoningRequest,
    ReasoningStep,
)

logger = logging.getLogger(__name__)


async def check_policy(
    request: ReasoningRequest, step: ReasoningStep
) -> ReasoningAssessment:
    """Check the proposed window against supplied policy, abstaining if absent."""
    policy = request.context.get("refund_policy")
    if not isinstance(policy, dict) or type(policy.get("window_days")) is not int:
        return ReasoningAssessment(
            source="supplied_refund_policy/v1", reason="No refund window supplied"
        )
    days = policy["window_days"]
    matches = (
        step.thought == f"Refunds are accepted within {days} days."
        and step.action == f"refund_window_days={days}"
    )
    return ReasoningAssessment(
        status="passed" if matches else "failed",
        score=float(matches),
        source="supplied_refund_policy/v1",
        evidence=(f"request.context.refund_policy.window_days={days}",),
        reason="Exact agreement with the supplied window; policy authenticity is not assessed",
    )


async def main() -> None:
    """Demonstrate rejection, revision, and a scoped stopping decision."""
    planner = MultiTurnAgent(
        AgentConfig(
            model_name="stub://policy-proposals",
            stub_responses=[
                json.dumps(
                    {
                        "thought": f"Refunds are accepted within {days} days.",
                        "action": f"refund_window_days={days}",
                        "confidence": confidence,
                        "evidence": [],
                    }
                )
                for days, confidence in ((90, 0.99), (30, 0.5))
            ],
        )
    )
    await planner.initialize()
    engine = ReasoningEngine(
        step_generator=ModelReasoningProvider(planner.generate_response),
        assessor=check_policy,
    )
    steps = await engine.reason(
        "Explain the refund window", {"refund_policy": {"window_days": 30}}
    )
    for step in steps:
        logger.info(
            "Proposal: %s | declared confidence: %.2f | policy check: %s",
            step.thought,
            step.confidence,
            step.assessment().status,
        )
    logger.info("Stopped because: %s", engine.get_status()["last_run"]["stop_reason"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
