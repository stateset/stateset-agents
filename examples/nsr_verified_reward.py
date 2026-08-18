"""Runnable demo of the StateSet NSR verifier integration.

Runs offline by default (an injected fake verifier), so it works with no
server and no keys:

    python examples/nsr_verified_reward.py

Set NSR_API_URL / STATESET_NSR_API_KEY / NSR_ORG_ID and pass --live to run
the same flow against a real NSR API (hosted or a local nsr-server):

    python examples/nsr_verified_reward.py --live

See docs/NSR_INTEGRATION.md for the full integration guide.
"""

import argparse
import asyncio

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.nsr_verifier import (
    NSROutcomeReporter,
    NSRVerifierReward,
    decision_request,
    fact,
    predicate,
    rule,
)

REQUEST = decision_request(
    "Can order A1 be refunded?",
    action="issue_refund",
    goal=predicate("permit_refund", "A1"),
    rules=[
        rule(
            "refund_ok",
            effect="permit",
            when=[
                predicate("return_received", "?o"),
                predicate("inspection_passed", "?o"),
            ],
            then=[predicate("permit_refund", "?o")],
        )
    ],
    facts=[
        fact("return_received", "A1", confidence=1.0),
        fact("inspection_passed", "A1", confidence=1.0),
    ],
    hydrate_org_context=False,
    external_ref="demo-episode-1",
)


async def fake_nsr(payload):
    """Offline stand-in: approves when both refund facts are present."""
    names = {f["predicate"]["name"] for f in payload.get("facts", [])}
    approved = {"return_received", "inspection_passed"} <= names
    return {
        "decision_id": "dec_demo",
        "decision": "approved" if approved else "refused",
        "confidence": 0.9,
        "plain_explanation": "demo verifier",
    }


async def main(live: bool) -> None:
    reward = NSRVerifierReward() if live else NSRVerifierReward(client=fake_nsr)

    async def score(response: str) -> None:
        result = await reward.compute_reward(
            [
                ConversationTurn(role="user", content="Refund order A1?"),
                ConversationTurn(role="assistant", content=response),
            ],
            context={"nsr_request": REQUEST},
        )
        print(f"  {response!r:50} -> score {result.score}  ({result.metadata['mode']})")

    print("Scoring policy responses against the verified decision:")
    await score("Approved: return received and inspection passed.")
    await score("Denied per policy.")
    await score("Interesting question!")

    if live:
        print("Recording the episode outcome (feeds NSR calibration):")
        reporter = NSROutcomeReporter()
        ok = await reporter.record(external_ref="demo-episode-1", outcome="honored")
        print(f"  outcome recorded: {ok}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use the real NSR API")
    asyncio.run(main(parser.parse_args().live))
