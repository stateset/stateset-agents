"""08 — The eval-driven dev loop (baseline → change → measure).

Whitepaper §11.7 is built on a simple rhythm: pick a rubric, score a baseline,
make a change, score again, keep what improves things. This example shows
the *shape* of that loop without a GPU — we compare two candidate "agents"
(both stub-backed, but with different canned-response heuristics) against
the customer-support rubric. The same pattern applies when one of the agents
is a real fine-tune.

Install:
    pip install stateset-agents

Run:
    python 08_eval_driven_loop.py

Expected output:
    Eval set: 8 scenarios
    Candidate A (no acknowledgement) rubric: 0.XX
    Candidate B (acknowledges + next step) rubric: 0.XX
    Δ = +0.XX (B is better — keep that change)
"""

import asyncio
from typing import Any

from stateset_agents.core.reward_base import RewardFunction
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data import SupportRewardComposite, load_support_scenarios

# We simulate "two candidate agents" with two pure-Python policies so the demo
# runs in CPU and the contrast between baseline and improved is reproducible.
# In your real loop, replace these with `await agent.generate_response(...)`.


async def policy_a(scenario: dict[str, Any]) -> str:
    """A weak policy: ignores the query and offers a generic apology."""
    return "Sorry for the inconvenience. Is there anything else I can help with?"


async def policy_b(scenario: dict[str, Any]) -> str:
    """A stronger policy: acknowledges the intent + offers a concrete next step."""
    intent = scenario.get("intent", "")
    must_ack = scenario.get("must_acknowledge", [])
    ack_phrase = " and ".join(must_ack) if must_ack else "your request"
    return (
        f"Thanks for reaching out about {ack_phrase}. "
        f"I'll get this {intent} request processed and confirm the next step right away."
    )


async def evaluate(
    policy, rubric: RewardFunction, scenarios: list[dict[str, Any]]
) -> float:
    scores: list[float] = []
    for s in scenarios:
        response = await policy(s)
        turns = [ConversationTurn(role="assistant", content=response)]
        result = await rubric.compute_reward(turns, context=s)
        scores.append(result.score)
    return sum(scores) / max(len(scores), 1)


async def main() -> None:
    rubric = SupportRewardComposite()
    eval_set = [s.to_scenario() for s in load_support_scenarios()[:8]]
    print(f"Eval set: {len(eval_set)} scenarios")

    a = await evaluate(policy_a, rubric, eval_set)
    print(f"Candidate A (no acknowledgement)         rubric: {a:.3f}")

    b = await evaluate(policy_b, rubric, eval_set)
    print(f"Candidate B (acknowledges + next step)   rubric: {b:.3f}")

    delta = b - a
    verdict = "B is better — keep that change" if delta > 0 else "A is better — revert"
    print(f"\nΔ = {delta:+.3f} ({verdict})")
    print()
    print("This is the loop: pick a rubric, run baseline, change one thing, measure.")
    print("Replace the two policy_* functions with real `agent.generate_response`")
    print("calls and you have the same shape as the §11.7 whitepaper protocol.")


if __name__ == "__main__":
    asyncio.run(main())
