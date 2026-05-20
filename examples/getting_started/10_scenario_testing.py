"""10 — Scenario-based assertions for an agent (regression-style testing).

The least-glamorous, highest-ROI use of the framework: pin a small handful
of behavioural assertions to specific scenarios, run them on every code or
checkpoint change, and break the build if any one regresses. This is the
shape of the assertion suite that ships with every starter template.

Three kinds of checks shown here:

  1. **Must-acknowledge** — the response mentions a required word (intent
     recognition).
  2. **Must-avoid** — the response avoids a forbidden phrase (safety / brand).
  3. **Reward ≥ floor** — the rubric score clears a tunable threshold.

Install:
    pip install stateset-agents

Run:
    python 10_scenario_testing.py

Exit code: 0 if all assertions pass, 1 otherwise (suitable for CI).

Expected output (with the bundled scenarios + a strong policy):
    ✓ [refund]    acknowledges 'refund'   in response
    ✓ [refund]    avoids       'impossible' in response
    ✓ [refund]    rubric ≥ 0.50           (got 0.XX)
    ...
    All 9 assertions passed.
"""

import asyncio
import sys
from typing import Any

from stateset_agents.core.reward_base import RewardFunction
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data import SupportRewardComposite, load_support_scenarios


# Replace this with your real agent: `await agent.generate_response(scenario["user_query"])`.
async def candidate_policy(scenario: dict[str, Any]) -> str:
    ack = " and ".join(scenario.get("must_acknowledge", [])) or "your request"
    intent = scenario.get("intent", "request")
    return (
        f"Thank you for reaching out about {ack}. I'll get this {intent} "
        f"processed and send confirmation to your email shortly."
    )


class Assertion:
    """A single scenario assertion. ``check()`` returns (passed, detail)."""

    def __init__(self, label: str, fn):
        self.label = label
        self.fn = fn

    async def check(self, scenario: dict[str, Any], response: str, rubric: RewardFunction):
        return await self.fn(scenario, response, rubric)


async def assert_acknowledges(scenario, response, _rubric):
    for word in scenario.get("must_acknowledge", []):
        if word.lower() not in response.lower():
            return False, f"missing '{word}'"
    return True, f"acknowledges {scenario.get('must_acknowledge', [])}"


async def assert_avoids(scenario, response, _rubric):
    for word in scenario.get("must_avoid", []):
        if word.lower() in response.lower():
            return False, f"contains forbidden '{word}'"
    return True, f"avoids {scenario.get('must_avoid', [])}"


async def assert_rubric_floor(floor: float):
    async def _check(scenario, response, rubric):
        turns = [ConversationTurn(role="assistant", content=response)]
        result = await rubric.compute_reward(turns, context=scenario)
        if result.score < floor:
            return False, f"rubric {result.score:.2f} < {floor}"
        return True, f"rubric {result.score:.2f} ≥ {floor}"
    return _check


async def main() -> int:
    rubric = SupportRewardComposite()
    eval_set = [s.to_scenario() for s in load_support_scenarios()[:3]]

    floor_check = await assert_rubric_floor(0.50)
    assertions = [
        Assertion("acknowledges required terms", assert_acknowledges),
        Assertion("avoids forbidden terms",      assert_avoids),
        Assertion("rubric ≥ 0.50",               floor_check),
    ]

    failures: list[str] = []
    total = 0
    for scenario in eval_set:
        response = await candidate_policy(scenario)
        for a in assertions:
            total += 1
            ok, detail = await a.check(scenario, response, rubric)
            marker = "✓" if ok else "✗"
            print(f"  {marker} [{scenario['intent']:9s}] {a.label:30s}  {detail}")
            if not ok:
                failures.append(f"[{scenario['intent']}] {a.label}: {detail}")

    print()
    if failures:
        print(f"FAILED {len(failures)}/{total}:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {total} assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
