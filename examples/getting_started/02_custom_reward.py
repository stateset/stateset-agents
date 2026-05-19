"""02 — Define a custom reward and score a conversation.

No training, no GPU. Demonstrates the RewardFunction contract documented in
whitepaper §4.1: `async compute_reward(turns, context) -> RewardResult`.

Install:
    pip install stateset-agents

Run:
    python 02_custom_reward.py

Expected output:
    rude + wrong     score = 0.00
    polite, wrong    score = 0.50
    polite + correct score = 1.00
"""

import asyncio
from typing import Any

from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn


class PoliteAndCorrectReward(RewardFunction):
    """Toy reward: 0.5 for politeness, 0.5 for the right number, summed."""

    name = "polite_and_correct"

    def __init__(self) -> None:
        super().__init__(weight=1.0, reward_type=RewardType.IMMEDIATE, name=self.name)

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        if not turns:
            return RewardResult(score=0.0, breakdown={"no_response": 1.0})
        text = (turns[-1].content or "").lower()

        polite_terms = ("please", "thank you", "happy to help", "of course")
        polite = 0.5 if any(p in text for p in polite_terms) else 0.0

        expected = (context or {}).get("expected_answer", "")
        correct = 0.5 if expected and expected.lower() in text else 0.0

        return RewardResult(
            score=polite + correct,
            breakdown={"polite": polite, "correct": correct},
        )


async def main() -> None:
    reward = PoliteAndCorrectReward()
    context = {"expected_answer": "42"}

    cases = [
        ("rude + wrong",      "Figure it out yourself."),
        ("polite, wrong",     "Thank you for your patience! I'll look into it."),
        ("polite + correct",  "Of course! The answer is 42, happy to help."),
    ]

    for label, response in cases:
        turns = [ConversationTurn(role="assistant", content=response)]
        result = await reward.compute_reward(turns, context=context)
        print(f"{label:20s} score = {result.score:.2f}  breakdown = {result.breakdown}")


if __name__ == "__main__":
    asyncio.run(main())
