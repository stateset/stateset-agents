"""Deterministic refund sandbox for outcome-based RL; no external services."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.environments.json_action import parse_json_action
from stateset_agents.core.trajectory import ConversationTurn


class RefundEnvironment(Environment):
    """Execute order lookup/refund/deny actions and score the resulting ledger.

    Actions are JSON objects with ``tool`` and ``args``. Money uses integer
    cents. A refund must be eligible, exact, and issued once. A denial is correct
    only for ineligible orders. Looking up the order is required before acting.
    """

    def __init__(self) -> None:
        super().__init__(max_turns=4)

    @staticmethod
    def validate_scenario(row: dict[str, Any]) -> None:
        """Validate a refund case without starting an episode."""
        if not isinstance(row, dict):
            raise ValueError("Refund scenario must be an object")
        if not isinstance(row.get("order_id"), str) or not row["order_id"].strip():
            raise ValueError("Refund scenario requires order_id")
        if type(row.get("amount_cents")) is not int or row["amount_cents"] <= 0:
            raise ValueError("Refund scenario amount_cents must be a positive integer")
        if type(row.get("eligible")) is not bool:
            raise ValueError("Refund scenario eligible must be boolean")

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        """Create a private ledger and a task prompt from a benchmark row."""
        row = copy.deepcopy(scenario or {})
        self.validate_scenario(row)
        prompt = (
            f"Resolve the refund request for order {row['order_id']}. "
            "Return one JSON action per turn: "
            '{"tool":"lookup_order","args":{"order_id":"..."}}, '
            '{"tool":"refund","args":{"order_id":"...","amount_cents":123}}, '
            '{"tool":"deny","args":{"order_id":"..."}}, or '
            '{"tool":"finish","args":{}}. '
            "Look up the order first. Refund eligible orders for their exact amount "
            "once; deny ineligible orders. Finish explicitly when resolved. "
            "Use unique JSON keys and no separate tool calls."
        )
        return EnvironmentState(
            episode_id=row["order_id"],
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "scenario": row,
                "looked_up": False,
                "refunds": [],
                "denied": False,
                "violations": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Apply one action; malformed or prohibited actions record violations."""
        if state.is_done:
            raise ValueError("Episode already completed")
        state.turn_count += 1
        context = state.context
        row = context["scenario"]
        finish = False
        try:
            tool, args = parse_json_action(action)
            if tool == "finish":
                if args:
                    raise ValueError("finish takes no args")
                finish = True
                observation: Any = {"finished": True}
            else:
                expected = (
                    {"order_id", "amount_cents"} if tool == "refund" else {"order_id"}
                )
                if set(args) != expected or args.get("order_id") != row["order_id"]:
                    raise ValueError("Wrong order or action arguments")
                if tool == "lookup_order":
                    context["looked_up"] = True
                    observation = {
                        key: row[key]
                        for key in ("order_id", "amount_cents", "eligible")
                    }
                elif not context["looked_up"]:
                    raise ValueError("Lookup required before resolution")
                elif tool == "refund":
                    if not row["eligible"] or context["refunds"] or context["denied"]:
                        raise ValueError("Ineligible or duplicate refund")
                    if (
                        type(args["amount_cents"]) is not int
                        or args["amount_cents"] != row["amount_cents"]
                    ):
                        raise ValueError("Incorrect refund amount")
                    context["refunds"].append(args["amount_cents"])
                    observation = {"refunded_cents": args["amount_cents"]}
                elif tool == "deny":
                    if row["eligible"] or context["denied"] or context["refunds"]:
                        raise ValueError("Incorrect or duplicate denial")
                    context["denied"] = True
                    observation = {"denied": True}
                else:
                    raise ValueError("Unknown tool")
        except (ValueError, TypeError) as exc:
            context["violations"] += 1
            observation = {"error": str(exc)}
        done = finish or state.turn_count >= self.max_turns
        resolved = (
            context["refunds"] == [row["amount_cents"]]
            if row["eligible"]
            else context["denied"]
        )
        success = bool(
            finish and resolved and context["looked_up"] and context["violations"] == 0
        )
        if done:
            state.status = EpisodeStatus.COMPLETED if finish else EpisodeStatus.TIMEOUT
        reward = (1.0 if success else -1.0) if done else 0.0
        return (
            state,
            reward,
            done,
            {
                "messages": [
                    {"role": "user", "content": json.dumps(observation, sort_keys=True)}
                ],
                "metrics": {
                    "task_success": float(success),
                    "policy_violations": context["violations"],
                    "tool_calls": state.turn_count,
                },
            },
        )


def refund_benchmark(
    split: str, count: int = 128, seed: int = 0
) -> list[dict[str, Any]]:
    """Generate balanced, shuffled cases with opaque disjoint order identities.

    Order IDs hide the split name and sequential counter. This remains a
    lookup/action smoke benchmark, not evidence of policy reasoning.
    """
    if (
        split not in ("train", "validation", "test")
        or type(count) is not int
        or count < 1
    ):
        raise ValueError("Use train, validation, or test and a positive count")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(f"refund-v1:{split}:{seed}")
    rows = [
        {
            "order_id": "refund-"
            + hashlib.sha256(f"refund-v1:{split}:{seed}:{i}".encode()).hexdigest()[:24],
            "amount_cents": rng.randint(100, 100000),
            "eligible": i % 2 == 0,
        }
        for i in range(count)
    ]
    rng.shuffle(rows)
    return rows
