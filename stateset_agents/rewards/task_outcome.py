"""Deterministic outcome checks for reference-labelled agent workflows."""

from __future__ import annotations

import json
from typing import Any

from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn


class TaskOutcomeReward(RewardFunction):
    """Score a trusted reference and executed tool trace, never helpful phrasing.

    References belong to the evaluator's scenario; callers must keep them out of
    observations. Missing references are errors, not neutral training rewards.
    """

    fail_on_error = True
    requires_environment_context = True

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE)

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Require final reference agreement and optional verified tool outcomes."""
        context = context or {}
        scenario = context.get("scenario", context)
        expected = scenario.get("expected_response")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(
                "Outcome rewards require a trusted expected_response reference"
            )
        assistant = [turn for turn in turns if turn.role == "assistant"]
        answer_matches = bool(
            assistant and (assistant[-1].content or "").strip() == expected.strip()
        )
        violations = context.get("policy_violations", 0)
        if type(violations) is not int or violations < 0:
            raise ValueError("policy_violations must be a nonnegative integer")
        expected_tools = scenario.get("expected_tool_calls")
        tools_match = True
        if expected_tools is not None:
            # Only environment-executed records count; generated tool text does not.
            executed = context.get("executed_tool_calls")
            if not isinstance(expected_tools, list) or not isinstance(executed, list):
                raise ValueError("Tool outcome rewards require an executed tool trace")
            # JSON identity preserves argument types: True must not equal 1,
            # and a float amount must not equal an integer-cent amount.
            tools_match = json.dumps(
                executed, sort_keys=True, allow_nan=False
            ) == json.dumps(expected_tools, sort_keys=True, allow_nan=False)
        success = answer_matches and tools_match and violations == 0
        return RewardResult(
            score=float(success),
            breakdown={
                "reference_agreement": float(answer_matches),
                "tool_outcome": float(tools_match),
                "task_success": float(success),
            },
            metadata={"reward_semantics": "verified_task_outcome_v1"},
        )
