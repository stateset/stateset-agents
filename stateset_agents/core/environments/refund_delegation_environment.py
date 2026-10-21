"""Explicit intent delegation over the unchanged synthetic refund sandbox."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.environments.json_action import parse_json_action
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)
from stateset_agents.core.refund_tools import (
    RefundOrderFacts,
    RefundResolutionRequest,
    SyntheticRefundResolver,
)
from stateset_agents.core.trajectory import ConversationTurn

DELEGATION_PROTOCOL = "refund-delegation-v1"


class RefundDelegationEnvironment(Environment):
    """Measure delegation compliance and tool execution as a whole agent system.

    The model may request one resolution and then finish. Arithmetic and policy
    priority are computed by the tool, not learned by the model. This isolated
    fixture uses an in-memory transaction, never a real payment service.
    """

    def __init__(self) -> None:
        super().__init__(max_turns=4)
        self.sandbox = RefundPolicyEnvironment()
        self._inner: EnvironmentState | None = None
        self._resolver: SyntheticRefundResolver | None = None
        self._outer: EnvironmentState | None = None
        self._execution: list[dict[str, Any]] = []
        self._receipts: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def observation_context(self, state: EnvironmentState) -> dict[str, Any]:
        """Keep private facts and nested ledger state out of model context."""
        return {}

    def execution_events(self) -> list[dict[str, Any]]:
        """Return detached execution evidence for the last model intent."""
        return copy.deepcopy(self._execution)

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        """Bind a new authorized synthetic order without exposing answer labels."""
        self._inner, self._resolver, self._outer = None, None, None
        self._execution, self._receipts = [], {}
        row = copy.deepcopy(scenario or {})
        inner = await self.sandbox.reset(row)
        order_id = row["order_id"]
        self._inner = inner
        self._resolver = SyntheticRefundResolver(
            order_id=order_id,
            idempotency_key=f"{DELEGATION_PROTOCOL}:{order_id}",
            lookup=self._lookup,
            submit=self._submit,
        )
        prompt = (
            "Resolve the refund request for the order bound to this session. "
            "The trusted resolve_refund tool looks up the authorized order and "
            "computes its policy action and exact amount. Delegate those decisions "
            "to the tool. Reply with one JSON object per turn, exactly tool and args, "
            "with unique keys, no prose, code fences, or separate tool calls. "
            'First reply: {"tool":"resolve_refund","args":{"intent":"resolve_refund"}}. '
            'After the tool receipt, finish with {"tool":"finish","args":{}}. '
            "Request exactly one resolution, then finish. Never supply an order ID, "
            "amount, reason, or order facts; those come from the trusted backend."
        )
        self._outer = EnvironmentState(
            episode_id=inner.episode_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "messages": [{"role": "user", "content": prompt}],
                "intent_violations": 0,
                "resolution_requested": False,
                "duplicate_intents": 0,
            },
        )
        return self._outer

    async def _execute(self, action: dict[str, Any]) -> dict[str, Any]:
        assert self._inner is not None
        self._inner, reward, done, info = await self.sandbox.step(
            self._inner, ConversationTurn(role="assistant", content=json.dumps(action))
        )
        self._execution.append(
            copy.deepcopy(
                {
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "status": self._inner.status.value,
                    "observations": info["messages"],
                    "metrics": info["metrics"],
                }
            )
        )
        observation = json.loads(info["messages"][0]["content"])
        if not isinstance(observation, dict):
            raise RuntimeError("Sandbox observation must be an object")
        if "error" in observation:
            raise RuntimeError("Delegated sandbox action rejected")
        return observation

    async def _lookup(self, order_id: str) -> dict[str, Any]:
        return await self._execute(
            {"tool": "lookup_order", "args": {"order_id": order_id}}
        )

    async def _submit(self, request: RefundResolutionRequest) -> dict[str, Any]:
        async with self._lock:
            assert self._inner is not None
            if request.idempotency_key in self._receipts:
                return copy.deepcopy(self._receipts[request.idempotency_key])
            current = self._inner.context["scenario"]
            if (
                RefundOrderFacts.from_lookup(current) != request.facts
                or self._inner.context["resolution"] is not None
            ):
                raise RuntimeError("Stale facts or already resolved order")
            observation = await self._execute(request.action())
            receipt = {"applied": True, "observation": observation}
            self._receipts[request.idempotency_key] = receipt
            return copy.deepcopy(receipt)

    async def step(
        self,
        state: EnvironmentState,
        action: ConversationTurn,
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Execute explicit intent; score syntax failures without fabricating effects."""
        if state is not self._outer or state.is_done or self._resolver is None:
            raise ValueError("Reset required before stepping the current episode")
        self._execution = []
        state.turn_count += 1
        ctx, finish = state.context, False
        # Only model syntax/argument errors are scored as policy violations.
        # Lookup, submit, and infrastructure exceptions must propagate.
        try:
            tool, args = parse_json_action(action)
            if tool == "resolve_refund":
                if args != {"intent": "resolve_refund"}:
                    raise ValueError("Only explicit resolve_refund intent is accepted")
                if ctx["resolution_requested"]:
                    ctx["duplicate_intents"] += 1
                    raise ValueError("Only one resolution intent is permitted")
            elif tool == "finish":
                if args:
                    raise ValueError("finish takes no arguments")
                finish = True
            else:
                raise ValueError("Unknown delegation tool")
        except (ValueError, TypeError) as exc:
            ctx["intent_violations"] += 1
            observation: Any = {"error": str(exc)}
        else:
            if finish:
                observation = await self._execute({"tool": "finish", "args": {}})
            else:
                observation = await self._resolver.resolve_refund(**args)
                ctx["resolution_requested"] = True
        assert self._inner is not None
        done = finish or state.turn_count >= self.max_turns
        success = bool(
            finish
            and ctx["resolution_requested"]
            and ctx["intent_violations"] == 0
            and self._inner.context["violations"] == 0
        )
        if done:
            state.status = EpisodeStatus.COMPLETED if finish else EpisodeStatus.TIMEOUT
        return (
            state,
            ((1.0 if success else -1.0) if done else 0.0),
            done,
            {
                "messages": [
                    {"role": "user", "content": json.dumps(observation, sort_keys=True)}
                ],
                "metrics": {
                    "task_success": float(success),
                    "policy_violations": ctx["intent_violations"]
                    + self._inner.context["violations"],
                    "tool_calls": state.turn_count,
                    "executed_tool_calls": self._inner.turn_count,
                    "duplicate_intents": ctx["duplicate_intents"],
                    "refunded_cents": sum(self._inner.context["refunds"]),
                },
            },
        )
