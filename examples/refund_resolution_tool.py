"""Run explicit refund delegation against the unchanged synthetic sandbox.

Run: python -m examples.refund_resolution_tool
No model, River session, payment service, or credentials are used. The scripted
intent tests tool integration; its success rate is not a model quality score.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.core.function_calling import FunctionCallingMixin, ToolCall
from stateset_agents.core.refund_tools import (
    RefundOrderFacts,
    RefundResolutionRequest,
    SyntheticRefundResolver,
)
from stateset_agents.core.trajectory import ConversationTurn


class SandboxToolHost(FunctionCallingMixin):
    """Exercise the same tool dispatcher an agent uses, with a scripted intent."""

    def __init__(self, resolver: SyntheticRefundResolver) -> None:
        self.__init_tools__([resolver.as_function()])


async def run_case(row: dict[str, Any]) -> dict[str, Any]:
    """Record intent separately from executed sandbox actions and observations."""
    env = RefundPolicyEnvironment()
    state = await env.reset(row)
    events: list[dict[str, Any]] = []
    backend_lock = asyncio.Lock()
    receipts: dict[str, dict[str, Any]] = {}

    async def execute(action: dict[str, Any]) -> dict[str, Any]:
        nonlocal state
        state, reward, done, info = await env.step(
            state, ConversationTurn(role="assistant", content=json.dumps(action))
        )
        observation = json.loads(info["messages"][0]["content"])
        events.append(
            {
                "executed_action": action,
                "observation": observation,
                "metrics": info["metrics"],
            }
        )
        if "error" in observation:
            raise ValueError("Sandbox rejected action")
        return {"observation": observation, "reward": reward, "done": done}

    async def lookup(order_id: str) -> dict[str, Any]:
        result = await execute({"tool": "lookup_order", "args": {"order_id": order_id}})
        return result["observation"]

    async def submit(request: RefundResolutionRequest) -> dict[str, Any]:
        # This lock and receipt map are an in-memory demo transaction only.
        # Real backends need persistent idempotency and atomic authorization /
        # current-fact checks in their own transaction.
        async with backend_lock:
            if request.idempotency_key in receipts:
                return receipts[request.idempotency_key]
            current = RefundOrderFacts.from_lookup(state.context["scenario"])
            if request.facts != current or state.context["resolution"] is not None:
                raise ValueError("Stale snapshot or order already resolved")
            result = await execute(request.action())
            receipt = {"applied": True, "observation": result["observation"]}
            receipts[request.idempotency_key] = receipt
            return receipt

    resolver = SyntheticRefundResolver(
        order_id=row["order_id"],
        idempotency_key=f"demo-resolution:{row['order_id']}",
        lookup=lookup,
        submit=submit,
    )
    intent = {"name": "resolve_refund", "arguments": '{"intent":"resolve_refund"}'}
    result = await SandboxToolHost(resolver).execute_tool_call(
        ToolCall(id="scripted_intent", function=intent)
    )
    if result.is_error:
        raise RuntimeError(result.content)
    terminal = await execute({"tool": "finish", "args": {}})
    success = terminal["done"] and terminal["reward"] == 1.0
    return {
        "case_id": row["order_id"],
        "scripted_intent": intent,
        "tool_result": json.loads(result.content),
        "executed_events": events,
        "sandbox_success": success,
    }


async def main() -> None:
    """Exercise 128 train fixtures and print an explicitly scoped contract result."""
    rows = refund_policy_benchmark("train", count=128, seed=20261020)
    results = [await run_case(row) for row in rows]
    report = {
        "kind": "synthetic_tool_contract_check",
        "model_evaluated": False,
        "provider_calls": 0,
        "cases": len(results),
        "passed": sum(result["sandbox_success"] for result in results),
        "example_trace": results[0],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["passed"] != report["cases"]:
        raise RuntimeError("Tool contract check failed")


if __name__ == "__main__":
    asyncio.run(main())
