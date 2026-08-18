"""
Tests for NSR rollout tools — nsr_decide / nsr_verify_plan exposed as
FunctionDefinitions so a ToolAgent policy can consult the verifier
mid-episode (and learn, via the NSR verifier reward, to do so).

Plan semantics mirror the NSR MCP server: one decision per step,
plan_verdict "approved" only when every step approves, evaluation stops at
the first blocking step, and an empty plan is an error — never a vacuous
approval.
"""

import json

import pytest

from stateset_agents.core.function_calling import (
    FunctionCallingMixin,
    FunctionDefinition,
    ToolCall,
)
from stateset_agents.tools.nsr import create_nsr_tools


class FakeNSRClient:
    def __init__(self, decisions=None):
        self.decisions = list(decisions or [])
        self.calls: list[dict] = []

    async def __call__(self, payload: dict) -> dict:
        self.calls.append(payload)
        decision = self.decisions.pop(0) if self.decisions else "approved"
        return {
            "decision_id": f"dec_{len(self.calls)}",
            "decision": decision,
            "confidence": 0.9,
            "plain_explanation": f"{decision} because test",
        }


def get_tool(tools, name):
    return next(t for t in tools if t.name == name)


class TestCreateNSRTools:
    def test_returns_both_function_definitions(self):
        tools = create_nsr_tools(client=FakeNSRClient())
        names = {t.name for t in tools}
        assert names == {"nsr_decide", "nsr_verify_plan"}
        assert all(isinstance(t, FunctionDefinition) for t in tools)
        assert all(t.handler is not None for t in tools)
        assert all(t.description for t in tools)


class TestNSRDecideTool:
    async def test_posts_query_and_returns_decision(self):
        client = FakeNSRClient(decisions=["denied"])
        decide = get_tool(create_nsr_tools(client=client), "nsr_decide")
        result = await decide.handler(
            query="Can order #A1 be refunded?", action="issue_refund"
        )
        assert result["decision"] == "denied"
        assert result["decision_id"] == "dec_1"
        assert client.calls[0]["query"] == "Can order #A1 be refunded?"
        assert client.calls[0]["action"] == "issue_refund"


class TestNSRVerifyPlanTool:
    async def test_all_steps_approved_approves_plan(self):
        client = FakeNSRClient(decisions=["approved", "approved"])
        verify = get_tool(create_nsr_tools(client=client), "nsr_verify_plan")
        result = await verify.handler(
            steps=[
                {"query": "Refund order #A1?", "action": "issue_refund"},
                {"query": "Notify customer C?", "action": "send_email"},
            ]
        )
        assert result["plan_verdict"] == "approved"
        assert len(result["steps"]) == 2
        assert [s["decision"] for s in result["steps"]] == ["approved", "approved"]

    async def test_blocking_step_denies_plan_and_stops(self):
        client = FakeNSRClient(decisions=["approved", "denied", "approved"])
        verify = get_tool(create_nsr_tools(client=client), "nsr_verify_plan")
        result = await verify.handler(
            steps=[
                {"query": "a?"},
                {"query": "b?"},
                {"query": "c?"},
            ]
        )
        assert result["plan_verdict"] == "denied"
        assert result["blocking_step"] == 1
        assert len(client.calls) == 2  # step c never evaluated

    async def test_refused_step_refuses_plan(self):
        client = FakeNSRClient(decisions=["refused"])
        verify = get_tool(create_nsr_tools(client=client), "nsr_verify_plan")
        result = await verify.handler(steps=[{"query": "a?"}])
        assert result["plan_verdict"] == "refused"

    async def test_empty_plan_is_an_error_never_approved(self):
        verify = get_tool(create_nsr_tools(client=FakeNSRClient()), "nsr_verify_plan")
        with pytest.raises(ValueError, match="non-empty"):
            await verify.handler(steps=[])

    async def test_step_without_query_is_an_error(self):
        verify = get_tool(create_nsr_tools(client=FakeNSRClient()), "nsr_verify_plan")
        with pytest.raises(ValueError, match="query"):
            await verify.handler(steps=[{"action": "issue_refund"}])


class TestFunctionCallingIntegration:
    async def test_tools_register_and_execute_through_the_mixin(self):
        class Host(FunctionCallingMixin):
            def __init__(self):
                self._tools = {}

        host = Host()
        for t in create_nsr_tools(client=FakeNSRClient(decisions=["approved"])):
            host.register_tool(t)

        call = ToolCall(
            id="tc_1",
            function={
                "name": "nsr_decide",
                "arguments": json.dumps({"query": "Refund order #A1?"}),
            },
        )
        result = await host.execute_tool_call(call)
        assert not result.is_error
        assert json.loads(result.content)["decision"] == "approved"
