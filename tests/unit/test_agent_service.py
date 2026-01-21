"""
Tests for AgentService conversation tracking.
"""

import pytest

from stateset_agents.api.schemas import ConversationRequest
from stateset_agents.api.services.agent_service import AgentService
from stateset_agents.utils.security import SecurityMonitor


class StubAgent:
    def __init__(self) -> None:
        self.tokenizer = None
        self.planning_manager = None
        self.last_messages = None

    async def generate_response(self, messages, context=None) -> str:
        self.last_messages = list(messages)
        return "stub-response"


@pytest.mark.asyncio
async def test_agent_service_tracks_conversations() -> None:
    service = AgentService(SecurityMonitor())
    agent_id = "agent-1"
    service.agents[agent_id] = StubAgent()

    request = ConversationRequest(messages=[{"role": "user", "content": "Hi"}])
    response = await service.get_conversation_response(agent_id, request)

    assert response.conversation_id
    assert agent_id in service.conversations
    assert len(service.agents[agent_id].last_messages) == 1

    conversations = service.conversations[agent_id]
    assert len(conversations) == 1

    record = conversations[0]
    assert record["id"] == response.conversation_id
    assert len(record["messages"]) == 2

    first_tokens = record["total_tokens"]

    followup = ConversationRequest(
        messages=[{"role": "user", "content": "Next"}],
        conversation_id=response.conversation_id,
    )
    await service.get_conversation_response(agent_id, followup)

    assert len(service.agents[agent_id].last_messages) == 3
    assert len(conversations) == 1
    assert len(record["messages"]) == 4
    assert record["total_tokens"] >= first_tokens


@pytest.mark.asyncio
async def test_agent_service_merges_full_history() -> None:
    service = AgentService(SecurityMonitor())
    agent_id = "agent-1"
    service.agents[agent_id] = StubAgent()

    request = ConversationRequest(messages=[{"role": "user", "content": "Hi"}])
    response = await service.get_conversation_response(agent_id, request)

    record = service.conversations[agent_id][0]
    full_history = [dict(msg) for msg in record["messages"]]
    full_history.append({"role": "user", "content": "Next"})

    followup = ConversationRequest(
        messages=full_history,
        conversation_id=response.conversation_id,
    )
    await service.get_conversation_response(agent_id, followup)

    assert len(record["messages"]) == 4
