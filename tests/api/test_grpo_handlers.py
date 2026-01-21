"""
Tests for GRPO handler context updates.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from stateset_agents.api.grpo.handlers import ConversationHandler
from stateset_agents.api.grpo.models import GRPOConversationRequest
from stateset_agents.api.grpo.state import reset_state_manager, get_state_manager


@pytest.mark.asyncio
async def test_grpo_handler_passes_context_updates():
    reset_state_manager()
    state = get_state_manager()
    conversation_id = "conv_123"
    state.create_conversation(conversation_id=conversation_id, user_id="user_1")

    multiturn_agent = MagicMock()
    multiturn_agent.continue_conversation = AsyncMock(
        return_value=[{"role": "assistant", "content": "ok"}]
    )
    multiturn_agent.get_conversation_summary = MagicMock(
        return_value={"conversation_id": conversation_id}
    )

    handler = ConversationHandler({"multiturn_agent": multiturn_agent})
    request = GRPOConversationRequest(
        message="What should we do next?",
        conversation_id=conversation_id,
        context={"plan_update": {"action": "advance"}},
    )

    response = await handler.handle_message(request, user_id="user_1")

    multiturn_agent.continue_conversation.assert_awaited_once()
    kwargs = multiturn_agent.continue_conversation.call_args.kwargs
    assert kwargs["context_update"] == {"plan_update": {"action": "advance"}}
    assert response.conversation_id == conversation_id
