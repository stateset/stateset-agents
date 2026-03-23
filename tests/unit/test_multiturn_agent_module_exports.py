"""Focused tests for the multi-turn agent module boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from stateset_agents.core.multiturn_agent import (
    ConversationContext,
    DialogueDatabase,
    MultiTurnAgent,
)
from stateset_agents.core.multiturn_context import (
    ConversationContext as SplitConversationContext,
)
from stateset_agents.core.multiturn_context import (
    DialogueDatabase as SplitDialogueDatabase,
)


def test_multiturn_agent_reexports_context_types():
    """Public multiturn module should keep exporting its context types."""
    assert ConversationContext is SplitConversationContext
    assert DialogueDatabase is SplitDialogueDatabase


@pytest.mark.asyncio
async def test_generate_multiturn_response_persists_assistant_turn():
    """Generated assistant replies should be written back to the transcript."""
    agent = MultiTurnAgent(model_config={"model_type": "test"})
    context = await agent.start_conversation()
    agent.generate_response = AsyncMock(return_value="Persisted response")

    response = await agent.generate_multiturn_response(
        context.conversation_id,
        "Hello there",
        strategy="default",
        use_tools=False,
    )

    assert response == "Persisted response"
    assert len(context.history) == 2
    assert context.history[0]["role"] == "user"
    assert context.history[1]["role"] == "assistant"
    assert context.history[1]["content"] == "Persisted response"
