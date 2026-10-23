"""Focused tests for the multi-turn agent module boundary."""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock

import pytest

# Deprecated with the api.grpo app (removed in v0.71.0); the warnings are
# asserted in test_dialogue_session_agent_is_deprecated below.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from stateset_agents.core import multiturn_agent
    from stateset_agents.core.multiturn_agent import (
        ConversationContext,
        DialogueDatabase,
        DialogueSessionAgent,
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
    agent = DialogueSessionAgent(model_config={"model_type": "test"})
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


def test_dialogue_session_agent_is_deprecated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The class no longer shares the stable ``core.agent.MultiTurnAgent`` name."""
    import importlib
    import sys

    from stateset_agents.core.agent import MultiTurnAgent as StableMultiTurnAgent

    assert "MultiTurnAgent" not in multiturn_agent.__all__
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.71\.0"):
        alias = multiturn_agent.MultiTurnAgent
    assert alias is DialogueSessionAgent
    assert alias is not StableMultiTurnAgent

    monkeypatch.delitem(sys.modules, "stateset_agents.core.multiturn_agent")
    with pytest.warns(DeprecationWarning, match="core.multiturn_agent"):
        importlib.import_module("stateset_agents.core.multiturn_agent")
