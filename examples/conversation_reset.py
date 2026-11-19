"""Reset a live conversation without accepting its late response; fully offline.

Run: python examples/conversation_reset.py
"""

from __future__ import annotations

import asyncio
import logging

from stateset_agents.core.agent import (
    AgentConfig,
    ConversationResetError,
    MultiTurnAgent,
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Discard a retired stream and continue with a new conversation."""
    agent = MultiTurnAgent(
        AgentConfig(
            model_name="stub://conversation-reset",
            stub_responses=[
                "This belongs to the previous conversation.",
                "Hello from the new conversation.",
            ],
        )
    )
    await agent.initialize()
    old_stream = agent.generate_response_stream("Start a response")
    try:
        logger.info("First chunk: %r", await anext(old_stream))
        await agent.reset()
        try:
            await anext(old_stream)
        except ConversationResetError:
            logger.info("Previous conversation stream was retired.")
    finally:
        await old_stream.aclose()
    logger.info("New answer: %s", await agent.generate_response("Hello"))
    logger.info("History: %s", agent.get_history())
    logger.info("Conversation status: %s", agent.get_conversation_status())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
