"""Recover a named conversation and continue with its history, entirely offline.

Run: python examples/durable_conversation.py

The temporary directory is for demonstration. Use an existing application-owned
directory when the snapshot should survive this process.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from stateset_agents.core.multiturn_agent import MultiTurnAgent

logger = logging.getLogger(__name__)


async def main() -> None:
    """Save, restore, continue, and publish using the acknowledged file revision."""
    with TemporaryDirectory(prefix="stateset-conversation-demo-") as directory:
        path = Path(directory) / "conversation.json"
        source = MultiTurnAgent(
            {}, response_backend=lambda prompt: "Your order is on its way."
        )
        context = await source.start_conversation(user_id="demo")
        await source.continue_conversation(context.conversation_id, "Track order 42")
        first = await source.save_conversation(context.conversation_id, path)

        def backend(prompt: str) -> str:
            # Demonstrate that the destination receives the recovered transcript.
            assert "Track order 42" in prompt
            assert "Your order is on its way." in prompt
            return "I still have the context for order 42."

        destination = MultiTurnAgent({}, response_backend=backend)
        restored = await destination.restore_conversation(
            path, expected_revision=first["revision"]
        )
        turns = await destination.continue_conversation(
            restored.conversation_id, "Do you remember the order?"
        )
        logger.info("Recovered conversation: %s", turns[-1]["content"])
        receipt = await destination.save_conversation(
            restored.conversation_id, path, expected_revision=first["revision"]
        )
        logger.info(
            "Saved %s: publication=%s, file_synced=%s, directory_synced=%s",
            receipt["revision"],
            receipt["publication"],
            receipt["file_synced"],
            receipt["directory_synced"],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
