"""Recover a conversation's plan without repeating completed steps.

Run: python examples/durable_planning.py
Uses local callbacks and temporary files; no credentials or model downloads.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from stateset_agents.core.multiturn_agent import MultiTurnAgent
from stateset_agents.experimental.long_term_planning import (
    PlanningConfig,
    PlanningManager,
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Restore a partially completed plan alongside its conversation history."""
    with TemporaryDirectory(prefix="stateset-planning-demo-") as directory:
        path = Path(directory) / "planned-conversation.json"
        source = MultiTurnAgent(
            {},
            planning_manager=PlanningManager(PlanningConfig(enabled=True)),
            response_backend=lambda prompt: "Ready for the next step.",
        )
        context = await source.start_conversation(user_id="demo")
        await source.continue_conversation(
            context.conversation_id, "Draft report, review it, then send it"
        )
        await source.continue_conversation(
            context.conversation_id,
            "The draft is done.",
            context_update={"plan_update": {"action": "advance"}},
        )
        receipt = await source.save_conversation(context.conversation_id, path)

        def backend(prompt: str) -> str:
            assert "[x] Draft report" in prompt
            assert "[>] review it" in prompt
            return "The draft is done. Review is the next step."

        planning = PlanningManager(PlanningConfig(enabled=True))
        destination = MultiTurnAgent(
            {}, planning_manager=planning, response_backend=backend
        )
        restored = await destination.restore_conversation(
            path, expected_revision=receipt["revision"]
        )
        turns = await destination.continue_conversation(
            restored.conversation_id, "Where were we?"
        )
        logger.info("%s", turns[-1]["content"])
        plan = planning.get_plan(restored.conversation_id)
        assert plan is not None
        logger.info("Restored plan progress: %.0f%%", 100 * plan.progress())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
