"""Commit related records together and inspect rollback, entirely offline.

Run: python examples/transactional_persistence.py
Use an application-owned file instead of a temporary directory for durable data.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from stateset_agents.api.persistence import (
    Agent,
    Conversation,
    DatabaseBackend,
    DatabaseConfig,
    UnitOfWork,
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Show atomic publication, rollback, and reopening the committed database."""
    with TemporaryDirectory(prefix="stateset-transactions-") as directory:
        config = DatabaseConfig(
            backend=DatabaseBackend.SQLITE,
            connection_url=str(Path(directory) / "records.db"),
        )
        async with UnitOfWork(config) as database:
            async with database.transaction():
                await database.agents.create(Agent(id="support", name="Support"))
                await database.conversations.create(
                    Conversation(id="order-42", agent_id="support", user_id="demo")
                )
            logger.info("Committed: %s", database.get_status()["last_transaction"])

            try:
                async with database.transaction():
                    await database.agents.delete("support")
                    raise ValueError("Keep the agent while its conversation exists")
            except ValueError:
                logger.info(
                    "Rolled back: %s", database.get_status()["last_transaction"]
                )

        async with UnitOfWork(config) as restored:
            assert await restored.agents.get("support") is not None
            assert await restored.conversations.get("order-42") is not None
            logger.info("Both committed records survived reopening.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
