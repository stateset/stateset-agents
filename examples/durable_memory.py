"""Save and recover portable memory data offline, without loading a model.

Run: python examples/durable_memory.py

This demo uses a temporary directory. Use an existing application-owned directory
for persistent storage. Initialize the destination first to rebuild dense vectors;
an uninitialized VectorMemory uses its existing lexical retrieval behavior.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from stateset_agents.core.enhanced import VectorMemory

logger = logging.getLogger(__name__)


async def main() -> None:
    """Recover entry data and replace a snapshot using its acknowledged revision."""
    with TemporaryDirectory(prefix="stateset-memory-demo-") as directory:
        path = Path(directory) / "memory.json"
        memory = VectorMemory()
        await memory.add_entry(
            "Refunds are accepted within 30 days.", {"policy": "demo"}
        )
        first = await memory.save_snapshot(path)
        recovered = VectorMemory()
        await recovered.restore_snapshot(path, expected_revision=first["revision"])
        match = (await recovered.retrieve_relevant("refunds"))[0]
        logger.info(
            "Recovered: %s (access count: %s)", match.content, match.access_count
        )
        second = await recovered.save_snapshot(
            path, expected_revision=first["revision"]
        )
        logger.info(
            "Saved revision %s: publication=%s, file_synced=%s, directory_synced=%s",
            second["revision"],
            second["publication"],
            second["file_synced"],
            second["directory_synced"],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
