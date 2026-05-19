"""01 — Hello, stub backend.

The smallest possible "did this install work?" example. Runs without a GPU,
without downloading any models, in about a second.

Install:
    pip install stateset-agents

Run:
    python 01_hello_stub.py

Expected output:
    Agent loaded OK. Version: 0.13.4
    Response: <some deterministic stub text>
"""

import asyncio

from stateset_agents import __version__
from stateset_agents.core import MultiTurnAgent
from stateset_agents.core.agent_config import AgentConfig


async def main() -> None:
    # use_stub_model=True swaps in a deterministic in-memory backend.
    # No model weights, no torch, no GPU — the framework is exercised end-to-end
    # but the underlying "model" returns canned text. This is the seam that lets
    # the test suite run without a GPU and is the right first sanity check after
    # a fresh `pip install`.
    agent = MultiTurnAgent(AgentConfig(
        model_name="stub://hello",   # the stub:// prefix routes to the in-memory backend
        use_stub_model=True,
    ))
    await agent.initialize()
    print(f"Agent loaded OK. Version: {__version__}")

    response = await agent.generate_response("Say hello.")
    print(f"Response: {response!r}")


if __name__ == "__main__":
    asyncio.run(main())
