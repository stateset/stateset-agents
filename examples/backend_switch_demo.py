"""Showcase switching between stub and full backends for MultiTurnAgent.

Run with:
    python examples/backend_switch_demo.py --stub
    python examples/backend_switch_demo.py --model gpt2
"""

import argparse
import asyncio
from typing import List, Dict

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent


def build_agent(args: argparse.Namespace) -> MultiTurnAgent:
    config = AgentConfig(
        model_name=args.model,
        use_stub_model=args.stub,
        stub_responses=[
            "Stub mode activated. Bring your own checkpoints!",
            "Responding from the lightweight backend.",
        ]
        if args.stub
        else None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    return MultiTurnAgent(config)


async def run_conversation(agent: MultiTurnAgent, messages: List[Dict[str, str]]) -> None:
    await agent.initialize()
    reply = await agent.generate_response(messages)
    print("Assistant:", reply)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backend switch demo for StateSet Agents")
    parser.add_argument("--stub", action="store_true", help="Use the stub backend")
    parser.add_argument("--model", default="gpt2", help="Model name or stub identifier")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=64)

    args = parser.parse_args()

    if args.stub and args.model == "gpt2":
        args.model = "stub://demo"

    agent = build_agent(args)
    history = [{"role": "user", "content": "Hello! Can you tell me about the latest update?"}]

    try:
        asyncio.run(run_conversation(agent, history))
    except Exception as exc:
        print("Failed to run conversation:", exc)
        if not args.stub:
            print("Hint: try `--stub` to run without downloading transformer weights.")


if __name__ == "__main__":
    main()
