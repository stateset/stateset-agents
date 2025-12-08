#!/usr/bin/env python3
"""
Hello World - StateSet Agents in 60 seconds

This example runs immediately after `pip install stateset-agents`.
No model downloads required - uses stub backend for instant feedback.

Run:
    python examples/hello_world.py

What this demonstrates:
- Creating a multi-turn conversational agent
- Running a simple conversation
- Basic reward evaluation
- The core GRPO training loop (in stub mode)
"""

import asyncio

# These imports work right after `pip install stateset-agents`
from stateset_agents import (
    MultiTurnAgent,
    ConversationEnvironment,
    HelpfulnessReward,
    SafetyReward,
    CompositeReward,
)
from stateset_agents.core.agent import AgentConfig


async def main():
    print("=" * 60)
    print("StateSet Agents - Hello World")
    print("=" * 60)
    print()

    # Step 1: Create an agent with stub backend (no model download)
    print("[1/4] Creating agent (stub mode - instant, no downloads)...")

    config = AgentConfig(
        model_name="stub://hello-world",
        use_stub_model=True,
        stub_responses=[
            "Hello! I'm a stub agent. In production, I'd use a real LLM.",
            "I can help explain how the GRPO training framework works.",
            "The key idea is: generate trajectories, compute rewards, update policy.",
            "Thank you for trying StateSet Agents!",
        ],
        system_prompt="You are a helpful AI assistant.",
        temperature=0.7,
        max_new_tokens=128,
    )

    agent = MultiTurnAgent(config)
    await agent.initialize()
    print("   Agent ready!")
    print()

    # Step 2: Have a conversation
    print("[2/4] Running a sample conversation...")
    print("-" * 40)

    conversation = [
        {"role": "system", "content": config.system_prompt},
    ]

    user_messages = [
        "Hello! What can you do?",
        "How does GRPO training work?",
        "Thanks for the explanation!",
    ]

    for user_msg in user_messages:
        print(f"User: {user_msg}")
        conversation.append({"role": "user", "content": user_msg})

        response = await agent.generate_response(conversation)
        print(f"Agent: {response}")
        print()

        conversation.append({"role": "assistant", "content": response})

    print("-" * 40)
    print()

    # Step 3: Show reward computation
    print("[3/4] Computing rewards for the conversation...")

    # Create trajectory turns from conversation
    from stateset_agents import ConversationTurn

    turns = []
    for msg in conversation:
        if msg["role"] in ("user", "assistant"):
            turns.append(ConversationTurn(role=msg["role"], content=msg["content"]))

    # Evaluate with composite reward
    reward_fn = CompositeReward([
        HelpfulnessReward(weight=0.7),
        SafetyReward(weight=0.3),
    ])

    result = await reward_fn.compute_reward(turns)
    print(f"   Total reward score: {result.score:.3f}")
    print(f"   Breakdown: {result.breakdown}")
    print()

    # Step 4: Show what training would look like
    print("[4/4] Training preview (what happens in real training)...")
    print()
    print("   In a real training run, StateSet Agents would:")
    print("   1. Generate multiple conversation trajectories")
    print("   2. Compute rewards for each trajectory")
    print("   3. Calculate advantages using GAE (Generalized Advantage Estimation)")
    print("   4. Update the policy using GRPO/GSPO gradient updates")
    print("   5. Apply KL divergence regularization to prevent drift")
    print()
    print("   To train with a real model, try:")
    print("   $ pip install 'stateset-agents[training]'")
    print("   $ python examples/train_with_trl_grpo.py")
    print()

    print("=" * 60)
    print("Done! You've seen the core concepts of StateSet Agents.")
    print()
    print("Next steps:")
    print("  - examples/customer_service_agent.py  # Domain-specific agent")
    print("  - examples/finetune_qwen3_gspo.py     # Real model training")
    print("  - stateset-agents train --help        # CLI training")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
