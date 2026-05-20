"""06 — Run a multi-turn episode against a ConversationEnvironment.

Builds on 01: rather than a single one-shot `generate_response`, we drive a
multi-turn episode against the bundled customer-support corpus. Stub-backed,
so it runs in <1 second on CPU. The shape is identical to a real training
rollout — `env.reset()` → loop `env.step()` until the episode terminates —
which is the same loop a GSPO trainer drives internally.

Install:
    pip install stateset-agents

Run:
    python 06_multi_turn_episode.py

Expected output:
    Loaded 24 support scenarios.
    Episode 1 / scenario 'I want my money back for order #4521'
      turn 1 | reward=0.X
      turn 2 | reward=0.X
      ...
    Episode finished. Total reward: 1.XX  Turns: N
"""

import asyncio

from stateset_agents.core import ConversationEnvironment, MultiTurnAgent
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.data import SupportRewardComposite, load_support_scenarios, make_support_scenarios


async def main() -> None:
    scenarios = load_support_scenarios()
    print(f"Loaded {len(scenarios)} support scenarios.")

    env = ConversationEnvironment(
        scenarios=make_support_scenarios(scenarios[:3]),  # first 3 for a quick demo
        reward_fn=SupportRewardComposite(),
        max_turns=3,
    )

    agent = MultiTurnAgent(AgentConfig(
        model_name="stub://multi-turn",
        use_stub_model=True,
    ))
    await agent.initialize()

    # Pick a specific scenario so the run is deterministic.
    initial_state = await env.reset(scenario=env.scenarios[0])
    user_query = initial_state.context["scenario"].get("user_query", "Hello.")
    print(f"\nEpisode 1 / scenario {user_query!r}")

    history: list[dict[str, str]] = [{"role": "user", "content": user_query}]
    total_reward = 0.0
    turn = 0
    while True:
        turn += 1
        response = await agent.generate_response(history)
        history.append({"role": "assistant", "content": response})

        # `step(response)` advances the stateful env and returns a payload dict.
        payload = await env.step(response)
        reward, done = float(payload["reward"]), bool(payload["done"])
        total_reward += reward
        print(f"  turn {turn} | reward={reward:.2f} | done={done}")

        if done:
            break

        # In a real evaluation loop you'd inject the simulated user's next turn here.
        # The stub backend doesn't care — feed it a generic follow-up to keep going.
        history.append({"role": "user", "content": "Can you elaborate?"})

    print(f"\nEpisode finished. Total reward: {total_reward:.2f}  Turns: {turn}")
    print("Tip: swap `use_stub_model=True` for a real `model_name=` to see real responses.")


if __name__ == "__main__":
    asyncio.run(main())
