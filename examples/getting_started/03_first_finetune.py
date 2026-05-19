"""03 — Your first real GSPO fine-tune.

A minimum-viable training run: small base model, LoRA, the safe-default
GSPOConfig from whitepaper §B.1 (use_reference_model=True, beta=0.05, 1 epoch).
Trains on the bundled customer-support corpus (24 scenarios; 16 train, 8 eval).

Install:
    pip install "stateset-agents[training]"

Run (requires CUDA — A100-40GB or similar; ~6 min wall clock for the trainee
alone, ~10 min including baseline + final eval):
    python 03_first_finetune.py

Expected output (approximate, depends on hardware + RNG):
    Baseline rubric: 0.506
    Trained rubric:  0.55-0.65 (positive transfer)
    Improvement:     +0.05 to +0.15

This is the same trainer/config that produced the canonical §11.7 result
in the whitepaper. If this works, you have a green light for the rest of
the framework.

For the multi-seed, LLM-judge-verified protocol, run the Colab notebook
at notebooks/customer_support_3seed_judge.ipynb instead.
"""

import asyncio
import time

import torch

from stateset_agents.core import ConversationEnvironment, MultiTurnAgent
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data import (
    SupportRewardComposite,
    load_support_scenarios,
    make_support_scenarios,
)
from stateset_agents.training import GSPOConfig, train_with_gspo
from stateset_agents.utils.reproducibility import set_all_seeds

TRAINEE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42


def prompt_for(scenario) -> str:
    return (
        "You are a helpful customer support agent. Respond to the user warmly, "
        "address their concern directly, and confirm the next step.\n\n"
        f"User: {scenario.user_query}\n\nAgent:"
    )


async def evaluate(agent: MultiTurnAgent, scenarios) -> float:
    rubric = SupportRewardComposite()
    scores: list[float] = []
    for s in scenarios:
        response = await agent.generate_response(prompt_for(s))
        turns = [ConversationTurn(role="assistant", content=response)]
        result = await rubric.compute_reward(turns, context=s.to_scenario())
        scores.append(result.score)
    return sum(scores) / max(len(scores), 1)


async def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "This example needs CUDA. For a GPU-free smoke test see 01_hello_stub.py."
        )

    set_all_seeds(SEED, deterministic_cuda=False)

    all_scenarios = load_support_scenarios()
    train_scenarios = all_scenarios[:16]
    eval_scenarios = all_scenarios[16:]

    # 1) Baseline eval — what does the untuned model score on the rubric?
    baseline_agent = MultiTurnAgent(AgentConfig(
        model_name=TRAINEE_MODEL,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",       # portable across cloud GPUs (no flash-attn dep)
        do_sample=False, temperature=0.0,
    ))
    await baseline_agent.initialize()
    baseline = await evaluate(baseline_agent, eval_scenarios)
    print(f"Baseline rubric: {baseline:.3f}")
    del baseline_agent
    torch.cuda.empty_cache()

    # 2) Train — GSPO with the §B.1 safe defaults.
    config = GSPOConfig(
        model_name=TRAINEE_MODEL,
        num_generations=4,
        clip_range_left=3e-4, clip_range_right=4e-4,
        learning_rate=5e-6,
        max_prompt_length=512, max_completion_length=320,
        use_lora=True, lora_r=16, lora_alpha=32,
        gradient_checkpointing=False,
        num_epochs=1, warmup_ratio=0.1,
        use_reference_model=True, beta=0.05,   # KL anchor — see §10.5 of the whitepaper
        output_dir="./outputs/getting_started_03",
    )
    agent = MultiTurnAgent(AgentConfig(
        model_name=TRAINEE_MODEL, torch_dtype="bfloat16", attn_implementation="sdpa",
    ))
    env = ConversationEnvironment(
        scenarios=make_support_scenarios(train_scenarios),
        reward_fn=SupportRewardComposite(),
        max_turns=4,
    )
    train_queries = [
        {"prompt": prompt_for(s), "context": {
            "must_acknowledge": list(s.must_acknowledge),
            "must_avoid": list(s.must_avoid),
            "intent": s.intent,
        }}
        for s in train_scenarios
    ]

    t0 = time.time()
    await train_with_gspo(
        config=config,
        agent=agent,
        environment=env,
        reward_model=env.reward_fn,
        train_queries=train_queries,
    )
    print(f"Training wall-clock: {time.time() - t0:.0f} s")

    # 3) Final eval — did training help?
    trained = await evaluate(agent, eval_scenarios)
    print(f"Trained rubric:  {trained:.3f}")
    print(f"Improvement:     {trained - baseline:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
