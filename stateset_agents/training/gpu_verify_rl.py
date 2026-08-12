"""GPU verification job for the RL (GSPO) training path.

The SFT path is verified on real rented hardware weekly (see
``.github/workflows/gpu-verify.yml``); this module gives the RL trainers —
the framework's namesake — the same treatment. It runs a SHORT real GSPO
training on a tiny real GPT-2 model and asserts the same convergence
property as the nightly CPU e2e test
(``tests/e2e/test_gspo_convergence_tiny.py``): the probability the policy
assigns to the preferred token strictly increases over training.

Runnable as ``python -m stateset_agents.training.gpu_verify_rl``. Uses CUDA
when available and falls back to CPU otherwise (same job, just slower), so
the exact code path that runs on the rented GPU is also unit-testable.

Exits 0 on success, 1 on failure, and always prints a single JSON summary
line prefixed with ``GPU_VERIFY_RL_SUMMARY`` as the last line of output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

__all__ = ["main", "run_verification"]

PROMPT = "Say A"
TARGET_RESPONSE = " A"
OFF_TARGET_RESPONSE = " B"
SUMMARY_PREFIX = "GPU_VERIFY_RL_SUMMARY"


def _tiny_gpt2() -> Any:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


class _ContainsTokenRewardModel:
    """Reward 1.0 if the response contains the target token, else 0.0."""

    async def compute_reward(self, turns: Any, context: Any) -> Any:
        from stateset_agents.core.reward_base import RewardResult

        score = 1.0 if "A" in turns[0].content else 0.0
        return RewardResult(score=score, breakdown={}, components={})


def _target_token_probability(
    model: Any, prompt_ids: Any, target_token_id: int
) -> float:
    """Probability the model assigns to ``target_token_id`` after the prompt."""
    import torch

    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        logits = model(prompt_ids.unsqueeze(0).to(device)).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
    model.train()
    return float(probs[target_token_id].item())


def run_verification(num_steps: int = 40) -> dict[str, Any]:
    """Run the short GSPO training and return the JSON-able summary dict."""
    import torch
    from transformers import GPT2Tokenizer

    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # nosec: B615
    tokenizer.pad_token = tokenizer.eos_token
    model = _tiny_gpt2().to(device)

    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0]
    target_token_id = tokenizer(TARGET_RESPONSE, add_special_tokens=False)["input_ids"][
        0
    ]

    initial_prob = _target_token_probability(model, prompt_ids, target_token_id)

    config = GSPOConfig(
        model_name="gpt2",
        num_generations=2,
        num_outer_iterations=1,
        num_iterations=1,
        max_prompt_length=32,
        max_completion_length=32,
        learning_rate=1e-2,
        beta=0.0,  # no KL penalty needed against a reference model
    )
    trainer = GSPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        agent=None,
        environment=None,
        reward_model=_ContainsTokenRewardModel(),
        ref_model=None,
    )

    responses = [TARGET_RESPONSE, OFF_TARGET_RESPONSE]

    async def fake_generate_group_responses(
        _prompt: str, _n: int
    ) -> list[tuple[str, float]]:
        # "Old" log probs must reflect the *current* policy (on-policy
        # rollout), matching what a real generator would produce.
        with torch.no_grad():
            old_log_probs, _ = trainer._compute_group_sequence_log_probs(
                PROMPT, responses
            )
        return list(zip(responses, old_log_probs.tolist(), strict=True))

    trainer.generator.generate_group_responses = fake_generate_group_responses

    avg_rewards: list[float] = []

    async def _train() -> None:
        for _ in range(num_steps):
            metrics = await trainer.train_step([PROMPT], num_groups=1)
            avg_rewards.append(float(metrics["average_reward"]))

    asyncio.run(_train())

    final_prob = _target_token_probability(model, prompt_ids, target_token_id)
    converged = final_prob > initial_prob
    return {
        "job": "gspo_gpu_verify",
        "device": device,
        "gpu": gpu_name,
        "num_steps": num_steps,
        "initial_target_prob": initial_prob,
        "final_target_prob": final_prob,
        "mean_reward_first": sum(avg_rewards[:5]) / max(len(avg_rewards[:5]), 1),
        "mean_reward_last": sum(avg_rewards[-5:]) / max(len(avg_rewards[-5:]), 1),
        "converged": converged,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify GSPO RL training converges on real hardware."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of GSPO training steps (default: 40).",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_verification(num_steps=args.steps)
    except Exception as exc:  # noqa: BLE001 — the summary line is the contract
        print(
            f"{SUMMARY_PREFIX} "
            + json.dumps(
                {"job": "gspo_gpu_verify", "converged": False, "error": str(exc)}
            )
        )
        return 1

    print(f"{SUMMARY_PREFIX} {json.dumps(summary)}")
    return 0 if summary["converged"] else 1


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess/live runs
    sys.exit(main())
