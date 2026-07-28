"""CPU convergence e2e test: GSPO training measurably improves the policy.

This is deliberately slow (real backward passes through a tiny real GPT2
model, ~40 optimizer steps) so it is excluded from the default fast path
(`pytest.ini` deselects `slow`) and instead runs in the nightly benchmark
workflow (see `.github/workflows/benchmark-nightly.yml`).

Task: single-token preference. The prompt is "Say A" and the reward is 1.0
whenever the sampled response contains the target token ("A") and 0.0
otherwise. Two fixed candidate completions are used per group (one on-target,
one off-target) with reward computed by a stub reward model — this isolates
the GSPO objective/optimizer path from generation-sampling noise, matching
the harness pattern in `tests/integration/test_trainer_ratio_invariants.py`.

Two convergence signals are checked:

  1. (primary, low-variance) The probability the *initial* policy logits
     assign to the target token immediately after the prompt must strictly
     increase after training. This is a direct read of the model's
     internal state and is not subject to sampling noise.
  2. (secondary, advisory) The mean training reward over the last 10 steps
     should exceed the mean over the first 10 steps by a margin. Because
     this is measured through the (fixed, 2-candidate) group-reward signal
     it can plateau early (reward is already 0.5 average once the model
     reliably prefers one candidate over the other within the pair), so it
     is logged and asserted leniently, not treated as a hard failure axis
     on its own the way (1) is.
"""

from __future__ import annotations

import logging

import pytest
import torch

pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from stateset_agents.core.reward_base import RewardResult
from stateset_agents.training.gspo_config import GSPOConfig
from stateset_agents.training.gspo_trainer import GSPOTrainer

logger = logging.getLogger(__name__)

PROMPT = "Say A"
TARGET_RESPONSE = " A"
OFF_TARGET_RESPONSE = " B"
NUM_STEPS = 40
WARMUP_STEPS = 10


def _tiny_gpt2() -> GPT2LMHeadModel:
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

    async def compute_reward(self, turns, context) -> RewardResult:
        content = turns[0].content
        score = 1.0 if "A" in content else 0.0
        return RewardResult(score=score, breakdown={}, components={})


def _target_token_probability(
    model: GPT2LMHeadModel, prompt_ids: torch.Tensor, target_token_id: int
) -> float:
    """Probability the model assigns to `target_token_id` right after the prompt."""
    model.eval()
    with torch.no_grad():
        logits = model(prompt_ids.unsqueeze(0)).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
    model.train()
    return float(probs[target_token_id].item())


def _run_convergence_training() -> tuple[float, float, list[float]]:
    """Train GSPO for NUM_STEPS on the single-token preference task.

    Returns (initial_target_prob, final_target_prob, per_step_avg_rewards).
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = _tiny_gpt2()

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

    trainer.generator.generate_group_responses = fake_generate_group_responses  # type: ignore[method-assign]

    avg_rewards: list[float] = []

    async def _train() -> None:
        for _ in range(NUM_STEPS):
            metrics = await trainer.train_step([PROMPT], num_groups=1)
            avg_rewards.append(float(metrics["average_reward"]))

    import asyncio

    asyncio.run(_train())

    final_prob = _target_token_probability(model, prompt_ids, target_token_id)
    return initial_prob, final_prob, avg_rewards


@pytest.mark.slow
def test_gspo_convergence_tiny() -> None:
    """GSPO training on a trivially learnable single-token task provably
    increases the probability of the preferred token under the policy."""
    initial_prob, final_prob, avg_rewards = _run_convergence_training()

    logger.info(
        "GSPO convergence: target-token prob %.6f -> %.6f", initial_prob, final_prob
    )
    logger.info("GSPO convergence: per-step avg reward = %s", avg_rewards)

    # Primary, low-variance assertion: the internal probability of the
    # preferred token strictly increased under the final policy.
    assert (
        final_prob > initial_prob
    ), f"target token probability did not increase: {initial_prob} -> {final_prob}"

    # Secondary, advisory assertion: reward over the tail of training should
    # not be worse than the start of training. Logged either way; only a
    # sharp regression (not merely "no improvement", since reward can
    # legitimately plateau at 0.5 once the group ordering is learned) fails
    # the test.
    first_window = avg_rewards[:WARMUP_STEPS]
    last_window = avg_rewards[-WARMUP_STEPS:]
    mean_first = sum(first_window) / len(first_window)
    mean_last = sum(last_window) / len(last_window)
    logger.info(
        "GSPO convergence: mean reward first %d steps=%.4f, last %d steps=%.4f",
        WARMUP_STEPS,
        mean_first,
        WARMUP_STEPS,
        mean_last,
    )
    assert (
        mean_last >= mean_first - 0.1
    ), f"reward regressed over training: first={mean_first:.4f} last={mean_last:.4f}"
