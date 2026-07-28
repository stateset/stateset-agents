"""Cross-trainer ratio/gradient invariants for DAPO, GEPO, and GSPO.

These integration tests exercise real (tiny) GPT2 models end-to-end through
each trainer's actual ``train_step``/log-prob/ratio machinery, checking the
invariants every importance-sampling trainer must satisfy:

  (a) On the very first (on-policy) evaluation, the mean importance ratio
      must be ~1 (log probs computed under identical, unchanged weights).
  (b) After one real optimizer step, recomputing the ratio against the
      frozen "old" log probs must diverge from 1 (the policy moved).
  (c) The training loss is finite (no NaN/Inf).
  (d) At least one model parameter has a nonzero gradient after a real
      ``train_step``/backward pass (the loss is actually connected to the
      parameters it's supposed to update).

Each trainer's public API differs (DAPO exposes token log probs + an inner
mu-loop; GEPO recomputes learner/sampler log probs every call; GSPO scores
whole sequences), so each factory below builds a small harness with a
uniform interface and the shared test body asserts the same four invariants
against it.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from stateset_agents.core.reward_base import RewardResult
from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer
from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer
from stateset_agents.training.gspo_config import GSPOConfig
from stateset_agents.training.gspo_trainer import GSPOTrainer


def _tiny_gpt2(vocab_size: int = 200) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    # Dropout disabled: two forward passes on unchanged weights must be
    # deterministic, or an on-policy ratio would spuriously differ from 1.
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=vocab_size,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


@dataclass
class TrainerHarness:
    model: torch.nn.Module
    run_train_step: Callable[[], Awaitable[dict[str, float]]]
    onpolicy_ratios: Callable[[], torch.Tensor]
    recompute_ratio_after_step: Callable[[], torch.Tensor]


# --------------------------------------------------------------------------
# DAPO
# --------------------------------------------------------------------------


def _make_dapo_harness() -> TrainerHarness:
    model = _tiny_gpt2()
    config = DAPOConfig(
        model_name="gpt2",
        group_size=2,
        num_gradient_updates=2,
    )

    def reward_fn(prompt: str, response: str) -> float:
        return 0.0

    trainer = DAPOTrainer(
        config=config, model=model, tokenizer=None, reward_fn=reward_fn
    )

    torch.manual_seed(1)
    input_ids = torch.randint(0, 200, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.ones_like(input_ids, dtype=torch.float)
    response_mask[:, :4] = 0.0

    captured_ratios: list[torch.Tensor] = []
    orig_ratio = trainer.compute_importance_ratio

    def spy(cur: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        ratio = orig_ratio(cur, old)
        captured_ratios.append(ratio.detach().clone())
        return ratio

    trainer.compute_importance_ratio = spy  # type: ignore[method-assign]

    async def run_train_step() -> dict[str, float]:
        captured_ratios.clear()
        sample = {
            "responses": [
                {
                    "input_ids": input_ids[i],
                    "attention_mask": attention_mask[i],
                    "response_mask": response_mask[i],
                    "sequence_length": int(attention_mask[i].sum()),
                }
                for i in range(2)
            ],
            "advantages": torch.tensor([0.5, -0.5]),
            "rewards": [1.0, 0.0],
            "accuracy": 0.5,
        }

        async def fake_collect(prompts: list[str], n: int) -> tuple[list[dict], float]:
            return [sample], 0.0

        trainer.collect_samples_with_dynamic_sampling = fake_collect  # type: ignore[method-assign]
        return await trainer.train_step(["q"])

    def onpolicy_ratios() -> torch.Tensor:
        # num_gradient_updates=2: the first inner update is always on-policy.
        return captured_ratios[0]

    def recompute_ratio_after_step() -> torch.Tensor:
        # The second inner update's ratio, captured inside the same
        # train_step call, is computed against the same frozen rollout-time
        # old log probs after one real optimizer step has already run.
        return captured_ratios[1]

    return TrainerHarness(
        model=model,
        run_train_step=run_train_step,
        onpolicy_ratios=onpolicy_ratios,
        recompute_ratio_after_step=recompute_ratio_after_step,
    )


# --------------------------------------------------------------------------
# GEPO
# --------------------------------------------------------------------------


def _make_gepo_harness() -> TrainerHarness:
    model = _tiny_gpt2(vocab_size=50257)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = GEPOConfig(model_name="gpt2", group_size=2)

    def reward_fn(prompt: str, response: str) -> float:
        return 1.0 if "ok" in response else 0.0

    trainer = GEPOTrainer(
        config=config, model=model, tokenizer=tokenizer, reward_fn=reward_fn
    )

    prompt_ids = tokenizer("hello", return_tensors="pt")["input_ids"][0]
    resp_a = torch.cat([prompt_ids, torch.tensor([1, 2, 3])])
    resp_b = torch.cat([prompt_ids, torch.tensor([4, 5, 6])])
    max_len = max(resp_a.shape[0], resp_b.shape[0])

    def _pad(t: torch.Tensor) -> torch.Tensor:
        pad = max_len - t.shape[0]
        return torch.nn.functional.pad(t, (0, pad), value=tokenizer.pad_token_id)

    fixed_responses = [
        {
            "response": "ok",
            "input_ids": _pad(resp_a),
            "attention_mask": torch.ones(max_len, dtype=torch.long),
            "response_start_idx": int(prompt_ids.shape[0]),
        },
        {
            "response": "nope",
            "input_ids": _pad(resp_b),
            "attention_mask": torch.ones(max_len, dtype=torch.long),
            "response_start_idx": int(prompt_ids.shape[0]),
        },
    ]

    async def fake_generate_group_responses(
        prompt: str, group_size: int
    ) -> list[dict[str, Any]]:
        return fixed_responses

    trainer.generate_group_responses = fake_generate_group_responses  # type: ignore[method-assign]

    captured_ratios: list[torch.Tensor] = []
    orig_coef = trainer.compute_gepo_coefficient

    def spy(learner: torch.Tensor, sampler: torch.Tensor) -> torch.Tensor:
        coef = orig_coef(learner, sampler)
        captured_ratios.append(coef.detach().clone())
        return coef

    trainer.compute_gepo_coefficient = spy  # type: ignore[method-assign]

    batch_input_ids = torch.stack([r["input_ids"] for r in fixed_responses])
    batch_attention_mask = torch.stack([r["attention_mask"] for r in fixed_responses])
    response_start_idx = fixed_responses[0]["response_start_idx"]

    frozen_old_log_probs: list[torch.Tensor] = []

    async def run_train_step() -> dict[str, float]:
        captured_ratios.clear()
        with torch.no_grad():
            _, old = trainer.compute_sequence_log_probs(
                batch_input_ids, batch_attention_mask, response_start_idx
            )
        frozen_old_log_probs.clear()
        frozen_old_log_probs.append(old)
        return await trainer.train_step(["hello"])

    def onpolicy_ratios() -> torch.Tensor:
        return captured_ratios[0]

    def recompute_ratio_after_step() -> torch.Tensor:
        with torch.no_grad():
            _, new_log_probs = trainer.compute_sequence_log_probs(
                batch_input_ids, batch_attention_mask, response_start_idx
            )
        return trainer.compute_gepo_coefficient(new_log_probs, frozen_old_log_probs[0])

    return TrainerHarness(
        model=model,
        run_train_step=run_train_step,
        onpolicy_ratios=onpolicy_ratios,
        recompute_ratio_after_step=recompute_ratio_after_step,
    )


# --------------------------------------------------------------------------
# GSPO
# --------------------------------------------------------------------------


class _StubRewardModel:
    async def compute_reward(
        self, turns: list[Any], context: dict[str, Any]
    ) -> RewardResult:
        score = 1.0 if turns[0].content == "ok" else 0.0
        return RewardResult(score=score, breakdown={}, components={})


def _make_gspo_harness() -> TrainerHarness:
    model = _tiny_gpt2(vocab_size=50257)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = GSPOConfig(
        model_name="gpt2",
        num_generations=2,
        num_outer_iterations=1,
        num_iterations=1,
        max_prompt_length=32,
        max_completion_length=32,
    )

    trainer = GSPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        agent=None,
        environment=None,
        reward_model=_StubRewardModel(),
        ref_model=None,
    )

    prompt = "hello"
    responses = ["ok", "nope"]

    with torch.no_grad():
        old_log_probs, _ = trainer._compute_group_sequence_log_probs(prompt, responses)

    async def fake_generate_group_responses(
        _prompt: str, _n: int
    ) -> list[tuple[str, float]]:
        return list(zip(responses, old_log_probs.tolist(), strict=True))

    trainer.generator.generate_group_responses = fake_generate_group_responses  # type: ignore[method-assign]

    captured_ratios: list[torch.Tensor] = []
    orig_ratio = trainer.compute_sequence_importance_ratio

    def spy(
        current: torch.Tensor, old: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        ratio = orig_ratio(current, old, lengths)
        captured_ratios.append(ratio.detach().clone())
        return ratio

    trainer.compute_sequence_importance_ratio = spy  # type: ignore[method-assign]

    async def run_train_step() -> dict[str, float]:
        captured_ratios.clear()
        return await trainer.train_step([prompt], num_groups=1)

    def onpolicy_ratios() -> torch.Tensor:
        return captured_ratios[0]

    def recompute_ratio_after_step() -> torch.Tensor:
        with torch.no_grad():
            new_log_probs, lengths = trainer._compute_group_sequence_log_probs(
                prompt, responses
            )
        return trainer.compute_sequence_importance_ratio(
            new_log_probs, old_log_probs, lengths
        )

    return TrainerHarness(
        model=model,
        run_train_step=run_train_step,
        onpolicy_ratios=onpolicy_ratios,
        recompute_ratio_after_step=recompute_ratio_after_step,
    )


_FACTORIES: dict[str, Callable[[], TrainerHarness]] = {
    "dapo": _make_dapo_harness,
    "gepo": _make_gepo_harness,
    "gspo": _make_gspo_harness,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("trainer_name", sorted(_FACTORIES.keys()))
async def test_ratio_and_gradient_invariants(trainer_name: str) -> None:
    harness = _FACTORIES[trainer_name]()

    metrics = await harness.run_train_step()

    # (c) Loss must be finite.
    loss_value = metrics.get("policy_loss")
    assert loss_value is not None
    assert math.isfinite(loss_value), f"{trainer_name}: non-finite loss {loss_value}"

    # (a) On-policy first evaluation: mean ratio in [0.99, 1.01].
    onpolicy = harness.onpolicy_ratios()
    mean_ratio = onpolicy.mean().item()
    assert (
        0.99 <= mean_ratio <= 1.01
    ), f"{trainer_name}: on-policy mean ratio {mean_ratio} not close to 1"

    # (b) After one real optimizer step, recomputed ratio must diverge from 1.
    ratio_after = harness.recompute_ratio_after_step()
    assert not torch.allclose(
        ratio_after, torch.ones_like(ratio_after), atol=1e-5
    ), f"{trainer_name}: ratio unchanged after optimizer step"

    # (d) At least one parameter must have a nonzero gradient after train_step.
    nonzero_grad = any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in harness.model.parameters()
    )
    assert nonzero_grad, f"{trainer_name}: no nonzero gradients after train_step"
