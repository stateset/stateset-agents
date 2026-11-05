"""Online DPO raises a judged behaviour's rate on held-out prompts.

The judge is ``DigitReward`` (the fraction of digit characters in the reply,
scored through the shared ``score_response`` path). Three online rounds each
sample eight 6-token replies per training prompt from the current tiny GPT-2
policy, pair the best against the worst and train DPO on those pairs. The
held-out digit rate (mean digit fraction of 32 fixed-seed samples per
held-out prompt, ~0.1 for the untrained policy) must rise by at least 0.1 with
either reference schedule; with the sign-flipped judge it must fall.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("transformers")

import torch  # noqa: E402

from tests.learning._toy import (  # noqa: E402
    HELDOUT_PROMPTS,
    TRAIN_PROMPTS,
    DigitReward,
    make_model,
    make_tokenizer,
    seeds,
    target_mass,
)

pytestmark = [pytest.mark.learning, pytest.mark.timeout(300)]

EXTRA_PROMPTS = [
    "alpha",
    "beta gamma",
    "zz top",
    "quick",
    "number please",
    "hey",
    "go on",
    "and then",
]
ROUNDS = 3


def _messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _run(seed: int, *, sign: float, reference_update: str) -> dict[str, Any]:
    from stateset_agents.training.preference import OnlineDPOConfig, OnlineDPOTrainer

    tokenizer = make_tokenizer()
    model = make_model(seed)
    config = OnlineDPOConfig(
        rounds=ROUNDS,
        samples_per_prompt=8,
        max_new_tokens=6,
        learning_rate=1e-2,
        beta=0.1,
        batch_size=4,
        num_epochs=4,
        lr_scheduler="constant",
        max_length=64,
        max_prompt_length=None,
        device="cpu",
        seed=seed,
        reference_update=reference_update,
        generation_batch_size=32,
    )
    heldout = [_messages(p) for p in HELDOUT_PROMPTS]
    trainer = OnlineDPOTrainer(
        model,
        tokenizer,
        DigitReward(sign),
        [_messages(p) for p in TRAIN_PROMPTS + EXTRA_PROMPTS],
        config,
    )
    # The held-out rate is always judged by the +1 digit reward.
    probe = OnlineDPOTrainer(
        model, tokenizer, DigitReward(1.0), [_messages("probe")], config
    )

    def rate() -> float:
        return probe.evaluate(heldout, samples_per_prompt=32)["eval/mean_score"]

    before, mass_before = rate(), target_mass(model, tokenizer)
    rounds = trainer.train()
    assert [entry.round for entry in rounds] == list(range(ROUNDS))
    assert rounds[0].pairs > 0  # later rounds may saturate (all ties)
    return {
        "before": before,
        "after": rate(),
        "mass_before": mass_before,
        "mass_after": target_mass(trainer.model, tokenizer),
        "pairs": [entry.pairs for entry in rounds],
        "round_scores": [round(float(entry.mean_score or 0.0), 3) for entry in rounds],
    }


@pytest.mark.parametrize("reference_update", ["fixed", "previous_round"])
@pytest.mark.parametrize("seed", seeds())
def test_online_rounds_raise_the_judged_rate(seed: int, reference_update: str) -> None:
    # Observed held-out digit rate (seeds 0/1/2), from ~0.10:
    # fixed 1.00 / 0.22 / 0.88; previous_round 1.00 / 0.48 / 1.00.
    torch.set_num_threads(1)
    result = _run(seed, sign=1.0, reference_update=reference_update)
    assert result["after"] >= result["before"] + 0.1, result


@pytest.mark.parametrize("seed", seeds())
def test_sign_flipped_judge_lowers_the_rate(seed: int) -> None:
    # Observed (seeds 0/1/2): ~0.10 -> 0.00 / 0.00 / 0.00.
    result = _run(seed, sign=-1.0, reference_update="fixed")
    assert result["after"] <= result["before"] - 0.05, result
