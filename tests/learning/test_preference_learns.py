"""The maintained preference trainers learn a digits-over-letters preference.

Training pairs prefer a random 4-digit reply over a random 4-letter reply;
held-out pairs use different prompts and strings. The policy's log-prob
margin ``log π(digits) - log π(letters)`` must grow under every method and
shrink when DPO's preference labels are swapped. A Bradley-Terry reward model
trained on the same pairs must rank held-out digits above letters (and below
them when its labels are swapped).
"""

from __future__ import annotations

import random
from typing import Any

import pytest

pytest.importorskip("transformers")

from tests.learning._toy import (  # noqa: E402
    HELDOUT_PROMPTS,
    TRAIN_PROMPTS,
    make_model,
    make_tokenizer,
    seeds,
)

pytestmark = pytest.mark.learning

STEPS = 10
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _triples(
    prompts: list[str], rng: random.Random, repeats: int
) -> list[tuple[str, str, str]]:
    return [
        (
            prompt,
            "".join(rng.choice(DIGITS) for _ in range(4)),
            "".join(rng.choice(LETTERS) for _ in range(4)),
        )
        for prompt in prompts
        for _ in range(repeats)
    ]


def _dataset(triples: list[tuple[str, str, str]], *, prefer_digits: bool) -> Any:
    from stateset_agents.training.preference import PreferenceDataset

    return PreferenceDataset.from_records(
        {
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": digits if prefer_digits else letters,
            "rejected": letters if prefer_digits else digits,
        }
        for prompt, digits, letters in triples
    )


HELDOUT = _triples(HELDOUT_PROMPTS, random.Random(1000), 2)


def _digit_margin(trainer: Any) -> float:
    margins = []
    for prompt, digits, letters in HELDOUT:
        chosen, rejected = trainer.sequence_logps(
            [{"role": "user", "content": prompt}], [digits, letters]
        )
        margins.append(chosen - rejected)
    return sum(margins) / len(margins)


def _train_policy(
    seed: int, method: str, *, prefer_digits: bool = True
) -> tuple[float, float]:
    from stateset_agents.training.preference import PreferenceConfig, PreferenceTrainer

    train = _dataset(
        _triples(TRAIN_PROMPTS, random.Random(seed), STEPS), prefer_digits=prefer_digits
    )
    config = PreferenceConfig(
        method=method,
        learning_rate=1e-2,
        beta=0.1,
        batch_size=len(TRAIN_PROMPTS),
        max_steps=STEPS,
        max_length=64,
        max_prompt_length=None,
        device="cpu",
        seed=seed,
    )
    trainer = PreferenceTrainer(make_model(seed), make_tokenizer(), config, train)
    before = _digit_margin(trainer)
    summary = trainer.train()
    assert summary.global_step == STEPS and summary.skipped_steps == 0
    return before, _digit_margin(trainer)


@pytest.mark.parametrize("seed", seeds())
def test_dpo_increases_chosen_over_rejected_margin(seed: int) -> None:
    # Observed (seeds 0/1/2): margin 0.08 / -0.04 / 0.10 -> 7.4 / 7.7 / 9.5.
    before, after = _train_policy(seed, "dpo")
    assert after > before + 3.0, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_dpo_swapped_preference_decreases_margin(seed: int) -> None:
    # Observed (seeds 0/1/2): margin 0.08 / -0.04 / 0.10 -> -5.7 / -6.7 / -5.1.
    before, after = _train_policy(seed, "dpo", prefer_digits=False)
    assert after < before - 3.0, (before, after)


@pytest.mark.parametrize("method", ["ipo", "simpo", "kto", "orpo"])
@pytest.mark.parametrize("seed", seeds())
def test_other_methods_increase_the_margin(seed: int, method: str) -> None:
    # Observed after (seeds 0/1/2): ipo 7.6/7.9/9.6, simpo 7.5/7.8/9.1,
    # kto 5.0/4.2/7.2, orpo 9.8/9.5/10.6 (from 0.08 / -0.04 / 0.10).
    before, after = _train_policy(seed, method)
    assert after > before + 2.0, (method, before, after)


def _train_reward_model(seed: int, *, prefer_digits: bool) -> dict[str, float]:
    from stateset_agents.training.preference import (
        RewardModelConfig,
        RewardModelTrainer,
    )

    train = _dataset(
        _triples(TRAIN_PROMPTS, random.Random(seed), STEPS), prefer_digits=prefer_digits
    )
    heldout = _dataset(HELDOUT, prefer_digits=True)
    config = RewardModelConfig(
        learning_rate=1e-2,
        batch_size=len(TRAIN_PROMPTS),
        max_steps=STEPS,
        max_length=64,
        max_prompt_length=None,
        device="cpu",
        seed=seed,
    )
    trainer = RewardModelTrainer(
        make_model(seed).transformer, make_tokenizer(), config, train, heldout
    )
    summary = trainer.train()
    assert summary.eval_metrics is not None
    return summary.eval_metrics


@pytest.mark.parametrize("seed", seeds())
def test_bradley_terry_reward_model_ranks_chosen_above_rejected(seed: int) -> None:
    # Observed (seeds 0/1/2): held-out accuracy 1.0 and AUC 1.0.
    metrics = _train_reward_model(seed, prefer_digits=True)
    assert metrics["eval/accuracy"] >= 0.875, metrics
    assert metrics["eval/auc"] >= 0.9, metrics


@pytest.mark.parametrize("seed", seeds())
def test_bradley_terry_swapped_labels_invert_the_ranking(seed: int) -> None:
    # Observed (seeds 0/1/2): accuracy 0.0; AUC 0.06 / 0.0 / 0.0.
    metrics = _train_reward_model(seed, prefer_digits=False)
    assert metrics["eval/accuracy"] <= 0.125, metrics
    assert metrics["eval/auc"] <= 0.2, metrics
