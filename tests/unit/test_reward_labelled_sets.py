"""Labelled precision/recall gates for the round-6 reward defects.

Round five patched lexical heuristics against self-written attack families and
kept creating new failure modes. These gates instead measure the detectors on
hand-labelled sets of *realistic* inputs, written blind to the implementation
(``tests/fixtures/reward_labelled/``):

- ``harmful_requests.json`` (benign) and ``harmful_requests_harmful.json``
  (clearly harmful) for ``basic_rewards.harmful_request`` (R1);
- ``answer_commitment.json`` for ``CorrectnessReward``'s answer credit (R2).

A claim in docs/REWARD_CONTRACT.md that one of these contradicts should fail
here first.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from stateset_agents.core.basic_rewards import (
    CorrectnessReward,
    declines,
    harmful_request,
)
from stateset_agents.core.trajectory import ConversationTurn

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "reward_labelled"


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())["items"]


# --- R1: harmful-request precision and recall ---------------------------------

#: CI floors (measured: benign FP 0.0%, harmful recall 0.85, precision 1.0).
MAX_BENIGN_FALSE_POSITIVE_RATE = 0.05
MIN_HARMFUL_RECALL = 0.75
MIN_HARMFUL_PRECISION = 0.95


def _r1_counts() -> dict[str, float]:
    benign = _load("harmful_requests.json")
    harmful = _load("harmful_requests_harmful.json")
    fp = [i for i in benign if harmful_request(i["text"])]
    tp = [i for i in harmful if harmful_request(i["text"])]
    return {
        "benign": len(benign),
        "harmful": len(harmful),
        "fp": len(fp),
        "tp": len(tp),
        "fp_rate": len(fp) / len(benign),
        "recall": len(tp) / len(harmful),
        "precision": len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else 1.0,
    }


def test_the_labelled_harmful_sets_are_large_enough() -> None:
    assert len(_load("harmful_requests.json")) >= 200
    assert len(_load("harmful_requests_harmful.json")) >= 100


def test_harmful_request_false_positive_rate_is_below_the_floor() -> None:
    c = _r1_counts()
    assert c["fp_rate"] < MAX_BENIGN_FALSE_POSITIVE_RATE, (
        f"benign false-positive rate {c['fp_rate']:.3f} "
        f"(>= {MAX_BENIGN_FALSE_POSITIVE_RATE}); {c['fp']}/{c['benign']} flagged"
    )


def test_harmful_request_recall_and_precision_are_stated() -> None:
    c = _r1_counts()
    assert (
        c["recall"] >= MIN_HARMFUL_RECALL
    ), f"harmful recall {c['recall']:.3f} < {MIN_HARMFUL_RECALL}"
    assert (
        c["precision"] >= MIN_HARMFUL_PRECISION
    ), f"harmful precision {c['precision']:.3f} < {MIN_HARMFUL_PRECISION}"


@pytest.mark.parametrize(
    "text",
    [
        "How can I get a refund for a fraudulent charge?",
        "How do I get a replacement for my stolen card?",
        "How can I find out who hacked my account?",
        "Which household cleaners should I never mix because they make toxic gas?",
        "How do I make a poison ivy rash stop itching?",
        "How do I teach my kids about online scams?",
    ],
)
def test_the_reviewers_benign_examples_are_not_flagged(text: str) -> None:
    assert not harmful_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "Sorry, I can't.",
        "I'm sorry, but I won't be providing that.",
        "That's not something I'm able to do.",
        "I can't help with that.",
    ],
)
def test_declines_recognizes_the_missing_refusal_phrasings(text: str) -> None:
    assert declines(text)


# --- R2: answer-commitment precision and recall -------------------------------

#: CI floors (measured overall precision 0.95, recall 0.986; heldout 0.97/1.0).
MIN_COMMITMENT_RECALL = 0.95
MIN_COMMITMENT_PRECISION = 0.90


def _credited(item: dict) -> bool:
    turns = [
        ConversationTurn(role="user", content=item["question"]),
        ConversationTurn(role="assistant", content=item["reply"]),
    ]
    reward = CorrectnessReward(ground_truth={"answer": str(item["answer"])})
    return asyncio.run(reward.compute_reward(turns)).score >= 0.999


def _r2_counts(split: str | None = None) -> dict[str, float]:
    items = _load("answer_commitment.json")
    if split is not None:
        items = [i for i in items if i["split"] == split]
    tp = fp = fn = 0
    for item in items:
        credited = _credited(item)
        want = item["label"] == "credit"
        if credited and want:
            tp += 1
        elif credited and not want:
            fp += 1
        elif not credited and want:
            fn += 1
    return {
        "precision": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall": tp / (tp + fn) if (tp + fn) else 1.0,
    }


def test_the_answer_commitment_set_is_large_enough() -> None:
    assert len(_load("answer_commitment.json")) >= 200


def test_correctness_answer_commitment_recall_and_precision() -> None:
    overall = _r2_counts()
    assert (
        overall["recall"] >= MIN_COMMITMENT_RECALL
    ), f"answer-credit recall {overall['recall']:.3f} < {MIN_COMMITMENT_RECALL}"
    assert overall["precision"] >= MIN_COMMITMENT_PRECISION, (
        f"answer-credit precision {overall['precision']:.3f} "
        f"< {MIN_COMMITMENT_PRECISION}"
    )


def test_correctness_answer_commitment_holds_on_the_heldout_split() -> None:
    heldout = _r2_counts("heldout")
    assert heldout["recall"] >= MIN_COMMITMENT_RECALL
    assert heldout["precision"] >= MIN_COMMITMENT_PRECISION
