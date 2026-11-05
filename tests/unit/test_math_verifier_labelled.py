"""MathAnswerVerifier against a blind-labelled set of chain-of-thought replies.

``tests/fixtures/reward_labelled/math_cot.json`` was written without reading
the verifier: realistic completions in seven model styles (``<think>``
self-corrections, numbered steps ending in ``\\boxed{}``, Minerva's "Final
Answer: ...", GSM8K's ``####``, markdown, plain prose), each labelled
``correct`` when its final answer is committed and equals the reference.
The verifier was tuned on the ``dev`` split only; the gate is on the whole
set (positive class: ``correct``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stateset_agents.rewards.math_verifier import verify_math_answer

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "reward_labelled"
    / "math_cot.json"
)
MIN_PRECISION = 0.98
MIN_RECALL = 0.95


def _items() -> list[dict[str, Any]]:
    return list(json.loads(FIXTURE.read_text(encoding="utf-8"))["items"])


def _scores(items: list[dict[str, Any]]) -> tuple[float, float, list[str]]:
    credited = correct = hits = 0
    misses = []
    for item in items:
        verdict = verify_math_answer(
            item["completion"], item["reference"], question=item["question"]
        )
        labelled = item["label"] == "correct"
        credited += verdict.correct
        correct += labelled
        hits += verdict.correct and labelled
        if verdict.correct != labelled:
            misses.append(f"{item['id']} ({item['split']}, {item['label']}): {verdict}")
    return hits / credited, hits / correct, misses


def test_the_labelled_set_is_large_and_balanced():
    items = _items()
    assert len(items) >= 200
    assert {item["split"] for item in items} == {"dev", "heldout"}
    assert {item["label"] for item in items} == {"correct", "incorrect"}


def test_precision_and_recall_on_the_labelled_set():
    items = _items()
    report = []
    for split in ("dev", "heldout"):
        precision, recall, _ = _scores([i for i in items if i["split"] == split])
        report.append(f"{split}: precision {precision:.3f}, recall {recall:.3f}")
    precision, recall, misses = _scores(items)
    report.append(f"overall: precision {precision:.3f}, recall {recall:.3f}")
    detail = "\n".join(report + misses)
    assert recall >= MIN_RECALL, detail
    assert precision >= MIN_PRECISION, detail
