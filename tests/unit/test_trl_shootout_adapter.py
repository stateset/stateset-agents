"""Unit tests for the independent upstream-TRL shootout adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.adapters.trl_grpo import (
    GSM8KTask,
    canonical_digest,
    completion_text,
    supported_kwargs,
)


def test_canonical_digest_is_stable_across_key_order() -> None:
    first = {"learning_rate": 5e-6, "max_steps": 4}
    second = {"max_steps": 4, "learning_rate": 5e-6}
    expected = hashlib.sha256(
        json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert canonical_digest(first) == canonical_digest(second) == expected


def test_supported_kwargs_filters_versioned_api() -> None:
    class CurrentConfig:
        def __init__(self, max_steps: int, num_generations: int) -> None:
            pass

    assert supported_kwargs(
        CurrentConfig,
        {"max_steps": 4, "num_generations": 8, "removed_option": True},
    ) == {"max_steps": 4, "num_generations": 8}


def test_completion_text_supports_current_trl_shapes() -> None:
    assert completion_text("answer") == "answer"
    assert completion_text([{"role": "assistant", "content": "answer"}]) == "answer"
    with pytest.raises(TypeError, match="unsupported TRL completion"):
        completion_text({"content": "answer"})


def test_gsm8k_task_is_self_contained_and_scores_answers() -> None:
    from stateset_agents.data.gsm8k import GSM8KExample

    example = GSM8KExample(
        question="What is one plus one?",
        answer_text="One plus one is two. #### 2",
        gold_answer=2.0,
    )
    task = GSM8KTask()

    assert task.format_prompt(example) == (
        "Solve this step by step.\n\nWhat is one plus one?\n\nAnswer:"
    )
    assert task.score_response(example, "The answer is 2") == (1.0, True)
    assert task.score_response(example, "I do not know") == (0.0, False)


def test_declared_and_locked_trl_versions_support_grpo() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    lock = (root / "requirements-dev-lock.txt").read_text(encoding="utf-8")
    assert pyproject.count('"trl>=0.14.0,<2.0.0"') == 3
    assert "trl==1.9.1" in lock
