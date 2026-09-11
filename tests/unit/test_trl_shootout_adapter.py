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
    require_supported_kwargs,
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


def test_required_objective_kwargs_fail_closed_on_trl_drift() -> None:
    class OldConfig:
        def __init__(self, epsilon: float) -> None:
            pass

    with pytest.raises(RuntimeError, match="importance_sampling_level"):
        require_supported_kwargs(
            OldConfig,
            {"epsilon": 3e-4, "importance_sampling_level": "sequence"},
            {"epsilon", "importance_sampling_level"},
        )


def test_pinned_trl_can_express_matched_gspo_objective(tmp_path: Path) -> None:
    trl = pytest.importorskip("trl")
    from trl import GRPOConfig

    values = {
        "importance_sampling_level": "sequence",
        "epsilon": 3e-4,
        "epsilon_high": 4e-4,
        "loss_type": "grpo",
        "scale_rewards": "group",
    }
    selected = require_supported_kwargs(GRPOConfig, values, set(values))
    config = GRPOConfig(
        output_dir=str(tmp_path), report_to=[], use_cpu=True, bf16=False, **selected
    )
    assert trl.__version__ == "1.12.0"
    assert config.importance_sampling_level == "sequence"
    assert config.epsilon == pytest.approx(3e-4)
    assert config.epsilon_high == pytest.approx(4e-4)
    assert config.loss_type == "grpo"


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
