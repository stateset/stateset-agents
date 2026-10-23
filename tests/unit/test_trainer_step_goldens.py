"""One native training step per trainer must reproduce its captured golden.

Each case runs a real ``train_step`` (``training_step`` / one-step ``train()``
for the GRPO trainers) on closed-form weights and scripted rollouts, and
compares the step's metrics, parameter delta, pre-/post-clip gradient norms,
learning rates, scheduler positions and step counters with
``tests/unit/goldens/step_goldens.json``.

These goldens are the numerical safety net for the unified RL spine refactor
(docs/superpowers/plans/2026-10-22-unified-rl-spine.md). Regenerate
deliberately with ``scripts/capture_step_goldens.py --only <case>`` and
explain the change in CHANGELOG.md; never regenerate to make a red test green
without understanding why the number moved.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "step_goldens.json").read_text()
)
_spec = importlib.util.spec_from_file_location(
    "capture_step_goldens",
    Path(__file__).resolve().parents[2] / "scripts" / "capture_step_goldens.py",
)
assert _spec is not None and _spec.loader is not None
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)

#: Relative tolerance for every float (fp32 CPU, one thread).
RTOL = 1e-5
#: Absolute floor for values that are zero in exact arithmetic (an on-policy
#: surrogate, a whitened mean) and so are pure fp32 rounding noise.
ATOL = 1e-6
#: Learning rates are closed-form Python floats: compare (almost) exactly.
_EXACT_FLOAT_KEYS = {"lr", "learning_rate", "actor_lr", "critic_lr"}


def _assert_same(got: Any, want: Any, path: str, exact: bool = False) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict), path
        # A trainer may report more metrics after a refactor; it must keep
        # every metric (and value) it reported when the golden was captured.
        missing = sorted(set(want) - set(got))
        assert not missing, f"{path}: missing {missing}"
        for key in want:
            _assert_same(
                got[key], want[key], f"{path}/{key}", exact or key in _EXACT_FLOAT_KEYS
            )
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (path, got, want)
        for index, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_same(g, w, f"{path}[{index}]", exact)
    elif isinstance(want, bool) or want is None or isinstance(want, (int, str)):
        assert type(got) is type(want) and got == want, (path, got, want)
    else:
        assert isinstance(got, float), (path, got)
        tolerance = {"rel_tol": 1e-12, "abs_tol": 0.0} if exact else {}
        assert math.isclose(
            got, want, **(tolerance or {"rel_tol": RTOL, "abs_tol": ATOL})
        ), f"{path}: got {got!r}, golden {want!r}"


def test_every_case_has_a_golden() -> None:
    assert sorted(GOLDENS) == sorted(capture.CASES)


@pytest.mark.parametrize("case", sorted(capture.CASES))
def test_train_step_reproduces_golden(case: str) -> None:
    _assert_same(capture.capture(case), GOLDENS[case], case)


def test_capture_restores_global_torch_state() -> None:
    clip = torch.nn.utils.clip_grad_norm_
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    capture.capture("dapo_k1")
    assert torch.nn.utils.clip_grad_norm_ is clip
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
