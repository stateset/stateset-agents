"""One native training step per trainer must reproduce its captured golden.

Each case runs a real ``train_step`` (``training_step`` / one-step ``train()``
for the GRPO trainers) on closed-form weights and scripted rollouts, and
compares the step's metrics, parameter delta, pre-/post-clip gradient norms,
learning rates, scheduler positions and step counters with
``tests/unit/goldens/step_goldens.json``.

These goldens are the numerical safety net for the unified RL spine refactor
(docs/superpowers/plans/2026-10-22-unified-rl-spine.md). Regenerate
deliberately with ``scripts/capture_step_goldens.py`` and explain the change
in CHANGELOG.md; never regenerate to make a red test green without
understanding why the number moved.

Tolerance policy. The JSON records the capture environment (python
major.minor, torch, transformers, numpy, platform system and machine):

* **strict** -- same environment as the capture: every float within rtol
  1e-5 (atol 1e-6 for values that are zero in exact arithmetic and so pure
  fp32 rounding noise). This is the refactor-equivalence guard a spine slice
  runs locally.
* **portable** -- any other environment (CI's torch versions, CPUs, BLAS
  builds, Windows): float drift between environments is real, so floats get
  rtol 5e-3 with absolute floors scaled to the case -- 1e-3 x the parameter
  delta norm for the sign-sensitive delta projection, 1e-3 x the case's
  largest gradient norm for gradient norms, 1e-4 for step metrics. That still
  catches real breakage (a gradient scaled by 1.05, a dropped prompt group, a
  skipped or extra update pass: see the mutation tests below).

In both modes learning rates, step counters, optimizer-step counts, metric
names, strings and flags must match exactly.
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

_spec = importlib.util.spec_from_file_location(
    "capture_step_goldens",
    Path(__file__).resolve().parents[2] / "scripts" / "capture_step_goldens.py",
)
assert _spec is not None and _spec.loader is not None
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)

_RAW = json.loads((Path(__file__).parent / "goldens" / "step_goldens.json").read_text())
RECORDED_ENVIRONMENT: dict[str, str] = _RAW[capture.ENVIRONMENT_KEY]
GOLDENS: dict[str, Any] = {
    case: value for case, value in _RAW.items() if case != capture.ENVIRONMENT_KEY
}

STRICT, PORTABLE = "strict", "portable"
#: Learning rates are closed-form Python floats: compare (almost) exactly.
_EXACT_FLOAT_KEYS = {"lr", "learning_rate", "actor_lr", "critic_lr"}
_NORM_KEYS = {"clip_norms", "grad_norms"}


def comparison_mode(recorded: dict[str, str], current: dict[str, str]) -> str:
    """``strict`` in the capture environment, ``portable`` anywhere else."""
    return STRICT if recorded == current else PORTABLE


def mismatches(got: dict[str, Any], want: dict[str, Any], mode: str) -> list[str]:
    """Every difference between a captured case and its golden under ``mode``."""
    delta_norm = float(want["param_delta_norm"])
    norm_scale = max(
        [*want["clip_norms"], *(n for v in want["grad_norms"].values() for n in v)],
        default=1.0,
    )
    found: list[str] = []

    def float_tolerance(top: str, key: str) -> dict[str, float]:
        if top == "lr" or key in _EXACT_FLOAT_KEYS:
            return {"rel_tol": 1e-12, "abs_tol": 0.0}
        if mode == STRICT:
            return {"rel_tol": 1e-5, "abs_tol": 1e-6}
        if top == "param_delta_probe":
            return {"rel_tol": 5e-3, "abs_tol": 1e-3 * delta_norm}
        if top in _NORM_KEYS:
            return {"rel_tol": 5e-3, "abs_tol": 1e-3 * norm_scale}
        if top == "metrics":
            return {"rel_tol": 5e-3, "abs_tol": 1e-4}
        return {"rel_tol": 5e-3, "abs_tol": 0.0}

    def visit(g: Any, w: Any, path: str, top: str, key: str) -> None:
        if isinstance(w, dict):
            if not isinstance(g, dict):
                found.append(f"{path}: got {g!r}, golden is a mapping")
                return
            # A trainer may report more metrics after a refactor; it must keep
            # every metric (and value) it reported when the golden was captured.
            for name in sorted(set(w) - set(g)):
                found.append(f"{path}/{name}: missing")
            for name in w:
                if name in g:
                    visit(g[name], w[name], f"{path}/{name}", top or name, name)
        elif isinstance(w, list):
            if not isinstance(g, list) or len(g) != len(w):
                found.append(f"{path}: got {g!r}, golden {w!r}")
                return
            for index, (a, b) in enumerate(zip(g, w, strict=True)):
                visit(a, b, f"{path}[{index}]", top, key)
        elif isinstance(w, bool) or w is None or isinstance(w, (int, str)):
            if type(g) is not type(w) or g != w:
                found.append(f"{path}: got {g!r}, golden {w!r}")
        elif not isinstance(g, float) or not math.isclose(
            g, w, **float_tolerance(top, key)
        ):
            found.append(f"{path}: got {g!r}, golden {w!r} ({mode})")

    for name in want:
        if name not in got:
            found.append(f"{name}: missing")
        else:
            visit(got[name], want[name], name, name, name)
    return found


MODE = comparison_mode(RECORDED_ENVIRONMENT, capture.environment_fingerprint())


def test_every_case_has_a_golden() -> None:
    assert sorted(GOLDENS) == sorted(capture.CASES)


@pytest.mark.parametrize("case", sorted(capture.CASES))
def test_train_step_reproduces_golden(case: str) -> None:
    found = mismatches(capture.capture(case), GOLDENS[case], MODE)
    assert not found, "\n".join([f"{case} ({MODE} comparison):", *found])


def test_capture_restores_global_torch_state() -> None:
    clip = torch.nn.utils.clip_grad_norm_
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    capture.capture("dapo_k1")
    assert torch.nn.utils.clip_grad_norm_ is clip
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic


# --------------------------------------------------------------------------
# the tolerance policy itself
# --------------------------------------------------------------------------


def test_any_fingerprint_difference_selects_portable_mode() -> None:
    recorded = dict(RECORDED_ENVIRONMENT)
    assert comparison_mode(recorded, dict(recorded)) == STRICT
    for key in recorded:
        assert comparison_mode(recorded, {**recorded, key: "other"}) == PORTABLE


def _drifted(value: Any, rel: float) -> Any:
    """``value`` with every non-learning-rate float scaled by ``1 + rel``."""
    if isinstance(value, dict):
        return {
            k: v if k in _EXACT_FLOAT_KEYS else _drifted(v, rel)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_drifted(v, rel) for v in value]
    if isinstance(value, float):
        return value * (1 + rel)
    return value


@pytest.mark.parametrize("case", sorted(capture.CASES))
def test_portable_mode_absorbs_cross_environment_drift(case: str) -> None:
    want = GOLDENS[case]
    assert mismatches(_drifted(want, 1e-3), want, PORTABLE) == []
    # A probe that cancels to ~0 may move by a small fraction of the delta.
    nudged = {
        **want,
        "param_delta_probe": want["param_delta_probe"]
        + 5e-4 * want["param_delta_norm"],
    }
    assert mismatches(nudged, want, PORTABLE) == []
    assert mismatches(_drifted(want, 1e-3), want, STRICT) != []


def _scale_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    original = capture.deterministic_weights

    def weights_with_scaled_gradients(module: Any, seed: int = 0) -> Any:
        module = original(module, seed)
        for parameter in module.parameters():
            parameter.register_hook(lambda gradient: gradient * 1.05)
        return module

    monkeypatch.setattr(capture, "deterministic_weights", weights_with_scaled_gradients)


def _drop_a_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture, "PROMPTS", capture.PROMPTS[:1])


def _skip_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    from stateset_agents.training.spine import update

    original = update.OptimizerStep.__call__
    calls = {"n": 0}

    def skip_second_pass(self: Any, **kwargs: Any) -> bool:
        calls["n"] += 1
        if calls["n"] == 2:
            self.discard_gradients()
            return False
        return bool(original(self, **kwargs))

    monkeypatch.setattr(update.OptimizerStep, "__call__", skip_second_pass)


@pytest.mark.parametrize(
    ("mutation", "cases"),
    [
        (_scale_gradients, ["dapo_k1", "multi_turn_grpo_k1", "vapo_k1", "gspo_k1"]),
        (_drop_a_group, ["dapo_k1", "gepo_k1", "gspo_token_k1", "multi_turn_grpo_k1"]),
        (_skip_a_pass, ["dapo_k2", "gspo_k2", "multi_turn_grpo_k2", "vapo_k2"]),
    ],
    ids=["gradient_x1.05", "dropped_group", "skipped_pass"],
)
def test_portable_mode_still_catches_a_wrong_refactor(
    monkeypatch: pytest.MonkeyPatch, mutation: Any, cases: list[str]
) -> None:
    for case in cases:
        with monkeypatch.context() as patch:
            mutation(patch)
            found = mismatches(capture.capture(case), GOLDENS[case], PORTABLE)
        assert found, f"portable comparison missed {mutation.__name__} in {case}"
