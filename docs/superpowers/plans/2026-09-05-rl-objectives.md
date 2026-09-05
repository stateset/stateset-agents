# RL Objectives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One declarative, externally verified policy-optimisation objective library (`stateset_agents/training/objectives.py`) that every native trainer routes through.

**Architecture:** `objectives.py` composes the existing pure-tensor primitives in `rl_losses.py` into a frozen `PolicyObjective` (advantage × ratio × clip × aggregation × KL) plus two functions, `compute_advantages` and `policy_loss`. Eleven named presets cover GRPO, Dr. GRPO, BNPO, DAPO, GSPO, GSPO-token, GEPO, RLOO, REINFORCE++-baseline, CISPO, and PPO. Verification is three-layered (loop references, TRL 1.12 pin, Hypothesis properties), then each trainer's private objective code is replaced by a call into the library under golden regression pins.

**Tech Stack:** Python 3.10+, torch (lazy via `trainer_utils.get_torch`), pytest + pytest-asyncio, hypothesis, trl 1.12 (project venv `.venv`, Python 3.12) for the external pin only.

**Spec:** `docs/superpowers/specs/2026-09-05-rl-objectives-design.md`

## Global Constraints

- Run every command with the project venv: `.venv/bin/python -m pytest ... -p no:cacheprovider -o addopts="" -q`. The `-o addopts=""` drops xdist/timeout so single files run fast and deterministically.
- `objectives.py` must import without torch (`from stateset_agents.training import objectives` in a torch-free process must not raise). Fetch torch inside functions via `_t()` exactly like `rl_losses.py`.
- No trainer public class, config field, or `train_step` signature changes.
- Numeric parity: every trainer golden matches to `atol=1e-6` except PPO (intended k3 KL change, golden regenerated in the same commit) and GSPO-token (gradient golden, not loss value — see Task 4).
- Lint gates after every task: `.venv/bin/ruff check stateset_agents tests scripts && .venv/bin/black --check stateset_agents tests scripts && .venv/bin/isort --check-only stateset_agents tests scripts`. Type gate before the PR: `.venv/bin/python scripts/check_types.py --all`.
- Commit after every task with a conventional-commit subject, ending the body with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01ATQWoEEmtLwgkpf1jddcPs
  ```
- Branch: `feat/rl-objectives` (already created, spec committed). Do not push.

---

## File map

| File | Responsibility |
|---|---|
| `stateset_agents/training/objectives.py` (new) | `PolicyObjective`, `PolicyLossResult`, `OBJECTIVES`, `compute_advantages`, `policy_loss` |
| `tests/unit/objective_reference.py` (new) | Pure-Python loop reference of every estimator/ratio/surrogate/aggregate/KL |
| `tests/unit/test_objectives.py` (new) | Validation, presets, parity with the loop reference |
| `tests/unit/test_objectives_properties.py` (new) | Hypothesis invariants |
| `tests/unit/test_objectives_trl_pin.py` (new) | External pin against TRL 1.12 |
| `scripts/capture_objective_goldens.py` (new) | Writes `tests/unit/goldens/objective_goldens.json` |
| `tests/unit/goldens/objective_goldens.json` (new) | Committed golden values |
| `tests/unit/test_objective_goldens.py` (new) | Asserts trainers reproduce goldens |
| `stateset_agents/training/{dapo,vapo,gspo,gspo_token,gepo,ppo}_trainer.py`, `loss_computation.py` | Objective assembly replaced by library calls |
| `stateset_agents/training/_registry.py` | Lazy exports |
| `docs/OBJECTIVES.md` (new), `docs/ADVANCED_RL_ALGORITHMS.md`, `docs/ARCHITECTURE.md`, `docs/COMPARISONS.md`, `README.md`, `CHANGELOG.md`, `contracts/component_maturity_v1.json` | Docs and evidence text |

---

### Task 1: `objectives.py` core with loop-reference parity tests

**Files:**
- Create: `stateset_agents/training/objectives.py`
- Create: `tests/unit/objective_reference.py`
- Create: `tests/unit/test_objectives.py`

**Interfaces:**
- Produces:
  - `PolicyObjective(name, advantage="group_norm", advantage_eps=1e-8, ratio="token", ratio_clamp=20.0, clip="clipped", clip_low=0.2, clip_high=0.2, delta=None, is_cap=5.0, aggregate="seq_mean", max_completion_length=None, kl="none", kl_coef=0.0, kl_bias_correction=False, entropy_coef=0.0)` frozen dataclass with `.with_(**changes)`.
  - `compute_advantages(rewards: Tensor[N], group_ids: Tensor[N], objective) -> Tensor[N]`
  - `policy_loss(*, logp_cur: Tensor[N,T], mask: Tensor[N,T], advantages: Tensor[N] | Tensor[N,T], objective, logp_old=None, logp_ref=None, group_ids=None, kl=None, entropy=None) -> PolicyLossResult(loss, ratio, metrics)`
  - `OBJECTIVES: Mapping[str, PolicyObjective]` with keys `grpo, dr_grpo, bnpo, dapo, gspo, gspo_token, gepo, rloo, reinforce_pp_baseline, cispo, ppo`.

- [ ] **Step 1: Write the loop reference module**

```python
# tests/unit/objective_reference.py
"""Explicit-loop reference implementations of every objective component.

Deliberately naive: Python lists of floats, no broadcasting, no torch. The
production code in ``stateset_agents.training.objectives`` must agree with
these to 1e-6. Keep this file boring — it is the oracle.
"""

from __future__ import annotations

import math


def _std0(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def ref_advantages(
    rewards: list[float], groups: list[int], kind: str, eps: float
) -> list[float]:
    n = len(rewards)
    out = [0.0] * n
    ids = sorted(set(groups))
    for gid in ids:
        idx = [i for i in range(n) if groups[i] == gid]
        rg = [rewards[i] for i in idx]
        if len(rg) <= 1:
            continue  # group of one -> advantage 0
        mean = sum(rg) / len(rg)
        if kind == "leave_one_out":
            for i, r in zip(idx, rg):
                out[i] = r - (sum(rg) - r) / (len(rg) - 1)
            continue
        centred = [r - mean for r in rg]
        if kind == "group_mean":
            for i, a in zip(idx, centred):
                out[i] = a
        elif kind == "group_norm":
            s = _std0(centred)
            if not math.isfinite(s) or s <= eps:
                continue
            for i, a in zip(idx, centred):
                out[i] = a / (s + eps)
        elif kind == "batch_norm":
            for i, a in zip(idx, centred):
                out[i] = a
        else:
            raise ValueError(kind)
    if kind == "batch_norm":
        s = _std0(rewards)
        if n <= 1 or not math.isfinite(s) or s <= eps:
            return [0.0] * n
        out = [a / (s + eps) for a in out]
    return out


def _seq_len(mask_row: list[float]) -> float:
    return max(sum(mask_row), 1.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ref_ratio(
    kind: str,
    logp_cur: list[list[float]],
    logp_old,  # list[list[float]] per-token, or list[float] sums
    mask: list[list[float]],
    groups: list[int] | None,
    clamp: float,
) -> list[list[float]]:
    """Return an [N][T] table (sequence-level ratios are repeated over T)."""
    n, t = len(logp_cur), len(logp_cur[0])
    per_token_old = isinstance(logp_old[0], list)
    if kind == "token":
        assert per_token_old
        return [
            [math.exp(_clamp(logp_cur[i][j] - logp_old[i][j], -clamp, clamp)) for j in range(t)]
            for i in range(n)
        ]
    if kind in ("sequence", "sequence_token"):
        out = []
        for i in range(n):
            if per_token_old:
                s = sum((logp_cur[i][j] - logp_old[i][j]) * mask[i][j] for j in range(t))
            else:
                s = sum(logp_cur[i][j] * mask[i][j] for j in range(t)) - logp_old[i]
            r = math.exp(_clamp(s / _seq_len(mask[i]), -clamp, clamp))
            out.append([r] * t)
        return out
    if kind == "group_expectation":
        assert not per_token_old and groups is not None
        out = [[0.0] * t for _ in range(n)]
        for gid in sorted(set(groups)):
            idx = [i for i in range(n) if groups[i] == gid]
            q = [logp_old[i] for i in idx]
            m2 = max(2 * x for x in q)
            m1 = max(q)
            lse2 = m2 + math.log(sum(math.exp(2 * x - m2) for x in q))
            lse1 = m1 + math.log(sum(math.exp(x - m1) for x in q))
            log_e = lse2 - lse1
            for i in idx:
                s = sum(logp_cur[i][j] * mask[i][j] for j in range(t))
                out[i] = [math.exp(_clamp(s - log_e, -clamp, clamp))] * t
        return out
    raise ValueError(kind)


def ref_surrogate(
    clip: str,
    ratio: list[list[float]],
    adv: list[list[float]],
    logp_cur: list[list[float]],
    clip_low: float,
    clip_high: float,
    delta: float | None,
    is_cap: float,
) -> list[list[float]]:
    n, t = len(ratio), len(ratio[0])
    out = [[0.0] * t for _ in range(n)]
    for i in range(n):
        for j in range(t):
            r, a = ratio[i][j], adv[i][j]
            if clip == "clipped":
                r1 = r if delta is None else min(r, delta)
                c = _clamp(r, 1 - clip_low, 1 + clip_high)
                out[i][j] = -min(r1 * a, c * a)
            elif clip == "cispo":
                out[i][j] = -min(r, is_cap) * a * logp_cur[i][j]
            elif clip == "none":
                out[i][j] = -r * a
            else:
                raise ValueError(clip)
    return out


def ref_k3(cur: float, ref: float) -> float:
    d = ref - cur
    return math.exp(d) - d - 1.0


def ref_aggregate(
    kind: str, loss: list[list[float]], mask: list[list[float]], max_len: int | None
) -> float:
    n, t = len(loss), len(loss[0])
    if kind == "seq_mean":
        rows = []
        for i in range(n):
            rows.append(sum(loss[i][j] * mask[i][j] for j in range(t)) / _seq_len(mask[i]))
        return sum(rows) / n
    total = sum(loss[i][j] * mask[i][j] for i in range(n) for j in range(t))
    if kind == "token_mean":
        return total / max(sum(mask[i][j] for i in range(n) for j in range(t)), 1.0)
    if kind == "seq_sum_const":
        assert max_len
        return total / (n * max_len)
    raise ValueError(kind)


def ref_policy_loss(
    obj,
    logp_cur: list[list[float]],
    mask: list[list[float]],
    advantages,  # list[float] or list[list[float]]
    logp_old=None,
    logp_ref=None,
    groups: list[int] | None = None,
    kl_ext: list[list[float]] | None = None,
    entropy: list[list[float]] | None = None,
) -> float:
    n, t = len(logp_cur), len(logp_cur[0])
    if logp_old is None:
        logp_old = [row[:] for row in logp_cur]
    ratio = ref_ratio(obj.ratio, logp_cur, logp_old, mask, groups, obj.ratio_clamp)
    if isinstance(advantages[0], list):
        adv = advantages
    else:
        adv = [[a] * t for a in advantages]
    per_token = ref_surrogate(
        obj.clip, ratio, adv, logp_cur, obj.clip_low, obj.clip_high, obj.delta, obj.is_cap
    )
    if obj.kl_coef > 0 and obj.kl == "k3_token":
        for i in range(n):
            for j in range(t):
                k = ref_k3(logp_cur[i][j], logp_ref[i][j])
                if obj.kl_bias_correction:
                    k *= ratio[i][j]
                per_token[i][j] += obj.kl_coef * k
    if obj.kl_coef > 0 and obj.kl == "external":
        for i in range(n):
            for j in range(t):
                per_token[i][j] += obj.kl_coef * kl_ext[i][j]
    loss = ref_aggregate(obj.aggregate, per_token, mask, obj.max_completion_length)
    if obj.kl_coef > 0 and obj.kl == "k3_sequence":
        rows = []
        for i in range(n):
            length = _seq_len(mask[i])
            cur = sum(logp_cur[i][j] * mask[i][j] for j in range(t)) / length
            ref = (
                sum(logp_ref[i][j] * mask[i][j] for j in range(t)) / length
                if isinstance(logp_ref[0], list)
                else logp_ref[i] / length
            )
            rows.append(ref_k3(cur, ref))
        loss += obj.kl_coef * sum(rows) / n
    if obj.entropy_coef > 0 and entropy is not None:
        num = sum(entropy[i][j] * mask[i][j] for i in range(n) for j in range(t))
        den = max(sum(mask[i][j] for i in range(n) for j in range(t)), 1.0)
        loss -= obj.entropy_coef * num / den
    return loss
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_objectives.py
"""PolicyObjective validation, presets, and parity with the loop reference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import objectives as O  # noqa: E402
from tests.unit import objective_reference as R  # noqa: E402

PRESETS = sorted(O.OBJECTIVES)


def _fixture(seed: int, n: int = 6, t: int = 7, groups: int = 2):
    g = torch.Generator().manual_seed(seed)
    logp_cur = -torch.rand(n, t, generator=g) * 3
    logp_old = logp_cur + 0.05 * torch.randn(n, t, generator=g)
    logp_ref = logp_cur + 0.1 * torch.randn(n, t, generator=g)
    mask = torch.ones(n, t)
    mask[0, 5:] = 0
    mask[1, 2:] = 0
    mask[2, :] = 0  # empty row
    group_ids = torch.arange(n) % groups
    rewards = torch.rand(n, generator=g)
    rewards[group_ids == 1] = 0.5  # one constant-reward group
    return logp_cur, logp_old, logp_ref, mask, group_ids, rewards


def _lists(x):
    return x.detach().tolist()


# --- construction ----------------------------------------------------------


def test_presets_exist_and_are_frozen():
    assert PRESETS == sorted(
        [
            "grpo", "dr_grpo", "bnpo", "dapo", "gspo", "gspo_token", "gepo",
            "rloo", "reinforce_pp_baseline", "cispo", "ppo",
        ]
    )
    with pytest.raises(Exception):
        O.OBJECTIVES["grpo"].clip_low = 0.5  # type: ignore[misc]
    with pytest.raises(TypeError):
        O.OBJECTIVES["grpo"] = None  # type: ignore[index]


def test_with_returns_modified_copy():
    base = O.OBJECTIVES["grpo"]
    new = base.with_(clip_high=0.28, name="mine")
    assert new.clip_high == 0.28 and new.name == "mine"
    assert base.clip_high == 0.2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"advantage": "bogus"}, "advantage"),
        ({"ratio": "bogus"}, "ratio"),
        ({"clip": "bogus"}, "clip"),
        ({"aggregate": "bogus"}, "aggregate"),
        ({"kl": "bogus"}, "kl"),
        ({"clip_low": -0.1}, "clip_low"),
        ({"clip_high": -0.1}, "clip_high"),
        ({"is_cap": 0.0}, "is_cap"),
        ({"ratio_clamp": 0.0}, "ratio_clamp"),
        ({"delta": 0.0}, "delta"),
        ({"kl": "none", "kl_coef": 0.1}, "kl_coef"),
        ({"kl": "k3_sequence", "kl_bias_correction": True}, "kl_bias_correction"),
        ({"max_completion_length": 0}, "max_completion_length"),
    ],
)
def test_invalid_objectives_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        O.PolicyObjective(name="x", **kwargs)


# --- advantages ------------------------------------------------------------


@pytest.mark.parametrize("kind", ["group_norm", "group_mean", "leave_one_out", "batch_norm"])
def test_compute_advantages_matches_reference(kind):
    _, _, _, _, group_ids, rewards = _fixture(1)
    obj = O.PolicyObjective(name="t", advantage=kind, advantage_eps=1e-8)
    got = O.compute_advantages(rewards, group_ids, obj)
    want = R.ref_advantages(_lists(rewards), _lists(group_ids), kind, 1e-8)
    torch.testing.assert_close(got, torch.tensor(want), atol=1e-6, rtol=0)


def test_compute_advantages_group_of_one_is_zero():
    obj = O.PolicyObjective(name="t", advantage="group_norm")
    got = O.compute_advantages(torch.tensor([0.3, 1.0, 2.0]), torch.tensor([0, 1, 1]), obj)
    assert got[0].item() == 0.0 and torch.isfinite(got).all()


def test_compute_advantages_external_raises():
    obj = O.PolicyObjective(name="t", advantage="external")
    with pytest.raises(ValueError, match="external"):
        O.compute_advantages(torch.zeros(2), torch.zeros(2, dtype=torch.long), obj)


def test_group_norm_matches_rl_losses_group_advantages():
    from stateset_agents.training import rl_losses

    rewards = torch.tensor([0.1, 0.9, 0.4, 0.4])
    obj = O.PolicyObjective(name="t", advantage="group_norm")
    got = O.compute_advantages(rewards, torch.zeros(4, dtype=torch.long), obj)
    torch.testing.assert_close(got, rl_losses.group_advantages(rewards))


# --- policy_loss parity ----------------------------------------------------


def _run(obj, seed=0, **overrides):
    logp_cur, logp_old, logp_ref, mask, group_ids, rewards = _fixture(seed)
    logp_cur = logp_cur.clone().requires_grad_(True)
    if obj.aggregate == "seq_sum_const":
        obj = obj.with_(max_completion_length=7)
    if obj.ratio == "group_expectation":
        logp_old = (logp_old * mask).sum(-1)  # sampler sequence sums
    adv_obj = obj if obj.advantage != "external" else obj.with_(advantage="group_norm")
    advantages = O.compute_advantages(rewards, group_ids, adv_obj)
    kwargs = dict(
        logp_cur=logp_cur, mask=mask, advantages=advantages, objective=obj,
        logp_old=logp_old, logp_ref=logp_ref, group_ids=group_ids,
    )
    kwargs.update(overrides)
    res = O.policy_loss(**kwargs)
    want = R.ref_policy_loss(
        obj, _lists(logp_cur), _lists(mask), _lists(advantages),
        logp_old=_lists(logp_old), logp_ref=_lists(logp_ref), groups=_lists(group_ids),
        kl_ext=_lists(kwargs["kl"]) if kwargs.get("kl") is not None else None,
        entropy=_lists(kwargs["entropy"]) if kwargs.get("entropy") is not None else None,
    )
    return res, want


@pytest.mark.parametrize("name", PRESETS)
def test_preset_matches_reference(name):
    res, want = _run(O.OBJECTIVES[name])
    assert res.loss.item() == pytest.approx(want, abs=1e-6)
    assert torch.isfinite(res.loss)
    res.loss.backward()  # differentiable


@pytest.mark.parametrize("name", ["grpo", "gspo", "gspo_token", "dapo", "cispo"])
def test_preset_with_kl_matches_reference(name):
    obj = O.OBJECTIVES[name]
    kl = "k3_sequence" if obj.ratio in ("sequence", "sequence_token") else "k3_token"
    obj = obj.with_(kl=kl, kl_coef=0.05)
    res, want = _run(obj)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)
    assert res.metrics["kl"] > 0


def test_k3_token_bias_correction_matches_reference():
    obj = O.OBJECTIVES["grpo"].with_(kl="k3_token", kl_coef=0.05, kl_bias_correction=True)
    res, want = _run(obj)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)


def test_external_kl_matches_reference():
    obj = O.OBJECTIVES["grpo"].with_(kl="external", kl_coef=0.1)
    kl = torch.rand(6, 7, generator=torch.Generator().manual_seed(3))
    res, want = _run(obj, kl=kl)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)


def test_entropy_bonus_matches_reference():
    obj = O.OBJECTIVES["dapo"].with_(entropy_coef=0.01)
    ent = torch.rand(6, 7, generator=torch.Generator().manual_seed(4))
    res, want = _run(obj, entropy=ent)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)
    assert res.metrics["entropy"] > 0


def test_delta_cap_matches_reference():
    obj = O.OBJECTIVES["dapo"].with_(delta=1.1)
    res, want = _run(obj)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)


def test_sequence_ratio_accepts_sum_logp_old():
    logp_cur, logp_old, _, mask, group_ids, rewards = _fixture(2)
    obj = O.OBJECTIVES["gspo"]
    adv = O.compute_advantages(rewards, group_ids, obj)
    per_token = O.policy_loss(
        logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj, logp_old=logp_old
    )
    sums = O.policy_loss(
        logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj,
        logp_old=(logp_old * mask).sum(-1),
    )
    torch.testing.assert_close(per_token.loss, sums.loss)


def test_token_ratio_rejects_sum_logp_old():
    logp_cur, logp_old, _, mask, group_ids, rewards = _fixture(2)
    obj = O.OBJECTIVES["dapo"]
    adv = O.compute_advantages(rewards, group_ids, obj)
    with pytest.raises(ValueError, match="per-token"):
        O.policy_loss(
            logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj,
            logp_old=logp_old.sum(-1),
        )


def test_dr_grpo_requires_max_completion_length():
    logp_cur, logp_old, _, mask, group_ids, rewards = _fixture(2)
    obj = O.OBJECTIVES["dr_grpo"]
    adv = O.compute_advantages(rewards, group_ids, obj)
    with pytest.raises(ValueError, match="max_completion_length"):
        O.policy_loss(logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj, logp_old=logp_old)


def test_missing_logp_old_defaults_to_detached_current():
    logp_cur, _, _, mask, group_ids, rewards = _fixture(5)
    logp_cur = logp_cur.requires_grad_(True)
    obj = O.OBJECTIVES["grpo"]
    adv = O.compute_advantages(rewards, group_ids, obj)
    a = O.policy_loss(logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj)
    b = O.policy_loss(
        logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj, logp_old=logp_cur.detach()
    )
    torch.testing.assert_close(a.loss, b.loss)
    torch.testing.assert_close(a.ratio, torch.ones_like(a.ratio))
    a.loss.backward()
    assert logp_cur.grad.abs().sum() > 0


def test_metrics_keys_and_types():
    res, _ = _run(O.OBJECTIVES["dapo"])
    for key in (
        "policy_loss", "kl", "entropy", "clip_fraction", "ratio_mean", "ratio_max",
        "advantage_mean", "advantage_std",
    ):
        assert isinstance(res.metrics[key], float), key


def test_per_token_advantages_are_accepted():
    logp_cur, logp_old, _, mask, _, _ = _fixture(6)
    obj = O.OBJECTIVES["ppo"]
    adv = torch.randn(6, 7, generator=torch.Generator().manual_seed(9))
    res = O.policy_loss(logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj, logp_old=logp_old)
    want = R.ref_policy_loss(obj, _lists(logp_cur), _lists(mask), _lists(adv), logp_old=_lists(logp_old))
    assert res.loss.item() == pytest.approx(want, abs=1e-6)


def test_module_imports_without_torch(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    sys.modules.pop("stateset_agents.training.objectives", None)
    mod = importlib.import_module("stateset_agents.training.objectives")
    assert mod.OBJECTIVES["grpo"].name == "grpo"
    sys.modules.pop("stateset_agents.training.objectives", None)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_objectives.py -p no:cacheprovider -o addopts="" -q`
Expected: collection error `ModuleNotFoundError: No module named 'stateset_agents.training.objectives'`

- [ ] **Step 4: Write the module**

```python
# stateset_agents/training/objectives.py
"""Declarative policy-optimisation objectives shared by every native trainer.

A :class:`PolicyObjective` names one point in the space
``advantage × ratio × clip × aggregate × kl``; :func:`compute_advantages`
and :func:`policy_loss` evaluate it on batched tensors. Presets in
:data:`OBJECTIVES` reproduce GRPO, Dr. GRPO, BNPO, DAPO, GSPO, GSPO-token,
GEPO, RLOO, REINFORCE++-baseline, CISPO, and PPO.

Pure tensor code: no trainer state, no model calls, torch fetched lazily so
importing this module never requires torch. Log-prob gathering stays with
the trainer (``rl_losses.gather_token_logprobs``); nothing here allocates a
full-vocabulary tensor.

Formulas and citations: ``docs/OBJECTIVES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

from . import rl_losses
from .trainer_utils import get_torch, require_torch

ADVANTAGE_KINDS = ("group_norm", "group_mean", "leave_one_out", "batch_norm", "external")
RATIO_KINDS = ("token", "sequence", "sequence_token", "group_expectation")
CLIP_KINDS = ("clipped", "cispo", "none")
AGGREGATE_KINDS = ("seq_mean", "token_mean", "seq_sum_const")
KL_KINDS = ("none", "k3_token", "k3_sequence", "external")


def _t() -> Any:
    return get_torch() or require_torch()


@dataclass(frozen=True)
class PolicyObjective:
    """One named policy-optimisation objective. See ``docs/OBJECTIVES.md``."""

    name: str
    advantage: str = "group_norm"
    advantage_eps: float = 1e-8
    ratio: str = "token"
    ratio_clamp: float = 20.0
    clip: str = "clipped"
    clip_low: float = 0.2
    clip_high: float = 0.2
    delta: float | None = None
    is_cap: float = 5.0
    aggregate: str = "seq_mean"
    max_completion_length: int | None = None
    kl: str = "none"
    kl_coef: float = 0.0
    kl_bias_correction: bool = False
    entropy_coef: float = 0.0

    def __post_init__(self) -> None:
        for fname, allowed in (
            ("advantage", ADVANTAGE_KINDS),
            ("ratio", RATIO_KINDS),
            ("clip", CLIP_KINDS),
            ("aggregate", AGGREGATE_KINDS),
            ("kl", KL_KINDS),
        ):
            value = getattr(self, fname)
            if value not in allowed:
                raise ValueError(f"{fname}={value!r}; expected one of {allowed}")
        for fname in ("clip_low", "clip_high", "advantage_eps", "kl_coef", "entropy_coef"):
            if float(getattr(self, fname)) < 0:
                raise ValueError(f"{fname} must be >= 0")
        for fname in ("is_cap", "ratio_clamp"):
            if float(getattr(self, fname)) <= 0:
                raise ValueError(f"{fname} must be > 0")
        if self.delta is not None and float(self.delta) <= 0:
            raise ValueError("delta must be > 0 or None")
        if self.max_completion_length is not None and int(self.max_completion_length) <= 0:
            raise ValueError("max_completion_length must be > 0 or None")
        if self.kl == "none" and self.kl_coef != 0:
            raise ValueError("kl_coef must be 0 when kl='none'")
        if self.kl_bias_correction and self.kl != "k3_token":
            raise ValueError("kl_bias_correction only applies to kl='k3_token'")

    def with_(self, **changes: Any) -> PolicyObjective:
        """Return a copy with ``changes`` applied (re-validated)."""
        return replace(self, **changes)


@dataclass(frozen=True)
class PolicyLossResult:
    """Scalar differentiable ``loss``, detached ``ratio``, float ``metrics``."""

    loss: Any
    ratio: Any
    metrics: dict[str, float] = field(default_factory=dict)


# --- advantages ------------------------------------------------------------


def compute_advantages(rewards: Any, group_ids: Any, objective: PolicyObjective) -> Any:
    """Advantages ``[N]`` from scalar rewards ``[N]`` and integer ``group_ids`` ``[N]``.

    Groups of size 1, constant rewards, and non-finite statistics give 0,
    never NaN. Always fp32.
    """
    torch = _t()
    if objective.advantage == "external":
        raise ValueError(
            f"objective {objective.name!r} uses external advantages; pass them to policy_loss"
        )
    r = torch.as_tensor(rewards).float().reshape(-1)
    g = torch.as_tensor(group_ids).long().reshape(-1)
    if r.shape != g.shape:
        raise ValueError(f"rewards {tuple(r.shape)} and group_ids {tuple(g.shape)} differ")
    adv = torch.zeros_like(r)
    if r.numel() == 0:
        return adv
    kind = objective.advantage
    eps = float(objective.advantage_eps)
    for gid in torch.unique(g):
        idx = (g == gid).nonzero(as_tuple=True)[0]
        rg = r[idx]
        if rg.numel() <= 1:
            continue
        if kind == "leave_one_out":
            adv[idx] = rg - (rg.sum() - rg) / (rg.numel() - 1)
        elif kind == "group_norm":
            adv[idx] = rl_losses.group_advantages(rg, normalize=True, eps=eps)
        else:  # group_mean, batch_norm: centre now, batch_norm scales below
            adv[idx] = rl_losses.group_advantages(rg, normalize=False)
    if kind == "batch_norm":
        if r.numel() <= 1:
            return torch.zeros_like(r)
        std = r.std(correction=0)
        if not torch.isfinite(std) or std <= eps:
            return torch.zeros_like(r)
        adv = adv / (std + eps)
    return adv


# --- ratio -----------------------------------------------------------------


def _sequence_log_ratio(logp_cur: Any, logp_old: Any, mask: Any) -> Any:
    torch = _t()
    lengths = torch.clamp(mask.sum(-1), min=1.0)
    if logp_old.dim() == 2:
        return ((logp_cur - logp_old) * mask).sum(-1) / lengths
    return ((logp_cur * mask).sum(-1) - logp_old) / lengths


def _compute_ratio(
    objective: PolicyObjective, logp_cur: Any, logp_old: Any, mask: Any, group_ids: Any | None
) -> Any:
    torch = _t()
    kind = objective.ratio
    clamp = float(objective.ratio_clamp)
    if kind == "token":
        if logp_old.dim() != 2:
            raise ValueError("ratio='token' requires per-token logp_old of shape [N, T]")
        return rl_losses.safe_exp_ratio(logp_cur - logp_old, clamp=clamp)
    if kind == "sequence":
        return rl_losses.safe_exp_ratio(
            _sequence_log_ratio(logp_cur, logp_old, mask), clamp=clamp
        ).unsqueeze(-1)
    if kind == "sequence_token":
        seq = rl_losses.safe_exp_ratio(
            _sequence_log_ratio(logp_cur, logp_old, mask), clamp=clamp
        ).detach()
        # Numerically equal to the sequence ratio; gradient flows per token.
        return seq.unsqueeze(-1) * torch.exp(logp_cur - logp_cur.detach())
    # group_expectation (GEPO)
    if logp_old.dim() != 1:
        raise ValueError("ratio='group_expectation' requires sampler sequence sums logp_old [N]")
    if group_ids is None:
        raise ValueError("ratio='group_expectation' requires group_ids")
    seq_cur = (logp_cur * mask).sum(-1)
    q = logp_old.detach()
    g = torch.as_tensor(group_ids).long().reshape(-1)
    out = torch.zeros_like(seq_cur)
    for gid in torch.unique(g):
        idx = (g == gid).nonzero(as_tuple=True)[0]
        log_e = torch.logsumexp(2 * q[idx], dim=0) - torch.logsumexp(q[idx], dim=0)
        out[idx] = rl_losses.safe_exp_ratio(seq_cur[idx] - log_e, clamp=clamp)
    return out.unsqueeze(-1)


# --- surrogate ---------------------------------------------------------------


def _surrogate(objective: PolicyObjective, ratio: Any, adv: Any, logp_cur: Any) -> Any:
    torch = _t()
    if objective.clip == "clipped":
        if objective.delta is None:
            return rl_losses.clipped_surrogate(
                ratio, adv, clip_low=objective.clip_low, clip_high=objective.clip_high
            )
        capped = torch.clamp(ratio, max=float(objective.delta))
        clipped = torch.clamp(ratio, 1.0 - objective.clip_low, 1.0 + objective.clip_high)
        return -torch.min(capped * adv, clipped * adv)
    if objective.clip == "cispo":
        weight = torch.clamp(ratio, max=float(objective.is_cap)).detach()
        return -weight * adv * logp_cur
    return -ratio * adv


def _aggregate(objective: PolicyObjective, per_token: Any, mask: Any) -> Any:
    torch = _t()
    if objective.aggregate == "seq_mean":
        return rl_losses.masked_mean(per_token, mask, mode="seq")
    if objective.aggregate == "token_mean":
        return rl_losses.masked_mean(per_token, mask, mode="token")
    if objective.max_completion_length is None:
        raise ValueError(
            f"objective {objective.name!r} aggregates with seq_sum_const and needs "
            "max_completion_length; use objective.with_(max_completion_length=...)"
        )
    n = per_token.shape[0]
    return (per_token * mask).sum() / float(n * int(objective.max_completion_length))


# --- policy loss -------------------------------------------------------------


def policy_loss(
    *,
    logp_cur: Any,
    mask: Any,
    advantages: Any,
    objective: PolicyObjective,
    logp_old: Any | None = None,
    logp_ref: Any | None = None,
    group_ids: Any | None = None,
    kl: Any | None = None,
    entropy: Any | None = None,
) -> PolicyLossResult:
    """Evaluate ``objective`` on one batch.

    ``logp_cur``/``mask`` are ``[N, T]``; ``logp_old`` is ``[N, T]`` per-token
    or ``[N]`` sequence sums (``None`` means ``logp_cur.detach()``);
    ``advantages`` is ``[N]`` or ``[N, T]``; ``logp_ref`` is ``[N, T]`` or
    ``[N]`` sums (only read when ``kl_coef > 0``); ``kl`` is the external
    per-token KL ``[N, T]`` for ``kl='external'``; ``entropy`` is ``[N, T]``.
    """
    torch = _t()
    if logp_cur.dim() != 2:
        raise ValueError("logp_cur must be [N, T]")
    mask = mask.to(logp_cur.dtype)
    if mask.shape != logp_cur.shape:
        raise ValueError("mask must match logp_cur shape")
    old = logp_cur.detach() if logp_old is None else logp_old.detach().to(logp_cur.dtype)
    adv = torch.as_tensor(advantages).to(logp_cur.dtype).detach()
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)
    elif adv.shape != logp_cur.shape:
        raise ValueError("advantages must be [N] or [N, T]")

    ratio = _compute_ratio(objective, logp_cur, old, mask, group_ids)
    per_token = _surrogate(objective, ratio, adv, logp_cur)

    kl_value = 0.0
    kl_seq_term = None
    if objective.kl_coef > 0:
        if objective.kl == "k3_token":
            if logp_ref is None or logp_ref.dim() != 2:
                raise ValueError("kl='k3_token' requires per-token logp_ref [N, T]")
            d = logp_ref.detach().to(logp_cur.dtype) - logp_cur
            k3 = torch.exp(d) - d - 1.0
            if objective.kl_bias_correction:
                k3 = k3 * ratio
            per_token = per_token + objective.kl_coef * k3
            kl_value = float(rl_losses.masked_mean(k3.detach(), mask, mode="token").item())
        elif objective.kl == "external":
            if kl is None:
                raise ValueError("kl='external' requires the kl tensor [N, T]")
            per_token = per_token + objective.kl_coef * kl.to(logp_cur.dtype)
            kl_value = float(rl_losses.masked_mean(kl.detach(), mask, mode="token").item())
        elif objective.kl == "k3_sequence":
            if logp_ref is None:
                raise ValueError("kl='k3_sequence' requires logp_ref")
            lengths = torch.clamp(mask.sum(-1), min=1.0)
            cur_seq = (logp_cur * mask).sum(-1) / lengths
            ref = logp_ref.detach().to(logp_cur.dtype)
            ref_seq = ((ref * mask).sum(-1) if ref.dim() == 2 else ref) / lengths
            kl_seq_term = rl_losses.k3_kl(cur_seq, ref_seq)
            kl_value = float(kl_seq_term.detach().item())

    per_token = per_token * torch.ones_like(mask)  # broadcast [N,1] -> [N,T]
    policy = _aggregate(objective, per_token, mask)
    loss = policy
    if kl_seq_term is not None:
        loss = loss + objective.kl_coef * kl_seq_term

    entropy_value = 0.0
    if entropy is not None:
        ent = rl_losses.masked_mean(entropy.to(logp_cur.dtype), mask, mode="token")
        entropy_value = float(ent.detach().item())
        if objective.entropy_coef > 0:
            loss = loss - objective.entropy_coef * ent

    with torch.no_grad():
        r = ratio.detach() * torch.ones_like(mask)
        m = mask > 0
        if objective.clip == "clipped":
            out = (r < 1.0 - objective.clip_low) | (r > 1.0 + objective.clip_high)
        elif objective.clip == "cispo":
            out = r > objective.is_cap
        else:
            out = torch.zeros_like(m)
        n_tok = float(m.sum().item())
        clip_fraction = float(out[m].float().sum().item() / n_tok) if n_tok else 0.0
        ratio_mean = float(r[m].mean().item()) if n_tok else 1.0
        ratio_max = float(r[m].max().item()) if n_tok else 1.0
        adv_flat = adv.reshape(-1)
        metrics = {
            "policy_loss": float(policy.detach().item()),
            "kl": kl_value,
            "entropy": entropy_value,
            "clip_fraction": clip_fraction,
            "ratio_mean": ratio_mean,
            "ratio_max": ratio_max,
            "advantage_mean": float(adv_flat.mean().item()) if adv_flat.numel() else 0.0,
            "advantage_std": (
                float(adv_flat.std(correction=0).item()) if adv_flat.numel() > 1 else 0.0
            ),
        }
    return PolicyLossResult(loss=loss, ratio=ratio.detach(), metrics=metrics)


# --- presets -----------------------------------------------------------------


def _preset(name: str, **kwargs: Any) -> PolicyObjective:
    return PolicyObjective(name=name, **kwargs)


OBJECTIVES: Mapping[str, PolicyObjective] = MappingProxyType(
    {
        "grpo": _preset("grpo", kl="k3_token"),
        "dr_grpo": _preset("dr_grpo", advantage="group_mean", aggregate="seq_sum_const"),
        "bnpo": _preset("bnpo", aggregate="token_mean"),
        "dapo": _preset("dapo", clip_high=0.28, aggregate="token_mean"),
        "gspo": _preset(
            "gspo", ratio="sequence", clip_low=3e-4, clip_high=4e-4, kl="k3_sequence"
        ),
        "gspo_token": _preset(
            "gspo_token", ratio="sequence_token", clip_low=3e-4, clip_high=4e-4, kl="k3_sequence"
        ),
        "gepo": _preset("gepo", ratio="group_expectation", ratio_clamp=30.0),
        "rloo": _preset("rloo", advantage="leave_one_out"),
        "reinforce_pp_baseline": _preset(
            "reinforce_pp_baseline", advantage="batch_norm", aggregate="token_mean"
        ),
        "cispo": _preset("cispo", clip="cispo", aggregate="token_mean"),
        "ppo": _preset("ppo", advantage="external", aggregate="token_mean", kl="k3_token"),
    }
)

__all__ = [
    "ADVANTAGE_KINDS",
    "AGGREGATE_KINDS",
    "CLIP_KINDS",
    "KL_KINDS",
    "OBJECTIVES",
    "RATIO_KINDS",
    "PolicyLossResult",
    "PolicyObjective",
    "compute_advantages",
    "policy_loss",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_objectives.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass. If `test_module_imports_without_torch` fails because `tests.unit` is not importable as a package, add an empty `tests/unit/__init__.py` only if one does not already exist (check first with `ls tests/unit/__init__.py`); otherwise import the reference via `importlib.util.spec_from_file_location`.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check stateset_agents/training/objectives.py tests/unit/test_objectives.py tests/unit/objective_reference.py
.venv/bin/black stateset_agents/training/objectives.py tests/unit/test_objectives.py tests/unit/objective_reference.py
.venv/bin/isort stateset_agents/training/objectives.py tests/unit/test_objectives.py tests/unit/objective_reference.py
git add stateset_agents/training/objectives.py tests/unit/test_objectives.py tests/unit/objective_reference.py
git commit -m "feat(training): add declarative PolicyObjective library with loop-reference parity tests"
```

---

### Task 2: Hypothesis property tests

**Files:**
- Create: `tests/unit/test_objectives_properties.py`

**Interfaces:**
- Consumes: `O.OBJECTIVES`, `O.policy_loss`, `O.compute_advantages` from Task 1.

- [ ] **Step 1: Write the tests**

```python
# tests/unit/test_objectives_properties.py
"""Invariants every objective must satisfy, checked with Hypothesis."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from stateset_agents.training import objectives as O  # noqa: E402

SETTINGS = settings(max_examples=40, deadline=None)
PRESETS = sorted(O.OBJECTIVES)


def _batch(seed: int, n: int = 4, t: int = 5):
    g = torch.Generator().manual_seed(seed)
    logp_cur = (-torch.rand(n, t, generator=g) * 3).requires_grad_(True)
    logp_old = logp_cur.detach() + 0.05 * torch.randn(n, t, generator=g)
    logp_ref = logp_cur.detach() + 0.1 * torch.randn(n, t, generator=g)
    mask = torch.ones(n, t)
    mask[0, 3:] = 0
    group_ids = torch.arange(n) % 2
    return logp_cur, logp_old, logp_ref, mask, group_ids


def _ready(name: str):
    obj = O.OBJECTIVES[name]
    if obj.aggregate == "seq_sum_const":
        obj = obj.with_(max_completion_length=5)
    return obj


def _call(obj, logp_cur, logp_old, mask, group_ids, advantages, **kw):
    if obj.ratio == "group_expectation":
        logp_old = (logp_old * mask).sum(-1)
    return O.policy_loss(
        logp_cur=logp_cur, mask=mask, advantages=advantages, objective=obj,
        logp_old=logp_old, group_ids=group_ids, **kw,
    )


@pytest.mark.parametrize("name", PRESETS)
@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_zero_advantage_gives_zero_gradient(name, seed):
    obj = _ready(name)
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    res = _call(obj, logp_cur, logp_old, mask, group_ids, torch.zeros(4))
    res.loss.backward()
    assert torch.allclose(logp_cur.grad, torch.zeros_like(logp_cur.grad))


@pytest.mark.parametrize("name", [n for n in PRESETS if O.OBJECTIVES[n].clip == "clipped"])
@SETTINGS
@given(seed=st.integers(0, 10_000), drift=st.floats(0.5, 2.0))
def test_out_of_region_sample_has_zero_gradient(name, seed, drift):
    """Ratio far above 1+clip_high with A>0 selects the clipped branch: no grad."""
    obj = _ready(name)
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    logp_old = logp_cur.detach() - drift  # ratio = exp(drift) >> 1 + clip_high
    adv = torch.ones(4)
    res = _call(obj, logp_cur, logp_old, mask, group_ids, adv)
    res.loss.backward()
    assert torch.allclose(logp_cur.grad, torch.zeros_like(logp_cur.grad), atol=1e-7)
    assert res.metrics["clip_fraction"] == pytest.approx(1.0)


@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_cispo_gradient_is_capped_weight_times_score(seed):
    obj = O.OBJECTIVES["cispo"]
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    adv = torch.randn(4, generator=torch.Generator().manual_seed(seed))
    res = _call(obj, logp_cur, logp_old, mask, group_ids, adv)
    res.loss.backward()
    weight = torch.clamp(torch.exp(logp_cur.detach() - logp_old), max=obj.is_cap)
    expected = -(weight * adv.unsqueeze(-1) * mask) / mask.sum()
    torch.testing.assert_close(logp_cur.grad, expected, atol=1e-6, rtol=1e-5)


@SETTINGS
@given(seed=st.integers(0, 10_000), coef=st.floats(0.01, 1.0))
def test_k3_kl_is_nonnegative_and_zero_at_equality(seed, coef):
    obj = O.OBJECTIVES["grpo"].with_(kl_coef=coef)
    logp_cur, logp_old, logp_ref, mask, group_ids = _batch(seed)
    res = _call(obj, logp_cur, logp_old, mask, group_ids, torch.zeros(4), logp_ref=logp_ref)
    assert res.metrics["kl"] >= 0
    same = _call(obj, logp_cur, logp_old, mask, group_ids, torch.zeros(4), logp_ref=logp_cur.detach())
    assert same.metrics["kl"] == pytest.approx(0.0, abs=1e-7)


@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_token_mean_equals_seq_mean_for_equal_lengths(seed):
    logp_cur, logp_old, _, _, group_ids = _batch(seed)
    mask = torch.ones(4, 5)
    adv = torch.randn(4, generator=torch.Generator().manual_seed(seed))
    a = _call(O.OBJECTIVES["bnpo"], logp_cur, logp_old, mask, group_ids, adv)
    b = _call(O.OBJECTIVES["grpo"].with_(kl="none"), logp_cur, logp_old, mask, group_ids, adv)
    torch.testing.assert_close(a.loss, b.loss)


@SETTINGS
@given(seed=st.integers(0, 10_000), length=st.integers(5, 64))
def test_seq_sum_const_is_token_mean_rescaled(seed, length):
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    adv = torch.randn(4, generator=torch.Generator().manual_seed(seed))
    dr = O.OBJECTIVES["dr_grpo"].with_(max_completion_length=length)
    bn = O.OBJECTIVES["bnpo"].with_(advantage="group_mean")
    a = _call(dr, logp_cur, logp_old, mask, group_ids, adv)
    b = _call(bn, logp_cur, logp_old, mask, group_ids, adv)
    scale = mask.sum() / (4 * length)
    torch.testing.assert_close(a.loss, b.loss * scale)


@SETTINGS
@given(rewards=st.lists(st.floats(-5, 5), min_size=2, max_size=8))
def test_leave_one_out_advantages_sum_to_zero(rewards):
    obj = O.OBJECTIVES["rloo"]
    r = torch.tensor(rewards)
    adv = O.compute_advantages(r, torch.zeros(len(rewards), dtype=torch.long), obj)
    assert adv.sum().item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("name", PRESETS)
@SETTINGS
@given(seed=st.integers(0, 10_000), scale=st.floats(10.0, 1000.0))
def test_no_preset_produces_non_finite_loss(name, seed, scale):
    obj = _ready(name)
    logp_cur, logp_old, _, mask, group_ids = _batch(seed)
    logp_old = logp_old - scale  # log-ratio up to +1000
    adv = torch.randn(4, generator=torch.Generator().manual_seed(seed))
    res = _call(obj, logp_cur, logp_old, mask, group_ids, adv)
    assert torch.isfinite(res.loss)
    res.loss.backward()
    assert torch.isfinite(logp_cur.grad).all()
```

- [ ] **Step 2: Run and confirm green**

Run: `.venv/bin/python -m pytest tests/unit/test_objectives_properties.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass. If `test_cispo_gradient_is_capped_weight_times_score` fails on masked positions, confirm `logp_cur.grad` at masked positions is exactly 0 in the implementation (it is, because `per_token * mask` zeroes them before aggregation) and that `expected` also multiplies by `mask` — the formula above already does.

- [ ] **Step 3: Lint and commit**

```bash
.venv/bin/ruff check tests/unit/test_objectives_properties.py && .venv/bin/black tests/unit/test_objectives_properties.py && .venv/bin/isort tests/unit/test_objectives_properties.py
git add tests/unit/test_objectives_properties.py
git commit -m "test(training): property-test PolicyObjective invariants with Hypothesis"
```

---

### Task 3: External pin against TRL 1.12

**Files:**
- Create: `tests/unit/test_objectives_trl_pin.py`

**Interfaces:**
- Consumes: `O.OBJECTIVES`, `O.policy_loss`, `O.compute_advantages`.

- [ ] **Step 1: Write the pin test**

```python
# tests/unit/test_objectives_trl_pin.py
"""Pin StateSet objectives to TRL's GRPO loss on identical tensors.

TRL's ``GRPOTrainer._compute_loss`` is bound to a bare namespace carrying only
the attributes it reads, with the per-token log-prob helper replaced by a
function that returns our fixture. Any TRL major-version change makes this
module skip loudly rather than silently pass.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

from stateset_agents.training import objectives as O  # noqa: E402

TRL_MAJOR = 1
if int(trl.__version__.split(".")[0]) != TRL_MAJOR:
    pytest.skip(
        f"objective pin targets trl {TRL_MAJOR}.x, found {trl.__version__}",
        allow_module_level=True,
    )

from trl.trainer.grpo_trainer import GRPOTrainer  # noqa: E402


def _fixture(seed: int, n: int = 6, t: int = 8, prompt: int = 3):
    g = torch.Generator().manual_seed(seed)
    logp = (-torch.rand(n, t, generator=g) * 3).requires_grad_(True)
    old = logp.detach() + 0.05 * torch.randn(n, t, generator=g)
    ref = logp.detach() + 0.1 * torch.randn(n, t, generator=g)
    mask = torch.ones(n, t)
    mask[0, 5:] = 0
    mask[1, 6:] = 0
    entropies = torch.rand(n, t, generator=g)
    rewards = torch.rand(n, generator=g)
    return logp, old, ref, mask, entropies, rewards, prompt


def _trl_loss(
    *, loss_type, level, eps_low, eps_high, beta, bias_correction, delta, fixture, adv, max_len
):
    logp, old, ref, mask, entropies, _, prompt = fixture
    n, t = logp.shape

    def fake_logps(self, model, input_ids, attention_mask, logits_to_keep, **kw):
        return logp, entropies, None

    fake = SimpleNamespace(
        aux_loss_enabled=False,
        top_entropy_quantile=1.0,
        off_policy_mask_threshold=None,
        importance_sampling_level=level,
        beta=beta,
        loss_type=loss_type,
        epsilon_low=eps_low,
        epsilon_high=eps_high,
        use_vllm=False,
        vllm_importance_sampling_correction=False,
        model=SimpleNamespace(training=False),
        current_gradient_accumulation_steps=1,
        max_completion_length=max_len,
        _entropy_bonus_enabled=False,
        _metrics={"eval": defaultdict(list), "train": defaultdict(list)},
        args=SimpleNamespace(use_bias_correction_kl=bias_correction, delta=delta, steps_per_generation=1),
        accelerator=SimpleNamespace(
            num_processes=1,
            sync_gradients=True,
            gather=lambda x: x,
            gather_for_metrics=lambda x: x,
            reduce=lambda x, reduction="sum": x,
        ),
    )
    fake._get_per_token_logps_and_entropies = fake_logps.__get__(fake)
    inputs = {
        "prompt_ids": torch.zeros(n, prompt, dtype=torch.long),
        "prompt_mask": torch.ones(n, prompt, dtype=torch.long),
        "completion_ids": torch.zeros(n, t, dtype=torch.long),
        "completion_mask": mask,
        "advantages": adv,
        "old_per_token_logps": old,
        "ref_per_token_logps": ref,
        "num_items_in_batch": mask.sum(),
    }
    return GRPOTrainer._compute_loss(fake, None, inputs)


CASES = [
    # (stateset preset, trl loss_type, importance level, eps_low, eps_high)
    ("grpo", "grpo", "token", 0.2, 0.2),
    ("bnpo", "bnpo", "token", 0.2, 0.2),
    ("dapo", "dapo", "token", 0.2, 0.28),
    ("dr_grpo", "dr_grpo", "token", 0.2, 0.2),
    ("cispo", "cispo", "token", 0.2, 5.0),
    ("gspo", "grpo", "sequence", 3e-4, 4e-4),
]


@pytest.mark.parametrize("preset, loss_type, level, lo, hi", CASES)
@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("bias_correction", [False, True])
def test_loss_matches_trl(preset, loss_type, level, lo, hi, beta, bias_correction):
    if bias_correction and (beta == 0.0 or level == "sequence"):
        pytest.skip("bias correction only meaningful for token-level k3")
    fixture = _fixture(seed=hash(preset) % 1000)
    logp, old, ref, mask, entropies, rewards, _ = fixture
    max_len = logp.shape[1]
    group_ids = torch.arange(logp.shape[0]) % 2

    obj = O.OBJECTIVES[preset].with_(advantage_eps=1e-4)
    if obj.aggregate == "seq_sum_const":
        obj = obj.with_(max_completion_length=max_len)
    if beta > 0:
        obj = obj.with_(
            kl="k3_token", kl_coef=beta, kl_bias_correction=bias_correction
        )
    adv = O.compute_advantages(rewards, group_ids, obj)

    ours = O.policy_loss(
        logp_cur=logp, mask=mask, advantages=adv, objective=obj, logp_old=old, logp_ref=ref
    ).loss
    theirs = _trl_loss(
        loss_type=loss_type, level=level, eps_low=lo, eps_high=hi, beta=beta,
        bias_correction=bias_correction, delta=None, fixture=fixture, adv=adv, max_len=max_len,
    )
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)


def test_delta_two_sided_clip_matches_trl():
    fixture = _fixture(seed=77)
    logp, old, ref, mask, _, rewards, _ = fixture
    group_ids = torch.arange(logp.shape[0]) % 2
    obj = O.OBJECTIVES["dapo"].with_(advantage_eps=1e-4, delta=1.05)
    adv = O.compute_advantages(rewards, group_ids, obj)
    ours = O.policy_loss(logp_cur=logp, mask=mask, advantages=adv, objective=obj, logp_old=old).loss
    theirs = _trl_loss(
        loss_type="dapo", level="token", eps_low=0.2, eps_high=0.28, beta=0.0,
        bias_correction=False, delta=1.05, fixture=fixture, adv=adv, max_len=logp.shape[1],
    )
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("scale, kind", [("group", "group_norm"), ("none", "group_mean"), ("batch", "batch_norm")])
def test_advantages_match_trl_scale_rewards(scale, kind):
    """Transcription of TRL 1.12 ``scale_rewards`` (nanstd, eps 1e-4)."""
    g = torch.Generator().manual_seed(5)
    num_generations = 4
    rewards = torch.rand(8, generator=g)
    grouped = rewards.view(-1, num_generations)
    mean_g = grouped.mean(dim=1).repeat_interleave(num_generations)
    if scale in ("group", "none"):
        std = grouped.std(dim=1, correction=0).repeat_interleave(num_generations)
    else:
        std = rewards.std(correction=0).expand_as(rewards)
    trl_adv = rewards - mean_g
    if scale != "none":
        trl_adv = trl_adv / (std + 1e-4)

    obj = O.PolicyObjective(name="t", advantage=kind, advantage_eps=1e-4)
    ours = O.compute_advantages(rewards, torch.arange(8) // num_generations, obj)
    torch.testing.assert_close(ours, trl_adv, atol=1e-6, rtol=1e-6)
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest tests/unit/test_objectives_trl_pin.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass. Known divergence to handle if it appears: TRL's `nanstd` uses `correction=1` in some versions. Check with `.venv/bin/python -c "import inspect,trl.trainer.utils as u;print(inspect.getsource(u.nanstd))"`. If it is unbiased, change the transcription's `correction=0` to `correction=1` **in the test only** and note it in the test docstring; the pin is about TRL's actual behaviour, and our `group_norm` keeps `correction=0` (the loss tests still pass because both sides receive the same `adv` tensor).

- [ ] **Step 3: Lint and commit**

```bash
.venv/bin/ruff check tests/unit/test_objectives_trl_pin.py && .venv/bin/black tests/unit/test_objectives_trl_pin.py && .venv/bin/isort tests/unit/test_objectives_trl_pin.py
git add tests/unit/test_objectives_trl_pin.py
git commit -m "test(training): pin PolicyObjective losses against TRL 1.12 GRPO"
```

---

### Task 4: Extract inline losses, then capture golden regression values

Trainers whose objective is inline in `train_step` (GSPO, GSPO-token, GEPO) first get a method with the *identical* math, so the golden pins a callable. The extraction is a pure refactor guarded by the existing behavioral tests.

**Files:**
- Modify: `stateset_agents/training/gspo_trainer.py:640-680` (inside `train_step`)
- Modify: `stateset_agents/training/gspo_token_trainer.py:222-290`
- Modify: `stateset_agents/training/gepo_trainer.py:575-590`
- Create: `scripts/capture_objective_goldens.py`
- Create: `tests/unit/goldens/objective_goldens.json`
- Create: `tests/unit/test_objective_goldens.py`

**Interfaces:**
- Produces:
  - `GSPOTrainer.compute_gspo_loss(importance_ratios: Tensor[G], advantages: Tensor[G], current_log_probs: Tensor[G], sequence_lengths: Tensor[G], ref_log_probs: Tensor[G] | None) -> Tensor` (scalar loss for one group)
  - `GSPOTokenTrainer.compute_gspo_token_loss(token_log_probs_list: list[Tensor[T_i]], sequence_lengths: Tensor[G], importance_ratios: Tensor[G], advantages: Tensor[G], current_log_probs: Tensor[G], ref_log_probs: Tensor[G] | None) -> Tensor`
  - `GEPOTrainer.compute_gepo_loss(learner_seq_log_probs: Tensor[G], sampler_seq_log_probs: Tensor[G], advantages: Tensor[G]) -> Tensor`

- [ ] **Step 1: Extract `GSPOTrainer.compute_gspo_loss`**

Add after `compute_group_advantages` in `gspo_trainer.py`:

```python
    def compute_gspo_loss(
        self,
        importance_ratios: torch.Tensor,
        advantages: torch.Tensor,
        current_log_probs: torch.Tensor,
        sequence_lengths: torch.Tensor,
        ref_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GSPO objective for one prompt group (clipped surrogate + optional k3 KL)."""
        policy_loss = rl_losses.clipped_surrogate(
            importance_ratios,
            advantages,
            clip_low=self.config.clip_range_left,
            clip_high=self.config.clip_range_right,
        ).mean()
        if self.config.beta > 0 and ref_log_probs is not None:
            kl_div = rl_losses.k3_kl(
                current_log_probs / sequence_lengths,
                ref_log_probs / sequence_lengths,
            )
            return policy_loss + self.config.beta * kl_div
        return policy_loss
```

Replace the block in `train_step` from `# Compute policy loss using GSPO objective` through `total_loss_item = policy_loss` (both branches) with:

```python
            ref_log_probs = None
            if self.config.beta > 0 and self.ref_model is not None:
                ref_log_probs = self._compute_batch_ref_log_probs(prompt, responses)
                if model_device is not None:
                    ref_log_probs = ref_log_probs.to(model_device)
            total_loss_item = self.compute_gspo_loss(
                importance_ratios,
                advantages,
                current_log_probs,
                sequence_lengths,
                ref_log_probs,
            )
```

Keep the existing `clipped_ratios`/`num_clipped` bookkeeping above it unchanged.

- [ ] **Step 2: Extract `GSPOTokenTrainer.compute_gspo_token_loss`**

Add to `gspo_token_trainer.py` after `compute_token_importance_ratio`:

```python
    def compute_gspo_token_loss(
        self,
        token_log_probs_list: list[torch.Tensor],
        sequence_lengths: torch.Tensor,
        importance_ratios: torch.Tensor,
        advantages: torch.Tensor,
        current_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GSPO-token objective for one prompt group.

        ``importance_ratios`` are the detached sequence ratios; gradients flow
        through ``token_log_probs_list`` only.
        """
        device = importance_ratios.device
        loss = torch.tensor(0.0, device=device)
        lo = 1 - self.config.clip_range_left
        hi = 1 + self.config.clip_range_right
        for i, token_log_probs in enumerate(token_log_probs_list):
            seq_ratio = importance_ratios[i]
            adv = advantages[i].detach()
            in_region = (seq_ratio >= lo) & (seq_ratio <= hi)
            push_out = ((adv > 0) & (seq_ratio > hi)) | ((adv < 0) & (seq_ratio < lo))
            gate = in_region | ~push_out
            gated_ratio = torch.where(gate, seq_ratio, torch.zeros_like(seq_ratio))
            token_loss = -(gated_ratio * adv * token_log_probs).sum() / sequence_lengths[i]
            loss = loss + token_loss / len(token_log_probs_list)
        if self.config.beta > 0 and ref_log_probs is not None:
            kl_div = rl_losses.k3_kl(
                current_log_probs / sequence_lengths, ref_log_probs / sequence_lengths
            )
            loss = loss + self.config.beta * kl_div
        return loss
```

In `train_step_token_level`, replace from `loss = torch.tensor(0.0, device=model_device)` through `loss += kl_penalty` (the whole inline loop plus KL block) with:

```python
            ref_log_probs = None
            if self.config.beta > 0 and self.ref_model is not None:
                ref_log_prob_values: list[float] = []
                for response in responses:
                    ref_log_prob_values.append(
                        await self._compute_ref_log_prob(query, response)
                    )
                ref_log_probs = torch.tensor(
                    ref_log_prob_values, dtype=torch.float32, device=model_device
                )
            loss = self.compute_gspo_token_loss(
                token_log_probs_list,
                sequence_lengths,
                importance_ratios,
                advantages,
                current_log_probs,
                ref_log_probs,
            )
```

- [ ] **Step 3: Extract `GEPOTrainer.compute_gepo_loss`**

Add after `compute_gepo_coefficient`:

```python
    def compute_gepo_loss(
        self,
        learner_seq_log_probs: torch.Tensor,
        sampler_seq_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """GEPO objective for one prompt group (clipped surrogate on group coefficients)."""
        gepo_coefs = self.compute_gepo_coefficient(
            learner_seq_log_probs, sampler_seq_log_probs
        )
        return rl_losses.clipped_surrogate(
            gepo_coefs,
            advantages,
            clip_low=self.config.clip_eps,
            clip_high=self.config.clip_eps,
        ).mean()
```

In `train_step`, replace the `gepo_coefs = ...` / `all_gepo_coefs.extend` / `policy_loss = rl_losses.clipped_surrogate(...).mean()` lines with:

```python
                with torch.no_grad():
                    all_gepo_coefs.extend(
                        self.compute_gepo_coefficient(
                            learner_seq_log_probs.detach(), rollout["sampler_seq_log_probs"]
                        ).tolist()
                    )
                policy_loss = self.compute_gepo_loss(
                    learner_seq_log_probs,
                    rollout["sampler_seq_log_probs"],
                    rollout["advantages"],
                )
```

- [ ] **Step 4: Run the behavioral suites to prove the refactor is inert**

Run: `.venv/bin/python -m pytest tests/unit/test_gspo_trainer.py tests/unit/test_gspo_token_trainer_behavioral.py tests/unit/test_gepo_trainer_behavioral.py tests/unit/test_advanced_rl_algorithms.py tests/integration/test_gspo_pipeline_integration.py tests/integration/test_trainer_ratio_invariants.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass.

- [ ] **Step 5: Commit the extraction**

```bash
git add stateset_agents/training/gspo_trainer.py stateset_agents/training/gspo_token_trainer.py stateset_agents/training/gepo_trainer.py
git commit -m "refactor(training): extract inline GSPO, GSPO-token, and GEPO losses into methods"
```

- [ ] **Step 6: Write the golden capture script**

```python
# scripts/capture_objective_goldens.py
"""Capture loss goldens for every native trainer's objective assembly.

Run BEFORE migrating a trainer to ``objectives.py`` and commit the JSON; the
paired test asserts the migrated trainer reproduces these numbers.

    .venv/bin/python scripts/capture_objective_goldens.py [--only dapo,...]

Deterministic: fixed seeds, CPU, dropout-free tiny GPT-2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

OUT = ROOT / "tests" / "unit" / "goldens" / "objective_goldens.json"


def tiny_model(vocab: int = 200):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32, n_layer=2, n_head=2, vocab_size=vocab, n_positions=64,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
    )


def tensors(seed: int, n: int = 4, t: int = 10):
    g = torch.Generator().manual_seed(seed)
    cur = (-torch.rand(n, t, generator=g) * 3).requires_grad_(True)
    old = cur.detach() + 0.05 * torch.randn(n, t, generator=g)
    ref = cur.detach() + 0.1 * torch.randn(n, t, generator=g)
    mask = torch.ones(n, t)
    mask[:, :3] = 0
    mask[0, 8:] = 0
    adv = torch.tensor([1.0, -0.5, 0.25, -0.75])
    return cur, old, ref, mask, adv


def golden_dapo() -> dict:
    from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer

    out = {}
    for token_level in (True, False):
        cfg = DAPOConfig(model_name="gpt2", group_size=4, use_token_level_loss=token_level)
        tr = DAPOTrainer(config=cfg, model=tiny_model(), tokenizer=None, reward_fn=lambda p, r: 0.0)
        cur, old, _, mask, adv = tensors(1)
        ratios = tr.compute_importance_ratio(cur, old)
        loss = tr.compute_dapo_loss(ratios, adv.unsqueeze(1).expand_as(ratios), mask)
        out[f"token_level={token_level}"] = loss.item()
    return out


def golden_vapo() -> dict:
    from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer

    out = {}
    for token_level in (True, False):
        cfg = VAPOConfig(
            model_name="gpt2", group_size=4, use_token_level_loss=token_level,
            per_device_train_batch_size=4,
        )
        tr = VAPOTrainer(config=cfg, model=tiny_model(), tokenizer=None, reward_fn=lambda p, r: 1.0)
        cur, old, _, mask, _ = tensors(2)
        g = torch.Generator().manual_seed(22)
        pol_adv = torch.randn(4, 10, generator=g)
        crit_adv = torch.randn(4, 10, generator=g)
        values = torch.randn(4, 10, generator=g)
        old_values = values + 0.1 * torch.randn(4, 10, generator=g)
        positive = torch.zeros(4, 10)
        positive[0] = mask[0]
        p, v, lm = tr.compute_vapo_losses(cur, old, pol_adv, crit_adv, values, old_values, mask, positive)
        out[f"token_level={token_level}"] = {"policy": p.item(), "value": v.item(), "positive_lm": lm.item()}
    return out


def golden_gspo() -> dict:
    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    out = {}
    for beta in (0.0, 0.05):
        cfg = GSPOConfig(model_name="gpt2", num_generations=4, beta=beta)
        tr = GSPOTrainer(
            config=cfg, model=tiny_model(), tokenizer=None, agent=None,
            environment=None, reward_model=None, ref_model=None,
        )
        cur_tok, old_tok, ref_tok, mask, adv = tensors(3)
        lengths = mask.sum(-1).clamp(min=1.0)
        cur = (cur_tok * mask).sum(-1)
        old = (old_tok * mask).sum(-1)
        ref = (ref_tok * mask).sum(-1)
        ratios = tr.compute_sequence_importance_ratio(cur, old, lengths)
        loss = tr.compute_gspo_loss(ratios, adv, cur, lengths, ref if beta > 0 else None)
        out[f"beta={beta}"] = loss.item()
    return out


def golden_gspo_token() -> dict:
    """GSPO-token: pin the GRADIENT w.r.t. token log-probs, not the loss value.

    The migrated objective reports the surrogate value (same quantity GSPO
    reports) instead of the log-prob-weighted quantity; gradients are
    identical, so that is what the golden pins.
    """
    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer

    out = {}
    for beta in (0.0, 0.05):
        cfg = GSPOConfig(model_name="gpt2", num_generations=4, beta=beta)
        tr = GSPOTokenTrainer(
            config=cfg, model=tiny_model(), tokenizer=None, agent=None,
            environment=None, reward_model=None, ref_model=None,
        )
        cur_tok, old_tok, ref_tok, mask, adv = tensors(4)
        lengths = mask.sum(-1).clamp(min=1.0)
        cur_tok = (cur_tok.detach() * mask).requires_grad_(True)
        cur = cur_tok.sum(-1)
        old = (old_tok * mask).sum(-1)
        ref = (ref_tok * mask).sum(-1)
        ratios = tr.compute_sequence_importance_ratio(cur, old, lengths).detach()
        # Make one sample sit outside the trust region on its advantage's side.
        ratios = ratios.clone()
        ratios[1] = 1.5
        token_lists = [cur_tok[i] for i in range(4)]
        loss = tr.compute_gspo_token_loss(token_lists, lengths, ratios, adv, cur, ref if beta > 0 else None)
        loss.backward()
        out[f"beta={beta}"] = cur_tok.grad.tolist()
    return out


def golden_gepo() -> dict:
    from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer

    cfg = GEPOConfig(model_name="gpt2", group_size=4)
    tr = GEPOTrainer(config=cfg, model=tiny_model(), tokenizer=None, reward_fn=lambda p, r: 0.0)
    cur_tok, old_tok, _, mask, adv = tensors(5)
    cur = (cur_tok * mask).sum(-1)
    old = (old_tok * mask).sum(-1)
    return {"default": tr.compute_gepo_loss(cur, old, adv).item()}


def golden_ppo() -> dict:
    from stateset_agents.training.ppo_trainer import PPOConfig, PPOTrainer

    cfg = PPOConfig(model_name="gpt2")
    tr = PPOTrainer(config=cfg, model=tiny_model(), tokenizer=None)
    cur, old, ref, mask, _ = tensors(6)
    g = torch.Generator().manual_seed(66)
    adv = torch.randn(4, 10, generator=g)
    loss, clip_fraction = tr.ppo_loss(cur, old, adv, mask)
    kl = tr.compute_kl_divergence(cur, ref, mask)
    return {"policy": loss.item(), "clip_fraction": clip_fraction.item(), "kl": kl.item()}


def golden_grpo() -> dict:
    from stateset_agents.training import loss_computation as lc

    from test_loss_computation_behavioral import _config, _make_agent, _make_group  # noqa: E402

    out = {}
    cases = {
        "reinforce": dict(group=_make_group(8, 4), cfg=_config(clip_ratio=0.0)),
        "clipped_inside": dict(group=_make_group(8, 4, log_probs=-(0.5 + 1e-4) * 4), cfg=_config(clip_ratio=0.2, seq_clip_ratio=3e-4)),
        "clipped_outside": dict(group=_make_group(8, 4, log_probs=-(0.5 + 1e-3) * 4), cfg=_config(clip_ratio=0.2, seq_clip_ratio=3e-4)),
    }
    for name, case in cases.items():
        agent = _make_agent(8, 4)
        loss, _ = lc._compute_group_policy_loss(case["group"], torch.tensor([1.0]), case["cfg"], agent)
        loss.backward()
        out[name] = {"loss": loss.item(), "grad": float(agent.model.p.grad)}
    return out


def golden_grpo_enhanced() -> dict:
    """Real tiny GPT-2 through compute_enhanced_grpo_loss (plain + exact KL)."""
    from stateset_agents.training import loss_computation as lc

    class _Tok:
        def apply_chat_template(self, messages, *, return_dict=False, return_assistant_tokens_mask=False, **kw):
            ids = torch.tensor([[5, 7, 9, 11, 13, 17, 19, 23, 29, 31]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "assistant_tokens_mask": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]),
            }

    def traj(reward: float, lp_sum: float | None):
        t = SimpleNamespace(
            turns=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            total_reward=reward, metadata={},
        )
        if lp_sum is not None:
            t.log_probs = lp_sum
        return t

    model = tiny_model()
    model.device = torch.device("cpu")
    ref = tiny_model()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.01)
    agent = SimpleNamespace(tokenizer=_Tok(), model=model)
    cfg = SimpleNamespace(
        max_prompt_length=32, max_completion_length=32, clip_ratio=0.2, seq_clip_ratio=3e-4,
        bf16=False, fp16=False,
    )
    out = {}
    for beta, use_ref in ((0.0, False), (0.05, True)):
        groups = [SimpleNamespace(trajectories=[traj(1.0, -20.0), traj(0.0, -21.0), traj(0.5, None)])]
        model.zero_grad()
        res = lc.compute_enhanced_grpo_loss(groups, beta, cfg, agent, reference_model=ref if use_ref else None)
        res["total_loss"].backward()
        grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
        out[f"beta={beta}"] = {
            "total": res["total_loss"].item(), "policy": res["policy_loss"].item(),
            "kl": float(res["kl_penalty"]), "grad_norm": grad_norm,
        }
    return out


CAPTURES = {
    "dapo": golden_dapo,
    "vapo": golden_vapo,
    "gspo": golden_gspo,
    "gspo_token": golden_gspo_token,
    "gepo": golden_gepo,
    "ppo": golden_ppo,
    "grpo": golden_grpo,
    "grpo_enhanced": golden_grpo_enhanced,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated subset to (re)capture")
    args = parser.parse_args()
    names = [n for n in args.only.split(",") if n] or list(CAPTURES)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    torch.use_deterministic_algorithms(True)
    for name in names:
        existing[name] = CAPTURES[name]()
        print(f"captured {name}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run the capture and inspect**

Run: `.venv/bin/python scripts/capture_objective_goldens.py && cat tests/unit/goldens/objective_goldens.json`
Expected: eight top-level keys, every value finite. If a trainer constructor rejects `tokenizer=None` or `reward_fn`, read that trainer's `__init__` and pass the minimal accepted arguments; do not change the trainer.

- [ ] **Step 8: Write the golden test**

```python
# tests/unit/test_objective_goldens.py
"""Trainers must reproduce the loss goldens captured before migration.

Regenerate deliberately with ``scripts/capture_objective_goldens.py --only X``
and explain the change in CHANGELOG.md; never regenerate to make a red test
green without understanding why the number moved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "objective_goldens.json").read_text()
)
_spec = importlib.util.spec_from_file_location(
    "capture_objective_goldens",
    Path(__file__).resolve().parents[2] / "scripts" / "capture_objective_goldens.py",
)
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)


def _assert_same(got, want, path="", atol=1e-6):
    if isinstance(want, dict):
        assert set(got) == set(want), path
        for k in want:
            _assert_same(got[k], want[k], f"{path}/{k}", atol)
    elif isinstance(want, list):
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_same(g, w, f"{path}[{i}]", atol)
    else:
        assert got == pytest.approx(want, abs=atol), path


@pytest.mark.parametrize("name", sorted(capture.CAPTURES))
def test_trainer_reproduces_golden(name):
    torch.use_deterministic_algorithms(True)
    _assert_same(capture.CAPTURES[name](), GOLDENS[name], name)
```

- [ ] **Step 9: Run the golden test (pre-migration, must be green)**

Run: `.venv/bin/python -m pytest tests/unit/test_objective_goldens.py -p no:cacheprovider -o addopts="" -q`
Expected: 8 passed.

- [ ] **Step 10: Lint and commit**

```bash
.venv/bin/ruff check scripts/capture_objective_goldens.py tests/unit/test_objective_goldens.py && .venv/bin/black scripts/capture_objective_goldens.py tests/unit/test_objective_goldens.py && .venv/bin/isort scripts/capture_objective_goldens.py tests/unit/test_objective_goldens.py
git add scripts/capture_objective_goldens.py tests/unit/goldens/objective_goldens.json tests/unit/test_objective_goldens.py
git commit -m "test(training): capture pre-migration loss goldens for every native trainer"
```

---

### Task 5: Migrate DAPO

**Files:**
- Modify: `stateset_agents/training/dapo_trainer.py` (`__init__`, `compute_dapo_loss`, `train_step`)
- Test: `tests/unit/test_dapo_trainer_behavioral.py` (append)

**Interfaces:**
- Produces: `DAPOTrainer.compute_dapo_loss_from_log_probs(current_log_probs [G,T], old_log_probs [G,T], advantages [G], response_mask [G,T]) -> tuple[Tensor, dict[str, float]]`
- Keeps: `DAPOTrainer.compute_dapo_loss(importance_ratios, advantages, response_mask)` (ratio-based, pinned by the golden)

- [ ] **Step 1: Add the objective in `__init__`** (after `self.config = config`), and `from . import objectives` next to the existing `from . import rl_losses`:

```python
        self._objective = objectives.OBJECTIVES["dapo"].with_(
            clip_low=float(config.clip_eps_low),
            clip_high=float(config.clip_eps_high),
            aggregate="token_mean" if config.use_token_level_loss else "seq_mean",
        )
```

- [ ] **Step 2: Add the log-prob entry point and re-express the ratio-based one**

Insert after `compute_group_advantages`:

```python
    def compute_dapo_loss_from_log_probs(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """DAPO objective from per-token log-probs via ``objectives.policy_loss``."""
        result = objectives.policy_loss(
            logp_cur=current_log_probs,
            mask=response_mask,
            advantages=advantages,
            objective=self._objective,
            logp_old=old_log_probs,
        )
        return result.loss, result.metrics
```

Replace the body of `compute_dapo_loss` (keep its signature and docstring first line) with:

```python
        # Ratio-based entry point kept for callers holding ratios; train_step
        # uses compute_dapo_loss_from_log_probs. Both evaluate self._objective.
        per_token = objectives.surrogate(
            self._objective, importance_ratios, advantages, importance_ratios
        )
        return objectives.aggregate(self._objective, per_token, response_mask)
```

This requires the two helpers to be public: in `objectives.py` rename `_surrogate` → `surrogate` and `_aggregate` → `aggregate`, update their internal callers, and add both names to `__all__`. (Task 1's tests do not reference the private names, so nothing else changes.)

- [ ] **Step 3: Route `train_step`**

Replace

```python
                importance_ratios = self.compute_importance_ratio(
                    current_token_log_probs, old_token_log_probs
                )

                # Expand advantages to token level
                # advantages: [batch_size] -> [batch_size, seq_len]
                token_advantages = advantages.unsqueeze(1).expand_as(importance_ratios)

                # Compute DAPO loss
                loss = self.compute_dapo_loss(
                    importance_ratios,
                    token_advantages,
                    batch_response_mask[:, 1:],  # Shift for next-token prediction
                )
```

with

```python
                loss, _objective_metrics = self.compute_dapo_loss_from_log_probs(
                    current_token_log_probs,
                    old_token_log_probs,
                    advantages,
                    batch_response_mask[:, 1:],  # Shift for next-token prediction
                )
```

- [ ] **Step 4: Wiring test**

Append to `tests/unit/test_dapo_trainer_behavioral.py`:

```python
def test_loss_from_log_probs_routes_through_policy_objective(dapo_trainer_factory, monkeypatch):
    from stateset_agents.training import objectives

    trainer = dapo_trainer_factory(tiny_model())
    seen = {}
    real = objectives.policy_loss

    def spy(**kw):
        seen["objective"] = kw["objective"]
        return real(**kw)

    monkeypatch.setattr(objectives, "policy_loss", spy)
    ids, am, rm = make_batch()
    with torch.no_grad():
        old, _ = trainer.compute_token_log_probs(ids, am, rm)
    cur, _ = trainer.compute_token_log_probs(ids, am, rm)
    loss, metrics = trainer.compute_dapo_loss_from_log_probs(
        cur, old, torch.tensor([1.0, -1.0]), rm[:, 1:]
    )
    assert seen["objective"].name == "dapo"
    assert seen["objective"].clip_high == trainer.config.clip_eps_high
    assert torch.isfinite(loss) and "clip_fraction" in metrics


def test_ratio_and_log_prob_entry_points_agree(dapo_trainer_factory):
    trainer = dapo_trainer_factory(tiny_model())
    ids, am, rm = make_batch()
    with torch.no_grad():
        old, _ = trainer.compute_token_log_probs(ids, am, rm)
    cur, _ = trainer.compute_token_log_probs(ids, am, rm)
    adv = torch.tensor([1.0, -1.0])
    via_lp, _ = trainer.compute_dapo_loss_from_log_probs(cur, old, adv, rm[:, 1:])
    ratios = trainer.compute_importance_ratio(cur, old)
    via_ratio = trainer.compute_dapo_loss(ratios, adv.unsqueeze(1).expand_as(ratios), rm[:, 1:])
    torch.testing.assert_close(via_lp, via_ratio)
```

- [ ] **Step 5: Run goldens, behavioral, and unit suites**

Run: `.venv/bin/python -m pytest tests/unit/test_objectives.py tests/unit/test_objective_goldens.py::test_trainer_reproduces_golden[dapo] tests/unit/test_dapo_trainer.py tests/unit/test_dapo_trainer_behavioral.py tests/unit/test_dapo_module_exports.py tests/unit/test_advanced_rl_algorithms.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass, golden unchanged.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check stateset_agents/training/dapo_trainer.py stateset_agents/training/objectives.py tests/unit/test_dapo_trainer_behavioral.py && .venv/bin/black stateset_agents/training/dapo_trainer.py stateset_agents/training/objectives.py tests/unit/test_dapo_trainer_behavioral.py && .venv/bin/isort stateset_agents/training/dapo_trainer.py stateset_agents/training/objectives.py tests/unit/test_dapo_trainer_behavioral.py
git add stateset_agents/training/dapo_trainer.py stateset_agents/training/objectives.py tests/unit/test_dapo_trainer_behavioral.py
git commit -m "refactor(training): route DAPO through PolicyObjective"
```

---

### Task 6: Migrate VAPO

**Files:**
- Modify: `stateset_agents/training/vapo_trainer.py:734-785` (`compute_vapo_losses`) and `__init__`

- [ ] **Step 1: Add the objective in `__init__`** (after `self.config = config`), plus `from . import objectives`:

```python
        self._objective = objectives.OBJECTIVES["ppo"].with_(
            name="vapo",
            clip_low=float(config.clip_eps_low),
            clip_high=float(config.clip_eps_high),
            aggregate="token_mean" if config.use_token_level_loss else "seq_mean",
            kl="none",
        )
```

- [ ] **Step 2: Replace the policy part of `compute_vapo_losses`**

Replace from `ratio = rl_losses.safe_exp_ratio(...)` through `policy_loss = rl_losses.masked_mean(...)` with:

```python
        policy_result = objectives.policy_loss(
            logp_cur=current_log_probs,
            mask=response_mask,
            advantages=policy_advantages,
            objective=self._objective,
            logp_old=old_log_probs,
        )
        policy_loss = policy_result.loss
```

Value loss and positive-LM loss lines stay exactly as they are.

- [ ] **Step 3: Run**

Run: `.venv/bin/python -m pytest tests/unit/test_objective_goldens.py::test_trainer_reproduces_golden[vapo] tests/unit/test_vapo_trainer.py tests/unit/test_vapo_trainer_behavioral.py tests/unit/test_vapo_module_exports.py tests/unit/test_vapo_rust_gae_parity.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
.venv/bin/ruff check stateset_agents/training/vapo_trainer.py && .venv/bin/black stateset_agents/training/vapo_trainer.py && .venv/bin/isort stateset_agents/training/vapo_trainer.py
git add stateset_agents/training/vapo_trainer.py
git commit -m "refactor(training): route VAPO policy loss through PolicyObjective"
```

---

### Task 7: Migrate GSPO

**Files:**
- Modify: `stateset_agents/training/gspo_trainer.py` (`__init__`, `compute_gspo_loss`)

- [ ] **Step 1: Objective in `__init__`** (after `self.ref_model = ref_model`), plus `from . import objectives`:

```python
        self._objective = objectives.OBJECTIVES["gspo"].with_(
            clip_low=float(config.clip_range_left),
            clip_high=float(config.clip_range_right),
            kl_coef=float(config.beta),
        )
```

- [ ] **Step 2: Replace `compute_gspo_loss` body**

`compute_gspo_loss` receives sequence-level tensors. Feed them to `policy_loss` as one-token rows: `logp_cur` of shape `[G,1]` holding `current_log_probs / sequence_lengths` is not right (the ratio would be exponentiated over a length of 1). Instead pass the sums with a synthetic mask whose row length equals the response length, so the sequence ratio normalises by the true length:

```python
    def compute_gspo_loss(
        self,
        importance_ratios: torch.Tensor,
        advantages: torch.Tensor,
        current_log_probs: torch.Tensor,
        sequence_lengths: torch.Tensor,
        ref_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GSPO objective for one prompt group via ``objectives.policy_loss``.

        The trainer scores sequences as summed log-probs plus lengths; the
        objective consumes per-token rows, so each sequence is presented as a
        single token carrying its length-normalised log-prob with a unit
        mask. ``importance_ratios`` is accepted for signature compatibility
        and cross-checked against the ratio the objective recomputes.
        """
        lengths = sequence_lengths.to(current_log_probs.dtype).clamp(min=1.0)
        logp_cur = (current_log_probs / lengths).unsqueeze(-1)
        # sequence ratio == exp((cur - old)/len); old is recovered from the
        # trainer's ratio so callers holding only ratios keep working.
        log_old = logp_cur.detach() - torch.log(importance_ratios.detach()).unsqueeze(-1)
        ref = None
        if self.config.beta > 0 and ref_log_probs is not None:
            ref = (ref_log_probs / lengths).unsqueeze(-1)
        result = objectives.policy_loss(
            logp_cur=logp_cur,
            mask=torch.ones_like(logp_cur),
            advantages=advantages,
            objective=self._objective,
            logp_old=log_old,
            logp_ref=ref,
        )
        return result.loss
```

Why this is numerically identical: with a unit mask, `_sequence_log_ratio` = `logp_cur − log_old` = `log(ratio)`, so `ratio` is reproduced exactly (the trainer already clamped it via `safe_exp_ratio`, and `log(exp(clamp(x)))` re-clamps at the same bound); `seq_mean` over one-token rows is the row mean; `k3_sequence` on per-length-normalised values equals the previous `k3_kl(cur/len, ref/len)`.

- [ ] **Step 3: Run**

Run: `.venv/bin/python -m pytest tests/unit/test_objective_goldens.py::test_trainer_reproduces_golden[gspo] tests/unit/test_gspo_trainer.py tests/unit/test_gspo_scoring_consistency.py tests/unit/test_gspo_module_exports.py tests/integration/test_gspo_pipeline_integration.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
.venv/bin/ruff check stateset_agents/training/gspo_trainer.py && .venv/bin/black stateset_agents/training/gspo_trainer.py && .venv/bin/isort stateset_agents/training/gspo_trainer.py
git add stateset_agents/training/gspo_trainer.py
git commit -m "refactor(training): route GSPO through PolicyObjective"
```

---

### Task 8: Migrate GSPO-token

**Files:**
- Modify: `stateset_agents/training/gspo_token_trainer.py` (`__init__`, `compute_gspo_token_loss`)

- [ ] **Step 1: Objective in `__init__`** (after `super().__init__(...)`):

```python
        self._objective = objectives.OBJECTIVES["gspo_token"].with_(
            clip_low=float(config.clip_range_left),
            clip_high=float(config.clip_range_right),
            kl_coef=float(config.beta),
        )
```

- [ ] **Step 2: Replace `compute_gspo_token_loss` body**

Token log-prob rows are ragged (`token_log_probs_list`), so pad them into `[G, T_max]` with a mask; sequence lengths already exclude the prompt, and the padded/prompt positions are zero in each row (the helper `gather_token_logprobs` masked them):

```python
    def compute_gspo_token_loss(
        self,
        token_log_probs_list: list[torch.Tensor],
        sequence_lengths: torch.Tensor,
        importance_ratios: torch.Tensor,
        advantages: torch.Tensor,
        current_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GSPO-token objective for one prompt group via ``objectives.policy_loss``.

        Gradients flow through the per-token log-probs; the sequence ratio is
        stop-gradient. Rows are padded to a common width with a response mask
        built from ``sequence_lengths`` counted from the end of each row (the
        gathered rows carry zeros on prompt and pad positions).
        """
        device = importance_ratios.device
        width = max(int(t.shape[-1]) for t in token_log_probs_list)
        logp_cur = torch.zeros(len(token_log_probs_list), width, device=device)
        mask = torch.zeros_like(logp_cur)
        for i, row in enumerate(token_log_probs_list):
            n = int(row.shape[-1])
            logp_cur[i, :n] = row
            start = max(n - int(sequence_lengths[i].item()), 0)
            mask[i, start:n] = 1.0
        lengths = sequence_lengths.to(logp_cur.dtype).clamp(min=1.0)
        # Recover sequence-sum old log-probs from the (detached) ratios.
        old_sums = current_log_probs.detach() - torch.log(importance_ratios.detach()) * lengths
        ref = ref_log_probs if (self.config.beta > 0 and ref_log_probs is not None) else None
        result = objectives.policy_loss(
            logp_cur=logp_cur,
            mask=mask,
            advantages=advantages,
            objective=self._objective,
            logp_old=old_sums,
            logp_ref=ref,
        )
        return result.loss
```

Subtlety: the old implementation divided each row's token sum by `sequence_lengths[i]`; `seq_mean` divides by `mask.sum(-1)`, which equals `sequence_lengths[i]` by construction above. The golden pins the gradient, which matches; the reported loss value changes (documented in CHANGELOG, Task 12).

- [ ] **Step 3: Run**

Run: `.venv/bin/python -m pytest tests/unit/test_objective_goldens.py::test_trainer_reproduces_golden[gspo_token] tests/unit/test_gspo_token_trainer_behavioral.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass. If the gradient golden differs only at prompt positions, confirm `start` is computed from the *row length* minus `sequence_lengths` (rows contain prompt positions as zeros before the response).

- [ ] **Step 4: Commit**

```bash
.venv/bin/ruff check stateset_agents/training/gspo_token_trainer.py && .venv/bin/black stateset_agents/training/gspo_token_trainer.py && .venv/bin/isort stateset_agents/training/gspo_token_trainer.py
git add stateset_agents/training/gspo_token_trainer.py
git commit -m "refactor(training): route GSPO-token through PolicyObjective"
```

---

### Task 9: Migrate GEPO

**Files:**
- Modify: `stateset_agents/training/gepo_trainer.py` (`__init__`, `compute_gepo_loss`)

- [ ] **Step 1: Objective in `__init__`**, plus `from . import objectives`:

```python
        self._objective = objectives.OBJECTIVES["gepo"].with_(
            clip_low=float(config.clip_eps), clip_high=float(config.clip_eps)
        )
```

- [ ] **Step 2: Replace `compute_gepo_loss` body**

```python
    def compute_gepo_loss(
        self,
        learner_seq_log_probs: torch.Tensor,
        sampler_seq_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """GEPO objective for one prompt group via ``objectives.policy_loss``."""
        logp_cur = learner_seq_log_probs.unsqueeze(-1)  # one "token" per sequence
        group_ids = torch.zeros(logp_cur.shape[0], dtype=torch.long, device=logp_cur.device)
        result = objectives.policy_loss(
            logp_cur=logp_cur,
            mask=torch.ones_like(logp_cur),
            advantages=advantages,
            objective=self._objective,
            logp_old=sampler_seq_log_probs.detach(),
            group_ids=group_ids,
        )
        return result.loss
```

With a unit mask, `seq_cur = learner sum`, and `group_expectation` reproduces `compute_gepo_coefficient_static` including the `clamp=30`.

- [ ] **Step 3: Run**

Run: `.venv/bin/python -m pytest tests/unit/test_objective_goldens.py::test_trainer_reproduces_golden[gepo] tests/unit/test_gepo_trainer_behavioral.py tests/unit/test_advanced_rl_algorithms.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
.venv/bin/ruff check stateset_agents/training/gepo_trainer.py && .venv/bin/black stateset_agents/training/gepo_trainer.py && .venv/bin/isort stateset_agents/training/gepo_trainer.py
git add stateset_agents/training/gepo_trainer.py
git commit -m "refactor(training): route GEPO through PolicyObjective"
```

---

### Task 10: Migrate PPO (intended k3 KL change)

**Files:**
- Modify: `stateset_agents/training/ppo_trainer.py` (`__init__`, `compute_kl_divergence`, `ppo_loss`, `train_step` KL block)
- Modify: `tests/unit/goldens/objective_goldens.json` (regenerate `ppo`)
- Test: `tests/unit/test_ppo_objective.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_ppo_objective.py
"""PPO must use the k3 KL estimator and the clamped ratio (objective library)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from stateset_agents.training import rl_losses  # noqa: E402
from stateset_agents.training.ppo_trainer import PPOConfig, PPOTrainer  # noqa: E402


def _trainer():
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(n_embd=32, n_layer=2, n_head=2, vocab_size=200, n_positions=64,
                   resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    )
    return PPOTrainer(config=PPOConfig(model_name="gpt2"), model=model, tokenizer=None)


def test_kl_is_k3_nonnegative_with_gradient_toward_ref():
    tr = _trainer()
    cur = (torch.randn(2, 5) - 1).requires_grad_(True)
    ref = cur.detach() + 0.3
    mask = torch.ones(2, 5)
    kl = tr.compute_kl_divergence(cur, ref, mask)
    assert kl.item() >= 0
    torch.testing.assert_close(kl, rl_losses.k3_kl(cur, ref, mask))
    kl.backward()
    # gradient descends toward ref: cur < ref so grad must be negative
    assert (cur.grad < 0).all()


def test_ratio_is_clamped_not_inf():
    tr = _trainer()
    cur = torch.zeros(1, 4, requires_grad=True)
    old = torch.full((1, 4), -200.0)
    loss, frac = tr.ppo_loss(cur, old, torch.ones(1, 4), torch.ones(1, 4))
    assert torch.isfinite(loss)
    assert frac.item() == pytest.approx(1.0)
```

Run: `.venv/bin/python -m pytest tests/unit/test_ppo_objective.py -p no:cacheprovider -o addopts="" -q`
Expected: FAIL (`compute_kl_divergence` returns a signed mean; `ppo_loss` gives inf/nan).

- [ ] **Step 2: Implement**

In `__init__` (after `self.config = config`), plus `from . import objectives, rl_losses`:

```python
        self._objective = objectives.OBJECTIVES["ppo"].with_(
            clip_low=float(config.clip_eps), clip_high=float(config.clip_eps), kl="none"
        )
```

Replace `compute_kl_divergence`:

```python
    def compute_kl_divergence(
        self,
        current_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """k3 estimate of KL(π_cur ‖ π_ref) over response tokens.

        Replaces the naive ``log π − log π_ref`` mean, whose gradient has zero
        expectation and therefore never pulled the policy toward the
        reference.
        """
        if mask is None:
            return rl_losses.k3_kl(current_log_probs, reference_log_probs)
        return rl_losses.k3_kl(current_log_probs, reference_log_probs, mask)
```

Replace `ppo_loss`:

```python
    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PPO clipped surrogate via ``objectives.policy_loss``; returns (loss, clip_fraction)."""
        if mask is None:
            mask = torch.ones_like(log_probs)
        result = objectives.policy_loss(
            logp_cur=log_probs,
            mask=mask,
            advantages=advantages,
            objective=self._objective,
            logp_old=old_log_probs,
        )
        clip_fraction = torch.tensor(result.metrics["clip_fraction"], device=log_probs.device)
        return result.loss, clip_fraction
```

The `train_step` KL block already calls `self.compute_kl_divergence(...)`; leave it.

- [ ] **Step 3: Regenerate the PPO golden and run**

```bash
.venv/bin/python scripts/capture_objective_goldens.py --only ppo
.venv/bin/python -m pytest tests/unit/test_ppo_objective.py tests/unit/test_objective_goldens.py tests/unit/test_advanced_rl_algorithms.py -p no:cacheprovider -o addopts="" -q
```
Expected: pass. Note the old and new `ppo.kl` values from `git diff tests/unit/goldens/objective_goldens.json` for the CHANGELOG line in Task 12. `policy` must be unchanged (same clipped surrogate, same token mean) unless the fixture overflowed, which it does not.

- [ ] **Step 4: Commit**

```bash
.venv/bin/ruff check stateset_agents/training/ppo_trainer.py tests/unit/test_ppo_objective.py && .venv/bin/black stateset_agents/training/ppo_trainer.py tests/unit/test_ppo_objective.py && .venv/bin/isort stateset_agents/training/ppo_trainer.py tests/unit/test_ppo_objective.py
git add stateset_agents/training/ppo_trainer.py tests/unit/test_ppo_objective.py tests/unit/goldens/objective_goldens.json
git commit -m "fix(training): PPO uses k3 KL and clamped ratios via PolicyObjective"
```

---

### Task 11: Migrate GRPO (plain and enhanced)

**Files:**
- Modify: `stateset_agents/training/loss_computation.py` (`compute_grpo_loss` baseline block, `_compute_group_policy_loss`, `compute_enhanced_grpo_loss`)

**Background:** both GRPO paths score a trajectory with `outputs.loss`, the mean per-token NLL over the `T = labels.ne(-100).sum()` response tokens, and hold the rollout-time log-prob as a *sum*. The previous clipped path recovered `new = -(loss * T)` and formed the sequence ratio `exp((new - old) / T)` clipped at `seq_clip_ratio`. The objective library reproduces this exactly when a trajectory is presented as one row of `T` identical tokens each holding `-loss`, with a unit mask: the sequence log-ratio is `(-loss*T - old)/T`, and `seq_mean` over identical tokens is the row value. The REINFORCE fallback (`advantage * loss`, no old log-probs) stays as is per the spec; only the clipped branch and the leave-one-out baseline are routed.

- [ ] **Step 1: Add helpers** after `_resolve_model_device` (and `from . import objectives` next to `from . import rl_losses`):

```python
def _grpo_objective(config: Any) -> Any:
    """GRPO trainers score a trajectory as summed log-probs, so the ratio is
    sequence-level and clipped at ``seq_clip_ratio`` (GSPO scale)."""
    seq_clip = float(getattr(config, "seq_clip_ratio", 3e-4))
    return objectives.OBJECTIVES["gspo"].with_(
        name="grpo_sequence", clip_low=seq_clip, clip_high=seq_clip, kl="none"
    )


def _clipped_trajectory_loss(
    nll: Any, token_count: int, advantage: Any, old_log_prob: Any, objective: Any
) -> Any:
    """Clipped sequence-ratio loss for one trajectory via ``objectives.policy_loss``.

    ``nll`` is the model's mean per-token NLL (differentiable); ``old_log_prob``
    the rollout-time summed log-prob. The trajectory is presented as
    ``token_count`` identical tokens so the sequence ratio normalises by the
    true length.
    """
    torch = require_torch()
    count = max(int(token_count), 1)
    logp_cur = (-nll).reshape(1, 1).expand(1, count)
    mask = torch.ones(1, count, dtype=logp_cur.dtype, device=logp_cur.device)
    adv = (
        advantage.reshape(1)
        if torch.is_tensor(advantage)
        else torch.tensor([float(advantage)], device=logp_cur.device)
    )
    result = objectives.policy_loss(
        logp_cur=logp_cur,
        mask=mask,
        advantages=adv.to(logp_cur.dtype),
        objective=objective,
        logp_old=old_log_prob.reshape(1).to(logp_cur.dtype),
    )
    return result.loss
```

- [ ] **Step 2: Route the clipped branch of `_compute_group_policy_loss`**

Add `objective = _grpo_objective(config)` next to `seq_clip = getattr(config, "seq_clip_ratio", 3e-4)` (delete the `seq_clip` line and the `compute_ppo_ratio` import usage below). Replace

```python
                    ratio = compute_ppo_ratio(new_log_prob, old_log_prob, token_count)
                    policy_loss = rl_losses.clipped_surrogate(
                        ratio, advantage, clip_low=seq_clip, clip_high=seq_clip
                    )
```

with

```python
                    policy_loss = _clipped_trajectory_loss(
                        outputs.loss, token_count, advantage, old_log_prob, objective
                    )
```

The `new_log_prob = -(outputs.loss * token_count)` line becomes unused; delete it and the `new_log_prob is not None` condition in the `if` above (the branch condition becomes `clip_ratio > 0 and old_log_prob is not None and token_count > 0`). `compute_ppo_ratio` itself stays exported (tests use it).

- [ ] **Step 3: Route the clipped branch of `compute_enhanced_grpo_loss`**

Add `objective = _grpo_objective(config)` before the group loop. Replace

```python
                if clip_ratio > 0 and old_log_prob is not None and token_count > 0:
                    new_log_prob = -(nll * token_count)
                    ratio = compute_ppo_ratio(new_log_prob, old_log_prob, token_count)
                    policy_loss = rl_losses.clipped_surrogate(
                        ratio, advantage, clip_low=seq_clip, clip_high=seq_clip
                    )
```

with

```python
                if clip_ratio > 0 and old_log_prob is not None and token_count > 0:
                    policy_loss = _clipped_trajectory_loss(
                        nll, token_count, advantage, old_log_prob, objective
                    )
```

and delete the now-unused `seq_clip = getattr(config, "seq_clip_ratio", 3e-4)` line. Leave the advantage computation at the top of the group loop untouched: it uses the unbiased std (`advantages.std()`), and the `grpo` preset uses `correction=0`; changing it would move the `grpo_enhanced` golden. Add the comment `# Historical unbiased-std normalisation; the grpo preset uses correction=0 (docs/OBJECTIVES.md).` above it. Leave the exact full-vocab KL block untouched.

- [ ] **Step 4: Route `compute_grpo_loss`'s `leave_one_out` baseline through the library**

Replace the `elif baseline_type == "leave_one_out":` block with:

```python
        elif baseline_type == "leave_one_out":
            # RLOO baseline: r_i minus the mean of the other rewards. For a
            # single sample the library returns advantage 0, i.e. baseline r.
            baseline = rewards - objectives.compute_advantages(
                rewards,
                torch.zeros_like(rewards, dtype=torch.long),
                objectives.OBJECTIVES["rloo"],
            )
```

- [ ] **Step 5: Run**

Run: `.venv/bin/python -m pytest tests/unit/test_objective_goldens.py tests/unit/test_loss_computation_behavioral.py tests/unit/test_loss_computation_masks.py tests/unit/test_grpo_complete.py tests/unit/test_multi_turn_trainer.py tests/unit/test_distributed_trainer.py tests/integration/test_stub_training_loop.py -p no:cacheprovider -o addopts="" -q`
Expected: all pass with goldens unchanged. `test_ppo_clip_uses_seq_clip_ratio` in `test_loss_computation_behavioral.py` spies on `rl_losses.clipped_surrogate` and expects `clip_low == clip_high == 7e-4`; the objective calls the same function with the same values, so it still passes.

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff check stateset_agents/training/loss_computation.py && .venv/bin/black stateset_agents/training/loss_computation.py && .venv/bin/isort stateset_agents/training/loss_computation.py
git add stateset_agents/training/loss_computation.py
git commit -m "refactor(training): route GRPO clipped losses and RLOO baseline through PolicyObjective"
```

---

### Task 12: Exports, docs, changelog, maturity contract, Rust parity

**Files:**
- Modify: `stateset_agents/training/_registry.py`
- Create: `docs/OBJECTIVES.md`
- Modify: `docs/ADVANCED_RL_ALGORITHMS.md` (after `## Algorithm Comparison` table), `docs/ARCHITECTURE.md` (training layer section), `docs/COMPARISONS.md` (TRL feature table row "Group-based RL algorithms"), `README.md` (trainer bullet), `CHANGELOG.md` (`[Unreleased]`), `contracts/component_maturity_v1.json` (GSPO/DAPO/GEPO limitation text)
- Test: `tests/unit/test_objectives_exports.py` (new), `tests/unit/test_objectives_rust_parity.py` (new)

- [ ] **Step 1: Exports test**

```python
# tests/unit/test_objectives_exports.py
def test_objectives_are_lazily_exported():
    import stateset_agents.training as training

    assert training.PolicyObjective.__name__ == "PolicyObjective"
    assert "grpo" in training.OBJECTIVES
    assert callable(training.policy_loss) and callable(training.compute_advantages)
    assert training.PolicyLossResult.__name__ == "PolicyLossResult"
```

Add to `OPTIONAL_EXPORTS` in `_registry.py` (after the `compute_importance_weights` entry):

```python
    # Declarative policy-optimisation objectives (torch-free import).
    "PolicyObjective": (f"{_PKG}.objectives", "PolicyObjective"),
    "PolicyLossResult": (f"{_PKG}.objectives", "PolicyLossResult"),
    "OBJECTIVES": (f"{_PKG}.objectives", "OBJECTIVES"),
    "compute_advantages": (f"{_PKG}.objectives", "compute_advantages"),
    "policy_loss": (f"{_PKG}.objectives", "policy_loss"),
```

Check `tests/unit/test_training_exports*.py` or similar for a hard-coded export list (`grep -rn "compute_importance_weights" tests/unit | grep -v objectives`) and add the five names there too.

- [ ] **Step 2: Rust parity test**

```python
# tests/unit/test_objectives_rust_parity.py
"""The optional Rust group-advantage kernel must agree with ``group_norm``."""

import pytest

torch = pytest.importorskip("torch")
rust = pytest.importorskip("stateset_rl_core")

from stateset_agents.training import objectives as O  # noqa: E402


def test_rust_group_advantages_match_group_norm():
    rewards = [0.1, 0.9, 0.4, 0.4, 0.7]
    got = rust.compute_advantages_for_group(rewards, "mean", True)
    want = O.compute_advantages(
        torch.tensor(rewards), torch.zeros(5, dtype=torch.long), O.OBJECTIVES["grpo"]
    )
    torch.testing.assert_close(torch.tensor(got, dtype=torch.float32), want, atol=1e-6, rtol=0)
```

Confirm the Python binding name with `grep -n "fn compute_advantages" rust_core/src/lib.rs` and adjust the call if the exported name differs (e.g. `compute_group_advantages`). The test skips when the extension is not built.

- [ ] **Step 3: Write `docs/OBJECTIVES.md`**

Contents (write in full, no abbreviations):

1. Title, one-paragraph purpose, import example:
   ```python
   from stateset_agents.training import OBJECTIVES, compute_advantages, policy_loss
   obj = OBJECTIVES["dapo"].with_(clip_high=0.3)
   adv = compute_advantages(rewards, group_ids, obj)
   out = policy_loss(logp_cur=lp, mask=m, advantages=adv, objective=obj, logp_old=old)
   out.loss.backward()
   ```
2. The five taxonomy tables from spec §1.2–§1.6 verbatim.
3. Preset table from spec §1.8 plus a citation column: GRPO (Shao et al. 2024, DeepSeekMath), Dr. GRPO (Liu et al. 2025, "Understanding R1-Zero-Like Training"), BNPO (TRL loss_type), DAPO (Yu et al. 2025), GSPO/GSPO-token (Zheng et al. 2025), GEPO (StateSet `gepo_trainer` docstring reference), RLOO (Ahmadian et al. 2024), REINFORCE++-baseline (Hu 2025; note: this preset follows TRL's `scale_rewards="batch"`, dividing group-centred rewards by the batch std of raw rewards, whereas verl normalises the centred scores), CISPO (Chen et al. 2025, MiniMax-M1), PPO (Schulman et al. 2017).
4. "Which preset" guide: verifiable single-turn math → `dapo` or `dr_grpo`; long multi-turn dialogue with stale rollouts → `gspo`; value-augmented → `ppo`/VAPO; off-policy async producers → `cispo` or `gepo`.
5. Verification section: the three layers and how to run them (`tests/unit/test_objectives*.py`), the golden pins and how to regenerate them, and the two known divergences: (a) enhanced GRPO keeps unbiased-std advantage normalisation, (b) GSPO-token now reports the surrogate value as `policy_loss`.
6. Trainer map: which preset each trainer instantiates and from which config fields.

- [ ] **Step 4: Cross-link docs**

- `docs/ADVANCED_RL_ALGORITHMS.md`: after the comparison table add a paragraph "Every trainer below evaluates a named `PolicyObjective`; see [OBJECTIVES.md](OBJECTIVES.md) for the exact formulas, presets, and the TRL pin."
- `docs/ARCHITECTURE.md`: in the training-layer description, add `objectives.py` above `rl_losses.py` with one line each.
- `docs/COMPARISONS.md`: change the TRL row "Group‑based RL algorithms | Yes (GRPO/GSPO/GEPO/DAPO/VAPO) | Partial" to "Yes — 11 presets in `training/objectives.py`, losses pinned to TRL 1.12 on CPU fixtures | grpo/bnpo/dr_grpo/dapo/cispo/sapo/vespo/luspo loss types" and add a sentence under "What the benchmark establishes" noting the pin is a correctness check, not a benchmark.
- `README.md`: in the trainer bullet, append "all trainers evaluate one declarative [`PolicyObjective`](docs/OBJECTIVES.md) (GRPO, Dr. GRPO, BNPO, DAPO, GSPO, GSPO-token, GEPO, RLOO, REINFORCE++-baseline, CISPO, PPO), verified against loop references, Hypothesis invariants, and TRL 1.12".
- `CHANGELOG.md` `[Unreleased]`:
  ```
  ### Added
  - `stateset_agents.training.objectives`: declarative `PolicyObjective` with eleven presets and `compute_advantages`/`policy_loss`; verified by loop references, Hypothesis property tests, and a numeric pin against TRL 1.12 (`docs/OBJECTIVES.md`).
  - Golden regression pins for every native trainer's objective (`scripts/capture_objective_goldens.py`).
  ### Changed
  - DAPO, VAPO, GSPO, GSPO-token, GEPO, PPO, and GRPO trainers evaluate their objective through `objectives.policy_loss`; numerics unchanged except below.
  - GSPO-token reports the clipped-surrogate value as `policy_loss` (gradients unchanged; previously the log-prob-weighted quantity).
  ### Fixed
  - PPO's KL penalty now uses the k3 estimator (the previous `log π − log π_ref` mean had a zero-expectation gradient) and clamps the log-ratio before exponentiating; PPO golden `kl` moved from <old> to <new>.
  ```
  Fill `<old>`/`<new>` from the Task 10 diff.
- `contracts/component_maturity_v1.json`: for the component named "GSPO, DAPO, and GEPO trainers", append to its limitations string: "; objectives are pinned against TRL 1.12 on CPU fixtures (tests/unit/test_objectives_trl_pin.py)". Run `make release-governance` afterwards; if the contract validator rejects the edit, read its error and adjust to the schema (evidence references may need to be a separate array field — follow the existing entries' shape).

- [ ] **Step 5: Run everything**

```bash
.venv/bin/python -m pytest tests/unit/test_objectives_exports.py tests/unit/test_objectives_rust_parity.py tests/unit/test_readme_cli_snippets.py -p no:cacheprovider -o addopts="" -q
make release-governance
.venv/bin/ruff check stateset_agents tests scripts && .venv/bin/black --check stateset_agents tests scripts && .venv/bin/isort --check-only stateset_agents tests scripts
.venv/bin/python scripts/check_types.py --all
.venv/bin/python scripts/check_repo_hygiene.py
.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -15
```
Expected: all green; coverage ratchet satisfied (new module is fully covered by its tests). `surrogate`/`aggregate` were made public in Task 5.

- [ ] **Step 6: Commit**

```bash
git add stateset_agents/training/_registry.py docs/OBJECTIVES.md docs/ADVANCED_RL_ALGORITHMS.md docs/ARCHITECTURE.md docs/COMPARISONS.md README.md CHANGELOG.md contracts/component_maturity_v1.json tests/unit/test_objectives_exports.py tests/unit/test_objectives_rust_parity.py
git commit -m "docs(training): document PolicyObjective presets, verification, and trainer map"
```

---

## Self-review

- **Spec coverage:** §1.1–1.8 → Task 1; §2 layers 1/2/3 → Tasks 1/3/2; §3 rows 1–8 → Tasks 5–11 with goldens from Task 4; §4 exports/docs/changelog/maturity → Task 12; Rust parity (out-of-scope note in spec) → Task 12 Step 2.
- **Deviation from spec, recorded:** spec §1.1 says `seq_sum_const` without `max_completion_length` is rejected in `__post_init__`; presets need `None` so the check lives in `policy_loss` (spec §1.8 already says so). Spec §3 says GSPO-token has no numeric change; the loss *value* changes while gradients are pinned — documented in Task 8 and the CHANGELOG. Enhanced GRPO keeps its unbiased-std advantages and the REINFORCE fallback stays `advantage * loss` (Task 11).
- **Type consistency:** `policy_loss` keyword names (`logp_cur, mask, advantages, objective, logp_old, logp_ref, group_ids, kl, entropy`) are identical in Tasks 1, 2, 3, 5–11. `PolicyLossResult.metrics["clip_fraction"]` used by Task 10 is defined in Task 1. `compute_gspo_loss`, `compute_gspo_token_loss`, `compute_gepo_loss` signatures match between Task 4 (extraction), the capture script, and Tasks 7–9.
