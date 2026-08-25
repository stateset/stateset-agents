# A+ Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take `stateset_agents` from B− to A+ by adding a shared RL-loss spine that fixes four algorithm bugs, turning every CI gate green, hardening checkpoint loading, fixing docs drift, and paying down the duplicated-trainer / copy-pasted-CLI architecture debt.

**Architecture:** A new pure-tensor module `stateset_agents/training/rl_losses.py` becomes the single implementation of token-logprob gathering, group advantages, k3 KL, clip gating and masked means; the five trainers call it. Everything else is targeted fixes plus meta-tests that keep the fixes from regressing (torch-import policy, core↔experimental layering, README CLI snippets).

**Tech Stack:** Python 3.10, torch 2.6 (CPU in tests), pytest + pytest-xdist, ruff, mypy, typer.

**Spec:** `docs/superpowers/specs/2026-08-24-a-plus-hardening-design.md`

## Global Constraints

- No CLI command name or flag may change (README snippet test + existing CLI tests enforce).
- `ruff check .`, `mypy --config-file mypy.ini`, and `pytest` must be green after every task; coverage `fail_under = 57` must hold.
- Torch is obtained via `from stateset_agents.training.trainer_utils import get_torch, require_torch`; new modules must not `import torch` at module level.
- Every test that needs torch uses `pytest.importorskip("torch")`.
- Commit after each task with a conventional-commit message.

---

## Phase 1 — RL loss spine

### Task 1.1: `rl_losses.py` — gather + masked_mean

**Files:**
- Create: `stateset_agents/training/rl_losses.py`
- Test: `tests/unit/test_rl_losses.py`

**Interfaces — Produces:**
```python
def gather_token_logprobs(logits, input_ids, response_mask) -> tuple[Tensor, Tensor]
    # logits [B,T,V], input_ids [B,T], response_mask [B,T] (1 on response tokens)
    # returns (token_logprobs [B,T-1] already multiplied by mask, shifted_mask [B,T-1])
def masked_mean(x, mask, *, mode: str = "token") -> Tensor
    # mode="token": sum(x*mask)/max(sum(mask),1); mode="seq": mean over rows of (row sum / max(row count,1))
```

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_rl_losses.py
import pytest
torch = pytest.importorskip("torch")
from stateset_agents.training import rl_losses as L


def _naive_gather(logits, ids, mask):
    lp = torch.log_softmax(logits[:, :-1], -1)
    out = torch.zeros(ids.shape[0], ids.shape[1] - 1)
    for b in range(ids.shape[0]):
        for t in range(ids.shape[1] - 1):
            out[b, t] = lp[b, t, ids[b, t + 1]] * mask[b, t + 1]
    return out, mask[:, 1:]


def test_gather_matches_naive_loop():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 7, generator=g)
    ids = torch.randint(0, 7, (2, 5), generator=g)
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]], dtype=torch.float32)
    got, got_mask = L.gather_token_logprobs(logits, ids, mask)
    want, want_mask = _naive_gather(logits, ids, mask)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_mask, want_mask)


def test_masked_mean_token_and_seq():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]])
    m = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    assert L.masked_mean(x, m, mode="token").item() == pytest.approx(10 / 4)
    assert L.masked_mean(x, m, mode="seq").item() == pytest.approx((2.0 + 4.0) / 2)


def test_masked_mean_empty_mask_is_zero_not_nan():
    x = torch.ones(2, 3)
    m = torch.zeros(2, 3)
    assert L.masked_mean(x, m).item() == 0.0
```

- [ ] **Step 2: Run** `pytest tests/unit/test_rl_losses.py -v` → FAIL (`No module named rl_losses`).

- [ ] **Step 3: Implement**

```python
"""Shared, stateless RL loss primitives used by every trainer.

Pure tensor functions. No trainer state, no model calls. torch is fetched
lazily so importing this module never requires torch.
"""
from __future__ import annotations

from typing import Any

from .trainer_utils import get_torch, require_torch


def _t() -> Any:
    return get_torch() or require_torch()


def gather_token_logprobs(logits: Any, input_ids: Any, response_mask: Any) -> tuple[Any, Any]:
    """Shift-by-one gather of per-token log-probs, masked to response tokens.

    Returns ``(token_logprobs, shifted_mask)`` both of shape ``[B, T-1]``.
    ``token_logprobs`` is already multiplied by ``shifted_mask``.
    """
    torch = _t()
    shift_logits = logits[..., :-1, :]
    shift_labels = input_ids[..., 1:]
    shifted_mask = response_mask[..., 1:].to(shift_logits.dtype)
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs * shifted_mask, shifted_mask


def masked_mean(x: Any, mask: Any, *, mode: str = "token") -> Any:
    """Mean of ``x`` over positions where ``mask`` is 1.

    ``mode="token"``: one global mean over all masked tokens (DAPO style).
    ``mode="seq"``: per-row mean, then mean over rows (GRPO/GSPO style).
    An all-zero mask yields 0, never NaN.
    """
    torch = _t()
    mask = mask.to(x.dtype)
    if mode == "token":
        return (x * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    if mode == "seq":
        per_row = (x * mask).sum(-1) / torch.clamp(mask.sum(-1), min=1.0)
        return per_row.mean()
    raise ValueError(f"unknown mode {mode!r}; expected 'token' or 'seq'")
```

- [ ] **Step 4: Run** tests → PASS. `ruff check stateset_agents/training/rl_losses.py tests/unit/test_rl_losses.py`.
- [ ] **Step 5: Commit** `feat(rl_losses): shared token-logprob gather and masked_mean`

### Task 1.2: `group_advantages`

**Files:** Modify `rl_losses.py`, `tests/unit/test_rl_losses.py`

**Produces:**
```python
def group_advantages(rewards, *, normalize: bool = True, eps: float = 1e-8) -> Tensor
    # rewards [G] for ONE group. mean-baseline; if normalize, divide by std(correction=0).
    # G==1 or non-finite/zero std -> returns zeros (no NaN).
```

- [ ] **Step 1: Tests**

```python
def test_group_advantages_matches_manual():
    r = torch.tensor([1.0, 2.0, 3.0, 6.0])
    a = L.group_advantages(r)
    want = (r - r.mean()) / (r.std(correction=0) + 1e-8)
    torch.testing.assert_close(a, want)
    assert a.mean().abs().item() < 1e-6


def test_group_advantages_single_sample_is_zero_not_nan():
    a = L.group_advantages(torch.tensor([0.7]))
    assert a.shape == (1,) and a.item() == 0.0 and torch.isfinite(a).all()


def test_group_advantages_constant_rewards_zero():
    a = L.group_advantages(torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(a, torch.zeros(3))


def test_group_advantages_unnormalized():
    r = torch.tensor([0.0, 2.0])
    torch.testing.assert_close(L.group_advantages(r, normalize=False), torch.tensor([-1.0, 1.0]))
```

- [ ] **Step 2:** run → FAIL. **Step 3: Implement**

```python
def group_advantages(rewards: Any, *, normalize: bool = True, eps: float = 1e-8) -> Any:
    """Group-relative advantages for one group of rewards ``[G]``.

    Groups of size 1, constant rewards, or non-finite statistics yield zeros
    rather than NaN (a NaN advantage silently poisons the whole batch).
    """
    torch = _t()
    rewards = rewards.float()
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    adv = rewards - rewards.mean()
    if not normalize:
        return adv
    std = adv.std(correction=0)
    if not torch.isfinite(std) or std <= eps:
        return torch.zeros_like(rewards)
    return adv / (std + eps)
```

- [ ] **Step 4:** PASS. **Step 5: Commit** `feat(rl_losses): NaN-safe group_advantages`

### Task 1.3: `clipped_surrogate` and `sequence_ratio`

**Produces:**
```python
def clipped_surrogate(ratio, advantages, *, clip_low: float, clip_high: float) -> Tensor
    # elementwise -min(ratio*A, clamp(ratio,1-lo,1+hi)*A)   (a LOSS, positive = bad)
def sequence_ratio(logp_cur, logp_old, mask) -> Tensor
    # per-row exp( sum((logp_cur-logp_old)*mask) / max(sum(mask),1) )  — GSPO length-normalised
def clip_fraction(ratio, *, clip_low, clip_high) -> float
```

- [ ] **Step 1: Tests**

```python
def test_clipped_surrogate_zero_advantage_zero_grad():
    logp = torch.zeros(3, requires_grad=True)
    ratio = torch.exp(logp - torch.tensor([0.1, -0.1, 0.0]))
    loss = L.clipped_surrogate(ratio, torch.zeros(3), clip_low=0.2, clip_high=0.2).sum()
    loss.backward()
    assert torch.equal(logp.grad, torch.zeros(3))


def test_clipped_surrogate_out_of_region_has_zero_grad_inside_has_grad():
    # ratio 1.5 with A>0 is above 1+clip_high -> clipped branch wins -> no grad
    logp = torch.tensor([0.0, 0.0], requires_grad=True)
    old = torch.tensor([-0.405465, 0.0])  # exp(0.405)=1.5 ; exp(0)=1.0
    ratio = torch.exp(logp - old)
    loss = L.clipped_surrogate(ratio, torch.tensor([1.0, 1.0]), clip_low=0.2, clip_high=0.2).sum()
    loss.backward()
    assert logp.grad[0].item() == 0.0
    assert logp.grad[1].item() != 0.0


def test_sequence_ratio_length_normalised():
    cur = torch.tensor([[0.0, -1.0, -1.0]])
    old = torch.tensor([[0.0, -2.0, -2.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    torch.testing.assert_close(L.sequence_ratio(cur, old, mask), torch.tensor([torch.e]))


def test_clip_fraction():
    ratio = torch.tensor([1.0, 1.5, 0.5, 1.1])
    assert L.clip_fraction(ratio, clip_low=0.2, clip_high=0.2) == pytest.approx(0.5)
```

- [ ] **Step 3: Implement**

```python
def clipped_surrogate(ratio: Any, advantages: Any, *, clip_low: float, clip_high: float) -> Any:
    """PPO/GSPO/DAPO clipped surrogate *loss* (elementwise, not reduced).

    ``-min(r·A, clip(r)·A)``. When the ratio leaves the trust region on the
    side the advantage would push it, the clipped branch is selected and,
    because ``clamp`` has zero gradient there, the sample contributes no
    gradient — that is the mechanism that bounds the policy step.
    """
    torch = _t()
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    return -torch.min(ratio * advantages, clipped * advantages)


def sequence_ratio(logp_cur: Any, logp_old: Any, mask: Any) -> Any:
    """GSPO length-normalised sequence importance ratio, one value per row."""
    torch = _t()
    mask = mask.to(logp_cur.dtype)
    log_ratio = ((logp_cur - logp_old) * mask).sum(-1) / torch.clamp(mask.sum(-1), min=1.0)
    return torch.exp(log_ratio)


def clip_fraction(ratio: Any, *, clip_low: float, clip_high: float) -> float:
    """Fraction of ratios outside the trust region (for logging)."""
    out = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    return float(out.float().mean().item()) if ratio.numel() else 0.0
```

- [ ] **Step 4:** PASS. **Step 5: Commit** `feat(rl_losses): clipped_surrogate, sequence_ratio, clip_fraction`

### Task 1.4: `k3_kl`

**Produces:**
```python
def k3_kl(logp_cur, logp_ref, mask=None) -> Tensor
    # r = logp_ref - logp_cur ; k3 = exp(r) - r - 1  (>=0, unbiased for KL(cur||ref))
    # if mask given: masked_mean(k3, mask, mode="seq") ; else mean over elements
```

- [ ] **Step 1: Tests**

```python
def test_k3_kl_nonnegative_and_zero_at_equality():
    cur = torch.tensor([[-1.0, -2.0]]); ref = torch.tensor([[-1.5, -1.0]])
    assert L.k3_kl(cur, ref).item() >= 0
    assert L.k3_kl(cur, cur).item() == 0.0


def test_k3_kl_gradient_pulls_toward_ref():
    ref = torch.tensor([[-1.0, -1.0]])
    cur = torch.tensor([[-2.0, -0.5]], requires_grad=True)
    before = L.k3_kl(cur, ref)
    before.backward()
    with torch.no_grad():
        cur2 = cur - 0.1 * cur.grad
    after = L.k3_kl(cur2, ref)
    assert after.item() < before.item()


def test_k3_kl_respects_mask():
    cur = torch.tensor([[0.0, -5.0]]); ref = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert L.k3_kl(cur, ref, mask).item() == 0.0
```

- [ ] **Step 3: Implement**

```python
def k3_kl(logp_cur: Any, logp_ref: Any, mask: Any | None = None) -> Any:
    """Schulman's k3 estimator of KL(π_cur ‖ π_ref) from sampled log-probs.

    ``k3 = exp(r) − r − 1`` with ``r = log π_ref − log π_cur``. It is
    non-negative, unbiased, and — unlike the naive ``log π_cur − log π_ref``
    — has a gradient whose expectation is the true KL gradient, so the
    penalty actually pulls the policy toward the reference.
    """
    torch = _t()
    r = logp_ref.detach() - logp_cur
    k3 = torch.exp(r) - r - 1.0
    if mask is None:
        return k3.mean()
    return masked_mean(k3, mask, mode="seq")
```

- [ ] **Step 4:** PASS. **Step 5: Commit** `feat(rl_losses): k3 KL estimator`

### Task 1.5: Wire GSPO + GSPO-token to `rl_losses`

**Files:** Modify `stateset_agents/training/gspo_trainer.py:387-416, 645-673`; `gspo_token_trainer.py:205-300`.

Changes (keep public method names — `compute_sequence_importance_ratio`, `compute_group_advantages` — as thin wrappers):

- `gspo_trainer.py:671-673` KL block becomes:
```python
                ref_log_probs = self._compute_batch_ref_log_probs(prompt, responses)
                if model_device is not None:
                    ref_log_probs = ref_log_probs.to(model_device)
                # k3 on the length-normalised sequence log-probs (one value per response)
                kl_div = rl_losses.k3_kl(
                    current_log_probs / sequence_lengths,
                    ref_log_probs / sequence_lengths,
                )
                kl_penalty = self.config.beta * kl_div
```
- `gspo_trainer.py:645-662` policy loss becomes `policy_loss = rl_losses.clipped_surrogate(importance_ratios, advantages, clip_low=self.config.clip_range_left, clip_high=self.config.clip_range_right).mean()`; keep `num_clipped` counting.
- `gspo_token_trainer.py:266-283`: replace the unconditional detached weight with a clip gate. Per response:
```python
                seq_ratio = importance_ratios[i]            # detached
                adv = advantages[i].detach()
                lo, hi = 1 - self.config.clip_range_left, 1 + self.config.clip_range_right
                # GSPO-token: gradient through token log-probs, weighted by the
                # stop-grad sequence ratio; gate to zero when the clipped
                # branch of min(r·A, clip(r)·A) is active.
                in_region = (seq_ratio >= lo) & (seq_ratio <= hi)
                push_out = ((adv > 0) & (seq_ratio > hi)) | ((adv < 0) & (seq_ratio < lo))
                gate = (in_region | ~push_out).to(token_log_probs.dtype)
                token_loss = -(gate * seq_ratio * adv * token_log_probs).sum() / sequence_lengths[i]
                loss += token_loss / len(responses)
```
- `gspo_token_trainer.py:298-299` KL → same k3 replacement as above.
- Add `from . import rl_losses` at the top of both modules.

- [ ] **Step 1: Tests** — add to `tests/unit/test_gspo_trainer.py` and `tests/unit/test_gspo_token_trainer_behavioral.py` (follow existing stub-model fixtures in those files):

```python
def test_gspo_kl_penalty_uses_k3(monkeypatch, gspo_trainer_with_ref):
    # k3 >= 0 always; the old estimator could be negative.
    seen = {}
    real = rl_losses.k3_kl
    def spy(*a, **k):
        out = real(*a, **k); seen["v"] = out.item(); return out
    monkeypatch.setattr(rl_losses, "k3_kl", spy)
    asyncio.run(gspo_trainer_with_ref.train_step(["hi"], num_groups=1))
    assert "v" in seen and seen["v"] >= 0.0


def test_gspo_token_out_of_region_sequence_gets_no_gradient(...):
    # Build a group where one response's old log-prob is far below current
    # (ratio >> 1+clip_right) with positive advantage; assert the parameter
    # gradient equals the gradient with that response removed.
```
(The fixture names must match those already in the two test files; read them first.)

- [ ] **Step 2–4:** run the two test files + `tests/unit/test_gspo_scoring_consistency.py` → PASS.
- [ ] **Step 5: Commit** `fix(gspo): k3 KL penalty and clip gate for GSPO-token; use rl_losses`

### Task 1.6: Wire DAPO / GEPO / VAPO to `rl_losses`

**Files:** `dapo_trainer.py:455-473, 494-527, 692-695`; `gepo_trainer.py:358-364`; `vapo_trainer.py:521-528, 745-760`.

- `compute_token_log_probs` in DAPO/VAPO → `return rl_losses.gather_token_logprobs(logits, input_ids, response_mask)` (VAPO's variant has no mask param; pass `torch.ones_like(input_ids)` and drop the second return).
- DAPO `compute_dapo_loss` body → `surrogate = -rl_losses.clipped_surrogate(...)`, then `masked_mean(surrogate, response_mask, mode="token" if self.config.use_token_level_loss else "seq")` negated. Keep the method.
- DAPO `:692-695` and GEPO `:358-364` → `advantages = rl_losses.group_advantages(rewards_tensor)`; GEPO keeps its `stats` dict (compute from `rewards` directly).
- VAPO `:745-760` → `clipped_surrogate` + `masked_mean` same as DAPO.

- [ ] **Step 1: Tests** — add to `tests/unit/test_dapo_trainer_behavioral.py` and `test_gepo_trainer_behavioral.py`:

```python
def test_group_of_one_advantage_is_finite(trainer):
    adv = trainer.compute_group_advantages(torch.tensor([0.3]))
    adv = adv[0] if isinstance(adv, tuple) else adv
    assert torch.isfinite(adv).all() and adv.item() == 0.0
```
Existing behavioral tests are the regression pin for the loss values.

- [ ] **Step 2–4:** `pytest tests/unit/test_dapo* tests/unit/test_gepo* tests/unit/test_vapo*` → PASS.
- [ ] **Step 5: Commit** `refactor(dapo,gepo,vapo): use rl_losses primitives; NaN-safe advantages`

### Task 1.7: GRPO fixes in `loss_computation.py`

**Files:** `stateset_agents/training/loss_computation.py:177-189, 252-341`; `training/config.py` (GRPO config clip defaults); `tests/unit/test_loss_computation_behavioral.py`.

- `_compute_group_policy_loss` `:333-341`: drop the `/ _tc` division entirely (outputs.loss is already a per-token mean). Replace with a comment: `# outputs.loss is already the per-token mean NLL; no further length division.`
- `:177-189` `compute_ppo_ratio` stays (sequence-mean ratio) but the default clip read at `:253` becomes `getattr(config, "seq_clip_ratio", 3e-4)` when the ratio is sequence-mean; add `seq_clip_ratio: float = 3e-4` to the GRPO `TrainingConfig` with a docstring saying the ±0.2 PPO clip is meaningless for a length-normalised ratio (cite GSPO).
- Use `rl_losses.clipped_surrogate(ratio, advantage, clip_low=clip, clip_high=clip)` instead of the inline min.

- [ ] **Step 1: Tests**

```python
def test_token_level_loss_not_double_normalised(stub_agent, grpo_config):
    grpo_config.token_level_loss = True
    # Two trajectories, same per-token NLL, lengths 4 and 8 -> equal loss.
    ...assert loss_short == pytest.approx(loss_long, rel=1e-4)


def test_sequence_ratio_clip_is_active():
    # ratio 1.001 with seq_clip_ratio 3e-4 and A>0 -> clipped branch -> zero grad
```

- [ ] **Step 5: Commit** `fix(grpo): remove 1/L² length bias; sequence-scale clip`

### Task 1.8: `distributed_trainer._compute_grpo_loss`

**Files:** `stateset_agents/training/distributed_trainer.py:375-391`; `tests/unit/test_distributed_trainer.py` (create if absent).

Replace the three placeholders: `_generate_trajectories` raises `NotImplementedError("DistributedTrainer requires an agent with generate_trajectories")` unless `self.agent` provides one; `_compute_rewards` calls `self.reward_fn`; `_compute_grpo_loss` builds a `TrajectoryGroup` and delegates to `loss_computation.compute_grpo_loss`. If the surrounding class cannot support that in a bounded change, delete the class and its export with a `DeprecationWarning` shim — a trainer that returns loss 0.0 must not exist.

- [ ] Test: `test_distributed_trainer_loss_is_not_constant` — two different reward lists give different losses.
- [ ] **Commit** `fix(distributed): real GRPO loss instead of 0.0 placeholder`

---

## Phase 2 — Green gates

### Task 2.1: `RemoteExecutor.undeploy`
**Files:** `stateset_agents/remote/executor.py:53+`, `stateset_agents/cli_remote.py:749`, `tests/unit/test_cli_remote.py` (or nearest existing CLI test file).
- Add to the ABC:
```python
    def undeploy(self, deployment_id: str) -> None:
        """Tear down a managed deployment. Providers without deployments raise."""
        raise NotImplementedError(f"{type(self).__name__} does not manage deployments")
```
- Test: `CliRunner().invoke(app, ["undeploy", "--provider", "fireworks", "dep-1"])` with `get_executor` monkeypatched to a stub recording the call; and a provider without `undeploy` exits non-zero with the message.
- **Commit** `fix(remote): undeploy on RemoteExecutor ABC + CLI test`

### Task 2.2: remaining mypy errors
- `rewards/nsr_verifier.py:58,202` — annotate the JSON access: `return cast(str | None, ...)` / `cast(dict[str, Any], ...)`.
- `training/harvest.py:141` — `return bool(...)`.
- `remote/river.py:95` — delete the unused `# type: ignore`.
- Verify: `mypy --config-file mypy.ini` → 0 errors. **Commit** `chore(types): zero mypy errors`

### Task 2.3: fix the 3 failing unit tests
- Get IDs from `scratchpad/unit.log` (`grep ^FAILED`). For each: use systematic-debugging; fix root cause, not the assertion. **Commit** per fix.

### Task 2.4: clean pytest exit
- In `tests/conftest.py` add a session-scoped autouse fixture that sets `os.environ.setdefault("LITELLM_DISABLE_ASYNC_CLIENT_CLEANUP", "1")` *before* litellm is imported (put it at module top), and a `pytest_sessionfinish` hook that calls `logging.shutdown()` after removing wandb's console-capture handlers. Verify `pytest tests/unit/test_rl_losses.py 2>&1 | grep -c Traceback` → 0.
- **Commit** `test: silence litellm/wandb atexit tracebacks`

### Task 2.5: xdist default
- `pytest.ini` `addopts` add `-n auto --dist loadfile`; document `-p no:xdist` in `TESTING.md`. Run the full suite; fix any test that fails only under xdist (shared tmp paths, global registries). **Commit** `test: run suite under xdist by default`

---

## Phase 3 — Security

### Task 3.1: `torch.load` hardening
**Files:** `training/multi_turn_checkpointing.py:102,134`, `training/single_turn_checkpointing.py:100`, `core/value_function.py:445`.
- Each becomes `torch.load(path, map_location=map_location or "cpu", weights_only=not trusted)` where `trusted: bool = False` is a new keyword on the enclosing function. Test: saving a dict with a lambda inside and loading with default raises; `trusted=True` loads. **Commit** `security: weights_only=True on checkpoint loads`

### Task 3.2: `SECURITY.md`
- Supported versions table → `0.35.x: yes`, `< 0.35: no`. Add "Trust boundaries" section: checkpoints (`trusted=True` required for pickled objects), Redis cache pickles (`enhanced_state_cache.py`, `api/distributed_cache.py` — only connect to a Redis you control), `API_REQUIRE_AUTH` must stay `true` in production. Remove `security-announce@stateset.ai`/RSS lines; keep GitHub Security Advisories as the channel. **Commit** `docs(security): real supported versions and trust boundaries`

### Task 3.3: remove root `sitecustomize.py`
- Move the `FixtureDef.unittest` patch into `tests/conftest.py` (top of file, guarded by `try/except AttributeError`). Delete `sitecustomize.py`; grep `MANIFEST.in`/`pyproject.toml` for references. Full suite passes. **Commit** `chore: fold sitecustomize into conftest`

---

## Phase 4 — Docs / DX

### Task 4.1: README snippet test
**Files:** `tests/unit/test_readme_cli_snippets.py`; fix `README.md:1241`.
```python
import re, subprocess, sys, pathlib, pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
SNIPPET = re.compile(r"^\s*stateset-agents\s+(.*)$", re.M)

def _snippets():
    out = []
    for name in ("README.md", "QUICKSTART.md"):
        text = (ROOT / name).read_text()
        for m in SNIPPET.finditer(text):
            line = m.group(1).split("#")[0].rstrip("\\ ").strip()
            if line: out.append((name, line))
    return out

@pytest.mark.parametrize("src,line", _snippets(), ids=lambda x: x if isinstance(x, str) else "")
def test_readme_command_flags_exist(src, line):
    words = line.split()
    sub = [w for w in words if not w.startswith("-")][:2]  # up to 2 levels of subcommand
    flags = {w.split("=")[0] for w in words if w.startswith("--")}
    r = subprocess.run([sys.executable, "-m", "stateset_agents.cli", *sub, "--help"],
                       capture_output=True, text=True, env={**os.environ, "COLUMNS": "200"})
    assert r.returncode == 0, f"{src}: `{line}` — {r.stderr[-300:]}"
    for f in flags:
        assert f in r.stdout, f"{src}: `{line}` — unknown flag {f}"
```
(Adjust the module invocation to whatever `python -m` entry works; add a `stateset_agents/__main__.py` that calls `cli.run()` if none exists — it's a DX win anyway.)
- **Commit** `docs: README snippet test; fix --no-dry-run example`

### Task 4.2: CHANGELOG + stale docs
- `[Unreleased]`: NSR verifier reward (`rewards/nsr_verifier.py`, `improve --reward nsr`), rl_losses spine + the four fixes, security hardening, xdist default.
- `docs/CLI_REFERENCE.md:242` wheel name → `0.35.1`; `docs/ARCHITECTURE.md:161` describe `training/rl_losses.py` and the `experimental/` rule from Task 5.2. **Commit** `docs: changelog and architecture refresh`

---

## Phase 5 — Architecture debt

### Task 5.1: `cli_train.py` generator
**Files:** `stateset_agents/cli_train.py`, `stateset_agents/core/model_presets.py`, existing `tests/unit/test_cli_train*.py`.
- Before touching code: `stateset-agents train --help` and each model subcommand `--help` → save to `scratchpad/cli_help_before/`.
- Diff two of the ten commands (`qwen3_5_0_8b` vs `qwen3_8`) to enumerate every differing literal; add those as fields on the preset dataclass in `model_presets.py` (e.g. `default_lr`, `default_lora_r`, `default_max_len`, `recommended_gpu`).
- Write `_register_model_command(app: typer.Typer, preset: ModelPreset) -> None` that closes over the preset and defines one Typer command with identical name/flags/help. Loop over `MODEL_PRESETS` at import time.
- After: regenerate `--help` outputs, `diff -r` against before → identical. Delete the ten bodies.
- **Commit** `refactor(cli_train): data-driven model commands (−2400 LOC)`

### Task 5.2: break core↔experimental
- `core/agent.py:15`, `core/multiturn_agent.py:16`: move the `experimental.long_term_planning` import inside the method(s) that use it.
- `stateset_agents/__init__.py:55`: remove experimental symbols from `_LAZY_EXPORTS`; add a `__getattr__` branch that warns `DeprecationWarning("import from stateset_agents.experimental")` and still returns them (one release grace).
- Meta-test `tests/unit/test_layering.py`: parse every `core/**/*.py` with `ast`; assert no module-level `Import`/`ImportFrom` whose name starts with `stateset_agents.experimental` or `..experimental`.
- **Commit** `refactor(core): stop importing experimental at module level`

### Task 5.3: delete `core/enhanced/advanced_rl_algorithms.py` duplicates
- Replace the module body with shims: `PPOConfig`, `PPOTrainer` → `from ..training.ppo_trainer import ...`; `GSPOConfig`, `GSPOTrainer` → `training.gspo_config/gspo_trainer`; `DPOTrainer`, `A2CTrainer`, `GSPOTrainerStub` → removed (grep tests/examples for users first; if any, keep a `DeprecationWarning` stub that raises on instantiation). Module-level `warnings.warn(..., DeprecationWarning)`.
- **Commit** `refactor: remove duplicate RL algorithms from core/enhanced`

### Task 5.4: `TrainingConfig` consolidation
- `training/config.py:TrainingConfig` is canonical. `core/types.py:53` → `from ..training.config import TrainingConfig` re-export; `core/type_system.py:119` renamed `TrainingConfigDict` (keep `TrainingConfig = TrainingConfigDict` alias with deprecation); `training/advanced_training_models.py:57` → subclass canonical, adding only its extra fields. Second `GSPOConfig` disappears with Task 5.3. mypy + tests green.
- **Commit** `refactor(config): one TrainingConfig`

### Task 5.5: torch import policy meta-test
**Files:** `tests/unit/test_torch_import_policy.py`, `tests/unit/torch_import_allowlist.txt`.
```python
def test_no_unguarded_module_level_torch_import():
    allow = set(ALLOWLIST.read_text().split())
    bad = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:               # module level only
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(n == "torch" or n.startswith("torch.") for n in names):
                    rel = str(path.relative_to(PKG.parent))
                    if rel not in allow: bad.append(rel)
    assert not bad, f"unguarded module-level torch imports (wrap in try/except or use get_torch()): {bad}"
```
- Fix the ten unguarded sites (`training/distributed.py:17`, `ppo_trainer.py:25`, `gspo_trainer.py:19`, `ema.py:26`, `utils/profiler.py:17`, + the rest the test lists) by wrapping in `try: import torch except ImportError: torch = None; TORCH_AVAILABLE = False`. Allowlist starts empty if all ten are fixable; otherwise commit the residue.
- **Commit** `chore: enforce guarded torch imports`

---

## Final gate
- `ruff check . && mypy --config-file mypy.ini && pytest` all green; coverage ≥ 57 (bump `fail_under` to the new floor if higher).
- Re-run the three audit reviewers from the grading session for a fresh grade.
