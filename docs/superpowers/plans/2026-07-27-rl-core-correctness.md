# RL Core Correctness Implementation Plan (A+ push, Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five advertised RL algorithms (GRPO, GSPO, GEPO, DAPO, VAPO) mathematically correct and verifiable by behavioral tests against the real trainers.

**Architecture:** The unifying defect is that "old" policy log-probs are recomputed from the *current* model instead of being captured at rollout time, plus probability-space arithmetic that belongs in log space. We fix each trainer in place (no restructuring), thread rollout-time log-probs through the sample dicts that already exist, and add a shared behavioral test pattern: *ratio ≈ 1 at step 0 on the first inner update, ratio ≠ 1 after a parameter update*, using a tiny real HF model (`hf-internal-testing/tiny-random-GPT2LMHeadModel`, already usable offline via the test stubs — if unavailable, construct a 2-layer `GPT2LMHeadModel(GPT2Config(n_embd=32, n_layer=2, n_head=2, vocab_size=1000))` directly; no network needed).

**Tech Stack:** Python 3.10, torch, transformers (tiny configs in tests), pytest + pytest-asyncio. Run tests with `python -m pytest <file> -v -p no:cacheprovider`.

## Global Constraints

- Never weaken existing passing tests; all new tests go in `tests/unit/` or `tests/integration/` per repo taxonomy.
- Follow `TESTING.md`: prefer real tiny models / StubBackend over MagicMock for trainer math.
- Coverage gate is `fail_under = 54` in pyproject.toml — do not lower it.
- Ruff config: `E,W,F,B,C4,UP`; keep line length and style consistent with surrounding code.
- Commit after each task with a conventional-commit message; do not push.
- All trainer `train_step` methods are async — tests must use `pytest.mark.asyncio` or `asyncio.run`.

---

### Task 1: DAPO — real old-policy log-probs and µ>1 inner updates

**Files:**
- Modify: `stateset_agents/training/dapo_trainer.py` (sample collection + `train_step`, ~lines 780–890)
- Test: `tests/unit/test_dapo_trainer_behavioral.py` (new)

**Interfaces:**
- Produces: each sample dict in `collect_samples_with_dynamic_sampling` gains key `"old_token_log_probs"` (tensor `[batch, seq-1]`, detached, computed once at collection time under `torch.no_grad()` via `self.compute_token_log_probs`).
- `train_step` runs `for _ in range(self.config.num_gradient_updates):` (remove the `min(..., 1)` cap) and recomputes `current_token_log_probs` each inner iteration against the stored old log-probs.

- [ ] **Step 1: Write failing behavioral tests**

```python
# tests/unit/test_dapo_trainer_behavioral.py
"""Behavioral tests: importance ratios must come from rollout-time log probs."""
import pytest, torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel


def tiny_model():
    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(n_embd=32, n_layer=2, n_head=2, vocab_size=200, n_positions=64)
    )


def make_batch():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 200, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.ones_like(input_ids, dtype=torch.float)
    response_mask[:, :4] = 0.0  # first 4 tokens are prompt
    return input_ids, attention_mask, response_mask


@pytest.mark.asyncio
async def test_ratio_diverges_from_one_after_update(dapo_trainer_factory):
    """After one optimizer step, ratios vs stored old log probs must != 1."""
    trainer = dapo_trainer_factory(tiny_model())
    ids, am, rm = make_batch()
    with torch.no_grad():
        old, _ = trainer.compute_token_log_probs(ids, am, rm)
    # take a real optimizer step on arbitrary loss
    cur, _ = trainer.compute_token_log_probs(ids, am, rm)
    loss = cur.sum()
    loss.backward()
    trainer.optimizer.step()
    cur2, _ = trainer.compute_token_log_probs(ids, am, rm)
    ratio = torch.exp(cur2 - old)
    assert not torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


@pytest.mark.asyncio
async def test_num_gradient_updates_respected(dapo_trainer_factory, monkeypatch):
    """train_step must run num_gradient_updates inner updates, not min(mu, 1)."""
    trainer = dapo_trainer_factory(tiny_model(), num_gradient_updates=3)
    steps = []
    orig = trainer.optimizer.step
    monkeypatch.setattr(trainer.optimizer, "step", lambda *a, **k: (steps.append(1), orig())[1])

    ids, am, rm = make_batch()
    sample = {
        "responses": [
            {"input_ids": ids[i], "attention_mask": am[i],
             "response_mask": rm[i], "sequence_length": int(am[i].sum())}
            for i in range(2)
        ],
        "advantages": torch.tensor([0.5, -0.5]),
        "rewards": [1.0, 0.0],
        "accuracy": 0.5,
    }

    async def fake_collect(prompts, n):
        return [sample], 0.0
    monkeypatch.setattr(trainer, "collect_samples_with_dynamic_sampling", fake_collect)
    await trainer.train_step(["q"])
    assert len(steps) == 3


@pytest.mark.asyncio
async def test_old_log_probs_captured_at_collection(dapo_trainer_factory, monkeypatch):
    """train_step must use sample['old_token_log_probs'] when present, and the
    second inner update must see a non-unit ratio (clipping can fire)."""
    trainer = dapo_trainer_factory(tiny_model(), num_gradient_updates=2)
    ids, am, rm = make_batch()
    with torch.no_grad():
        old, _ = trainer.compute_token_log_probs(ids, am, rm)
    sample = {
        "responses": [
            {"input_ids": ids[i], "attention_mask": am[i],
             "response_mask": rm[i], "sequence_length": int(am[i].sum())}
            for i in range(2)
        ],
        "advantages": torch.tensor([1.0, -1.0]),
        "rewards": [1.0, 0.0],
        "accuracy": 0.5,
        "old_token_log_probs": old,
    }
    seen_ratios = []
    orig_ratio = trainer.compute_importance_ratio
    def spy(cur, old_):
        r = orig_ratio(cur, old_)
        seen_ratios.append(r.detach().clone())
        return r
    monkeypatch.setattr(trainer, "compute_importance_ratio", spy)

    async def fake_collect(prompts, n):
        return [sample], 0.0
    monkeypatch.setattr(trainer, "collect_samples_with_dynamic_sampling", fake_collect)
    await trainer.train_step(["q"])
    assert len(seen_ratios) == 2
    # first inner update: on-policy, ratio ~ 1
    assert torch.allclose(seen_ratios[0], torch.ones_like(seen_ratios[0]), atol=1e-4)
    # second inner update: policy moved, ratio must not be identically 1
    assert not torch.allclose(seen_ratios[1], torch.ones_like(seen_ratios[1]), atol=1e-5)
```

Also add a `dapo_trainer_factory` fixture in this file (not conftest): builds `DAPOTrainer` with a stub tokenizer-free path — inspect `DAPOTrainer.__init__` and construct it the same way `tests/unit/test_dapo_trainer.py` does, passing the tiny model, an `AdamW(model.parameters(), lr=1e-3)`-equivalent via its config, and `num_gradient_updates` override.

- [ ] **Step 2: Run tests, verify they fail** (the µ test and old-log-probs test fail; the divergence test may pass — keep it as a guard)
- [ ] **Step 3: Implement**
  - In `collect_samples_with_dynamic_sampling` (or wherever the per-sample dict is assembled), after building the batch tensors for a sample, compute and store `sample["old_token_log_probs"]` under `torch.no_grad()`.
  - In `train_step` (dapo_trainer.py:843-850): use `sample.get("old_token_log_probs")`; only if absent, compute once before the inner loop (preserving backward compat). Replace `range(min(self.config.num_gradient_updates, 1))` with `range(max(1, self.config.num_gradient_updates))`. Ensure batch padding in train_step matches the shape stored at collection (store per-sample padded to that sample's max len — compute it from the same `batch_input_ids` built in train_step by moving the batch-build before collection storage, or simplest: compute old log probs inside train_step **before** the inner loop, once per sample, and keep the collection-time key optional).
- [ ] **Step 4: Run new tests + `python -m pytest tests/unit/test_dapo_trainer.py -v` — all pass**
- [ ] **Step 5: Commit** `fix(dapo): honor num_gradient_updates and freeze old-policy log probs before inner updates`

---

### Task 2: GEPO — log-space arithmetic and mask off-by-one

**Files:**
- Modify: `stateset_agents/training/gepo_trainer.py` (`compute_sequence_log_probs` ~:260-299, `compute_gepo_coefficient` ~:301-337, callers ~:495-560)
- Test: `tests/unit/test_gepo_trainer_behavioral.py` (new)

**Interfaces:**
- `compute_gepo_coefficient(learner_seq_log_probs, sampler_seq_log_probs)` now takes **log**-probs and computes everything with `torch.logsumexp`; returns coefficients (linear space, safe because ratios are computed as differences of logs before exponentiation).
- `compute_sequence_log_probs(..., response_start_idx)` applies the mask at `max(response_start_idx - 1, 0)` on the shifted axis (matching `gspo_trainer.py:500`).

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_gepo_trainer_behavioral.py
import math, pytest, torch

from stateset_agents.training.gepo_trainer import GEPOTrainer


def test_gepo_coefficient_no_underflow_long_sequences():
    """Sums of token log probs for realistic sequences (~ -600 nats) must not
    produce 0/NaN coefficients."""
    learner = torch.tensor([-600.0, -610.0, -605.0, -595.0])
    sampler = torch.tensor([-601.0, -609.0, -606.0, -594.0])
    coef = GEPOTrainer.compute_gepo_coefficient_static(learner, sampler)
    assert torch.isfinite(coef).all()
    assert (coef > 0).all()


def test_gepo_coefficient_matches_linear_space_on_small_values():
    """On numerically safe values, log-space result equals the linear formula
    coef_i = p_i / E_qhat[q], E_qhat[q] = sum(q^2)/sum(q)."""
    learner_lp = torch.log(torch.tensor([0.30, 0.20, 0.10]))
    sampler_lp = torch.log(torch.tensor([0.25, 0.25, 0.10]))
    q = sampler_lp.exp()
    expected = learner_lp.exp() / ((q * q).sum() / q.sum())
    got = GEPOTrainer.compute_gepo_coefficient_static(learner_lp, sampler_lp)
    assert torch.allclose(got, expected, rtol=1e-5)


def test_response_mask_offset_matches_gspo_convention():
    """With prompt length P, the shifted-label mask must start at P-1."""
    trainer = object.__new__(GEPOTrainer)  # no init needed for pure method
    mask = GEPOTrainer.build_response_mask(
        attention_mask=torch.ones(1, 10, dtype=torch.long), response_start_idx=4
    )
    # shifted axis has length 9; positions 0..2 are prompt-only, 3.. are response
    assert mask.shape == (1, 9)
    assert mask[0, :3].sum() == 0
    assert mask[0, 3:].sum() == 6
```

- [ ] **Step 2: Run, verify fail** (methods don't exist yet)
- [ ] **Step 3: Implement**
  - Add `@staticmethod compute_gepo_coefficient_static(learner_seq_log_probs, sampler_seq_log_probs)`:

```python
@staticmethod
def compute_gepo_coefficient_static(learner_seq_log_probs, sampler_seq_log_probs):
    sampler_lp = sampler_seq_log_probs.detach()
    # log E_qhat[q] = log( sum(q^2)/sum(q) ) = logsumexp(2*lq) - logsumexp(lq)
    log_group_expectation = torch.logsumexp(2 * sampler_lp, dim=0) - torch.logsumexp(
        sampler_lp, dim=0
    )
    log_coef = learner_seq_log_probs - log_group_expectation
    return torch.exp(torch.clamp(log_coef, min=-30.0, max=30.0))
```

  - Rewire instance method `compute_gepo_coefficient` to delegate to the static, and change its **call sites** (~:503-535) to pass `sequence_log_probs` directly — delete the `torch.exp(...)` conversions at gepo_trainer.py:503-507/514. Keep the `[0.8, 1.2]`-style clipping but apply it to `log_coef` bounds equivalently (`clamp(log_coef, log(0.8+eps)... )`) or clamp the returned linear coef as before — returned values are now O(1) by construction on-policy.
  - Add `@staticmethod build_response_mask(attention_mask, response_start_idx)` implementing the `max(response_start_idx - 1, 0)` shifted-index convention, and use it in `compute_sequence_log_probs` (replacing lines ~:290-292).
- [ ] **Step 4: Run new + existing GEPO tests — pass**
- [ ] **Step 5: Commit** `fix(gepo): compute group-expectation weights in log space; fix response-mask off-by-one`

---

### Task 3: GSPO — score the sampled tokens, normalize loss by group count, remove test scaffolding

**Files:**
- Modify: `stateset_agents/training/gspo_generation.py` (`_generate_with_hf` :194-206, `_compute_sequence_log_prob` :243+)
- Modify: `stateset_agents/training/gspo_trainer.py` (loss accumulation ~:660-675; fake-param injection :356-359)
- Test: `tests/unit/test_gspo_scoring_consistency.py` (new)

**Interfaces:**
- `_generate_with_hf` builds the prompt text via `tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)` when the tokenizer has a chat template, and passes that same rendered string to `_compute_sequence_log_prob` as the prompt; falls back to raw `prompt` otherwise. This makes generation and scoring share one tokenization convention.
- `_compute_sequence_log_prob(prompt_text, response)` concatenates **without** the injected `" "` separator: `full_text = prompt_text + response` (generation continues the prompt directly; the space inserts a token that was never sampled).
- GSPO trainer divides accumulated loss by the number of query groups before `backward()`.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_gspo_scoring_consistency.py
import pytest, torch

pytest.importorskip("transformers")


def test_scoring_uses_rendered_prompt_no_space_join(monkeypatch):
    """The text scored must be exactly rendered_prompt + response."""
    from stateset_agents.training import gspo_generation as gg
    captured = {}

    class Tok:
        chat_template = "{{messages}}"
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "<user>" + msgs[0]["content"] + "<assistant>"
        def __call__(self, text, **kw):
            captured.setdefault("texts", []).append(text)
            import torch
            return {"input_ids": torch.ones(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long)}

    gen = object.__new__(gg.GSPOResponseGenerator)  # adjust to actual class name
    rendered = Tok().apply_chat_template([{"role": "user", "content": "hi"}])
    assert gg.build_scoring_text(rendered, "there") == "<user>hi<assistant>there"


@pytest.mark.asyncio
async def test_gspo_loss_normalized_by_group_count():
    """Doubling the number of identical query groups must not change the loss
    magnitude handed to backward()."""
    # Use the trainer's loss-accumulation helper: refactor train_step so the
    # per-group loop calls self._accumulate_group_loss and final loss =
    # total / num_groups; test the helper arithmetic directly.
    from stateset_agents.training.gspo_trainer import normalize_total_loss
    one = normalize_total_loss(torch.tensor(6.0), num_groups=1)
    three = normalize_total_loss(torch.tensor(18.0), num_groups=3)
    assert torch.allclose(one, three)


def test_no_fake_parameter_injection():
    import inspect
    from stateset_agents.training import gspo_trainer
    src = inspect.getsource(gspo_trainer)
    assert "dummy" not in src.lower() or "nn.Parameter(torch.zeros" not in src
```

(Adapt the first test to the real generator class name after reading the file; the essential assertions are: `build_scoring_text` exists and does not insert `" "`, and chat-template rendering is used when available.)

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement**
  - Add module-level `def build_scoring_text(prompt_text: str, response: str) -> str: return prompt_text + response` in `gspo_generation.py`; use it in `_compute_sequence_log_prob` (replacing `prompt + " " + response` at :250). In `_generate_with_hf`, render the chat template once (guard with `getattr(tokenizer, "chat_template", None)`), pass `rendered` to `_compute_sequence_log_prob`, and compute `prompt_length` from tokenizing `rendered` (with `add_special_tokens=False`).
  - In `gspo_trainer.py`: add `def normalize_total_loss(total_loss, num_groups): return total_loss / max(num_groups, 1)` and apply before `backward()` (~:667-671).
  - Delete the fake-parameter injection at :356-359; in the test suite, any test relying on parameterless models gets a 1-param `nn.Linear(1,1)` model instead (fix those fixtures where they live).
  - vLLM path: at `gspo_generation.py:185`, add a code comment is NOT enough — record sampling params alongside: return `(result.response, result.cumulative_logprob)` unchanged but ensure `generate_groups` is called with `logprobs` computed at the same temperature used for scoring; if the vLLM generator exposes sampling params, set `temperature`/`top_p` into a `scoring_temperature` config check: when `config.temperature != 1.0`, rescale is not possible post-hoc, so instead request vLLM's `prompt_logprobs`-style raw logprobs by passing `logprobs=1` with `temperature`-independent scoring if supported; otherwise document the residual bias in the docstring and emit a one-time `logger.warning` when `temperature != 1.0` and vLLM logprobs are used as old-policy logprobs. (Full fix = re-score rollouts with an HF forward pass at T=1; wire this behind `config.rescore_old_log_probs: bool = True` defaulting True, implemented by calling `_compute_sequence_log_prob` on each vLLM response.)
- [ ] **Step 4: Run new tests + `tests/unit/test_gspo_trainer.py` + `tests/integration/test_gspo_pipeline_integration.py` — pass**
- [ ] **Step 5: Commit** `fix(gspo): score sampled text consistently, normalize loss by group count, rescore vLLM rollouts`

---

### Task 4: GSPO-token — restore the gradient path

**Files:**
- Modify: `stateset_agents/training/gspo_token_trainer.py` (:105, :131-135, :152-236)
- Test: `tests/unit/test_gspo_token_trainer_behavioral.py` (new)

**Interfaces:**
- Token log probs used in the loss are computed **with** gradients; only the sequence-ratio used for clipping is detached.
- Response tokens only: mask out prompt positions using prompt length (same convention as Task 2's `build_response_mask`).
- Rewards computed via `compute_turn_reward` (the documented single-turn entry point on `RewardFunction`) instead of the `trajectory=None` kwargs call.
- Device via the parent's `_get_model_device(self.model)`, not `self.model.device`.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_gspo_token_trainer_behavioral.py
import pytest, torch

pytest.importorskip("transformers")


@pytest.mark.asyncio
async def test_train_step_produces_gradients(gspo_token_trainer_tiny):
    """After train_step, at least one model parameter must have a non-None,
    nonzero grad — the no_grad bug made backward() a no-op/raise."""
    trainer = gspo_token_trainer_tiny  # fixture: tiny GPT2 + stub generator + constant reward
    metrics = await trainer.train_step(["hello"], num_groups=1)
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.asyncio
async def test_loss_excludes_prompt_tokens(gspo_token_trainer_tiny):
    """Token loss must be masked to response positions."""
    trainer = gspo_token_trainer_tiny
    # spy on the masked log-prob tensor via the helper introduced in impl
    lp = torch.arange(9, dtype=torch.float).unsqueeze(0)
    masked = trainer.mask_prompt_tokens(lp, prompt_length=4)
    assert masked[0, :3].sum() == 0  # shifted convention: P-1 leading zeros
```

The fixture builds the trainer the way `examples/train_with_gspo.py` does but with the tiny GPT2 from Task 1 and a monkeypatched `generator.generate_group_responses` returning `[("ok", -5.0)]`, and a reward function whose `compute_turn_reward` returns a fixed `RewardResult`-compatible value (see `core/reward_base.py`).

- [ ] **Step 2: Run, verify fail** (backward raises / no grads)
- [ ] **Step 3: Implement**
  - Remove `torch.no_grad()` around the scoring forward (:163-165); keep `old_log_probs` detached.
  - Do not `.item()` the sequence log prob for the current policy — keep tensors: build `current_log_probs = torch.stack([t.sum() for t in masked_token_log_probs])` so gradients survive; pass to `compute_sequence_importance_ratio` but **detach** the ratio before clipping (the GSPO-token objective uses stop-gradient sequence ratios, gradient flows only through `token_log_probs` at :232-234 — that part is already right).
  - Add `def mask_prompt_tokens(self, token_log_probs, prompt_length)` implementing the shifted `max(prompt_length-1,0)` mask; compute `prompt_length` by tokenizing the query alone (mirror `gspo_generation._compute_sequence_log_prob`). Use it before summing (:179) and in the loss (:232).
  - Replace the reward call (:131-135) with `await self.reward_model.compute_turn_reward(turn=turn, context={"user_query": query})` — check the exact signature in `core/reward_base.py:97-150` first and match it.
  - Replace `self.model.device` (:105, :122, :139 etc.) with `device = _get_model_device(self.model)` (import from where the parent gets it).
- [ ] **Step 4: Run new tests + any existing gspo_token tests — pass**
- [ ] **Step 5: Commit** `fix(gspo-token): restore gradient path, mask prompt tokens, fix reward call and device lookup`

---

### Task 5: GRPO loss path — per-token ratios, working entropy bonus, narrow exceptions

**Files:**
- Modify: `stateset_agents/training/loss_computation.py` (:150-336, `compute_enhanced_grpo_loss` :339-465, `_estimate_policy_entropy` ~:190-210)
- Test: `tests/unit/test_loss_computation_behavioral.py` (new)

**Interfaces:**
- PPO ratio computed on **length-normalized** log probs: `ratio = exp((new_lp - old_lp)/token_count)` (sequence-mean ratio, bounded), replacing raw-sum exponentials at :288-296. Store `token_count` once; old log probs from trajectories are likewise divided by their own token counts when a `log_probs` list/tensor is present (`old_lp_mean = old_log_prob / token_count`).
- Entropy bonus contributes gradient: computed from the same `outputs.logits` inside the grad-enabled forward, as `-(p * logp).sum(-1)` masked-mean over response tokens, added to the loss as a tensor.
- `compute_enhanced_grpo_loss` gains ratio clipping identical to the main path and skips the full-vocab `log_softmax` when `beta == 0`.
- Exception handling around the forward pass narrowed to `(RuntimeError, ValueError)`; `AttributeError/KeyError/TypeError` propagate.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_loss_computation_behavioral.py
import math, pytest, torch

from stateset_agents.training import loss_computation as lc


def test_ratio_is_length_normalized():
    """A 200-token response with per-token drift 0.01 must give a finite,
    O(1) ratio — not exp(2.0) vs exp(sum) overflow behavior."""
    new_lp_sum = torch.tensor(-400.0)
    old_lp_sum = torch.tensor(-402.0)
    ratio = lc.compute_ppo_ratio(new_lp_sum, old_lp_sum, token_count=200)
    assert math.isfinite(ratio.item())
    assert abs(ratio.item() - math.exp(2.0 / 200)) < 1e-6


def test_entropy_bonus_has_gradient():
    logits = torch.randn(1, 6, 50, requires_grad=True)
    mask = torch.ones(1, 6)
    ent = lc.compute_entropy_bonus(logits, mask)
    ent.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_attribute_errors_propagate():
    """Systematic bugs must not be swallowed into zero loss."""
    assert AttributeError not in lc.LOSS_EXCEPTIONS
    assert KeyError not in lc.LOSS_EXCEPTIONS
    assert TypeError not in lc.LOSS_EXCEPTIONS
```

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement**
  - Add `def compute_ppo_ratio(new_log_prob_sum, old_log_prob_sum, token_count): return torch.exp((new_log_prob_sum - old_log_prob_sum) / max(token_count, 1))` and use it at :296. Remove the second division at :317-321 when the ratio path was taken (the length normalization now lives in the ratio; keep the `/_tc` normalization only for the no-old-log-probs REINFORCE branch).
  - Add `def compute_entropy_bonus(logits, response_mask)` returning masked mean entropy as a differentiable tensor; call it where `_estimate_policy_entropy` was used and delete the no-grad float version (keep a deprecation alias if it's exported).
  - Note: `outputs.loss`-based `new_log_prob` requires the forward with `labels=`; entropy needs `outputs.logits` — ensure the forward requests logits (it does by default).
  - Update `LOSS_EXCEPTIONS` (:22-29) to `(RuntimeError, ValueError, OSError)` — check `stateset_agents/exceptions.py` canonical tuples first and reuse/define there per repo convention, then fix any tests that asserted the old membership.
  - In `compute_enhanced_grpo_loss`: wrap `policy_loss` in the same clip (`compute_ppo_ratio` + `torch.min`) when old log probs exist; guard the `log_softmax` at :414 with `if beta > 0:`.
- [ ] **Step 4: Run new tests + `python -m pytest tests/unit -k "loss" -v` — pass**
- [ ] **Step 5: Commit** `fix(grpo-loss): length-normalized PPO ratios, differentiable entropy bonus, narrow exception tuple`

---

### Task 6: VAPO — real value clipping, terminal rewards, wired critic advantages

**Files:**
- Modify: `stateset_agents/training/vapo_trainer.py` (`compute_vapo_losses` :682-760, `train_step` :770-920)
- Test: `tests/unit/test_vapo_trainer_behavioral.py` (new)

**Interfaces:**
- `compute_vapo_losses(..., old_values: torch.Tensor)` — new required arg: rollout-time value predictions (already computed at :838); clipping compares fresh `values` against `old_values`, not against itself.
- `critic_advantages` is used for the value-loss weighting per the VAPO paper's decoupled-GAE (`returns = critic_advantages + old_values`); if the paper wiring is ambiguous in this codebase, minimum bar: `returns` for the value target are built from the critic-λ GAE, and `policy_advantages` from the policy-λ GAE — both already computed; connect them.
- Rewards placed on the **terminal** response token only (use the existing `dones` tensor at :847-851); remove the broadcast at :842-844.
- Optimizer step moved out of the per-prompt loop: accumulate loss over prompts, single `backward()`/`step()` per train_step (matching GSPO after Task 3).

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_vapo_trainer_behavioral.py
import pytest, torch

from stateset_agents.training.vapo_trainer import VAPOTrainer


def test_value_clipping_uses_rollout_values(vapo_trainer_tiny):
    """With old_values far from current values, the clipped branch must differ
    from the unclipped one (self-clipping made them identical)."""
    t = vapo_trainer_tiny  # config.value_clip = 0.2
    values = torch.tensor([[1.0, 1.0]])
    old_values = torch.tensor([[0.0, 0.0]])
    returns = torch.tensor([[0.5, 0.5]])
    mask = torch.ones(1, 2)
    v_loss_far = t.compute_value_loss(values, old_values, returns, mask)
    v_loss_self = t.compute_value_loss(values, values.detach(), returns, mask)
    assert not torch.allclose(v_loss_far, v_loss_self)


def test_reward_on_terminal_token_only(vapo_trainer_tiny):
    t = vapo_trainer_tiny
    rewards = t.build_token_rewards(scalar_reward=1.0, response_mask=torch.tensor([[0., 1., 1., 1.]]))
    assert rewards.tolist() == [[0.0, 0.0, 0.0, 1.0]]
```

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement**
  - Extract `compute_value_loss(values, old_values, returns, response_mask)` from :725-745 with the fix: `clipped_values = old_values + clamp(values - old_values, ±value_clip)`, `old_values` passed in from the rollout (:838) and threaded through `compute_vapo_losses`.
  - Add `build_token_rewards(scalar_reward, response_mask)`: zeros everywhere, `scalar_reward` at each row's last nonzero mask index; use it at :842-844 before GAE.
  - Wire `critic_advantages`: value targets `returns = critic_advantages + old_values` (decoupled-GAE); policy surrogate keeps `policy_advantages`.
  - Move `optimizer.zero_grad()/step()` and `scheduler.step()` outside the per-prompt loop; divide accumulated loss by the number of prompts.
- [ ] **Step 4: Run new tests + existing `tests/unit/test_vapo*` — pass**
- [ ] **Step 5: Commit** `fix(vapo): clip values against rollout predictions, terminal-token rewards, wire decoupled GAE, batch optimizer step`

---

### Task 7: Cross-trainer ratio regression suite + wire the Rust kernels or delete their dead path

**Files:**
- Test: `tests/integration/test_trainer_ratio_invariants.py` (new)
- Modify: `stateset_agents/core/rust_accelerator.py` + `stateset_agents/training/vapo_trainer.py` (GAE call site)

**Interfaces:**
- A parametrized integration test asserting, for each of DAPO/GEPO/GSPO with a tiny real model: (a) on-policy first-update mean ratio in `[0.99, 1.01]`; (b) after one optimizer step, recomputed ratio ≠ 1; (c) loss is finite; (d) at least one parameter grad is nonzero after `train_step`.
- VAPO's Python GAE gains an optional fast path: `from stateset_agents.core.rust_accelerator import compute_gae` guarded by availability (`try/except ImportError → None`), used when present, with a unit test asserting the Python and Rust implementations agree on a fixed input **when the extension is installed**, and `pytest.importorskip("stateset_rl_core")` otherwise.

- [ ] **Step 1: Write the invariant test** (reuse the tiny-model + monkeypatched-generator fixtures from Tasks 1/4; parametrize over trainer factories; each case runs one real `train_step`)
- [ ] **Step 2: Run — some cases pass only after Tasks 1–6; run after those merge, fix any residual failures in the trainers (not the tests)**
- [ ] **Step 3: Wire Rust GAE into VAPO** with graceful fallback; add the parity test
- [ ] **Step 4: Full suite: `python -m pytest tests/unit tests/integration -m "not performance and not slow and not gpu" -q` — green; coverage gate holds**
- [ ] **Step 5: Commit** `test: cross-trainer ratio/gradient invariants; feat: use rust GAE kernel in VAPO when available`

---

## Self-Review notes

- Spec coverage: findings 1a–1g from the review each map to a task (1a→T1, 1b→T2, 1c→T3, 1d→T4, 1e→T5, 1f→T6, 1g+1h→T7). The vLLM temperature-mismatch bias is handled in T3 via rescoring behind `rescore_old_log_probs`.
- Implementers must adapt exact fixture construction to the real `__init__` signatures — read the trainer's existing unit-test file first; the plan's test code states required behavior, constructor plumbing may differ.
- Type consistency: `build_response_mask` (T2) and `mask_prompt_tokens` (T4) both use the shifted `max(P-1,0)` convention; `normalize_total_loss` (T3) and VAPO's per-prompt averaging (T6) are analogous but separate functions.
