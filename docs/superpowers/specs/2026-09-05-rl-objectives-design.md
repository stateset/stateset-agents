# RL objectives — design

Date: 2026-09-05. Track 1 of the "best RL library" programme: make the core
policy-optimisation math a single, declarative, externally verified library
and route every native trainer through it.

## Why

`stateset_agents/training/rl_losses.py` (170 lines, 27 tests) holds the
shared primitives added in the August hardening pass. Every trainer still
assembles its own objective from them:

| Trainer | Ratio | Clip | KL | Aggregation | Private code |
|---|---|---|---|---|---|
| GRPO (`loss_computation.py`) | sequence-mean via `outputs.loss` | 3e-4 symmetric | none (plain) / exact full-vocab (enhanced) | per-trajectory mean | ~300 lines |
| GSPO | sequence-mean | 3e-4 / 4e-4 | k3 on sequence-mean log-probs | mean over rows | ~60 lines |
| GSPO-token | sequence, token gradient | 3e-4 / 4e-4 | k3 (inherited) | mean over rows | ~100 lines |
| DAPO | token | 0.2 / 0.28 | none | token-mean or seq-mean | ~40 lines |
| GEPO | group-expectation | 0.2 symmetric | none | mean over groups | ~50 lines |
| VAPO | token | 0.2 / 0.28 | none | token-mean or seq-mean | ~50 lines |
| PPO | token, **unclamped exp** | 0.2 symmetric | **naive `log π − log π_ref`** (zero-expectation gradient) | token-mean | ~70 lines |

Defects and gaps this design removes:

- PPO still carries the naive KL estimator that the GSPO fix replaced with k3,
  and exponentiates the log-ratio without the overflow clamp every other
  trainer uses.
- Dr. GRPO, BNPO, RLOO (as a full objective, not only a baseline switch),
  REINFORCE++-baseline, and CISPO are absent from the native trainers. CISPO
  exists only as a string forwarded to the River remote service.
- No trainer's objective is verified against an external implementation.
  TRL 1.12.0 is installed in the project venv and implements grpo, bnpo,
  dr_grpo, dapo, cispo, token and sequence ratios, and k3 KL, so an external
  pin is available.
- The GRPO path documented as "GRPO" actually computes a GSPO-style
  sequence-mean ratio clipped at 3e-4; the taxonomy below makes such choices
  explicit and nameable instead of implicit.

## 1. Module: `stateset_agents/training/objectives.py`

Pure tensor code. No trainer state, no model calls, torch fetched lazily via
`trainer_utils.get_torch()` exactly as `rl_losses.py` does, so importing the
module never requires torch. `rl_losses.py` stays as the primitive layer;
`objectives.py` composes it.

### 1.1 `PolicyObjective` (frozen dataclass)

| Field | Values | Meaning |
|---|---|---|
| `name` | str | Preset or user label, carried into metrics |
| `advantage` | `group_norm`, `group_mean`, `leave_one_out`, `batch_norm`, `external` | How rewards become advantages (§1.2) |
| `advantage_eps` | float, default `1e-8` | Added to the std in normalised estimators (TRL uses `1e-4`; the pin tests pass that) |
| `ratio` | `token`, `sequence`, `sequence_token`, `group_expectation` | Importance-ratio level (§1.3) |
| `ratio_clamp` | float, default `20.0` | Log-ratio clamp before `exp` (`rl_losses.safe_exp_ratio`); `group_expectation` uses `30.0` to preserve GEPO numerics |
| `clip` | `clipped`, `cispo`, `none` | Surrogate family (§1.4) |
| `clip_low`, `clip_high` | float | Trust region `[1 − low, 1 + high]` for `clipped` |
| `delta` | float or None | Optional upper cap on the *unclipped* ratio branch (TRL two-sided clipping) |
| `is_cap` | float, default `5.0` | Importance-weight cap for `cispo` |
| `aggregate` | `seq_mean`, `token_mean`, `seq_sum_const` | Reduction from per-token loss to scalar (§1.5) |
| `max_completion_length` | int or None | Required by `seq_sum_const` |
| `kl` | `none`, `k3_token`, `k3_sequence`, `external` | KL penalty estimator (§1.6) |
| `kl_coef` | float | β |
| `kl_bias_correction` | bool | Multiply per-token k3 by the ratio (TRL `use_bias_correction_kl`) |
| `entropy_coef` | float | Subtracts `entropy_coef · masked_mean(entropy)` when the caller supplies an entropy tensor |

`__post_init__` validates every field value and raises `ValueError` naming
the offending field: unknown enum values; `clip_low` or `clip_high`
negative; `kl_coef` non-zero with `kl="none"`; `kl_bias_correction` outside
`k3_token`, and so on. Shape-dependent checks (`seq_sum_const` without
`max_completion_length`, `group_expectation` without sequence-sum old
log-probs, `token` ratio without per-token old log-probs) are raised by
`policy_loss`, so presets can carry `None` placeholders.
`with_(**changes)` returns a modified copy so presets are easy to tweak.

### 1.2 `compute_advantages(rewards, group_ids, objective) -> Tensor[N]`

Operates on the flat batch; `group_ids` is an integer tensor `[N]` naming
each sample's prompt group. All estimators use `std(correction=0)`, treat a
group of size 1 or a non-finite/zero std as advantage 0 (never NaN), and
return fp32.

| `advantage` | Formula |
|---|---|
| `group_norm` | `(r_i − mean_g) / (std_g + eps)` — GRPO, GSPO, DAPO, CISPO |
| `group_mean` | `r_i − mean_g` — Dr. GRPO (`scale_rewards="none"`) |
| `leave_one_out` | `r_i − mean_{j∈g, j≠i} r_j` — RLOO |
| `batch_norm` | `(r_i − mean_g) / (std_batch + eps)` — REINFORCE++-baseline, TRL `scale_rewards="batch"` |
| `external` | Caller passes `advantages` to `policy_loss`; calling `compute_advantages` raises |

The `leave_one_out` branch already exists inside `loss_computation.py` as
`baseline_type="leave_one_out"`; it moves here and that code path delegates.

### 1.3 Ratio levels

Inputs are `logp_cur [N,T]` (with grad), `mask [N,T]`, and `logp_old`, which
may be `[N,T]` per-token or `[N]` per-sequence *sums*. When `logp_old` is
`None`, it defaults to `logp_cur.detach()` (TRL convention): the ratio is
numerically 1 and the gradient is the vanilla policy gradient, which is how
REINFORCE-style objectives are expressed without a special case.

| `ratio` | Shape | Formula | Requires |
|---|---|---|---|
| `token` | `[N,T]` | `exp(logp_cur − logp_old)` | per-token `logp_old` |
| `sequence` | `[N,1]` | `exp( Σ_t m(logp_cur − logp_old) / Σ_t m )` | per-token or sum `logp_old` |
| `sequence_token` | `[N,T]` | `sg[s_i] · exp(logp_cur − sg[logp_cur])` — GSPO-token | per-token or sum `logp_old` |
| `group_expectation` | `[N,1]` | `exp( S_i − (logsumexp_g(2·q) − logsumexp_g(q)) )` with `S_i = Σ_t m·logp_cur`, `q` the sampler sequence sums — GEPO | sum `logp_old`, `group_ids` |

Passing a `[N]` `logp_old` with `ratio="token"` raises; the trainer must
carry per-token old log-probs to ask for a token ratio.

### 1.4 Surrogate

| `clip` | Per-element loss |
|---|---|
| `clipped` | `−min(r·A, clamp(r, 1−low, 1+high)·A)`; if `delta` is set, `r` in the first term is `min(r, delta)` |
| `cispo` | `−sg[min(r, is_cap)] · A · logp_cur` |
| `none` | `−r·A` |

`rl_losses.clipped_surrogate` remains the implementation of the `clipped`
branch.

### 1.5 Aggregation

Per-token loss `l [N,T]` and mask `m`:

| `aggregate` | Formula | Matches |
|---|---|---|
| `seq_mean` | `mean_i( Σ_t l·m / max(Σ_t m, 1) )` | GRPO, GSPO, TRL `grpo` |
| `token_mean` | `Σ l·m / max(Σ m, 1)` | DAPO, BNPO, CISPO, TRL `bnpo`/`dapo` |
| `seq_sum_const` | `Σ l·m / (N · max_completion_length)` | Dr. GRPO, TRL `dr_grpo` |

A `[N,1]` per-token loss (sequence ratios) is broadcast across the mask
before aggregation, so `seq_mean` of a sequence-level loss equals the plain
row mean the GSPO trainer computes today.

### 1.6 KL

| `kl` | Formula | Added |
|---|---|---|
| `k3_token` | `exp(ref − cur) − (ref − cur) − 1` per token, optionally `· r` | to the per-token loss before aggregation (TRL) |
| `k3_sequence` | k3 on length-normalised sequence sums, one value per row | mean over rows, added after aggregation — preserves current GSPO numerics |
| `external` | caller passes `kl [N,T]` (enhanced GRPO's exact full-vocab KL) | per token before aggregation |
| `none` | — | — |

### 1.7 `policy_loss(...) -> PolicyLossResult`

```python
def policy_loss(
    *,
    logp_cur, mask, advantages, objective,
    logp_old=None, logp_ref=None, group_ids=None, kl=None, entropy=None,
) -> PolicyLossResult
```

`advantages` is `[N]` (broadcast over tokens) or `[N,T]` (external per-token
credit, PPO/VAPO). `PolicyLossResult` is a frozen dataclass with `loss`
(scalar, differentiable), `ratio` (detached), `metrics` (`policy_loss`,
`kl`, `entropy`, `clip_fraction`, `ratio_mean`, `ratio_max`, `advantage_mean`,
`advantage_std`, all floats). The function never calls a model and never
allocates a full-vocabulary tensor; log-prob gathering stays with the
trainer via `rl_losses.gather_token_logprobs`.

### 1.8 Presets (`OBJECTIVES: Mapping[str, PolicyObjective]`)

| Name | advantage | ratio | clip (low/high) | aggregate | kl |
|---|---|---|---|---|---|
| `grpo` | group_norm | token | clipped 0.2/0.2 | seq_mean | k3_token, β=0 |
| `dr_grpo` | group_mean | token | clipped 0.2/0.2 | seq_sum_const | none |
| `bnpo` | group_norm | token | clipped 0.2/0.2 | token_mean | none |
| `dapo` | group_norm | token | clipped 0.2/0.28 | token_mean | none |
| `gspo` | group_norm | sequence | clipped 3e-4/4e-4 | seq_mean | k3_sequence, β=0 |
| `gspo_token` | group_norm | sequence_token | clipped 3e-4/4e-4 | seq_mean | k3_sequence, β=0 |
| `gepo` | group_norm | group_expectation | clipped 0.2/0.2 | seq_mean | none |
| `rloo` | leave_one_out | token | clipped 0.2/0.2 | seq_mean | none |
| `reinforce_pp_baseline` | batch_norm | token | clipped 0.2/0.2 | token_mean | none |
| `cispo` | group_norm | token | cispo cap 5.0 | token_mean | none |
| `ppo` | external | token | clipped 0.2/0.2 | token_mean | k3_token, β=0 |

`dr_grpo` requires `with_(max_completion_length=...)` before use; the preset
itself carries `None` and `policy_loss` raises a clear error otherwise.
Presets are documented with their paper citation in `docs/OBJECTIVES.md`.

## 2. Verification (three layers, `tests/unit/test_objectives*.py`)

1. **Loop references.** `tests/unit/objective_reference.py` implements every
   estimator, ratio, surrogate, aggregation, and KL as explicit Python loops
   over lists of floats, with no torch broadcasting. Each preset is checked
   against it on seeded random batches with ragged masks, groups of size 1,
   constant-reward groups, and sum-vs-token `logp_old`. Tolerance `1e-6`.
2. **External pin against TRL 1.12.** `tests/unit/test_objectives_trl_pin.py`
   skips unless `trl` imports with major version 1. It binds
   `GRPOTrainer._compute_loss` to a `SimpleNamespace` carrying only the
   attributes that method reads (`loss_type`, `epsilon_low/high`,
   `importance_sampling_level`, `beta`, `args`, `_get_per_token_logps_and_entropies`
   returning our fixture tensors, `model.training=False`, and the logging
   stubs) and asserts our `grpo`, `bnpo`, `dr_grpo`, `dapo`, `cispo`, and
   sequence-ratio `gspo` losses match TRL's to `1e-5` on the same tensors, with
   and without k3 KL and bias correction. Advantage estimators are pinned
   against a transcription of TRL's `scale_rewards` block with `eps=1e-4`.
   A version drift makes the test skip with a message naming the pinned
   version, never silently pass.
3. **Property tests (Hypothesis, already a dev dependency).** Zero advantage
   ⇒ zero gradient for every preset; a `clipped` sample outside its trust
   region on the advantage's side ⇒ zero gradient; `cispo` gradient equals
   `−sg[min(r,cap)]·A·∇logp`; `k3 ≥ 0` and zero at equality; `token_mean` and
   `seq_mean` agree when every row has equal length; `seq_sum_const` equals
   `token_mean · (Σm)/(N·L)`; `leave_one_out` advantages sum to zero within a
   group; `logp_old=None` reproduces `logp_old=logp_cur.detach()` exactly; no
   preset produces a non-finite loss on log-ratios up to ±1e3.

## 3. Trainer migration

Each trainer keeps its public class, config fields, and `train_step`
signature. Only the objective assembly is replaced. Migration order and the
pinned behaviour:

| Step | Trainer | Objective used | Numeric change |
|---|---|---|---|
| 1 | DAPO | `dapo` with config `clip_eps_low/high`, aggregate from `use_token_level_loss` | none |
| 2 | VAPO | `ppo`-family with `advantage="external"`, config clip, aggregate from `use_token_level_loss`; value loss and positive-LM loss stay in VAPO | none |
| 3 | GSPO | `gspo` with `clip_range_left/right`, `kl="k3_sequence"`, `kl_coef=beta` | none |
| 4 | GSPO-token | `gspo_token`, same config | **reported `policy_loss` value** becomes the clipped-surrogate value (as GSPO reports) instead of the log-prob-weighted sum; gradients identical and pinned |
| 5 | GEPO | `gepo` with `clip_eps` | none |
| 6 | PPO | `ppo` with `clip_eps`, `kl="k3_token"`, adaptive controller still scales `kl_coef` per step | **intended**: k3 replaces naive KL; ratio gains overflow clamp |
| 7 | GRPO plain (clipped branch only; the REINFORCE fallback stays `advantage * loss`) | per-token log-probs gathered from the same forward pass (equal to the mean NLL it uses today); `ratio="sequence"` always, since `trajectory.log_probs` is stored as a sum, clip `seq_clip_ratio`; `advantage` from `baseline_type` / `advantage_normalization` | none for the clipped path; the no-old-log-prob REINFORCE path stays REINFORCE |
| 8 | GRPO enhanced | clipped branch as above; the exact full-vocab KL and the historical unbiased-std advantage normalisation stay in place (routing the KL through `kl="external"` was not needed for parity) | none |

`compute_grpo_loss` and `compute_enhanced_grpo_loss` keep their signatures,
so `multi_turn_trainer`, `single_turn_trainer`, and `distributed_trainer`
are untouched.

**Golden regression pins.** Before step 1, `scripts/capture_objective_goldens.py`
runs each of the seven trainers on a deterministic tiny model
(`GPT2Config(n_embd=32, n_layer=2, n_head=2, vocab_size=200)`, seeded) with
fixed token batches and writes the loss values to
`tests/unit/goldens/objective_goldens.json`. `tests/unit/test_objective_goldens.py`
asserts each trainer still produces those numbers after migration
(tolerance `1e-6`). The PPO entry is regenerated at step 6 in the same commit
as the fix, with the CHANGELOG naming the change. The goldens file is
committed and the script stays for future regeneration.

## 4. Public surface and docs

- `_registry.OPTIONAL_EXPORTS` gains `PolicyObjective`, `PolicyLossResult`,
  `OBJECTIVES`, `compute_advantages`, `policy_loss` (lazy, torch-free import).
- `docs/OBJECTIVES.md`: the taxonomy tables above, one worked formula per
  preset with its citation, the verification story, and a "which preset"
  guide. Linked from `ADVANCED_RL_ALGORITHMS.md`, `ARCHITECTURE.md`,
  `COMPARISONS.md` (TRL row updated to say objectives are pinned against
  TRL 1.12), and `README.md`'s trainer bullet.
- `CHANGELOG.md` `[Unreleased]`: new module, presets, PPO KL fix, TRL pin.
- `contracts/component_maturity_v1.json`: no tier changes; the GSPO/DAPO/GEPO
  limitation text gains "objectives pinned against TRL 1.12 on CPU fixtures"
  so the evidence reference is machine-validated.

## Out of scope

`TrainingConfig.objective` and a CLI flag (chosen for a later pass); TRL's
off-policy sequence masking and entropy-quantile masking; DAPO overlong
reward shaping (stays in `DAPORewardShaper`, it is reward-side); the async
control plane's staleness importance weights (a different quantity); the
Rust advantage kernel (its formula already matches `group_norm`; a parity
test is added, no Rust change); `core/enhanced/advanced_rl_algorithms.py`
(deprecated shim).

## Order and gates

Branch `feat/rl-objectives` from `master`. Commits: (1) objectives module +
loop references + property tests, (2) TRL pin, (3) golden capture + goldens,
(4)–(11) one trainer per commit, (12) docs, exports, changelog. After every
commit: `ruff check`, `black --check`, `isort --check`, `python
scripts/check_types.py --all`, and the affected unit tests; the full suite
with the coverage ratchet before the PR.

## Implementation record (2026-09-05)

Implemented on `feat/rl-objectives` per `docs/superpowers/plans/2026-09-05-rl-objectives.md`.
Deviations from the text above are folded in: shape checks live in
`policy_loss`; GSPO-token's reported loss value changed (gradients pinned);
enhanced GRPO keeps its unbiased-std advantages; `surrogate` and `aggregate`
are public helpers used by DAPO's ratio-based entry point.
