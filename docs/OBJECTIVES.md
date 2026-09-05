# Policy objectives

`stateset_agents.training.objectives` is the single place where StateSet's
policy-optimisation math lives. A `PolicyObjective` names one point in the
space **advantage × ratio × clip × aggregation × KL**; two pure functions,
`compute_advantages` and `policy_loss`, evaluate it on batched tensors. Every
native trainer (GRPO, GSPO, GSPO-token, DAPO, GEPO, VAPO, PPO) builds its
objective from a preset and calls `policy_loss`, so a formula exists once,
is tested once, and is pinned against an external implementation once.

```python
from stateset_agents.training import OBJECTIVES, compute_advantages, policy_loss

obj = OBJECTIVES["dapo"].with_(clip_high=0.3)          # tweak a preset
adv = compute_advantages(rewards, group_ids, obj)        # [N]
out = policy_loss(
    logp_cur=logp_cur,   # [N, T], with grad
    mask=response_mask,  # [N, T]
    advantages=adv,      # [N] or [N, T]
    objective=obj,
    logp_old=logp_old,   # [N, T] per token, or [N] sequence sums
)
out.loss.backward()
out.metrics  # policy_loss, kl, entropy, clip_fraction, ratio_mean, ...
```

The module imports without torch; torch is fetched lazily inside the
functions. Log-prob gathering stays with the trainer
(`rl_losses.gather_token_logprobs`); nothing here allocates a
full-vocabulary tensor.

## Taxonomy

### Advantage estimators (`compute_advantages`)

`rewards` and `group_ids` are flat `[N]`; a group is one prompt's samples.
All estimators use `std(correction=0)`, and a group of size 1, constant
rewards, or a non-finite statistic yields advantage 0 (never NaN).

| `advantage` | Formula | Used by |
|---|---|---|
| `group_norm` | `(r_i − mean_g) / (std_g + eps)` | GRPO, GSPO, DAPO, CISPO |
| `group_mean` | `r_i − mean_g` | Dr. GRPO (TRL `scale_rewards="none"`) |
| `leave_one_out` | `r_i − mean_{j∈g, j≠i} r_j` | RLOO |
| `batch_norm` | `(r_i − mean_g) / (std_batch(r) + eps)` | REINFORCE++-baseline (TRL `scale_rewards="batch"`) |
| `external` | caller supplies `advantages` (per-token GAE) | PPO, VAPO |

`advantage_eps` defaults to `1e-8`; TRL uses `1e-4`, and the pin tests pass
that value.

### Ratio levels

`logp_old` may be per-token `[N, T]` or per-sequence sums `[N]`. When it is
`None`, it defaults to `logp_cur.detach()`: the ratio is numerically 1 and
the gradient is the vanilla policy gradient, which is how REINFORCE-style
objectives are expressed without a special case.

| `ratio` | Shape | Formula | Needs |
|---|---|---|---|
| `token` | `[N, T]` | `exp(logp_cur − logp_old)` | per-token `logp_old` |
| `sequence` | `[N, 1]` | `exp( Σ_t m·(logp_cur − logp_old) / Σ_t m )` | per-token or sums |
| `sequence_token` | `[N, T]` | `sg[s_i] · exp(logp_cur − sg[logp_cur])` (GSPO-token) | per-token or sums |
| `group_expectation` | `[N, 1]` | `exp( S_i − (logsumexp_g(2q) − logsumexp_g(q)) )`, `S_i = Σ_t m·logp_cur`, `q` = sampler sums (GEPO) | sums and `group_ids` |

Every log-ratio is clamped to `±ratio_clamp` (default 20; GEPO 30) before
`exp`, so one wildly off-policy token cannot make the loss non-finite.

### Surrogate

| `clip` | Per-element loss |
|---|---|
| `clipped` | `−min(r·A, clamp(r, 1−clip_low, 1+clip_high)·A)`; with `delta`, the first `r` is `min(r, delta)` (TRL two-sided clipping) |
| `cispo` | `−sg[min(r, is_cap)] · A · logp_cur` |
| `none` | `−r·A` |

### Aggregation

| `aggregate` | Formula | Matches |
|---|---|---|
| `seq_mean` | `mean_i( Σ_t l·m / max(Σ_t m, 1) )` | GRPO, GSPO, TRL `grpo` |
| `token_mean` | `Σ l·m / max(Σ m, 1)` | DAPO, BNPO, CISPO, TRL `bnpo`/`dapo` |
| `seq_sum_const` | `Σ l·m / (N · max_completion_length)` | Dr. GRPO, TRL `dr_grpo` |

### KL penalty

| `kl` | Estimator | Where it is added |
|---|---|---|
| `k3_token` | `exp(ref − cur) − (ref − cur) − 1` per token, optionally `· r` (`kl_bias_correction`) | per token, before aggregation (TRL) |
| `k3_sequence` | k3 on length-normalised sequence log-probs, one value per row | mean over rows, after aggregation (GSPO) |
| `external` | caller passes a per-token `kl` tensor (enhanced GRPO's exact full-vocab KL) | per token, before aggregation |
| `none` | — | — |

`entropy_coef` subtracts `entropy_coef · masked_mean(entropy)` when the
caller supplies a differentiable per-token entropy tensor.

## Presets

| Name | advantage | ratio | clip (low / high) | aggregate | kl | Reference |
|---|---|---|---|---|---|---|
| `grpo` | group_norm | token | clipped 0.2 / 0.2 | seq_mean | k3_token, β=0 | Shao et al. 2024, *DeepSeekMath* |
| `dr_grpo` | group_mean | token | clipped 0.2 / 0.2 | seq_sum_const | none | Liu et al. 2025, *Understanding R1-Zero-Like Training* |
| `bnpo` | group_norm | token | clipped 0.2 / 0.2 | token_mean | none | TRL `loss_type="bnpo"` |
| `dapo` | group_norm | token | clipped 0.2 / 0.28 | token_mean | none | Yu et al. 2025, *DAPO* |
| `gspo` | group_norm | sequence | clipped 3e-4 / 4e-4 | seq_mean | k3_sequence, β=0 | Zheng et al. 2025, *GSPO* |
| `gspo_token` | group_norm | sequence_token | clipped 3e-4 / 4e-4 | seq_mean | k3_sequence, β=0 | Zheng et al. 2025, *GSPO* (token variant) |
| `gepo` | group_norm | group_expectation | clipped 0.2 / 0.2 | seq_mean | none | StateSet GEPO (`training/gepo_trainer.py`) |
| `rloo` | leave_one_out | token | clipped 0.2 / 0.2 | seq_mean | none | Ahmadian et al. 2024, *Back to Basics (RLOO)* |
| `reinforce_pp_baseline` | batch_norm | token | clipped 0.2 / 0.2 | token_mean | none | Hu 2025, *REINFORCE++* (baseline variant) |
| `cispo` | group_norm | token | cispo, cap 5.0 | token_mean | none | Chen et al. 2025, *MiniMax-M1* |
| `ppo` | external | token | clipped 0.2 / 0.2 | token_mean | k3_token, β=0 | Schulman et al. 2017, *PPO* |

Notes:

- `dr_grpo` needs `with_(max_completion_length=L)` before use; `policy_loss`
  raises a clear error otherwise.
- `reinforce_pp_baseline` follows TRL's `scale_rewards="batch"`: group-centred
  rewards divided by the batch std of the raw rewards. verl's
  `reinforce_plus_plus_baseline` instead normalises the centred scores by
  their own batch std; the two differ by a constant factor per batch.
- KL coefficients default to 0 in every preset; trainers set `kl_coef` from
  their `beta` config field.

## Which preset

| Situation | Start with |
|---|---|
| Verifiable single-turn tasks (math, code, tool calls), long completions | `dapo` (clip-higher, token mean) or `dr_grpo` (no length bias) |
| Multi-turn dialogue with long or stale rollouts | `gspo` (sequence-level ratio, tiny trust region) |
| Value-augmented training with a critic | `ppo` with external GAE advantages (what VAPO does) |
| Off-policy or asynchronous producers | `cispo` (bounded importance weight, no gradient cut-off) or `gepo` |
| Small groups where the mean baseline is noisy | `rloo` |

## Verification

Three layers, all CPU-only and part of the default suite:

1. **Loop references** (`tests/unit/objective_reference.py`,
   `tests/unit/test_objectives.py`): every estimator, ratio, surrogate,
   aggregation, and KL is implemented as explicit Python loops over lists of
   floats, and each preset is checked against it on ragged masks, empty rows,
   groups of size 1, constant-reward groups, and sum-versus-token
   `logp_old`, to `1e-6`.
2. **External pin** (`tests/unit/test_objectives_trl_pin.py`): the
   `grpo`, `bnpo`, `dr_grpo`, `dapo`, `cispo`, and sequence-ratio `gspo`
   losses, with and without k3 KL and bias correction, and the two-sided
   `delta` clip, match TRL 1.12's `GRPOTrainer._compute_loss` on identical
   tensors to `1e-5`; advantage estimators match a transcription of TRL's
   `scale_rewards` block. The module skips, with a message naming the pinned
   version, if TRL's major version changes; it never silently passes.
3. **Property tests** (`tests/unit/test_objectives_properties.py`,
   Hypothesis): zero advantage ⇒ zero gradient for every preset; a clipped
   sample outside its trust region on the advantage's side ⇒ zero gradient;
   the CISPO gradient equals `−sg[min(r, cap)]·A·∇logp`; k3 ≥ 0 and zero at
   equality; `token_mean` equals `seq_mean` for equal lengths;
   `seq_sum_const` is `token_mean` rescaled; RLOO advantages sum to zero;
   no preset produces a non-finite loss for log-ratios up to ±1e3.

**Golden regression pins.** `scripts/capture_objective_goldens.py` runs every
trainer's objective assembly on a deterministic dropout-free tiny GPT-2 and
writes `tests/unit/goldens/objective_goldens.json`;
`tests/unit/test_objective_goldens.py` asserts each trainer still produces
those numbers. The goldens were captured before the trainers were migrated
onto `policy_loss`, so they prove the migration changed nothing except the
two documented cases below. Regenerate a single entry deliberately with
`--only <name>` and explain the change in `CHANGELOG.md`.

Known differences, on purpose:

- **Enhanced GRPO** (`compute_enhanced_grpo_loss`) keeps its historical
  advantage normalisation with the unbiased std; the `grpo` preset uses
  `correction=0`.
- **GSPO-token** reports the clipped-surrogate value as `policy_loss` (the
  same quantity GSPO reports) instead of the log-prob-weighted sum. Gradients
  are identical and are what its golden pins.
- **PPO** now uses the k3 KL estimator and clamps the log-ratio; the previous
  `log π − log π_ref` mean had a zero-expectation gradient. Its golden was
  regenerated in the same commit as the fix.

## Trainer map

| Trainer | Preset | Config fields it reads |
|---|---|---|
| `DAPOTrainer` | `dapo` | `clip_eps_low`, `clip_eps_high`, `use_token_level_loss` |
| `VAPOTrainer` (policy half) | `ppo` with `advantage="external"`, `kl="none"` | `clip_eps_low`, `clip_eps_high`, `use_token_level_loss` |
| `GSPOTrainer` | `gspo` | `clip_range_left`, `clip_range_right`, `beta` |
| `GSPOTokenTrainer` | `gspo_token` | `clip_range_left`, `clip_range_right`, `beta` |
| `GEPOTrainer` | `gepo` | `clip_eps` |
| `PPOTrainer` | `ppo` with `kl="none"` (KL added in `train_step` with the adaptive coefficient) | `clip_eps`, `beta` |
| GRPO (`loss_computation.py`, plain and enhanced) | `gspo` renamed `grpo_sequence` (sequence ratio, since rollouts store summed log-probs) | `seq_clip_ratio`, `clip_ratio`, `baseline_type` (`leave_one_out` → `rloo` advantages) |

VAPO's value loss and positive-example LM loss, PPO's value loss, and
enhanced GRPO's exact full-vocabulary KL remain in their trainers; the
objective owns the policy term only.
