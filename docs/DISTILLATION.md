# Knowledge distillation

`stateset_agents.training.distillation` trains a small *student* causal LM to
match a frozen *teacher*'s token distribution: offline logit distillation on
fixed completions, and on-policy distillation (GKD) on completions the student
samples itself.

Stability: **provisional / beta** (`distillation` in
[COMPONENT_MATURITY.md](COMPONENT_MATURITY.md)). The names are importable from
`stateset_agents.training` and `stateset_agents.training.distillation` but are
not part of the v1 stable contract ([API_COMPATIBILITY.md](API_COMPATIBILITY.md)).
The learning evidence is CPU toy models only (see [Evidence](#evidence)).

## When to use which

| You have | Use | Notes |
|---|---|---|
| Completions (ground truth, or teacher samples = sequence-level KD) and a teacher that fits next to the student | offline, `divergence="forward_kl"` | Classic logit KD. Mean-seeking: the student covers all of the teacher's modes. |
| The same, but a student too small to cover the teacher | offline, `reverse_kl` or `jsd` | Mode-seeking; `jsd` interpolates (`jsd_beta`). |
| No memory for the teacher during training | offline, `teacher_top_k=k` | Score the teacher once into a top-k cache, release it. |
| Exposure bias (the student drifts off the teacher's data at inference) | on-policy, `lmbda > 0` (GKD) | The student samples; the teacher scores every sampled position. |
| On-policy with the textbook reverse-KL objective | `lmbda=1`, `on_policy_loss="sampled"` | Policy-gradient form of the per-token reverse KL on the sampled tokens. |
| Only a reward, no teacher | — | Use online DPO ([PREFERENCE_TRAINING.md](PREFERENCE_TRAINING.md#online-dpo)) or the GRPO family. |

## Losses

At every response position, with `S = softmax(z_student / τ)` and
`T = softmax(z_teacher / τ)` (temperature `τ = temperature`):

| `divergence` | GKD `β` | Per-token loss |
|---|---|---|
| `forward_kl` | 0 | `KL(T ‖ S) = Σ_v T(v) (log T(v) − log S(v))` |
| `reverse_kl` | 1 | `KL(S ‖ T) = Σ_v S(v) (log S(v) − log T(v))` |
| `jsd` | `jsd_beta ∈ (0, 1)` | `β KL(T ‖ M) + (1 − β) KL(S ‖ M)`, `M = β T + (1 − β) S` |

This is the generalized Jensen-Shannon divergence of GKD (Agarwal et al.
2024, "On-Policy Distillation of Language Models", eq. 1) with TRL's endpoint
convention. As in TRL, logits are divided by `τ` and the loss is **not**
rescaled by `τ²`.

- **Response tokens only.** Records are tokenized as prompt ids then response
  ids by the preference encoder (chat template via
  `core.chat_masking.final_turn_token_mask`, `MultiTurnAgent`'s plain format,
  or raw prompts), so the mask is exact; position `t` predicts token `t+1`.
- **Generation length.** On-policy sampling caps each completion at
  `min(max_new_tokens, max_length - prompt_tokens)` before generation.
  Teacher scoring and training use the entire sampled completion. Equal
  budgets are batched together; shorter prompts keep their own budgets.
  Set `max_length` within both models' supported context windows.
- **Reduction.** The token mean: each example contributes
  `weight × response tokens`, and gradient accumulation divides by the
  window's total, so micro-batches add up to exactly the large-batch loss
  (the same normalization as TRL's `num_items_in_batch`).
- **Sampled reverse KL** (`on_policy_loss="sampled"`, on-policy batches):
  per sampled token `x_t ~ S`, the surrogate is
  `sg(log S(x_t) − log T(x_t)) · log S(x_t)`; its gradient in expectation is
  `∇ KL(S ‖ T)` at that position, and `log S(x_t) − log T(x_t)` is reported
  as the unbiased `kl_estimate`. Each position is credited on its own (no
  credit to earlier tokens for later positions), like GKD. It needs
  `sampling_temperature == temperature` and `top_p == 1`, so samples come
  from the distribution being differentiated.
- **Top-k cache** (`teacher_top_k=k`): both distributions are coarse-grained
  to the teacher's `k` tokens plus one bucket for the remaining mass, and the
  divergence is taken on those `k + 1` categories. That lower-bounds the full
  divergence (data-processing inequality) and equals it as the teacher's
  remaining mass goes to zero; bucket masses below `1e-7` are floored there.

The pure functions live in `stateset_agents.training.distillation.losses`
(`divergence_from_logprobs`, `token_divergence`, `generalized_jsd_loss`,
`topk_logprobs`, `topk_divergence`, `sampled_reverse_kl`).

### Pinned to TRL

`tests/unit/test_distillation_trl_pin.py` compares against
`trl.experimental.gkd.GKDTrainer` on identical tensors:

- `generalized_jsd_loss` vs `GKDTrainer.generalized_jsd_loss` for
  `β ∈ {0, 0.1, 0.5, 0.9, 1}`, `τ ∈ {1, 2}`, three seeds, with `-100`
  prompt/padding labels: values (rel 1e-5) and gradients (rel 1e-4); `sum`
  and `batchmean` reductions.
- The trainer's own path (`DistillationTrainer._compute` on real tiny GPT-2
  student and teacher, two micro-batches reduced over the window) vs
  `GKDTrainer.compute_loss` with `num_items_in_batch` = the window's token
  count, for `forward_kl`, `jsd` and `reverse_kl`: loss and every student
  parameter gradient.

The pin passes on trl 1.9.1 (the lock file) and 1.16.0; a TRL major version
other than 1 skips loudly.

## Vocabularies

Teacher and student must agree on what each logit column means.
`vocab_alignment` (`resolve_vocab_mapping`) decides:

- `strict` (default): the tokenizers' token-to-id maps must be identical (when
  a `teacher_tokenizer` is given) and the logit widths equal. Anything else
  raises `VocabularyMismatchError`; nothing is truncated.
- `shared`: tokens are aligned **by string**. The loss runs over the tokens
  both vocabularies contain (each side renormalized over that shared
  support), sequences are translated into the teacher's ids, and on-policy
  sampling suppresses student tokens outside the support. A sequence holding
  a token the teacher lacks raises. Use it for the common padded-embedding
  case (same tokenizer, e.g. 151,936 vs 152,064 logit rows) or tokenizers
  that number the same tokens differently. The shared/student-only/
  teacher-only counts are reported in `summary.data["vocab"]`.

Tokenizers that split text differently (different merges) are not supported:
that needs sequence alignment (ULD/GOLD-style), which this package does not
attempt.

## On-policy distillation (GKD)

`lmbda` is GKD's λ: each micro-batch, with probability `lmbda` (a
deterministic draw per seed, update and micro-batch, so resumes are exact),
the student samples one completion per prompt (`max_new_tokens`,
`sampling_temperature`, `top_p`; plain sampling from an explicit
`GenerationConfig`, no top-k or repetition penalty inherited from the
checkpoint) and the batch trains on those exact sampled token ids; otherwise
it trains on the dataset completions. `lmbda=1` needs only prompts; any
`lmbda < 1` needs a completion for every example. Samples are drawn at the
start of the optimizer window, so they are exactly on-policy for that update.

## Data format

Strict JSON Lines (the shared reader and strict parser: NaN, Infinity and
duplicate keys are rejected; errors name file and line):

```jsonl
{"prompt": "Q: 2+2=", "completion": "4"}
{"prompt": [{"role": "user", "content": "Where is my order?"}], "completion": "It ships tomorrow.", "weight": 2.0, "group_id": "case-17"}
{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Refund?"}, {"role": "assistant", "content": "Approved."}]}
{"prompt": [{"role": "user", "content": "Summarize the policy."}]}
```

`messages` ending in an assistant turn split into prompt and completion;
records without a completion are prompt-only (on-policy). Preference fields
(`chosen`, `rejected`, `label`) are refused. Optional `weight`, `group_id`,
`source_id`, `metadata` and a `split` declaration. Train/held-out splits are
prompt-disjoint on the shared request identity, `group_id` and `source_id`
(`split_distillation_dataset`, `validation_fraction`), exactly as for
preference data.

## Python API

```python
from stateset_agents.training.distillation import (
    DistillationConfig,
    TeacherCache,
    precompute_teacher_cache,
    train_distillation,
)

# On-policy GKD: half the batches on student samples, JSD with beta 0.5.
result = train_distillation(
    "Qwen/Qwen3.5-0.8B",                      # student (id or loaded + tokenizer=)
    "distill.jsonl",
    teacher="Qwen/Qwen3.5-9B",
    config=DistillationConfig(
        divergence="jsd", jsd_beta=0.5, temperature=1.0, lmbda=0.5,
        max_new_tokens=256, learning_rate=1e-5, batch_size=4,
        gradient_accumulation_steps=8, validation_fraction=0.05,
        vocab_alignment="shared",            # padded embedding widths differ
        use_lora=True, output_dir="outputs/kd",
    ),
)
print(result.summary.eval_metrics)  # eval/forward_kl, eval/reverse_kl, eval/divergence

# Offline from a top-k cache: run the teacher once, train later without it.
config = DistillationConfig(teacher_top_k=32)
cache = precompute_teacher_cache(teacher_model, tokenizer, "distill.jsonl", config,
                                 student=student_model)
cache.save("teacher_top32.pt")
train_distillation(student_model, "distill.jsonl", tokenizer=tokenizer, config=config,
                   teacher_cache=TeacherCache.load("teacher_top32.pt"))
```

`DistillationTrainer(model, tokenizer, config, train_dataset, eval_dataset,
teacher=..., teacher_tokenizer=..., teacher_cache=...)` is the underlying
class (`train()`, `evaluate(dataset=None, on_policy=False)`,
`divergence_report(prompt, response)`, `save_checkpoint()`,
`load_checkpoint()`).

The teacher is frozen (`requires_grad=False`, eval mode) and never receives
gradients; it must be a separate object from the student. Optimization is the
preference trainers' loop: AdamW, warmup + linear/cosine/constant schedule,
`max_grad_norm` clipping through `trainer_utils.safe_optimizer_step`, a
`raise`/`skip` policy for non-finite losses or gradients, bf16 autocast,
optional LoRA on the student.

### Metrics

Training history: `loss`, token-mean `forward_kl`, `reverse_kl` and
`divergence` (the configured one), `kl_estimate` (sampled batches),
`on_policy_fraction`, `tokens`. `evaluate()` reports the same as `eval/*`,
teacher-forced on held-out completions, or `eval/on_policy/*` on one
fixed-seed student sample per held-out prompt (`on_policy=True`; automatic
for prompt-only held-out data). Against a cache the KLs are the top-k
coarse-grained ones.

### Checkpoints

`save_checkpoint(dir)` writes `model/` (`save_pretrained`, an adapter under
LoRA, plus the tokenizer), `trainer_state.pt` (optimizer, scheduler, torch
RNG) and `distillation_checkpoint.json` (written last). `load_checkpoint` /
`train(resume_from=dir)` load through `training.model_checkpoint` and
`core.checkpoint_io` (weights-only unless `trusted=True`). Before loading
weights or training state, they require matching training settings, including
the divergence, temperature, `lmbda`, `on_policy_loss`, sampling, optimizer,
batching, seed, encoding and LoRA settings. Missing training settings,
invalid configurations and mismatches raise `ConfigurationError` (also a
`ValueError`). You may change `device`, `output_dir`, `logging_steps`,
`eval_steps` and `save_steps`, and adjust `max_steps` and `num_epochs`.
The saved optimizer and scheduler are restored unchanged: extending the
run limit does not extend the original learning-rate schedule.

Version-2 checkpoints also require matching input fingerprints before loading
weights or training state. These cover ordered training and evaluation token
sequences, prompt/response boundaries, example weights, student tokenizer
vocabulary and available fast-tokenizer rules, special tokens, chat template,
generation stop tokens and the student/teacher vocabulary mapping. Raw
examples are not stored in the fingerprints. Annotation-only metadata,
tokenizer file paths and transient tokenizer padding/truncation state are
excluded. Custom tokenizers without `get_vocab()` have their encoded examples
checked, but their full vocabulary and custom implementation cannot be verified.

Version-3 checkpoints additionally require matching fingerprints of the
teacher's parameters and persistent buffers, or the exact token ids and
log-probabilities in the teacher-cache entries used by training/evaluation.
Unused cache entries are ignored. LoRA students also require the same frozen
base state, which an adapter checkpoint does not contain. All checks finish
before restoring student weights, optimizer state or progress.

State fingerprints are recomputed at save/load, include names, dtypes, shapes
and values, and use bounded tensor chunks (4 MiB by default) without copying
the whole teacher to CPU. They are device-independent but require exact values
and dtypes. Unsupported state, such as unmaterialized meta tensors, fails
before checkpoint writes. Model code and forward-pass configuration are not
identified by these fingerprints; use the original teacher implementation and
settings. Teacher caches save with `TeacherCache.save` and load weights-only.

Version-1 and version-2 checkpoints remain readable with warnings for missing
identity checks. Version 2 still requires matching input fingerprints; version
3 requires both input and model fingerprints. Older readers cannot load
version-3 checkpoints.

## CLI

```bash
# Validate data, split and settings without loading a model (torch-free)
stateset-agents distill --train-data distill.jsonl --validation-fraction 0.05 --dry-run
# Offline KD
stateset-agents distill --model Qwen/Qwen3.5-0.8B --teacher Qwen/Qwen3.5-9B \
  --train-data distill.jsonl --validation-fraction 0.05 --vocab-alignment shared \
  --lora --output-dir outputs/kd
# On-policy GKD on prompts only, sampled reverse KL
stateset-agents distill --model student --teacher teacher --train-data prompts.jsonl \
  --lmbda 1 --divergence reverse_kl --on-policy-loss sampled --max-new-tokens 256
# Build and save a top-k cache, then train from it without the teacher
stateset-agents distill --model student --teacher teacher --train-data distill.jsonl \
  --teacher-top-k 32 --save-teacher-cache top32.pt
stateset-agents distill --model student --teacher-cache top32.pt --teacher-top-k 32 \
  --train-data distill.jsonl
```

See [CLI_REFERENCE.md](CLI_REFERENCE.md#stateset-agents-distill).

## Evidence

`tests/learning/test_distillation_learns.py` runs with `LEARNING_SEEDS=0,1,2`.
A tiny GPT-2 teacher is fine-tuned for 60 supervised steps to emit digits;
the student starts untrained. The teacher generates training data on four
prompts and held-out data on four different prompts. Every positive mode
must reduce both held-out KLs by at least 65% and raise the first-token digit
mass above 0.6. The forward KL uses teacher completions; the reverse KL uses
fixed-seed student samples.

The 2026-11-06 run below reports the **worst** result across the three seeds:
maximum after/before ratio for each KL (must be ≤ 35%) and minimum final
digit mass (must be ≥ 0.6). All 21 positive cases pass.

| mode | learning rate | updates | worst forward-KL ratio | worst reverse-KL ratio | lowest digit mass |
|---|---|---|---|---|---|
| `gkd_mixed` | 0.005 | 30 | 12.8% | 23.5% | 0.833 |
| `gkd_on_policy_jsd` | 0.001 | 240 | 9.0% | 7.3% | 0.948 |
| `gkd_on_policy_reverse_kl` | 0.001 | 240 | 18.8% | 19.1% | 0.950 |
| `offline_forward_kl` | 0.005 | 30 | 14.4% | 17.2% | 0.748 |
| `offline_jsd` | 0.005 | 30 | 18.3% | 27.4% | 0.889 |
| `offline_topk_cache` | 0.005 | 30 | 19.8% | 19.2% | 0.751 |
| `on_policy_sampled_reverse_kl` | 0.001 | 960 | 12.5% | 7.0% | 0.962 |

Shuffled-teacher controls preserve the teacher's confidence while permuting
its logits over vocabulary tokens. They use the same update budget as their
positive counterparts. Neither KL to the real teacher may decrease, and
digit mass must stay below 0.4. All six controls pass; the table shows the
smallest KL increase and largest final digit mass across seeds.

| control | minimum forward-KL increase | minimum reverse-KL increase | highest digit mass |
|---|---|---|---|
| `offline_forward_kl` | +1.893 | +1.829 | 0.260 |
| `on_policy_sampled_reverse_kl` | +2.889 | +1.343 | 0.274 |

### Workload calibration

The earlier 30-update fixture missed thresholds on seed 1 in pure on-policy
JSD, full-distribution reverse KL, and sampled reverse KL, including on
unmodified v0.71.7. Optimization settings were selected using training
prompts before evaluating the revised workloads on held-out prompts. The
quality assertions, teacher construction, evaluation data, seeds, and
negative controls are unchanged.

At a learning rate of 1e-3, the two full-distribution modes satisfy the
training criteria at 240 updates. The sampled estimator needs 960 updates:
seed 1 remains stalled at 480 before converging. Larger batches, a detached
state-baseline prototype, and temperature 2 did not resolve that short-run
stall and were not adopted. The estimator and trainer implementations and
defaults remain unchanged. Longer runs increase test runtime.

The results above use CPU PyTorch 2.13.0 and Transformers 5.14.1; the same
27 learning checks pass on PyTorch 2.11.0. [Calibration trials and all
per-seed held-out measurements](../benchmark_results/training_contracts/20261106/learning-gates/README.md)
are recorded with the online-DPO results. These toy checks do not certify
large-model training quality.

## Limitations

- Single process: no DDP/FSDP, no gradient checkpointing, no fp16
  GradScaler; sampling uses `model.generate` (no vLLM).
- The full teacher forward runs on every batch, including for
  `on_policy_loss="sampled"` (which only needs the sampled tokens' teacher
  log-probs); the top-k cache is offline only.
- Cross-tokenizer distillation needs shared token strings; differently
  segmented text is not aligned.
- Evidence is CPU toy models; large-model runs are graduation criteria.
