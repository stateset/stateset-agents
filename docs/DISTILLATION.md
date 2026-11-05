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
`core.checkpoint_io` (weights-only unless `trusted=True`) and refuse a
checkpoint with a different divergence, `jsd_beta`, temperature or `lmbda`;
resuming reproduces the uninterrupted run exactly, on-policy sampling
included. Teacher caches save with `TeacherCache.save` and load weights-only.

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

`tests/learning/test_distillation_learns.py` (`LEARNING_SEEDS=0,1,2`, a few
seconds of single-threaded CPU per run). The teacher is a tiny GPT-2 fine-tuned for 60 steps to
answer any prompt with digits (~0.99 of its first-token mass on digits); the
student is an untrained tiny GPT-2 (~0.10). Training data are teacher samples
on 4 training prompts; held-out data are teacher samples on 4 other prompts.
30 updates per run. Every mode must cut the held-out forward KL (teacher-forced
on teacher completions) and the on-policy reverse KL (student samples) by at
least 65% and lift the digit mass above 0.6:

| seed | mode | fwd KL before → after | on-policy rev KL before → after | digit mass |
|---|---|---|---|---|
| 0 | offline forward KL | 2.423 → 0.224 | 5.343 → 0.055 | 0.100 → 0.942 |
| 1 | offline forward KL | 2.798 → 0.240 | 5.047 → 0.062 | 0.096 → 0.963 |
| 2 | offline forward KL | 2.461 → 0.168 | 3.445 → 0.172 | 0.099 → 0.959 |
| 0 | offline JSD (β 0.5) | 2.423 → 0.177 | 5.343 → 0.234 | 0.100 → 0.898 |
| 1 | offline JSD (β 0.5) | 2.798 → 0.180 | 5.047 → 0.135 | 0.096 → 0.960 |
| 2 | offline JSD (β 0.5) | 2.461 → 0.315 | 3.445 → 0.354 | 0.099 → 0.929 |
| 0 | offline, top-8 cache | 2.423 → 0.559 | 5.343 → 0.542 | 0.100 → 0.796 |
| 1 | offline, top-8 cache | 2.798 → 0.272 | 5.047 → 0.220 | 0.096 → 0.899 |
| 2 | offline, top-8 cache | 2.461 → 0.300 | 3.445 → 0.319 | 0.099 → 0.911 |
| 0 | GKD λ 0.5, forward KL | 2.423 → 0.113 | 5.343 → 0.163 | 0.100 → 0.903 |
| 1 | GKD λ 0.5, forward KL | 2.798 → 0.166 | 5.047 → 0.132 | 0.096 → 0.937 |
| 2 | GKD λ 0.5, forward KL | 2.461 → 0.323 | 3.445 → 0.693 | 0.099 → 0.769 |
| 0 | GKD λ 1, JSD (β 0.5, lr 2e-3) | 2.423 → 0.365 | 5.343 → 0.723 | 0.100 → 0.792 |
| 1 | GKD λ 1, JSD (β 0.5, lr 2e-3) | 2.798 → 0.747 | 5.047 → 1.160 | 0.096 → 0.779 |
| 2 | GKD λ 1, JSD (β 0.5, lr 2e-3) | 2.461 → 0.590 | 3.445 → 0.719 | 0.099 → 0.840 |
| 0 | GKD λ 1, reverse KL | 2.423 → 0.254 | 5.343 → 0.227 | 0.100 → 0.951 |
| 1 | GKD λ 1, reverse KL | 2.798 → 0.488 | 5.047 → 0.788 | 0.096 → 0.932 |
| 2 | GKD λ 1, reverse KL | 2.461 → 0.367 | 3.445 → 0.450 | 0.099 → 0.927 |
| 0 | λ 1, sampled reverse KL | 2.423 → 0.331 | 5.343 → 0.362 | 0.100 → 0.914 |
| 1 | λ 1, sampled reverse KL | 2.798 → 0.745 | 5.047 → 1.050 | 0.096 → 0.869 |
| 2 | λ 1, sampled reverse KL | 2.461 → 0.706 | 3.445 → 0.651 | 0.099 → 0.931 |

Negative control: the same runs against a shuffled teacher (its logits
permuted over the vocabulary: equally sharp, wrong tokens) must not lower
either KL to the real teacher and keep the digit mass below 0.4:

| seed | mode | fwd KL before → after | on-policy rev KL before → after | digit mass |
|---|---|---|---|---|
| 0 | shuffled teacher, offline forward KL | 2.423 → 4.721 | 5.343 → 7.867 | 0.100 → 0.100 |
| 1 | shuffled teacher, offline forward KL | 2.798 → 5.009 | 5.047 → 7.907 | 0.096 → 0.204 |
| 2 | shuffled teacher, offline forward KL | 2.461 → 5.613 | 3.445 → 6.017 | 0.099 → 0.004 |
| 0 | shuffled teacher, λ 1, sampled reverse KL | 2.423 → 4.447 | 5.343 → 6.895 | 0.100 → 0.115 |
| 1 | shuffled teacher, λ 1, sampled reverse KL | 2.798 → 5.385 | 5.047 → 6.456 | 0.096 → 0.209 |
| 2 | shuffled teacher, λ 1, sampled reverse KL | 2.461 → 5.542 | 3.445 → 5.824 | 0.099 → 0.005 |

Observed failure mode: pure on-policy JSD(0.5) at the common learning rate
(5e-3) collapsed on seed 2 (held-out forward KL 2.46 → 7.97, digit mass to
0): the student's own samples left the teacher's support, where the bounded
JSD saturates (≤ ln 2 per token) and its gradient vanishes, while the
later-position terms kept training on junk contexts. Forward KL, reverse KL,
JSD with β 0.1 or 0.9, or JSD at 2e-3 all converge on that seed; the test
runs JSD at 2e-3.

## Limitations

- Single process: no DDP/FSDP, no gradient checkpointing, no fp16
  GradScaler; sampling uses `model.generate` (no vLLM).
- The full teacher forward runs on every batch, including for
  `on_policy_loss="sampled"` (which only needs the sampled tokens' teacher
  log-probs); the top-k cache is offline only.
- Cross-tokenizer distillation needs shared token strings; differently
  segmented text is not aligned.
- Evidence is CPU toy models; large-model runs are graduation criteria.
