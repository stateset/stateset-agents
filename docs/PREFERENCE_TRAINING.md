# Preference training

`stateset_agents.training.preference` trains a policy directly on preference
data (DPO, conservative DPO, IPO, SimPO, KTO, ORPO), runs online (iterative)
DPO against a judge, and trains Bradley-Terry pairwise reward models (full
backbone or LoRA) that plug into the RL trainers as a `RewardFunction`.

Stability: **provisional / beta** (`preference-optimization` in
[COMPONENT_MATURITY.md](COMPONENT_MATURITY.md)). The names are importable
from `stateset_agents.training` and `stateset_agents.training.preference`
but are not part of the v1 stable contract
([API_COMPATIBILITY.md](API_COMPATIBILITY.md)). It replaces the deprecated
research `stateset_agents.core.enhanced.DPOTrainer` (removed in v0.71.0).

## When to use which

| You have | Use | Notes |
|---|---|---|
| Pairs (chosen vs rejected) and a reference model you can afford | `dpo` | The default. `label_smoothing` (cDPO) if some labels are noisy. |
| Pairs, and DPO overfits (rewards and margins keep growing, held-out quality drops) | `ipo` | Regresses the per-token log-ratio gap to `1/(2β)` instead of pushing it to infinity. |
| Pairs, no memory for a reference model, length bias matters | `simpo` | Reference-free, length-normalized, target margin `simpo_gamma`. Typical `β≈2`, `γ≈0.5–1.5`. |
| Thumbs-up/down (unpaired) labels, or imbalanced data | `kto` | Pairs are split into one desirable and one undesirable example. |
| Pairs and no SFT stage yet | `orpo` | Reference-free; SFT on `chosen` plus an odds-ratio penalty. |
| Pairs, and you want an RL reward rather than a policy | `rm` | Bradley-Terry reward model, exported as `PairwiseRewardFunction`; `use_lora` for large backbones. |
| Prompts and a judge (a reward model, a verifier, an LLM judge), no fixed pairs | online DPO | `OnlineDPOTrainer`: sample, judge, pair best vs worst, DPO; repeat. |

The offline methods never sample from the policy. [Online DPO](#online-dpo)
builds its pairs from the current policy's own samples each round. For
on-policy RL with a verifiable or judged reward use the GRPO-family trainers
([OBJECTIVES.md](OBJECTIVES.md)); a reward model trained here is one way to
supply that reward. To match a teacher model's distribution instead, see
[DISTILLATION.md](DISTILLATION.md).

## Methods

All methods score **response tokens only**. A record is tokenized as prompt
ids followed by response ids, so the response mask is exact; log-probs are
gathered shift-by-one with `rl_losses.gather_token_logprobs` (position `t`
predicts token `t+1`; padding and prompt tokens are excluded).

| Method | Loss (per pair) | Log-probs | Reference |
|---|---|---|---|
| `dpo` | `-(1-ε) log σ(βh) - ε log σ(-βh)`, `h = Δ_w - Δ_l`, `Δ = log π - log π_ref` | summed | frozen |
| `ipo` | `(h̄ - 1/(2β))²` with `h̄` the per-token average of `Δ_w`, `Δ_l` | summed, then ÷ tokens | frozen |
| `simpo` | `-log σ(β(ℓ̄_w - ℓ̄_l) - γ)`, `ℓ̄` mean token log-prob (with optional `ε`) | mean | none |
| `kto` | desirable `λ_D(1 - σ(β(Δ - z)))`, undesirable `λ_U(1 - σ(β(z - Δ)))` | summed | frozen |
| `orpo` | `NLL(y_w) - β log σ(log odds(y_w) - log odds(y_l))` | mean | none |
| `rm` | `-log σ(r_w - r_l - m)` (+ `c (r_w + r_l)²`) | — | — |

KTO's reference point `z` is the batch mean of `log π - log π_ref` on
*mismatched* prompt/completion pairs (each prompt paired with another
example's completion through a fixed derangement), clamped at zero and
detached. `m` is a pair's optional `margin`; `c` is
`center_rewards_coefficient`.

The losses are pure functions in `stateset_agents.training.preference.losses`
(`dpo_loss`, `ipo_loss`, `simpo_loss`, `kto_loss`, `kto_kl_estimate`,
`orpo_loss`, `bradley_terry_loss`). They validate `beta` (finite, positive),
`label_smoothing` (`[0, 0.5)`), `gamma` (finite, nonnegative), shapes and
finiteness, and are pinned numerically to TRL 1.12 (`DPOTrainer` sigmoid and
IPO, `KTOTrainer`, `RewardTrainer`, CPO's SimPO and ORPO's odds ratio) in
`tests/unit/test_preference_trl_pin.py`. Our ORPO NLL is the mean of
per-sequence token means; TRL's is one token mean over the batch.

## Data format

JSON Lines, one record per line, read by the shared JSONL reader (records are
separated by `\n` only) and the strict parser used for fine-tuning data (NaN,
Infinity and duplicate keys are rejected). Errors name the file and line.

```jsonl
{"prompt": "Q: 2+2=", "chosen": "4", "rejected": "5"}
{"prompt": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Where is my order?"}], "chosen": "It ships tomorrow.", "rejected": "I don't know.", "margin": 1.0, "weight": 2.0, "group_id": "case-17"}
{"messages": [{"role": "user", "content": "Refund?"}], "chosen": {"role": "assistant", "content": "Approved."}, "rejected": [{"role": "assistant", "content": "No."}]}
{"chosen": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}], "rejected": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "What."}]}
{"prompt": "Write a haiku", "completion": "...", "label": true}
```

- Context: `prompt` (a raw string or a list of messages) or `messages`, or
  TRL's implicit-prompt form (`chosen`/`rejected` are full conversations that
  share every message but the last). Message prompts need a user message and
  must not end with an assistant turn.
- Responses: a string, an assistant message, or a one-element list holding
  one. `chosen` and `rejected` must differ.
- Unpaired (KTO only): `completion` plus a boolean `label`.
- Optional: `margin` (pairs; used by `rm`), `weight` (positive; scales the
  loss of that record in every method), `group_id`, `source_id`, `metadata`,
  and a `split` declaration (`train`/`validation`/`test`, checked like the
  SFT data). Other keys are ignored.

### Prompt-disjoint splits

`split_preference_dataset(dataset, validation_fraction, seed)` (and
`validation_fraction` in the configs and CLI) groups records that share the
request identity used everywhere else in the repo
(`stateset_agents.data.identity.prompt_identity` of the first user message,
or of a raw prompt: NFKC, case and whitespace insensitive), a `group_id`, or
a `source_id`, transitively, and assigns whole groups to one side. The result
is deterministic and independent of input order; exact duplicate records are
dropped. Explicit evaluation data is checked with `check_preference_overlap`
and rejected if it shares any identity with training data.

### Rendering

`prompt_format` (default `auto`):

- `chat_template` — the tokenizer's chat template; the response is the final
  assistant turn located by `core.chat_masking.final_turn_token_mask`
  (assistant-only labels including the end-of-turn tokens).
- `plain` — `MultiTurnAgent`'s format without a template
  (`System: …\nUser: …\nAssistant: `), then the response and EOS. Only
  system/user/assistant turns render.
- `raw` — a raw prompt string followed directly by the response and EOS.
- `auto` — message prompts use the chat template when the tokenizer has one
  (`plain` otherwise); raw prompt strings use `raw`.

`max_prompt_length` keeps the end of an overlong prompt; a sequence longer
than `max_length` keeps the end of the prompt (leaving room for at least one
response token) and the start of the response. Truncations are counted in the
training summary.

## Python API

```python
from stateset_agents.training.preference import (
    PreferenceConfig,
    train_preference,
    train_reward_model,
)

result = train_preference(
    "Qwen/Qwen3.5-0.8B",                 # or a loaded model + tokenizer=...
    "pairs.jsonl",                        # path, PreferenceDataset or records
    config=PreferenceConfig(
        method="dpo",
        beta=0.1,
        learning_rate=5e-7,
        batch_size=4,
        gradient_accumulation_steps=8,
        validation_fraction=0.05,         # prompt-disjoint held-out split
        use_lora=True,                    # reference = adapters disabled
        output_dir="outputs/dpo",
    ),
)
print(result.summary.eval_metrics)      # eval/rewards/accuracies, margins, ...

rm = train_reward_model("Qwen/Qwen3.5-0.8B", "pairs.jsonl", validation_fraction=0.05,
                        output_dir="outputs/rm")
reward = rm.reward_function(temperature=1.0)
```

`PreferenceTrainer(model, tokenizer, config, train_dataset, eval_dataset,
ref_model=None)` is the underlying class (`train()`, `evaluate()`,
`save_checkpoint()`, `load_checkpoint()`, `sequence_logps()`).

### Reference model

`dpo`, `ipo` and `kto` need a frozen reference: an explicit `ref_model`, a
frozen deep copy of the policy (default), or — with `use_lora` — the policy
with its adapters disabled (no second copy). Reference log-probs are computed
under `no_grad` per batch, or once up front with
`precompute_reference_logps=True`, which then releases the copy (evaluate new
data by passing it to the constructor). `simpo` and `orpo` are
reference-free.

### Optimization

AdamW (no weight decay on 1-D parameters), linear/cosine/constant schedule
with `warmup_steps`, gradient accumulation whose micro-batches add up to the
window's exact weighted mean, `max_grad_norm` clipping, `bf16` autocast
(`mixed_precision="bf16"`), dropout disabled by default (`disable_dropout`).
Updates go through `trainer_utils.safe_optimizer_step`. `nonfinite_policy`
is `raise` (default) or `skip` for a window whose loss or gradients are not
finite. `max_steps` (update attempts) overrides `num_epochs`.

### Metrics

Training history and `evaluate()` report each method's implicit rewards —
`β(log π - log π_ref)` for DPO/IPO/KTO, `β` × mean token log-prob for
SimPO/ORPO — as `rewards/chosen`, `rewards/rejected`, `rewards/margins` and
`rewards/accuracies` (fraction of pairs whose chosen reward is higher), plus
`logps/*`, the loss and (KTO) `kl`. Reward models report `eval/accuracy`,
`eval/auc` (ROC AUC of chosen vs rejected scores), `eval/margin` and the
Bradley-Terry loss.

### Checkpoints

`save_checkpoint(dir)` writes `model/` (`save_pretrained`, an adapter under
LoRA, plus the tokenizer), `trainer_state.pt` (optimizer, scheduler, RNG) and
`preference_checkpoint.json` (written last). `load_checkpoint(dir)` /
`train(resume_from=dir)` restore weights through `training.model_checkpoint`
and state through `core.checkpoint_io` (weights-only unless `trusted=True`);
resuming an interrupted run reproduces the uninterrupted one exactly.
Construct the trainer from the original model before resuming: the reference
comes from it.

## Online DPO

`OnlineDPOTrainer(model, tokenizer, reward_fn, prompts, OnlineDPOConfig(...),
ref_model=None, eval_prompts=None)` / `train_online_dpo(...)` run `rounds`
rounds of:

1. **Sample**: the current policy samples `samples_per_prompt` (K ≥ 2)
   completions per prompt (`max_new_tokens`, `sampling_temperature`,
   `top_p`; plain sampling from an explicit `GenerationConfig`, seeded per
   round), kept as exact token ids.
2. **Judge**: each decoded completion is scored by `reward_fn`, any
   `RewardFunction`, through the shared
   `core.reward_context.score_response` path (the reply in a
   `[user, assistant]` conversation with the standard context; a prompt's
   `scenario` mapping, e.g. a `ground_truth`, is lifted into it). A
   `PairwiseRewardFunction` (below) is one judge. Reward failures follow the
   repo contract: only `RewardUnavailableError` drops that completion (unless
   the reward sets `fail_on_error`); any other exception, a `None` and a
   non-finite score raise. A round whose every completion is unavailable
   raises `RewardUnavailableError`.
3. **Pair**: each prompt's best completion is paired against its worst when
   the gap is positive and at least `min_score_margin`; ties (`tie`), small
   gaps (`below_margin`), fewer than two scored completions
   (`too_few_scores`) and pairs that decode to identical text
   (`identical_text`) are skipped and counted. Among equal scores the
   earliest sample wins, a uniform choice since samples are i.i.d.
   (`select_pair`).
4. **Train**: `method` (`dpo` or `ipo`) on that round's pairs with a fresh
   `PreferenceTrainer` (the settings above: `beta`, learning rate,
   `num_epochs` or `max_steps` per round, LoRA, ...) on the exact sampled
   token ids, not a re-tokenization of their text. The reference is
   `reference_update="fixed"` (the initial policy every round: a frozen copy,
   an explicit `ref_model`, or the adapter-disabled policy under LoRA) or
   `"previous_round"` (a frozen copy of the policy as the previous round left
   it). A round without pairs trains nothing.

Each round's summary (`OnlineRoundSummary`) records samples, scored and
unavailable counts, pairs, skip reasons, mean/chosen/rejected judge scores,
the DPO training summary and, with `eval_prompts` (prompt-disjoint from the
training prompts), `evaluate()`'s mean judge score of fixed-seed policy
samples. With `output_dir` every round is checkpointed to
`<output_dir>/round-<r>` (`model/`, the round's `pairs.jsonl`, the round
history) and `train(resume_from=...)` continues from the next round exactly
(construct the trainer from the original model first; a `fixed` reference
comes from it).

```python
from stateset_agents.training.preference import (
    OnlineDPOConfig, load_reward_function, train_online_dpo,
)

judge = load_reward_function("outputs/rm/final")            # or any RewardFunction
result = train_online_dpo(
    "Qwen/Qwen3.5-0.8B", judge,
    [{"prompt": [{"role": "user", "content": q}]} for q in questions],
    config=OnlineDPOConfig(rounds=3, samples_per_prompt=4, max_new_tokens=256,
                           beta=0.1, learning_rate=5e-7, num_epochs=1,
                           reference_update="previous_round", use_lora=True,
                           output_dir="outputs/online-dpo"),
    eval_prompts=heldout_questions,
)
for entry in result.rounds:
    print(entry.round, entry.pairs, entry.mean_score, entry.eval_metrics)
```

## Reward models as RL rewards

`RewardModelTrainer` puts a linear head on a backbone's final hidden state at
the last non-padding token. `PairwiseRewardFunction` scores the newest
in-scope assistant reply of a conversation (the latest turn under
`reward_scope="latest_turn"`) in the context of every turn before it,
rendered exactly as in training:

- score `σ((r - offset) / temperature)` in `[0, 1]` (`score_range`), raw `r`
  in `metadata["raw_reward"]`;
- deterministic (eval mode, inference mode), never mutates turns or context;
- a non-finite model output raises `ValueError`.

`load_reward_function(checkpoint_dir)` reloads a saved model (local files
only, no remote code).

### LoRA reward models

`RewardModelConfig(use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
lora_target_modules=None)` (or `--method rm --lora`) wraps the backbone in
`peft` LoRA adapters (`FEATURE_EXTRACTION`; targets inferred like the SFT
path when not given) and trains only the adapters and the scalar head. `peft`
is an optional dependency (`pip install 'stateset-agents[training]'`); a clear
`ImportError` names it when missing. A LoRA reward model saves its adapter,
head, tokenizer and the base model's `name_or_path` (`reward_model.json`
records `"lora": true`); `load_reward_function(dir, base_model=...)` loads the
adapter onto `base_model` (a loaded backbone, a local path, or by default the
recorded base from the local Hugging Face cache, no downloads), and adapter
weights load through `training.model_checkpoint` (weights-only unless
`trusted=True`). `save_reward_model(..., merge_lora=True)` writes a merged,
standalone backbone instead. Trainer checkpoints resume exactly and refuse a
`use_lora` mismatch. Use it like any other reward, preferably composed with
an outcome or verifier reward (see [REWARD_CONTRACT.md](REWARD_CONTRACT.md)):
learned rewards can be exploited by the policy they train.

## CLI

```bash
stateset-agents preference --train-data pairs.jsonl --validation-fraction 0.05 --dry-run
stateset-agents preference --method dpo --model Qwen/Qwen3.5-0.8B \
  --train-data pairs.jsonl --validation-fraction 0.05 --lora --output-dir outputs/dpo
stateset-agents preference --method rm --model Qwen/Qwen3.5-0.8B \
  --train-data pairs.jsonl --eval-data heldout.jsonl --lora --output-dir outputs/rm
```

Online DPO has a Python API only (it needs a judge object).

See [CLI_REFERENCE.md](CLI_REFERENCE.md#stateset-agents-preference).

## Limitations

- Single process: no DDP/FSDP, no gradient checkpointing, no fp16 GradScaler.
- Online DPO samples with `model.generate` (no vLLM), scores each completion
  with its own reward call, and pairs only best vs worst per prompt.
- A LoRA reward model's adapter needs its base model available locally to
  load (or save it merged).
- Learning is verified on CPU toy models (`tests/learning/test_preference_learns.py`,
  `tests/learning/test_online_dpo_learns.py`, three seeds); large-model
  evidence is a graduation criterion.

### Online DPO evidence

`tests/learning/test_online_dpo_learns.py` (seeds 0, 1, 2): a tiny GPT-2
policy, the `DigitReward` judge (fraction of digit characters), 12 training
prompts, K = 8 six-token samples, 3 rounds of 4 DPO epochs. Held-out digit
rate (32 fixed-seed samples on each of 4 unseen prompts):

| seed | judge | reference | held-out digit rate | pairs per round |
|---|---|---|---|---|
| 0 | +digits | fixed | 0.105 → 0.999 | 12, 12, 12 |
| 0 | +digits | previous_round | 0.105 → 0.999 | 12, 12, 12 |
| 0 | −digits (control) | fixed | 0.105 → 0.000 | 12, 12, 6 |
| 1 | +digits | fixed | 0.093 → 0.224 | 12, 12, 10 |
| 1 | +digits | previous_round | 0.093 → 0.484 | 12, 12, 11 |
| 1 | −digits (control) | fixed | 0.093 → 0.001 | 12, 12, 0 |
| 2 | +digits | fixed | 0.093 → 0.884 | 11, 12, 12 |
| 2 | +digits | previous_round | 0.093 → 0.999 | 11, 12, 12 |
| 2 | −digits (control) | fixed | 0.093 → 0.000 | 11, 9, 2 |

The test asserts a rise of at least 0.1 with the real judge and a fall of at
least 0.05 with the sign-flipped one. Seed 1 with a fixed reference shows the
toy's limit: the training prompts reach a 0.97 judged rate by round 3 while
the held-out rate only doubles. Rounds without pairs (the control saturating
at zero digits, so every group ties) train nothing, as designed.
