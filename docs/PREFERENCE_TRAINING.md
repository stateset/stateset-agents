# Preference training

`stateset_agents.training.preference` trains a policy directly on preference
data (DPO, conservative DPO, IPO, SimPO, KTO, ORPO) and trains Bradley-Terry
pairwise reward models that plug into the RL trainers as a `RewardFunction`.

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
| Pairs, and you want an RL reward rather than a policy | `rm` | Bradley-Terry reward model, exported as `PairwiseRewardFunction`. |

Preference optimization is offline: it never samples from the policy. For
on-policy RL with a verifiable or judged reward use the GRPO-family trainers
([OBJECTIVES.md](OBJECTIVES.md)); a reward model trained here is one way to
supply that reward.

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
only, no remote code). Use it like any other reward, preferably composed with
an outcome or verifier reward (see [REWARD_CONTRACT.md](REWARD_CONTRACT.md)):
learned rewards can be exploited by the policy they train.

## CLI

```bash
stateset-agents preference --train-data pairs.jsonl --validation-fraction 0.05 --dry-run
stateset-agents preference --method dpo --model Qwen/Qwen3.5-0.8B \
  --train-data pairs.jsonl --validation-fraction 0.05 --lora --output-dir outputs/dpo
stateset-agents preference --method rm --model Qwen/Qwen3.5-0.8B \
  --train-data pairs.jsonl --eval-data heldout.jsonl --output-dir outputs/rm
```

See [CLI_REFERENCE.md](CLI_REFERENCE.md#stateset-agents-preference).

## Limitations

- Single process: no DDP/FSDP, no gradient checkpointing, no fp16 GradScaler.
- Reward models train the whole backbone (no LoRA).
- Learning is verified on CPU toy models (`tests/learning/test_preference_learns.py`,
  three seeds); large-model evidence is a graduation criterion.
