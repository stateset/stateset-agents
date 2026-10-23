# Improving a deployed policy

The improvement loop connects curated conversations, supervised preparation,
offline policy extraction, held-out evaluation, reusable exports, and reload
checks. Live quality, backend comparison, and distributed reliability remain
separate evidence gates in [the roadmap](ROADMAP.md).

## Run the local example

Install the training extra and run from the repository:

```bash
OMP_NUM_THREADS=1 python examples/offline_improvement_pipeline.py --output /tmp/offline-improvement
```

The example uses a tiny random transformer and a local tokenizer. It performs
real SFT and IQL-weighted assistant-token updates, exports both policies, reloads
each for held-out predictions, and executes local inference from the candidate
export. It requires no model downloads. Its synthetic cases verify mechanics;
the output explicitly records `production_approved=false`.

The [retained wheel verification](../benchmark_results/training_contracts/20261014/offline-improvement-wheel/README.md)
includes raw reports, actual optimizer counts, source hashes and the full local
verification result. Its toy predictions do not isolate RL gains from SFT.

`run_improvement_pipeline()` accepts backend-specific callbacks. SFT and RL
receive only train and validation partitions. Evaluation receives the frozen
artifact path and a detached test partition. Exporters must return the requested
path. Evaluation and serving must preserve the artifact bytes. The output keeps
the split manifest, baseline/candidate reports and artifact hashes, serving
result, and a stage journal that records failures and cancellation.

## Curate production conversations

Each JSONL row needs `source_id`, a timezone-aware ISO `timestamp`, and text
`messages` with supported roles. A finite `reward` is optional. Use a source ID
that groups related conversations, such as an account or incident, before
exporting them:

```bash
python -m stateset_agents.data.curation \
  --input production-traces.jsonl --output outputs/curated \
  --validation-after 2026-08-01T00:00:00Z \
  --test-after 2026-09-01T00:00:00Z --minimum-reward 0.5
```

The output directory must be new. Known credential and contact patterns and
sensitive fields are redacted recursively. Source IDs are replaced with hashes.
This baseline does not recognize every form of sensitive data; apply the
domain-specific review required for your source before exporting it.

Exact redacted message duplicates join their source groups. Each joined group
moves entirely to its newest temporal partition, preventing the same source
or duplicate conversation from crossing a split. The reward threshold filters
training only; failed held-out conversations remain available for evaluation.
The manifest retains counts and fingerprints for every partition.

Pass `--decontaminate-against EVAL` (repeatable: a path, `NAME=PATH` or
`builtin:NAME`, e.g. `builtin:gsm8k-test`) to drop training rows that share a
13-gram with an external eval set; `--ngram-size` changes the window. Held-out
partitions are not filtered. The manifest's `decontamination` block records
each eval set's SHA-256 and contaminated-row count and the number of training
rows removed. See [Decontamination](DECONTAMINATION.md).

## Train on actual logged actions

`ConversationDataset.policy_transitions()` defines ordered assistant actions.
The next observation includes intervening tool/user messages; terminal rewards
are retained. Its token rows and offline state/action features use that same
ordering. Only the current assistant action is supervised; earlier assistant
turns remain context. Truncating an action fails instead of silently changing
the target. Configure a real embedding cache before converting offline features.
The high-level `train()` entrypoint creates a sentence-transformer cache for
conversation datasets that lack one and uses its actual feature dimension;
install `sentence-transformers` for that path. Custom caches can avoid downloads.

`OfflineGRPOTrainer.train()` performs **IQL-weighted policy extraction from logged
actions**. It is an offline objective, with detached `min(Q1, Q2) - V` weights,
not an on-policy GRPO rollout objective. CQL/BCQ/BEAR/DT value pretraining surfaces
remain separate; language-model extraction currently requires IQL.
`policy_temperature` and `policy_max_weight` control exponential weighting.
Assistant-token loss is averaged per action before weighting. Nonfinite losses
or gradients fail before the optimizer changes weights.
The policy optimizer honors configured Adam betas and weight decay. The public
offline entrypoint rejects online `objective` overrides rather than silently
ignoring them.

Training averages the weighted loss over actions in each optimizer update,
including uneven microbatches and a final partial window. Changing microbatch
boundaries preserves the objective; assistant tokens are still averaged within
each action. Final-window metrics include the optimizer update and gradient norm.
Complete checkpoints contain policy/value networks, optimizers, pending
gradients, their action count and normalization scale, RNG state, dataset
fingerprint and the exact mid-epoch cursor. Older checkpoints can resume at
completed optimizer updates. Older in-flight accumulation lacks the counts needed
to preserve action weighting and is rejected before changing policy weights.
The [retained action-weighting regression](../benchmark_results/training_contracts/20261014/offline-action-normalization/README.md)
compares uneven accumulation with a combined-batch gradient and includes pending
checkpoint recovery, legacy compatibility, and complete wheel verification.
Resuming with changed logged data fails. `train(..., training_mode="offline",
save_path=...)` exports reusable weights and complete training state; legacy
value-only saves cannot satisfy an explicit policy resume.

The tiny-transformer device contract checks real logged-policy updates, complete
checkpoint recovery and inference after reloading the export:

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python benchmarks/offline_policy_device_contract.py \
  --device cuda --output /tmp/offline-device-contract
```

CUDA is mandatory unless `--device cpu` is explicitly selected. The output
directory must be new. Installed-wheel CI runs the CPU contract against minimum,
latest and locked training dependencies. The [retained device evidence](../benchmark_results/training_contracts/20261014/native-offline-device/README.md)
includes the initial CUDA RNG restore failure and the corrected wheel's passing
RTX A5000 run. These are synthetic execution checks; they do not establish
representative policy quality.

## Diagnose learning and measure cost

Native trainers reject nonfinite scalar diagnostics. Five consecutive measured
zero-advantage observations raise `TrainingSignalError` by default. Set
`zero_signal_patience` to a different nonnegative count, or zero to disable the
guard for intentional constant-reward or post-convergence experiments. Missing
measurements do not count as zero signal.

Multi-turn profiling separates environment rollout, reward evaluation and
learner host-wall time. Recent p95 samples are bounded; totals cover the run.
These timings do not synchronize CUDA kernels. Success counts require explicit
evaluator booleans; positive reward alone does not establish success. Optional
costs are caller-supplied and phase estimates exclude provisioning/storage;
retain the provider ledger for total spend.

## Check general capability before promotion

Domain fine-tuning can quietly cost general ability. With the optional
lm-evaluation-harness extra installed, score the base model and the candidate
adapter on the same public tasks, then gate on a bounded regression:

```bash
pip install 'stateset-agents[eval-harness]'
python -m stateset_agents.evaluation.capability_regression run \
  --checkpoint Qwen/Qwen3.5-0.8B --tasks hellaswag,arc_easy,gsm8k \
  --num-fewshot 0 --limit 500 --output outputs/capability-base.json
python -m stateset_agents.evaluation.capability_regression run \
  --checkpoint Qwen/Qwen3.5-0.8B --adapter outputs/sft_v1 \
  --tasks hellaswag,arc_easy,gsm8k --num-fewshot 0 --limit 500 \
  --output outputs/capability-candidate.json
python -m stateset_agents.evaluation.capability_regression compare \
  --baseline outputs/capability-base.json \
  --candidate outputs/capability-candidate.json \
  --max-regression 0.02 --task-max-regression gsm8k=0.03 \
  --output outputs/capability-gate.json
```

`run` drives lm-eval's `hf` model (`peft=` loads the adapter) with fixed seeds
and writes a report in the repo's evaluation shape (`schema_version`,
`base_model`, `seed`, `settings`, `selected_checkpoint`, per-task metrics and
stderr, and a headline metric per task: `acc_norm`, then `acc`, then
`exact_match`, and so on). `compare` exits `1` when a task's headline metric
gets worse by more than the allowed amount, in the metric's own units (`0.02` =
two accuracy points; `--relative` makes it a fraction of the baseline score).
Lower-is-better metrics such as perplexity are handled using lm-eval's
`higher_is_better`. Reports with different tasks, few-shot counts, limits,
seeds or chat-template settings are refused. Batch size is ignored. The gate
report carries both reports' content hashes. From Python:
`run_capability_eval(checkpoint, CapabilityEvalSettings(tasks=...), adapter=...)`
and `compare_capability_reports(baseline, candidate, max_regression=...)`. Any
object with lm-eval's `simple_evaluate` can be passed as `harness=`.

Without the extra, `run` exits `2` with the install command. The hook is tested
against a fake harness only. No retained lm-eval run backs it yet, and a pass
bounds regression on the chosen tasks only. Compare deltas with the reported
stderr before reading a small move as real.

## Promote evaluated bytes

Write required lineage with immutable `model_revision`, `dataset_sha256` and
`reward_sha256` before freezing the artifact. Required writes fail on error.
Promotion evidence contains those dataset/reward hashes, `artifact_sha256`,
review provenance (`source="reviewed"` and `reviewer`), and raw `baselines` and
`candidates` reports from `evaluation.agent_runs.evaluation_report()`.
Each report's settings must match all three lineage identities. Reports need
at least 32 distinct held-out case hashes per seed, six paired seeds, and distinct
candidate checkpoint paths and content hashes. The selected candidate's `selected_checkpoint.inventory_sha256`
must equal `training.promotion.artifact_digest()` for the actual policy tree.

```bash
python -m stateset_agents.training.promotion \
  --artifact outputs/policy --evidence outputs/paired-evidence.json \
  --output outputs/promotion.json
```

The receipt lives outside the artifact. Verification recomputes raw-outcome
quality and safety gates and hashes actual policy files. Symlinks, empty policy
trees, changed weights and edited receipts fail validation. The seed-level
significance test assumes independent training runs and a protocol fixed before
test inspection. A reviewer attestation does not independently prove that work.

Managed inference's `--production --promotion-receipt ... --artifact-directory ...`
checks the receipt before allocation, requires the same base model, and requires
the approved digest as a segment in the weights URI path. Uploading identical
bytes and verifying the remote object remain the deployment operator's
responsibility; local validation does not prove what a remote server loaded.
