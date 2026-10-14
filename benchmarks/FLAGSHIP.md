# Flagship benchmark — the one number that sells the framework

**Goal:** a reproducible, headline-grade result on a real (7–8B) model on the
framework's differentiator task — multi-turn customer support — with receipts
that anyone can re-run from a fresh clone. The existing whitepaper result
(judge +0.079 on a 0.5B model, §11.7) proves the pipeline; this proves the
*product*.

## The run

Three seeds × GSPO on an immutable 7–9B instruct checkpoint, multi-turn
customer-support composite reward, and cross-family judge-scored evaluation
before/after. Copy `flagship_manifest.example.json`, replace every placeholder,
and use one command for the complete roster:

```bash
cp benchmarks/flagship_manifest.example.json benchmarks/flagship_manifest.json
# Fill immutable revisions, exact hardware, provider billing source, cost limit,
# cross-family judge, and the provider driver before continuing.

make benchmark-flagship-run \
  MANIFEST=benchmarks/flagship_manifest.json \
  OUTPUT_DIR=benchmark_results/flagship_v1/preflight \
  EXTRA_ARGS=--preflight

make benchmark-flagship-run \
  MANIFEST=benchmarks/flagship_manifest.json \
  OUTPUT_DIR=benchmark_results/flagship_v1/measured
```

The driver is invoked as an argv list with `shell=False`. It must report the
provider-derived cost and create its policy artifact inside the runner-owned
directory. The outer runner measures wall time, hashes the artifact itself,
retains stdout/stderr and failed attempts, and refuses partial matrices.

## Publish gates (do not publish a number that fails these)

1. **Three seeds, all reported** — no seed selection. Mean ± std in the
   headline; per-seed JSONs committed under `benchmark_results/flagship_v1/`
   (the schema in `benchmark_results/SCHEMA.md` applies).
2. **Judge stability** — every seed records the cross-family judge's measured
   self-disagreement; any value above 0.05 invalidates the matrix.
3. **Baseline parity** — the pre-training eval uses the identical prompt
   template, decoding params, and judge as the post-training eval
   (`--skip-baseline` is forbidden for the flagship).
4. **Provenance** — harness commit, manifest/config digests, model, dataset and
   judge revisions, exact GPU/CUDA/driver, policy digest, portable artifact
   path, external wall time, and provider-derived cost are bound to schema-v2
   evidence. Publication re-hashes each retained artifact and rejects symlinks,
   path escapes, mixed harnesses, or mixed manifests.
5. **A negative or null result still gets committed** to
   `benchmark_results/flagship_v1/` — the credibility of every other number
   in this repo depends on not silently discarding runs.

## Where the number goes once it exists

- README badge + "What's new" headline (replace the §11.7 badge).
- `docs/WHITEPAPER.md` new section, anchored to the exact commit.
- The PyPI project description (long_description picks up README).

## Hardware notes

- The flagship gate accepts only 7–9B models. A smaller diagnostic is useful,
  but must use a separate output directory and cannot satisfy this gate.
- CI never runs this (GPU); the nightly `-m slow` convergence test
  (`tests/e2e/test_gspo_convergence_tiny.py`) is the automated proxy that the
  training loop still learns.

After collection, independently revalidate the retained bundle without
executing a provider command:

```bash
python benchmarks/run_flagship_matrix.py benchmarks/flagship_manifest.json \
  --validate-existing benchmark_results/flagship_v1/measured/evidence \
  --output-dir benchmark_results/flagship_v1/validated
```

## RunPod manifest (`flagship_manifest.json`) and pending pins

`benchmarks/flagship_manifest.json` is the real (non-example) v1 contract:
Qwen2.5-7B-Instruct policy, Mistral-Small-24B-Instruct-2501 cross-family judge, one
`NVIDIA H100 80GB HBM3` per seed, provider cost from the RunPod API. Each seed
runs `flagship_runpod_driver.py`, which provisions one pod, runs
`flagship_workload.py` on it (baseline eval -> multi-turn GSPO+LoRA -> final
eval -> judge, each GPU phase in its own process), downloads every artifact,
and terminates the pod unconditionally.

### Training protocol: on-policy multi-turn GSPO

Training is genuine on-policy multi-turn RL, not single-turn GSPO on
flattened histories. The workload drives `MultiTurnGRPOTrainer` with
`objective="gspo"` step by step (`generate_trajectories` then
`training_step`):

- **Episodes.** `ConversationEnvironment` (subclassed only to put the
  scenario's product and store policy in the system turn and to end episodes
  solely when the script runs out) opens with the scenario's first scripted
  customer message and replays each later scripted message after the agent's
  reply. Every assistant turn is sampled from the *current* policy
  (`rollout_temperature`, top-k/top-p off, no repetition penalty), so later
  turns condition on the policy's own earlier replies.
- **Objective.** Each assistant turn is one row on the trainer's batched token
  path; GSPO's length-normalized sequence ratio is taken per assistant turn
  (clip `clip_range_left`/`clip_range_right` = 3e-4/4e-4), weighted by the
  episode's group-normalized advantage over the `num_generations` episodes of
  the same scenario, with a sequence-level k3 KL to the base model
  (`beta`). A step that falls back to the sequence-level path fails the run.
- **Reward.** `ConversationSupportReward` = `SupportRewardComposite` scored on
  each assistant reply (with the scenario context and the preceding customer
  turns) and averaged over the episode's replies, so every turn earns credit
  and the composite's 10-120 word brand-voice window applies per reply rather
  than to the whole conversation. The cross-family judge is evaluation-only.
- **Schedule.** `scenarios_per_step` train scenarios per step, drawn in order
  from seeded per-epoch permutations of the train split; a run whose groups
  have no reward variance for `zero_signal_steps` consecutive steps, or whose
  rollout groups come back incomplete, fails closed.
- **Evaluation parity.** Baseline and final evaluation roll out the same
  environment (same system turn, scripted turns, prompt rendering through the
  training agent's chat template, greedy decoding at `temperature`,
  `max_completion_length`), via vLLM, the final one with the trained LoRA
  adapter. The judge scores exactly those multi-turn conversations.
- **Text fidelity.** The policy's replies end only at the chat template's
  EOS and enter the conversation verbatim; the base agent's blank-line stop,
  speaker-label truncation and keyword-injection hints are disabled for the
  flagship so training rows, conversation history and judged transcripts
  agree.

### Runtime and cost (1x H100 80GB at $3.99/h)

Training rollouts are lockstep-batched: each conversation turn of all
`scenarios_per_step x num_generations` = 32 episodes is one HF `generate`
call (`MultiTurnGRPOTrainer` with `batched_rollouts=True`), and the LoRA
update backpropagates one `train_forward_rows` chunk at a time. Estimated per
step on one H100 (unmeasured until the preflight): ~3.2 batched decoding
rounds of at most 192 tokens (~30-60 s) plus the update over ~100 turn rows
(~30-60 s), so ~1-2 min. The pinned 64 steps (2,048 episodes, 256 scenario
draws, about half an epoch of the 500 train scenarios) take ~1-2 h; with setup
(~25 min) and the two vLLM evaluations plus the judge (~25 min) a seed is
~2-3 h nominal. The 6 h `max_wall_clock_seconds` pod lifetime bounds spend at
<= $23.94 per seed even if throughput is 2-3x below the estimate. Run the
one-seed `--preflight` first and re-pin `max_steps` from its measured step
time before the measured matrix.

**Both revisions are pinned.**

- `dataset_revision` is commit `db0fd73c` of
  [`stateset/customer-support-multiturn-v1`](https://huggingface.co/datasets/stateset/customer-support-multiturn-v1)
  (560 train / 240 eval; generator export sha256 `59ccd505...`; loading it at
  that revision reproduces `build_support_dataset()` exactly).
- `judge.rubric_revision` is pinned to `rubric_revision()` (the git blob SHA
  of `stateset_agents/evaluation/rubrics/support_rubric_v4.md`); editing the
  rubric changes it, and the workload refuses a mismatch.

**Judge.** `mistralai/Mistral-Small-24B-Instruct-2501` (Apache-2.0, ungated,
cross-family vs the Qwen policy) at commit `9527884b`. It replaced
Mistral-7B-Instruct-v0.3 after the H100 preflight measured that judge as far
too lenient (base-model mean 0.901, self-disagreement 0.08 against the 0.05
gate; it scored policy_compliance 5 on replies that granted a must_avoid
reimbursement). Rubric v2 adds hard caps with concrete anchors (forbidden
commitments, unfilled placeholders, contradicted outcomes, redundant requests,
repetition), and `support_judge.detect_findings` puts deterministic findings
(must_avoid phrases present, template placeholders, near-verbatim repeated
assistant turns) into both passes as verified facts and into every per-item
judge record. The judge phase loads the local snapshot (HF-format shards only;
the duplicate `consolidated.safetensors` is skipped) with vLLM at
`gpu_memory_utilization=0.85` in its own process after the policy is freed;
v1 stays in the package for history.

**Rubric v3 (checklist).** The v2 judge measured self-disagreement 0.072 on
the H100 preflight (baseline group; final 0.058): 1-5 Likert scores moved
with criterion order, most for `empathy_tone` (pass 1, where it is listed
fourth, averaged +0.38 points above pass 2, where it is second) and
`resolution` (27% of items differed). v3 replaces each criterion's score
with 3-5 yes/no checks (21 total); the criterion score is the fraction that
hold and the item score the mean over criteria and passes. Pass 2 reverses
the criteria and the checks inside each criterion. Verified findings are
applied after parsing so the judge cannot contradict them: exact checks
(`policy_no_must_avoid_phrase`, `conciseness_reply_length` <= 120 words) take
the detector's value, and veto checks (`accuracy_no_placeholder`,
`conciseness_no_repetition`) are forced false when the detector fires.
Whether a must_avoid *mention* is a forbidden *commitment*
(`policy_no_forbidden_commitment`) stays a judge decision: on the preflight,
about a third of the 28 items with a must_avoid hit only mentioned the phrase
in a correct explanation ("no carrier-declared
weather emergency", "not returnable unless ..."). Each judge record keeps the
final `checks`, the `forced` values and the judge answers they `overrides`.

To measure a rubric on transcripts that already exist, without training:

```bash
# free: cheapest 80 GB A100/H100 in the catalog, worst case for 1.5 h
python benchmarks/flagship_rejudge.py --input-dir RUN/artifacts/seed-42 \
  --output-dir out/rejudge --plan
# paid (one GPU, judge phase only); the harness commit must be pushed
STATESET_REJUDGE_CONFIRM_MAX_COST_USD=5 python benchmarks/flagship_rejudge.py \
  --input-dir RUN/artifacts/seed-42 --output-dir out/rejudge
# or on a machine that already has an 80 GB GPU
python benchmarks/flagship_rejudge.py --input-dir ... --output-dir ... --local
```

Spending needs an explicit ceiling no larger than `max_cost_usd_per_seed`
(the manifest command is fixed, so pass it through the environment):

```bash
# free: price the worst case for one seed from the public catalog
python benchmarks/flagship_runpod_driver.py <placeholder args> --plan
# paid: the matrix runner invokes the driver once per seed
export RUNPOD_API_KEY=...            # never commit it
export HF_TOKEN=...                  # optional: the judge is ungated
export STATESET_FLAGSHIP_CONFIRM_MAX_COST_USD=25
make benchmark-flagship-run MANIFEST=benchmarks/flagship_manifest.json \
  OUTPUT_DIR=benchmark_results/flagship_v1/preflight EXTRA_ARGS=--preflight
```

`STATESET_FLAGSHIP_GPU` overrides the GPU (the manifest's `hardware.gpu` must
then change too, or the evidence is rejected).

### Preflight manifest

`benchmarks/flagship_preflight_manifest.json` is the measured manifest with
`max_steps` 16 and `max_wall_clock_seconds` 12600, so one diagnostic seed
exercises the whole pipeline (rollouts, training, both evaluations, judge)
with a worst case of about $13.30 on one H100 at $3.99/h. Its config hash
differs from the measured manifest, and `run_flagship_matrix.py` never accepts
preflight evidence for publication. Use its measured step time to re-pin
`max_steps` in `flagship_manifest.json`.

## Flagship v2: Qwen3.5-9B policy, Gemma 4 31B judge

v2 is a **separate protocol** (`stateset-flagship-multiturn-v2`) next to v1.
The v1 manifests, config hashes, behaviour and evidence are unchanged. v2
evidence goes under `benchmark_results/flagship_v2/` and is never mixed
with v1. Until the judge is calibrated, v2 scores are not comparable to v1
scores (see the risks below).

| | v1 | v2 |
| --- | --- | --- |
| Policy | `Qwen/Qwen2.5-7B-Instruct` @ `a09a3545` | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) @ `c2022362` (Apache-2.0, ungated) |
| Judge | `mistralai/Mistral-Small-24B-Instruct-2501` @ `9527884b` | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) @ `842da379` (Apache-2.0, ungated) |
| Dataset | `stateset/customer-support-multiturn-v1` @ `db0fd73c` | unchanged |
| Rubric | v4 (`rubric_revision` `25fa6b2a`) | unchanged, with the same 0.05 self-disagreement gate |
| Manifests | `flagship_manifest.json`, `flagship_preflight_manifest.json` | `flagship_v2_manifest.json`, `flagship_v2_preflight_manifest.json` |

`model_parameter_count` is 8,953,803,264. That is the text decoder that is
trained and evaluated (`model.language_model.*` plus `lm_head`), counted
from the pinned safetensors headers. The full checkpoint has 9,653,104,368
parameters, because it also holds a 456M-parameter vision tower and a
243M-parameter MTP head. Neither is trained or used.

What v2 changes, all switched on from the pinned config, so v1 configs
(which do not have these keys) behave exactly as before:

- **`runtime_profile: "flagship-v2"`.** The RunPod driver installs
  `.[training,vllm,flagship-v2]`, which brings in Transformers >= 5.5,
  vLLM >= 0.29 and `flash-linear-attention`. It also defaults to a 250 GB
  container disk, for the ~19 GB policy, the ~62 GB judge, a merged policy
  copy and caches. v1 still installs `.[training,vllm]` on 150 GB.
- **`chat_template_kwargs: {"enable_thinking": false}`.** This is bound on
  the policy tokenizer, so training rollouts, the trainer's token path and
  both evaluations render the same non-thinking prompt. Qwen3.5 thinks by
  default.
- **`forbid_think_blocks: true`.** A `<think>` or `<|think|>` marker in any
  training-rollout or evaluation completion fails the run instead of
  scoring reasoning text.
- **`lora_target_modules`.** These are the full-attention `q/k/v/o_proj`,
  the GatedDeltaNet linear-attention `in_proj_qkv`, `in_proj_a`,
  `in_proj_b`, `in_proj_z` and `out_proj`, and the MLP `gate/up/down_proj`.
  The names were checked against the pinned checkpoint's weight map; 24 of
  the 32 layers are linear attention. None of these names occur in the
  vision tower.
- **`merge_adapter_for_eval: true`.** vLLM silently ignores LoRA on hybrid
  Qwen3.5 checkpoints (docs/PROOFS.md). For the final evaluation, the
  adapter is merged into full weights with
  `stateset_agents.training.merge_adapter`, which refuses an adapter whose
  merged greedy probe matches the base model. vLLM then serves the merged
  weights. The result records `final_eval_weights: "merged"`.
- **`trust_remote_code: false`.** Both architectures are native in
  Transformers 5.5+, so no remote code runs. The flag is plumbed through to
  the agent, the tokenizer and vLLM in case a future pin needs it.
- **`judge.loader: "gemma4"`.** The Gemma 4 judge loads text-only in vLLM
  (no image slots), with `enable_thinking=False` and
  `gpu_memory_utilization=0.92` for its ~62 GB of bf16 weights. The v1
  Mistral path (`loader` absent or `"mistral"`) is unchanged.
  `flagship_rejudge.py` reads `judge.loader` from `--manifest` (or
  `--judge-loader`) and installs the v2 dependencies for it.

### v2 preflight (one seed, paid, needs explicit approval)

```bash
# free: validate the manifest and print the seed plan
python benchmarks/run_flagship_matrix.py \
  benchmarks/flagship_v2_preflight_manifest.json --dry-run --preflight
# paid: one H100 seed, capped at the manifest's $28 per-seed limit
export RUNPOD_API_KEY=...            # never commit it
export STATESET_FLAGSHIP_CONFIRM_MAX_COST_USD=28
make benchmark-flagship-run \
  MANIFEST=benchmarks/flagship_v2_preflight_manifest.json \
  OUTPUT_DIR=benchmark_results/flagship_v2/preflight EXTRA_ARGS=--preflight
```

**Worst-case cost.** The preflight keeps v1's 250-step config but sets
`max_wall_clock_seconds` to 25,200. That gives a 24,600 s pod lifetime, so
the worst case is **$27.27 at $3.99/h** for one H100 SECURE, under the $28
`max_cost_usd_per_seed`. The driver's `--plan` re-prices it from the live
catalog. The measured manifest allows 28,800 s per seed ($31.26 worst case,
$32 cap, so at most **$93.77 for three seeds**). These limits are
estimates, not measurements. A 9B hybrid policy with a 31B judge is slower
than v1. Expected time is
~35 min of setup and downloads, the training steps, ~10 min for the merge,
and ~30 min for the judge. Re-pin `max_steps` and the wall-clock limit from
the preflight's measured times before the measured matrix.

### v2 risks

- **Judge calibration.** Rubric v4 and the 0.05 gate were tuned against the
  Mistral judge. Gemma 4's self-disagreement and its leniency are
  unmeasured, and the preflight is the first measurement. Gemma 4 versus
  Mistral scores on the same transcripts can be compared, at the cost of a
  judge-only GPU run, with `flagship_rejudge.py --manifest
  benchmarks/flagship_v2_manifest.json` on existing v1 transcripts.
- **Memory.** The 31B bf16 judge leaves about 10 GB of an 80 GB GPU for KV
  cache at `max_model_len` 8192. If vLLM refuses to start, the judge phase
  fails closed.
- **Dependencies.** vLLM >= 0.29 pulls its own torch build over the
  container image's torch 2.7.1. The first v2 pod install is unverified.
- **Merge probe.** A very small adapter whose greedy probe completion
  matches the base model fails the final evaluation by design, instead of
  reporting base-model scores as trained ones.
