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
  of `stateset_agents/evaluation/rubrics/support_rubric_v2.md`); editing the
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
