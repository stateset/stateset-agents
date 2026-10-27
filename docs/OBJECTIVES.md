# Policy objectives

## Training contracts

Native trainers call `TrainingConfig.validate_for_training()` before model or
optimizer allocation. `validate()` retains advisory guidance. CLI, API and Python
training share a torch-free configuration resolver; checkpoints retain the
resolved configuration.

`train()` inherits both `model_name` and `model_revision` from the supplied
agent configuration in online, offline, and hybrid modes. Explicit settings must
agree with that identity; configure a different model or revision on the agent
before training. A training override cannot retroactively pin loaded weights.
Nonempty string revisions are accepted for exploration; use immutable commits
for reproducible studies.
For API training, set `training_config_overrides.model_revision`; embedded and
durable workers apply it to the agent before loading its tokenizer and weights.

Required outcome rewards retain their failure and evaluator-context requirements
when nested inside `CompositeReward` or `UncertaintyWeightedReward`. Missing
references or executed tool evidence still stop training. Ordinary composite
components keep their scenario context; outcome components receive final
environment state. Composite metadata retains each component's breakdown and
metadata without treating a positive aggregate score as verified task success.
Compositions containing a sparse or cumulative child run at episode end, with
the complete trace. Required step-reward errors also propagate to the caller.
`Environment.get_reward()` supplies a completed trajectory's final state when
the reward requires evaluator context, so rescoring retains the same evidence.
Episode execution deep-copies public observation contexts before calling the
policy, including contexts supplied by custom observation hooks. Policy-side
mutations therefore leave evaluator tool traces and nested source data intact.
Uncertainty weighting preserves the original evidence and explanation, validates
finite nonnegative estimates and coefficients, and rejects nonfinite base scores.

Exact prompt and sampled token IDs are required by default. Legacy text-only
trajectories require `allow_sequence_fallback=True`; if the sequence path
would then have to supervise every token (prompt included), it also needs
`allow_unmasked_labels=True`. Requesting sampler
importance correction without sampler probabilities raises unless
`allow_missing_sampler_logprobs=True`; this opt-in is reported as
`old_logprobs_source="recompute_fallback"`. Metadata is validated without
coercing malformed IDs or probabilities. Sequence compatibility reports its
effective objective and fallback explicitly. On that path, a tokenizer without
a chat template that cannot map assistant text to tokens (no offset mapping)
would supervise every token, user and system included; that raises
`RolloutContractError` unless `allow_unmasked_labels=True`. A trajectory whose
sequence-path loss fails raises `TrajectoryLossError` rather than being dropped
from its group's mean.

Sequence objectives sum assistant-turn losses within each trajectory before
averaging trajectories, so early termination cannot reweight other episodes.
Across prompt groups, sequence objectives weight actual trajectory counts and
token objectives weight actual sampled response-token counts. Entropy remains
one global response-token mean, independent of the policy reduction; KL,
importance-ratio and clipping diagnostics also weight sampled tokens. Explicit
batch-normalized advantages center rewards within each prompt group and use the
reward standard deviation across the whole training batch, after reward clipping.
Forward chunk sizes do not change these measures. Uneven groups contribute
according to their actual trajectory/token counts.

Native GRPO gradient accumulation carries those counts across microbatches.
Policy terms use their configured trajectory/token measure, while entropy keeps
its response-token measure. Each microbatch graph is released after backward;
sequence objectives with entropy retain a second gradient bank. Both sums are
normalized before clipping, including a final incomplete accumulation window.
Advantage estimation still belongs to each rollout batch: accumulating gradients
does not merge independently sampled groups or recompute their reward statistics.
Checkpoint state records pending counts, scales and entropy gradients. Legacy
pending scalar windows finish with their original microbatch-mean estimator;
subsequent native token windows use count weighting. Custom scalar and sequence
compatibility losses retain their existing estimator.
Pending count-weighted windows use accumulation schema 2; completed or scalar
windows retain schema 1.
Keep custom/sequence compatibility losses in separate accumulation windows;
mixing them into a pending count-weighted window raises an error rather than
silently changing its measure.

Native token-path losses under `DistributedDataParallel` use counts summed over
the wrapper's own process group. A shared window scale and the global
trajectory/token denominator compensate for DDP's rank averaging. Policy and
entropy banks are both reduced before they are merged, unscaled and clipped;
the second backward on a retained graph cannot rely on DDP's first reduction.
These reductions also reconcile pending no-sync gradients. They add a final
bank reduction even when DDP has already averaged the policy bank. Flattened
reduction payloads are limited to 25 MiB, including when an individual gradient
is larger; noncontiguous inputs can require additional bounded copy storage.
FP16/BF16 bank reductions accumulate in FP32, and values are divided by the
group size before summation to avoid overflowing an otherwise finite mean.
`DistributedTrainer` synchronizes registered model buffers once at the learner
step boundary, then suppresses per-forward buffer broadcasts while local chunks
run. It restores the user's broadcast preference even when the loss fails.
This avoids collective-order deadlocks from unequal forward counts; buffer
layouts must remain consistent across workers.

Single-step, complete/partial accumulation windows and repeated inner updates
follow the same count rule. A fixed-size integer handshake rejects differing
trajectory/token measures, entropy-bank settings and invalid counts on every
participating rank. Manual CPU/Gloo setup uses CPU DDP device arguments and
validates an existing process group's topology/backend before reusing it.
Workers must enter the same update/window schedule with nonempty token-path
batches. Custom scalar/sequence compatibility losses retain their earlier
rank-mean estimator. The two-worker CPU contract does not certify NCCL, FSDP,
DeepSpeed or distributed checkpoint recovery. A prompt's complete group must
stay on one worker. Multi-rank Accelerate optimizer/partial-flush behavior
remains a separate verification requirement.

Native token-path `batch_norm` advantages (REINFORCE++ baseline) subtract each
complete prompt group's mean, then divide by the population standard deviation
of **all clipped rewards in that learner batch across the DDP process group**.
Group-normalized, median and leave-one-out estimators retain their complete
local prompt-group populations. An accumulation window does not become a new
reward-normalization population.

The legacy `global_mean` baseline uses the previous running mean, shared across
DDP ranks. Each rollout batch adds its global clipped reward mean/count once.
Both plain and KL-enhanced losses receive this state. Repeated inner updates
reuse a fixed baseline snapshot and do not observe the same rewards again,
including when a pending accumulation window must first be flushed.
Legacy `advantage_normalization` still centers and scales within each complete
group; when that scale is nonzero, centering cancels the running-mean offset.

A fixed-size count/configuration-digest handshake checks reward validity,
estimator settings, clipping, update intent and prior running state before the
token-loss model forward. FP64 central moments provide the global population
scale; counts remain exact int64. Nonfinite raw rewards are rejected even if
clipping could hide them. Peers also reject nonfinite computed advantages
together before baseline mutation or backward. Reward-level agreement alone
requires valid token-path batches on the same schedule. The native
learner admission check below additionally coordinates malformed or missing
metadata and incompatible path selection before old-policy scoring.
These semantics apply to the native token path, not sequence compatibility
fallbacks, FSDP or DeepSpeed.

`DistributedTrainer.training_step` admits a multi-worker native DDP batch before
buffer synchronization, old-policy snapshots or policy/reference scoring. Its
`stateset-native-learner-preflight-v5` handshake exchanges one 194-element int64
tensor per worker on DDP's own process group: protocol version, validity bits
and six SHA-256 digests. The components cover the effective loss, update schedule
and exact counters, model/reference layouts, optimizer groups and scheduler/AMP
state, active baseline/continual settings, and effective rollout path. The model
component also binds StateSet package source hashes, policy/reference submodule
source and loaded method revisions, learner/optimizer/scheduler implementations,
runtime dependency versions and numerical precision controls. Errors
identify components and process-group ranks without transporting sampled IDs,
prompt text, parameter values or pickled objects.

Admission checks sampled IDs, aligned finite nonpositive sampler likelihoods,
policy-version metadata, finite raw rewards and clipped FP32 bounds. Vocabulary
bounds are checked when the model exposes embedding sizes or a vocabulary size.
Explicit compatibility permissions still select the sequence or missing-sampler
fallback; peers must select the same effective path. Complete group sizes and
forward chunk counts may differ, and chunk preferences are validated locally
without requiring identical values. Rejected admission preserves model buffers,
optimizer state, pending gradients/banks, counters and running reward statistics;
corrected inputs/settings can continue on the same learner and process group.

The v3 admission descriptor distinguishes genuine empty contributors from
malformed rollout groups. Empty workers adopt populated peers' token path and
old-policy source, participate in global reward statistics and explicit
gradient-bank reductions, and perform no actor scoring forwards. Local sample
counts weight both the objective and metrics. Parameters unused on every worker
retain absent gradients, preserving optimizer momentum and weight-decay behavior.
A globally empty batch returns a no-op before buffer, gradient, optimizer,
scheduler, counter or running-baseline mutation. Mixed empty/positive sequence
compatibility batches reject before scoring; this guarantee covers token training.
Timing and cost remain observable for no-ops, but they do not consume or reset
the diagnostic monitor's policy-signal patience.

The native training loop coordinates local setup, rollout generation, ordinary
callbacks/logging, early-stop decisions and tracking cleanup. Generation uses
the unwrapped actor under inference mode; local token loss/backward and optimizer
phases report application failures to all live peers. A failed native loop is
poisoned until a complete checkpoint reload, including when an earlier optimizer
update already committed. All peers must still enter the same operation;
callbacks that perform their own incompatible collectives and failures inside
in-flight backend collectives are outside this protocol.

New multi-turn checkpoints persist `_next_episode`: zero before the first episode,
advanced after successful learner work or a globally empty episode, and unchanged
after an unexecuted transition. Resume uses that cursor; snapshots without it
preserve the older `current_epoch + 1` rule. Reporting, evaluation and periodic
saves fire once when batch-reported committed updates cross their configured
interval, including inner updates that cross a boundary without landing exactly
on it. Pending gradient accumulation batches do not repeat the previous step's
events.
Task-boundary and final accumulation flushes now construct committed-update
receipts and use the same log/evaluation/periodic-save cadence as batch updates.
The optional `on_optimizer_step(step, metrics)` callback and callable
`optimizer_step` event receive one receipt for each successful operation that
committed updates. `optimizer_step_start` is exclusive and `optimizer_step_end`
inclusive; `optimizer_updates` records the actual counter delta, including
multiple inner updates. `update_source` is `batch`, `task_boundary` or
`final_flush`; `update_task_id` identifies the outgoing active continual task,
or is null when no continual task is active. Episode callbacks continue to
describe generated episodes.

Flush receipts contain update counters and learning rate, without a newly
measured loss, reward or trajectory population. Their empty `trajectory_metrics`
is labelled `trajectory_metric_scope=not_measured_flush`. Loggers receive
numeric flush counts, including `task_boundary_updates` or `final_flush_updates`.
The event is separate from `step_end`, so it does not reset signal-guard history.
Only actual committed updates trigger cadence; GradScaler overflow skips and
globally empty work produce no update receipt. Native task-window admission
allows scaled overflow to reach GradScaler, while checkpoint publication still
rejects nonfinite pending gradients.

Boundary evaluation can stop before the incoming task generates data. Final
flushes evaluate before the final checkpoint, and a successful periodic save
after final consolidation satisfies the final save. Resumes with an exhausted episode budget produce
no additional update events. Native loop v3 agrees on receipt controls before
callbacks and rejects older loop generations at startup. All workers must enter
the same operation. Callback delivery is not transactional: failures after a
commit can leave an applied update, and loading an older checkpoint may replay
work. Native failures require a complete healthy reload. Failed task-memory
consolidation can also leave an earlier flush committed before reporting.

The [2026-10-16 source/wheel acceptance](../benchmark_results/training_contracts/20261016/reward-recovery/README.md)
verifies sixteen reporting and recovery scenarios per peer within twelve real
two-worker CPU/Gloo contracts. It includes real scheduler recovery and matched
initial/continual reference architectures. The retained failed attempts show
the fixture corrections. GPU/NCCL, multi-node and model-quality gains remain
unverified; the diagnostic acceptance below predates these reporting changes.

The native runtime now finishes local timing/cost recording and signal-guard
observation through coordinated DDP status phases. A local nonfinite metric or
profiling error rejects every live peer before the next learner step. Because
the optimizer may already have committed, these failures require a complete
checkpoint reload. Preflight v4 agrees on guard patience, streak, last-step
identity, retained failure and observation count before model scoring; differing
controls reject without learner mutation. Timing values and reward logs stay
rank-local. All participants must use this protocol; older native preflight
versions reject before the new post-step collectives.

Version-two learning state retains native diagnostic history across runtime and
asynchronous checkpoints: counters, bounded recent rewards, finite scalar
metrics, last-step identity and failures. Restore validates this state before
applying weights, preserves existing monitor references held by callbacks, and
clears transient rollout timing. Failed snapshots retain their failure; healthy
snapshots repair stale target failures. Process-local elapsed clocks restart.
Serial version-one learning state remains readable and starts fresh diagnostic
history, rather than borrowing history from the target trainer. Pending batches
retain the existing step-identity deduplication rule; globally empty batches
neither consume nor reset signal patience. Native DDP checkpoint v3 binds the
guard's shared controls while retaining each rank's diagnostic logs. Native v1
directories fail closed; this change does not migrate them.

The [diagnostic recovery acceptance record](../benchmark_results/training_contracts/20261015/native-ddp-diagnostics/summary.json)
retains 2,034 targeted source regressions (7 skips), 767 isolated-wheel
regressions, 411 type-checked modules and eleven actual two-worker contracts
against source and wheel. Eight diagnostic scenarios per peer cover pending
history and empty no-ops, threshold failure/reload, one-worker nonfinite metrics
and profiling errors, divergent finite guard decisions, three pre-update control
mismatches, and complete error cleanup. Corrected inputs remain usable after
admission rejection; post-update diagnostic errors require a healthy full reload.
The next optimizer update after reload matches an unwrapped combined-population
reference. Unit contracts also cover malformed history, retained failures,
callback monitor references and serial version-one learning-state compatibility
through runtime and asynchronous single/multi-turn checkpoint APIs. Peer, source
and wheel diagnostic reports agree. Package/test Python hashes remain fixed,
the wheel matches source, and workflow lint passes. This evidence uses tiny
CPU/Gloo learners and synthetic rewards; GPU and quality gains remain unverified.
Native episode callbacks now use global newly generated raw rewards:
`total_reward`, `episode_reward`, `episode_reward_std` and
`episode_num_trajectories`, labelled `episode_reward_scope=ddp_global_new_rollouts`.
Replay rewards remain separate. On reporting steps, `trajectory_metrics` covers
the global learner population including replay, labelled
`trajectory_metric_scope=ddp_global_training_batch`. W&B accepts these precomputed
scalars even when its worker generated no episodes. Inspectable older logger
signatures receive the same global scalar names through their existing metric
argument, without local trajectory objects or logger-body retries.

Rollout reports retain sample-weighted mean/std/extrema, group-weighted reward
diversity and an exact global median. Diversity excludes groups with fewer than
two reward observations. The model's process group exchanges four
int64 counts and eight float64 summaries after bounded protocol/status admission.
Detailed median selection uses at most 64 reductions of two cumulative counts
over ordered float64 keys, with no reward-vector, trajectory or text gathering.
Only local keys are sorted; even medians average two exact ratios before one
float64 rounding. Detail selection runs only on reporting steps. Count reporting
rejects populations above `2**53 - 1`; native loop v2 detects mixed v1/v2 loops
before starting episodes. Callback metric dictionaries and their nested rollout
summaries are copied so callback edits do not alter later logs.

The [global rollout reporting acceptance record](../benchmark_results/training_contracts/20261015/native-ddp-rollout-metrics/summary.json)
retains 2,014 source regressions (7 skips), 741 isolated-wheel regressions,
411 type-checked modules and ten actual two-worker contracts against source and
wheel. Twenty-one new scenarios per peer cover twelve reward populations, five
coherent rejection/recovery cases and four complete callback/logger loops.
Independent high-precision Decimal references cover unequal/empty workers,
float64 extrema, subnormals and skewed medians. A main worker with only replay
reports global new-episode mean 1.5/count 2 and learner-population mean 0.6/count 5.
Policy updates still agree with an unwrapped combined-population reference;
direct observation preserves learner state and RNG. Peer/source/wheel reports
agree, package/test hashes stay fixed, and workflow lint passes. This evidence
is CPU/Gloo with synthetic rewards; it does not establish GPU or quality gains.

The [empty-worker and loop-recovery acceptance record](../benchmark_results/training_contracts/20261015/native-ddp-empty-loop/summary.json)
retains 1,990 targeted source regressions (7 skips), 717 isolated-wheel regressions,
410 type-checked source modules, and nine real two-worker contracts against both
source and wheel. Its new 128-case update matrix crosses eight objective settings,
entropy on/off, one/two inner updates, fixed/alternating empty workers and SGD/AdamW.
Combined-population updates and metrics agree within `rtol=2e-5, atol=2e-6`;
pending-checkpoint next parameters and optimizer state agree exactly. A direct
CPU GradScaler resume/flush and two coherent overflow skips bring that contract
to 131 scenarios per peer.

Seventeen loop scenarios cover initial/completed resume, a failed task-B teacher
snapshot followed by exact recovery from task A, and fifteen coordinated local
failure cases. Source/test hashes stay fixed; wheel Python matches source; new
peer/source/wheel reports agree. CI includes these regressions and workflow lint
passes. These synthetic tiny CPU/Gloo cases do not establish GPU/NCCL,
multi-node or model-quality gains; the lifecycle gaps above remain open.

The guard compares layouts and scalar settings/state, including scalar tensor
learning rates. Scheduler identity additionally binds Python class code/source,
lambda/partial arguments, defaults, closures and referenced globals. Composite
schedulers include their children. Identical saved `LambdaLR` scalar state does
not establish the same future learning rates; differing callable bindings now
reject before scoring. Unsupported opaque bindings fail closed. This is not a
proof of equivalence for arbitrary callbacks or external mutable services.

At first admission after model wrapping, and at admission after a legacy/runtime
checkpoint load attempt, workers also compare dense policy/reference parameter
values and optimizer slots. Any worker's invalidation requests this check on all peers;
a successful cached check is skipped on routine learner steps. Values are
fingerprinted in bounded CPU tensor chunks (4 MiB by default, at least one
element), without making a full host copy. Device placement and strides do not
change identity; dtype, shape and exact bytes do. Nonfinite tensors, meta/sparse/
PyTorch-quantized layouts and unsupported opaque state reject coherently. The
value phase sends one additional 98-element int64 tensor of validity bits and
SHA-256 digests. Buffers and pending
gradient banks can legitimately differ between workers and remain rank-local.
The fixed-width admission version rejects mixed v1/v2 headers before this
conditional phase, so peers cannot disagree about entering it.

These admission checks protect completed-checkpoint startup and subsequent
restore attempts. Native DDP now also has the separate collective checkpoint
format described below. Every peer must reach admission; process/network
failures require a separate recovery protocol.
Single-worker DDP and non-DDP execution retain their local paths. Direct loss
helpers do not provide this learner-boundary admission guarantee.
Native token-path diagnostics under DDP now use their actual global measures.
`policy_objective_loss` includes the configured KL term and uses the policy's
trajectory/token denominator. `total_loss` and the historical `policy_loss`
alias additionally include token-weighted entropy; an EWC penalty is reported
separately with the policy measure. Entropy, KL, importance ratios and clipping
fractions weight sampled tokens. Advantage mean, nonzero fraction and population
standard deviation combine all workers' already-computed advantages using
counts and stable central moments, independently of the reward-population
statistics used to construct those advantages.

Reports expose global `num_trajectories`, `num_rows`, `num_tokens` and
`policy_denominator`, with `metric_scope="ddp_global_batch"` and
`metric_reduction_protocol="stateset-native-metrics-v1"`. These describe one
learner call; repeated inner updates report the final inner loss, and an
accumulation window is not retrospectively merged into a diagnostic batch.
`entropy_measured=false` marks the legacy zero placeholder when entropy was not
calculated. Integer optimizer counters retain their exact types and values.

A fixed-size integer/schema-hash handshake rejects mismatched names/types,
native/compatibility paths, reduction measures, nonfinite values and divergent
control state before the diagnostic payload collective. Counts remain int64;
values and moments travel as float64, without pickled objects or application
strings. Custom scalar metrics retain their rank-mean estimator. Diagnostic
validation follows the learner update and cannot roll back a committed optimizer
step. FSDP/DeepSpeed reporting and multi-rank Accelerate remain unverified.

The [retained native-DDP evidence](../benchmark_results/training_contracts/20261015/native-ddp-weighting/summary.json)
records 919 passing targeted regressions (7 skips), 355 passing wheel
regressions, and the same real two-process contract against source and wheel.
Each contract runs 288 update scenarios across six objectives, variable chunks,
entropy, complete/partial accumulation, repeated inner updates and `no_sync()`.
It also verifies registered buffers, rejection/recovery, missing gradients,
bounded payloads, and FP16 overflow/subnormal handling. The largest observed
parameter error from the combined-batch reference is about 1.2e-7. Counts overlap;
these are generated tiny CPU policies, not quality or GPU scaling measurements.

The newer [diagnostic verification](../benchmark_results/training_contracts/20261015/native-ddp-metrics/summary.json)
records 1,793 passing targeted regressions (7 skips), 399 type-checked modules,
371 passing wheel regressions and 294 real two-process update/report scenarios
against both source and wheel, including six nonzero-KL cases. Nine malformed
metric/schema cases reject on both ranks and recover afterward. The maximum
observed report difference from the combined-batch reference is about 1.6e-7.

The [reward-population verification](../benchmark_results/training_contracts/20261015/native-ddp-advantages/summary.json)
records 1,826 passing targeted regressions (7 skips), 400 type-checked modules,
495 passing wheel regressions, and both real two-process contracts against
source and installed wheel. The new contract exercises 48 population scenarios
across plain/KL-enhanced loss, clipping, normalized/running baselines, accumulation
and repeated inner updates, in addition to the existing 294 update/report
scenarios. Independent reward formulas match the distributed advantages; running
baseline states are identical across workers. Four edge populations cover
constant rewards, singletons, large offsets and below-epsilon scales. Twelve
bad reward/estimator/state cases reject on both peers and recover on the same
process group. The largest parameter difference from the combined-batch
reference is about 1.2e-7; the largest diagnostic difference is about 1.2e-6.
These CPU results do not certify split prompt groups, GPUs, multi-node execution
or improved model quality. Regression and wheel counts overlap.

The [native learner admission evidence](../benchmark_results/training_contracts/20261015/native-ddp-preflight/summary.json)
records 1,848 passing targeted regressions (7 skips), 401 type-checked modules,
517 passing wheel regressions and three real two-process contracts against
source and installed wheel. Forty admission rejection/recovery cases include
loss/schedule/optimizer drift, malformed sampled metadata, clipping, divergent
buffer preferences and trainable layouts, missing references, and pending
accumulation state. They verify no policy/reference scoring and unchanged
buffers, optimizer state, gradients/banks, counters and running baselines on
rejection, then continue on the same learner/process group. The wire capture
verifies one fixed-size int64 digest collective per rejection. The accepted
control preserves a counter above 2^53 while workers use different chunk sizes.
The existing 294 update/report and 48 reward-population scenarios also pass.
These are CPU contracts with overlapping regression counts; they do not prove
GPU execution, checkpoint tensor agreement or model-quality improvement.

The [replica and scheduler verification](../benchmark_results/training_contracts/20261015/native-ddp-replica-consensus/summary.json)
records 1,874 passing targeted regressions (7 skips), 404 type-checked modules,
543 passing wheel regressions, and all four two-process contracts against
source and installed wheel. Package and test sources stay unchanged throughout
acceptance. Thirty-two new scenarios use completed one-worker checkpoints staged
per worker and restored before DDP wrapping through both checkpoint APIs.
Twenty-six cases reject value drift/nonfinite values, scheduler callable
differences, source-file I/O failure or mixed protocol headers without scoring
or changing learner state. Corrected next updates match the combined reference
exactly in these generated tiny CPU tests. Asymmetric invalidation, failed-load
invalidation and skipped routine rechecks are covered. The existing 294 update,
48 population and 40 admission scenarios also pass. Counts overlap; this does
not certify the legacy parallel checkpoint writer, multi-rank pending-window
recovery, GPUs, multi-node execution or quality gains.

The Hugging Face fallback batches groups up to `generation_batch_size`, trims
padding after EOS, and restores model mode and KV-cache settings on failure.
Token scoring gathers selected logits before normalization, avoiding a full
vocabulary log-softmax output.

GSPO, GSPO-token, DAPO, GEPO, VAPO, PPO and the base trainer share versioned
`TrainingRuntime` checkpoints. Publication stages a complete directory, syncs
its inventory and replaces it atomically. Loading restores policy and critic
weights, optimizers, schedulers, pending gradients, counters and Python/NumPy/
Torch RNGs. Native single/multi-turn GRPO retain their asynchronous checkpoint
API and share the same publication and RNG primitives. Legacy model exports
are not silently treated as complete runtime checkpoints.

The shared runtime validates optimizer slots, registered component schemas,
counters, pending gradients and isolated RNG states before applying restored
learner state. Nonfinite policy parameters or optimizer slots cannot be
published as a new runtime checkpoint. Restore checks policy parameters after
loading; a late model, component or rollout-publication failure quarantines the
trainer. Its managed training, generation and checkpoint publication paths stay
blocked until a complete validated runtime reload succeeds. This containment
does not roll back a partially applied restore or guard direct calls to a raw
optimizer. Unsupported stateful components or load hooks reject explicitly.
The verified optimizer surface is standard PyTorch SGD, Adam and AdamW;
custom optimizer classes and load hooks reject. Built-in scheduler/scaler state
must pass its schema and binding checks. Restored metric histories own their
containers, and NumPy numeric metrics serialize as plain Python values without
changing the live histories or rounding integer metrics through floating point.
Conflicting saved task identifiers reject before learner mutation.

Local native single/multi-turn checkpoint recovery also stages supported
optimizer, scheduler, scaler, counter, pending-gradient and RNG state before
changing the policy. Later commit failures keep managed training and publication
quarantined until a complete reload succeeds. Restored policy publication must
succeed when rollout synchronization is enabled; explicit `rollout_sync=False`
retains its opt-out. The collective native DDP protocol remains a separate path.

Model-only imports retain their `False` return and can change weights for
inspection or inference. They do not establish learner recovery, and a mutated
trainer remains quarantined; construct a fresh trainer for an intentional warm
start. Structurally complete legacy checkpoints without RNG state remain a
best-effort recovery path, with a warning that exact stochastic continuation is
unavailable. Metadata alone does not establish successful training recovery.
New native payloads declare `native_checkpoint_schema: 1`, requiring complete
learner records and atomic publication. The older publication marker alone
does not assert that payload schema. After successful rollout publication, RNGs
restore last so backend setup cannot advance the resumed learner's stream.

Adaptive PPO runtime checkpoints retain the current KL coefficient, target and
horizon. A configured adaptive controller requires that saved state; old
checkpoints without it cannot establish exact adaptive continuation. Supported
legacy checkpoints without an adaptive controller remain readable. Objective
and estimator configuration must match, including the PPO GAE/value/clipping
and inner-update settings.

Dedicated `train_with_gspo`, `train_with_gspo_token`, `train_with_dapo` and
`train_with_gepo` convenience loops, plus their customer-service/reasoning
wrappers, reject explicit resume requests before expensive setup. They do not
restore an iteration/query cursor. Native single/multi-turn GRPO resume remains
supported, including the studio's local GSPO objective path. Dataset/environment
cursors and arbitrary user callbacks are not supplied by the low-level runtime.

When `DistributedTrainer.agent.model` is an actual DDP wrapper, both checkpoint
APIs use `stateset-native-ddp-checkpoint-v3`. All process-group workers must
call the same save or load, with the same absolute shared-filesystem path.
The asynchronous API uses `DistributedConfig.checkpoint_dir`; the runtime API
uses its explicit directory. Native DDP always saves each worker's state,
including when the legacy `checkpoint_parallel` setting is false. Unwrapped
single-worker checkpoints retain their existing local format.

The atomic directory holds one `training_state.pt` for policy/reference
parameters, optimizer slots/groups and scheduler state, a `native-ddp.json`
generation record, and a `rank-NNNNN.pt` shard for every process-group rank.
Local shards retain persistent and nonpersistent buffers, pending policy and
entropy gradients with their normalization counts, RNGs, estimator/replay/EWC
history, counters, AMP state and rollout publication versions. The publication
inventory binds every file. Save checks shared values and control agreement
before publishing; a failed save preserves the previous completed generation.

Initialize the supplied model, optimizer, scheduler and DDP wrapper before
loading. Restore requires the same ordered process-group ranks, Python/Torch
versions, device family, architecture, training configuration and tokenizer
identity. Each peer validates its shard before any peer applies state. Model
parameters and the DDP wrapper retain their identities. Scheduler validation
binds future callable behavior as well as saved scalars. CPU optimizer slots
receive owned storage so later updates cannot modify mapped checkpoint files.
Optimizer load hooks are unsupported because staging them could mutate live
state. This directory is a learner checkpoint; export model weights separately
for Hugging Face serving.

Version 3 additionally requires matching implementation revisions, including
when all workers agree with each other but differ from the checkpoint. Older
native DDP checkpoints lack this binding and are rejected; they must be resumed
with their original implementation. Unwrapped single-worker formats are
unchanged. Source identities exclude installation paths, so identical source
and wheel installations agree. Source files must be readable; edits and missing
sources are checked at admission, without a permanently cached startup identity.
Use immutable installations and restart workers after code changes. This is
source/runtime provenance, not hardware attestation or complete equivalence of
arbitrary mutable globals, hooks, external services or native library binaries.

Validation errors preserve live learner state. An unexpected apply or rollout
publication failure blocks training and checkpoint saving on all peers until
a coordinated reload succeeds. This does not roll back a partially applied
restore. Every worker must participate in the operation; a lost worker or
broken process group requires process-group recovery outside this protocol.
Dataset/environment cursors and arbitrary user callbacks are also outside the
checkpoint. GPU/AMP, multi-node, FSDP/DeepSpeed and multi-rank Accelerate recovery
require their own acceptance evidence.

The [native DDP checkpoint acceptance record](../benchmark_results/training_contracts/20261015/native-ddp-checkpoint/summary.json)
retains 1,890 passing targeted regressions (7 skips), 406 type-checked modules,
559 installed-wheel regressions and five real two-worker contracts against
both source and wheel. The new contract runs 84 resume scenarios per peer:
both checkpoint APIs, SGD with momentum and AdamW, six explicit objectives plus
the default objective, completed/partial accumulation and repeated inner
updates. Resumed parameters, optimizer state and rank-local RNG draws match
exactly. Fourteen rejection/recovery scenarios cover write/read/validation/apply
faults, re-signed malformed shards, differing paths/operations and shared-value
drift. Unit checks also exercise replay, EWC name translation, frozen LwF
references, malformed optimizer slots and mapped-storage ownership. Package
and test hashes stay fixed throughout acceptance, and the wheel matches every
package source file. These generated tiny CPU tests verify learner mechanics;
they do not establish model quality or complete training-loop task/evaluation
coordination. Counts overlap with the earlier contracts.

Both checkpoint APIs retain the global reward baseline's mean, count and running
sum, private replay/reservoir and fractional-replay RNG streams, the current task,
frozen LwF reference weights, and EWC statistics. Learning-state validation occurs
before policy weights change. Legacy checkpoints without baseline history cannot
resume a `global_mean` estimator; legacy LwF snapshots without reference weights
are also rejected. Legacy snapshots used with group baselines reset unused global
statistics, and older replay snapshots without RNG state retain their supplied
generator stream without claiming exact historical replay recovery.

EWC's empirical Fisher estimate averages squared trajectory-loss gradients over
accepted examples. `ewc_num_samples` caps attempted trajectories; failed examples
do not dilute the denominator or trigger extra attempts. Empty/truncated assistant
targets are skipped before forward, and nonfinite losses/gradients are excluded.
Zero connected gradients remain valid observations. If no example is accepted,
the prior Fisher statistics and task anchor remain unchanged.

FP16/BF16 gradients are squared and averaged in FP32, with FP32 anchors; FP64
policies retain FP64 statistics. A running mean avoids overflowing a finite
estimate's intermediate sum. Consolidation validates all candidate statistics
before replacing task memory. Existing optimizer gradients and each submodule's
train/eval flag are preserved; OOM and interruption errors propagate. Inspect
`manager.last_fisher_estimate` for attempted, accepted, skipped and failure counts.

For a native DDP policy, `ContinualLearningManager.on_task_end()` is collective
and requires the standard manager on every live process-group worker. Teacher
snapshots copy the unwrapped policy and freeze it; they retain no DDP reducer
or forward collectives. Fisher estimation also forwards the unwrapped policy,
enables autograd inside `no_grad`, and never accumulates into live parameter
gradients. Peers may contribute different numbers of accepted examples or none.
Global Fisher is the accepted-count-weighted mean of local estimates. Empty
global populations preserve prior Fisher and anchors without applying decay.
Reduction prescales FP64 buckets capped at 4 MiB before writing FP32/FP64 master
statistics. Invalid examples remain visible in the population report.

Consolidation stages teacher/EWC state and private replay RNG streams before
committing. Local buffers, module modes, global RNGs, optimizer state and pending
policy/entropy banks survive rejected staging. Buffer synchronization honors the
wrapper preference once, then restores local values. Every worker validates the
candidate before replacing the public manager's contents. Partial memory
application or cleanup failures block learning, evaluation and saving until a
full native checkpoint reload succeeds.

Automatic task switches close the previous task's pending optimizer window
before constructing its teacher and anchors. A successful flush remains a
committed update if later memory estimation fails; the old memory and task ID
stay unchanged, and retry does not repeat that update. Partial flush failure
requires full reload. The scheduler budget uses the captured training task
sequence, including each task's partial window. Finalization uses the same
coordinated flush. Checkpoints retain a task/policy-step/replay-sequence receipt
so a completed-run resume does not consolidate the final task twice. Older
snapshots without this receipt cannot prove that final consolidation occurred.

Native checkpoint tokenizer identity excludes incidental fast-backend padding
and truncation state left by preprocessing. It still binds vocabulary, added
tokens, processing, templates and tokenizer sides; training lengths are bound
in configuration. Loading a generation without a reference model clears a
stale teacher, including after a failed task-memory commit.

Task labels and transition order must agree across workers. These task-memory
phases have their own evidence boundary. The surrounding loop now coordinates
local generation and ordinary callback failures as described above; callback
collectives, environment-owned state, lost processes and GPU/multi-node
acceptance remain separate requirements.

The [native DDP continual-learning acceptance record](../benchmark_results/training_contracts/20261015/native-ddp-continual/summary.json)
retains 1,965 targeted source regressions (7 skips), 674 installed-wheel
regressions, 409 type-checked modules and seven real two-worker contracts against
source and wheel. Its new contract contains 36 population scenarios across LwF,
EWC and combined replay/LwF/EWC, with buffer synchronization enabled/disabled
and unequal, empty, masked, capped or privately sampled replay populations.
Fisher applies in 24 cases and frozen teachers in 24. Reference Fisher uses real
offline HF tokenization, assistant masks and causal autograd losses.

Eight rejection/recovery cases cover snapshot/estimation failures, divergent
configuration/task/window state, partial memory commit, failed optimizer flush,
and estimation failure after a completed flush. Two additional complete training
and resume cases execute five task segments with one or two inner updates,
commit exactly five or ten optimizer steps, and preserve final task memory on
completed-run resume. Pending-window checkpoints resume exact next parameters
and optimizer state. All source/test hashes stay fixed, the wheel matches the
source, and peer/source/wheel outcomes agree. These generated tiny FP32 CPU/Gloo
cases do not certify general loop failure orchestration, GPUs or quality gains;
scenario and regression counts overlap with prior records.

Run a sampled multi-turn regression without downloads:

```bash
python -m stateset_agents.training.sampled_workflow \
  --output /tmp/sampled-workflow --seeds 42 43 44
```

To exercise gradient accumulation with the same 80 optimizer-update budget per
seed, use `--updates 160 --gradient-accumulation-steps 2` and a new output path.
`--updates` counts rollout batches; the report records committed optimizer
updates. A final partial accumulation window is flushed before checkpointing.

The tiny transformer receives supervised tool-syntax preparation shared by the
baseline and candidate. Balanced decision labels leave policy learning to RL.
Actions are sampled through unmodified Hugging Face generation and executed
against an isolated ledger. Evidence retains paired held-out outcomes, case
hashes, optimizer updates, reload checks and a bootstrap interval across seeds.
This is synthetic mechanics evidence, not representative industry quality.
Use reviewed cases and `stateset_agents.verticals.agent_study` for that claim.
The retained [three-seed CPU regression with accumulation](../benchmark_results/training_contracts/20261015/evaluation-isolation/native_workflow/summary.json)
includes raw before/after outcomes, training metrics, and exact-weight reload
checks for this workflow.

`training-contracts.yml` checks installed wheels with minimal, API, workers and
training extras; deliberate minimum/latest/locked dependency combinations; the
sampled regression; and the PostgreSQL worker contract.

For a representative native study on reviewed refund cases:

```bash
python -m stateset_agents.training.native_refund_study \
  --cases reviewed-refund-cases.json --output outputs/refund-study \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775
```

The JSON contains `provenance` (`source="reviewed"`, a nonempty `reviewer`) and
`train`, `validation`, `test` lists of refund-policy order facts. Each partition
needs at least 32 independent cases, with no repeated IDs or observable order
facts across partitions. Changing annotations or IDs does not create a distinct
case. Reviewed studies require at least six training seeds for an exact seed-level sign
test; the final policy is selected exclusively on validation results. Raw test
outcomes are retained before training and once after reloading the selected
checkpoint. Failure to meet the quality gate exits nonzero. `--synthetic`
exercises the same pipeline while always refusing a representative-quality claim.

Native refund evaluation uses `sha256-case-v1` randomness: the first 64 bits of
the canonical SHA-256 of protocol, training seed, case ID and exact case hash
seed each episode independently. Reports retain that seed and case hash.
Python and Torch use the full 64-bit seed; NumPy's legacy generator uses its
low 32 bits. Case order and an earlier episode's token consumption do not shift
later cases.
Both policies start each matched case from the same streams; divergent action
paths can still consume different random draws within a case. Historical
sequential-stream reports are a different protocol and cannot be mixed with
these reports in a comparison.

`evaluation.isolation.isolated_evaluation()` preserves Python, NumPy, CPU Torch
and available CUDA RNGs, grad mode, each module's train/eval flag, and the model's
KV-cache setting on normal exit, error or cancellation. Imports remain lazy.
Local evaluations must run serially with training and other global RNG consumers;
overlapping isolated contexts from different tasks or threads raise before
reseeding. Private agent/provider RNGs need their own explicit isolation.
Invalid revisions, nonfinite learning rates, insufficient reviewed seed counts
and group-relative budgets with fewer than two generations fail before model
allocation. Synthetic runs may use fewer seeds but cannot pass the representative
quality gate.

Validation reports use `split="validation"`; held-out comparison rejects them.
The native study retains the initial validation report and every scheduled
validation, then writes a selection receipt with their hashes, the winning step
(earliest on ties), committed optimizer-update count and checkpoint inventory.
The candidate report binds that receipt. Missing, inconsistent or zero committed
updates stop the study before candidate test evaluation.

The shared `training.evaluation.evaluate_agent()` aborts on episode/scoring
errors by default. Concurrent episodes are cancelled and drained before failure
returns. Required reward failures and OOM propagate even in diagnostic mode.
Rewards needing evaluator context receive final executed state; ordinary rewards
receive the scenario. Nonfinite rewards are invalid, and an unavailable override
never falls back to the environment's score.
With `EvaluationConfig(fail_on_error=False)`, reports include attempted, valid
and failed episode counts, error rate and completeness. Failed attempts stay
in the success-rate denominator; reward/length means describe valid samples only.
`eval_success_rate` retains its legacy `reward > 0.5` heuristic. It does not
establish verified task success; use executed-outcome evidence for quality claims.

`MultiTurnGRPOTrainer.evaluate()` now isolates local evaluation from training
RNGs, grad mode, module modes and KV-cache preferences. Empty or incomplete
results neither select a best checkpoint nor consume early-stopping patience.
A failed best-checkpoint save restores the previous best score and patience.

For an actual native DDP wrapper, `DistributedTrainer.evaluate()` is collective.
Each worker's scenario list is its local partition, and `num_eval_episodes` is
a per-worker limit. Lists are not automatically sharded or deduplicated; counts
describe attempted samples rather than unique reviewed cases. Workers agree on
evaluation settings, policy values and selection counters before inference.
Local generation uses the unwrapped policy, allowing unequal forward counts
and empty partitions. Buffer synchronization follows the DDP preference once
at entry, and local buffers are restored before selection, including custom
evaluation buffer writes. Pending policy/entropy gradient banks stay intact.

Global reward/length means and population reward variance use valid sample
counts; success/error rates retain all attempts in their denominators. Native
metrics carry an exact private integer success count without adding public
JSON fields. Counts travel as four int64 values and moments as three float64
values, after a fixed 66-element int64 status/identity handshake. Every peer
receives the same metrics and enters the same collective best save. Local
errors and cancellation notify peers before returning; cancellation retains
its local exception type. A buffer-restoration failure blocks further learning
and evaluation until a complete native checkpoint reload succeeds.

Every live peer must enter evaluation, and process-group timeouts must cover
the slowest local partition. Serialize evaluation with training and other
process-global RNG consumers. Environment-owned state/RNGs, external providers,
arbitrary user callback collectives, lost workers, and complete training-loop
task transitions require separate coordination. CPU/Gloo evidence does not
establish GPU, multi-node or other distributed-backend behavior.

The [native DDP evaluation acceptance record](../benchmark_results/training_contracts/20261015/native-ddp-evaluation/summary.json)
retains 1,917 targeted source regressions (7 skips), 586 installed-wheel
regressions, 408 type-checked modules and six real two-worker contracts against
source and wheel. The new contract checks 28 scenarios per peer with buffer
broadcasting enabled and disabled: unequal populations, nonimprovement/ties,
empty partitions, local errors/cancellation, policy/settings/counter drift,
malformed statistics, failed best publication, failed buffer restoration and
different collective operations. Eighteen reject and then recover on the same
process group. Best checkpoints preserve pending windows and resume the same
next optimizer update. Wire shapes/dtypes are checked, source and test hashes
stay fixed, and source/wheel outcomes agree. These generated tiny CPU tests
verify evaluator mechanics; they do not establish model quality or complete
training-loop orchestration. Test counts overlap with earlier acceptance runs.

Shared runtime metrics separate rollout time from update time, count exact
generated tokens and verified successful episodes where available, and report
process peak VRAM on CUDA. Supplying a measured `compute_cost_per_hour` adds an
estimated compute cost for those phases and successful episodes per dollar.
These phase estimates exclude provisioning, storage and network charges; the
provider ledger remains the source for whole-job accounting.

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

## Selecting an objective

Every trainer evaluates its native preset unless you name another one.

```python
from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer

config = DAPOConfig(model_name="Qwen/Qwen2.5-0.5B-Instruct", objective="cispo")
config = DAPOConfig(objective="dapo", objective_overrides={"clip_high": 0.3})
config = DAPOConfig(objective="dr_grpo")  # max_completion_length is filled in
```

```bash
stateset-agents train --list-objectives
stateset-agents train --objective rloo --no-dry-run
stateset-agents model qwen3-5-0-8b --objective gspo_token --write-config run.json
```

Rules, implemented by `objectives.resolve_objective`:

- `objective` unset: the trainer's native preset with the trainer's own clip
  and aggregation fields (`clip_eps_low`, `use_token_level_loss`, ...), so
  nothing changes by default.
- `objective` set: that preset with its own clip and aggregation choices; the
  trainer's native clip fields are not applied. `beta` becomes `kl_coef` when
  the preset has a KL estimator, `max_completion_length` is filled in for
  `dr_grpo`, and `objective_overrides` (any `PolicyObjective` field) is
  applied last.
- The preset's advantage estimator is used for the group advantages, so
  `rloo`, `dr_grpo`, and `reinforce_pp_baseline` change more than the
  surrogate.
- Trainers whose rollouts store sequence-level log-probs (GSPO, GSPO-token,
  GEPO, and the GRPO trainers) accept only sequence-level presets
  (`gspo`, `gspo_token`, and `gepo` for GEPO). Trainers with per-token
  log-probs (DAPO, VAPO, PPO) accept everything except `gepo`. An
  incompatible choice raises at construction and names the compatible presets.
- VAPO and PPO keep their external GAE advantages and separate value losses;
  the preset only changes the policy surrogate.
- An unknown name warns in `TrainingConfig.validate()` and raises at trainer
  construction and on the CLI, listing the valid presets.

## The GRPO trainers: per-token path and sequence fallback

`MultiTurnGRPOTrainer` and `SingleTurnGRPOTrainer` ask the agent for a
full assistant turn (`MultiTurnAgent.generate_turn`) whose metadata carries
the exact prompt token ids, the sampled response ids, and the model's
log-probs of those ids. When every trajectory in a group has that data, the
loss takes the **token path**: one row per assistant turn, one padded
forward pass per group in chunks of `generation_batch_size` rows, per-token
log-probs from the stored ids (no re-tokenisation drift), advantages from
the objective's estimator broadcast to each trajectory's rows, and
`policy_loss` with `logp_old=None` (the trainers take one optimizer step per
rollout batch, so this is exactly on-policy and equals the TRL convention).
The native objective on this path is the `grpo` preset with `clip_ratio` as
the symmetric trust region; `objective` selects any preset, token-level ones
included. With `beta > 0` and a reference model the enhanced path adds
`k3_token` KL from one no-grad reference forward on the same batch;
`entropy_coef` uses the differentiable entropy of the same logits.

`num_gradient_updates` (default 1) turns one rollout batch into that many
optimizer steps: the old policy's per-token log-probs are frozen once
(`compute_token_old_logprobs`, a no-grad forward on the stored ids) and each
update re-evaluates the clipped objective against them, so ratios move away
from 1 and the trust region engages exactly as in PPO and DAPO. Each inner
update is a full optimizer step; gradient accumulation applies to the
single-update default only. Groups on the sequence fallback ignore the
setting (a warning is logged once).

### Memory: one chunk's graph at a time

Inside `MultiTurnGRPOTrainer.training_step` the token path backpropagates
each chunk of `generation_batch_size` rows as soon as its loss share is
computed, instead of keeping every chunk of every group alive for one final
`backward`. Each share is the chunk's part of the full loss: means use the
whole group's row count (`seq_mean`, `seq_sum_const`, `k3_sequence` KL) or
response-token count (`token_mean`, entropy) as the denominator
(`policy_loss(total_rows=, total_tokens=)`), and the result is divided by the
number of groups, so the summed gradient equals the single-backward
gradient for every preset (`group_expectation` couples rows and is rejected
on this path). Log-probs are computed only at response positions as
`logit[y] - logsumexp(logits)` in fp32 (one `log_softmax` when
`entropy_coef > 0`), so the largest per-chunk tensor is
`[response tokens, vocab]` rather than a full-width fp32 log-softmax. At
~600-token rows and a 152k vocabulary the old layout kept roughly
0.4-0.55 GB per row alive until the end of the step; peak activation memory
is now bounded by one chunk. Lower `generation_batch_size` to trade speed for
memory. The loss dict reports `backward_done=True` and a detached
`total_loss`. Calling `compute_grpo_loss` directly (outside `training_step`)
still returns a differentiable loss with all chunk graphs kept.

### Driving the trainer by hand

Loops that call `generate_trajectories` and `training_step` themselves
(instead of `train()`) get the configured warmup and decay schedule from
`initialize()`, sized from `num_episodes` exactly as `train()` sizes it;
call `trainer._setup_scheduler(n)` after `initialize()` for a different
budget. `TrainingConfig` defaults to watch in such loops:

- `gradient_accumulation_steps` defaults to **4**: only every fourth
  `training_step` takes an optimizer step. Set it to 1 for one update per
  rollout batch.
- `entropy_coef` defaults to **0.01**: the token path then adds an entropy
  bonus (and computes a full-vocabulary softmax at response positions).
  Set it to 0.0 for the plain objective.
- `gradient_checkpointing=True` is applied to the policy in `initialize()`
  (non-reentrant; PEFT policies also get `enable_input_require_grads`).
- `bf16=True` uses autocast without a `GradScaler`; only `fp16` is scaled.
- Fields the trainer cannot act on (model, LoRA, sampling, vLLM settings,
  which belong to `AgentConfig`, and other trainers' fields such as
  `num_epochs`) are reported once in a warning when they differ from their
  defaults.

Groups with any trajectory lacking token metadata (text-only agents, stub
backends, legacy transcripts) take the **sequence fallback**: per-trajectory
forward passes, sequence-level ratio clipped at `seq_clip_ratio`, and the
REINFORCE branch when no old log-probs exist. The loss dict reports `path`
(`token` or `sequence`), `objective`, and `num_rows`.

## Rollouts are sampled in inference mode

Every `MultiTurnAgent` generation runs inside
`stateset_agents.core.generation_mode.inference_mode`: the model is put in
eval mode with the KV cache enabled for the duration of `generate`, and the
trainer's `train()` state is restored afterwards. Sampling in train mode with
LoRA dropout and gradient checkpointing active produced garbage rollouts on a
real 1.5B model (and therefore zero reward), which is how the earlier
native-GSPO comparison rows went wrong; the guard makes that impossible for
any trainer that samples through the agent.

## Rollout text is exactly what the policy sampled

An assistant turn's `content` (what the conversation history, the reward and
the next prompt see) equals `tokenizer.decode(token_ids,
skip_special_tokens=True)` for the `token_ids` the token path trains on:

- Generation ends at EOS or `max_new_tokens`. Stop strings are explicit
  (`AgentConfig.stop_sequences`, default empty); when one fires, text and ids
  are cut at the same token boundary, and a merged token that straddles the
  stop string is dropped whole (it cannot be split). The criterion also
  matches stop strings inside merged tokens, so generation still ends there.
  `LEGACY_STOP_SEQUENCES` (`"User:"`, `"System:"`, `"\n\n"`) restores the
  old stop set for no-chat-template models that would otherwise continue
  into an invented next turn.
- No whitespace stripping or speaker-label cutting is applied to sampled
  text; rollout-engine results get the same stop handling.
- `AgentConfig.enable_persona_hints` (canned billing/technical sentences,
  default off) only affects `generate_response`; `generate_turn` and
  `generate_turns` ignore it and log a warning, so injected text can never
  reach a training row or a reward.
- `top_k=0` / `top_k=None` disables top-k (both map to HF `top_k=0` and to
  `top_k=-1` for a rollout engine), so the sampling distribution is the one
  the config names. `top_p` and `repetition_penalty` still apply when set.
- With `enable_reasoning`, the `<think>` block is removed from `content` and
  kept in `metadata["reasoning"]`; the ids still cover it.

### Rollouts should be sampled from the policy itself

Policy-gradient objectives assume each reply was drawn from the policy's own
distribution. `AgentConfig` defaults (`top_p=0.9`, `top_k=50`,
`repetition_penalty=1.1`) suit serving but transform that distribution, which
biases the gradient. `MultiTurnGRPOTrainer.initialize()` logs a warning naming
every such setting; for unbiased training set `temperature=1.0`, `top_p=1.0`,
`top_k=0`, `repetition_penalty=1.0`, and `do_sample=True`.

## Group sampling in one call

`MultiTurnAgent.generate_turns(messages, n)` samples the whole group of a
group-relative objective in one batched generate (HF `num_return_sequences`,
or a batch of `n` prompts to the attached engine); each turn carries the same
exact token ids and sampler log-probs as `generate_turn`. Native GSPO uses it
for its HF path and, on a fresh rollout batch, takes its old-policy log-probs
from the current forward pass detached (`rescore_old_log_probs`, default),
so the sequence ratio is exactly 1 and the narrow GSPO band gates only
genuine drift; the gap between the sampler's log-probs and that pass is
reported as `generation_log_prob_gap`. Both GSPO and GSPO-token honor this setting
and store the diagnostic in step metrics and training history. GSPO-token's
token log-probabilities retain gradients; only the old-policy scores and its
sequence ratio are detached.

Set `rescore_old_log_probs=False` when using the generator's original scores
for importance correction, such as rollouts from a lagging policy. In this mode
the score-gap diagnostic is zero because the selected old-policy scores are
the generator's own scores; it does not mean the generator matches the current
model. Same-pass rescoring assumes fresh rollouts and cannot verify that an
external engine is synchronized. GSPO-token previously ignored the setting;
its default now follows GSPO, while `False` retains generator-score behavior.

## Rollouts from an inference engine

`MultiTurnAgent.set_rollout_backend(engine)` routes generation through any
engine exposing `async generate_with_logprobs(prompts, **kwargs)` that
returns objects shaped like `training.vllm_backend.GenerationResult`
(`training.vllm_backend.VLLMGenerator` does). The turns it returns carry the
engine's exact prompt and response token ids and per-token log-probs, so the
GRPO trainers' per-token path trains on what the engine generated while
rollouts stop being bottlenecked on sequential Hugging Face `generate`.
Engine failures fall back to native generation and are recorded on the turn
as `rollout_backend_error`.

**Staying on-policy.** After every optimizer step both GRPO trainers call
`agent.sync_rollout_backend()` (`TrainingConfig.rollout_sync`, default on),
which pushes the policy's current weights into the engine through its
`sync_weights(model)` hook. `VLLMGenerator.sync_weights` merges a PEFT
adapter for the duration of the read, strips wrapper prefixes, and streams
`(name, tensor)` pairs into the in-process engine's `load_weights`; an
unfamiliar engine layout is adapted by setting `VLLMGenerator.weight_loader`
(the engine model is found by a breadth-first search over the attribute names
vLLM has used across V0/V1; `VLLMConfig.in_process_weight_sync` keeps the V1
engine core in this process so there is a model object to reach).
`benchmarks/vllm_rollout_check.py` is the live proof: it samples through the
engine, takes real GRPO steps, and records the engine-vs-policy log-prob gap
before the step, after it, and after the sync, with versions and hardware.
Retained on vLLM 0.28 / NVIDIA A40 / Qwen2.5-0.5B-Instruct
(`benchmark_results/vllm_rollout_check/`): full weights 0.17 → 43.0 → 0.29
nats (max over the sampled tokens; the large middle number is three Adam
steps at lr 5e-4 on all weights), LoRA r=8 0.17 → 3.88 → 0.10 nats, one
sync each, and the first LoRA attempt caught a real bug (PEFT's
`.base_layer` names) that the stale-reporting path surfaced instead of
hiding.
Every turn records `rollout_backend_version` (successful syncs so far) and
`rollout_backend_stale`. A backend without `sync_weights`, or a sync that
raises, is reported once as stale rather than silently sampling from an old
policy; `agent.rollout_backend_error` keeps the reason and the trainers expose
the last outcome as `last_rollout_sync`.

**Correcting for a stale engine.** `TrainingConfig.old_logprobs_source`
chooses the old policy on the token path. `"recompute"` (default) treats
each rollout batch as exactly on-policy. `"sampler"` uses the log-probs the
engine reported at sampling time as `logp_old`, so the importance ratio
`exp(logp_cur - logp_sampler)` corrects for an engine that lags the policy or
differs numerically (TRL's vLLM importance-sampling correction); the loss
dict reports `old_logprobs_source` (`recompute`, `sampler`, or `snapshot`
when `num_gradient_updates` froze its own old log-probs, which takes
precedence). Missing requested sampler log-probs raise by default. Explicit
`allow_missing_sampler_logprobs=True` permits recomputation and reports
`"recompute_fallback"` in the loss metadata.
Use `"sampler"` with `rollout_sync=False` for a deliberately asynchronous
engine, and note that the engine's log-probs must be of the raw model
distribution (temperature 1, no top-p) to match the training forward pass.

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

## Preference optimization (offline)

The policy objectives above are on-policy: they score sampled rollouts.
Offline preference optimization trains directly on chosen/rejected (or
desirable/undesirable) responses with no sampling: DPO, conservative DPO,
IPO, SimPO, KTO and ORPO in `stateset_agents.training.preference`
(`PreferenceTrainer`, `train_preference`, `stateset-agents preference`). The
same package trains Bradley-Terry reward models whose
`PairwiseRewardFunction` can supply the reward for the trainers above. Those
losses use response-token log-probs gathered by the same
`rl_losses.gather_token_logprobs` and are pinned to TRL 1.12. See
[PREFERENCE_TRAINING.md](PREFERENCE_TRAINING.md).

## PPO reference update limits

Canonical `PPOTrainer` supports experimental token-exact online training through
`train()` and the `train_ppo()` convenience loader. The online protocol samples
real categorical actions and retains their token IDs, behavior log probabilities,
critic values, terminal rewards and EOS/horizon boundaries. It never decodes and
re-tokenizes actions for training. Sampling, current-policy scoring and reference
KL use the same positive temperature. Online sampling requires `top_p=1`,
`top_k=0` and `gradient_accumulation_steps=1`, now the PPO-specific defaults.

Each update cycles through `mini_batch_size` prompts and samples
`num_generations` completions per prompt. `num_episodes` passed to `train()` is a
total update target, including completed updates. Without that argument the
target is `config.num_episodes * config.num_epochs`. A complete runtime checkpoint
restores both optimizers, the critic, schedulers, update cursor and RNGs. The
ordered prompt population, actual prompt token IDs and EOS setting are bound in
the checkpoint configuration; changed input populations reject. For a fresh
trainer, set `resume_from_checkpoint` and call `train()` with the same prompts.
The convenience loader pins both model and tokenizer to `model_revision`, seeds
initialization/sampling from `seed`, and saves a final checkpoint. With an
already-created model, seed initialization before constructing the trainer;
online collection consumes the active PyTorch RNG stream.

`output_dir/ppo-rollouts/run-*/` retains one raw rollout and matching commit
receipt per successful update. Receipts bind the full behavior learner state
and exact sampled records. Reward callbacks may be synchronous or asynchronous;
they receive the original prompt and decoded completion. Reentrant training,
saving and loading reject while a batch is active. Callback state mutations
quarantine the learner, and unchanged-learner collection failures restore RNGs
for retry. External reward-service state is not checkpointed; callers must keep
the reward definition and external dependencies stable for exact continuation.

This initial online implementation uses complete-prefix scoring and full learner
fingerprints. It targets correctness, with CPU tiny-transformer and synthetic
learning contracts; large-model throughput, GPU memory, distributed execution,
quantized training and independent quality gains remain unverified. EOS and the
configured token horizon both end an episode, with no truncation bootstrap.

`train_step()` remains available for externally supplied text batches. It
requires equal nonempty populations, finite numeric rewards, a positive inner
epoch count, prompts and completions within their configured limits and at least
one response action per row. The separately tokenized prompt must be an exact
prefix of the combined text's tokens; ambiguous BPE joins and special-token
boundaries reject before model scoring. Completion overflow rejects rather than
applying a full-response reward to truncated text. It rejects nonfinite
advantages, returns, losses and gradients
before the affected inner update can step either optimizer. A positive KL
penalty requires a reference model. Adaptive coefficients commit only after both
optimizers succeed, so rejected updates preserve the controller. Old/current
policy, critic and reference scoring use evaluation mode with gradients enabled
for current scores; dropout cannot manufacture an initial importance ratio.
Every submodule's original mode is restored on success or failure.

These guards do not roll back earlier inner updates or make failures between
optimizer commits atomic. Any failure after an optimizer starts quarantines
training and checkpoint publication until a complete runtime checkpoint reload.
The external text interface still does not establish exact sampled token identity
or behavior-policy provenance; use the online path to retain those values.
The shared declarative PPO objective and its numeric contracts remain available.

## Sampled workflow and representative studies

`python -m stateset_agents.training.sampled_workflow --device cuda` runs three
seeds through real sampling, tool execution, native GRPO updates and checkpoint
reload. Retained CPU/CUDA results and provider accounting are in
`benchmark_results/training_contracts/20261014/`. These synthetic mechanics
checks passed; they do not establish representative domain quality.

`python -m stateset_agents.training.native_refund_study --help` describes the
pretrained-model study runner. It requires reviewed, disjoint train/validation/test
cases, pins model and implementation identity, selects checkpoints on validation,
and retains per-case test traces. Its six-seed default supports the quality gate.
The retained one-seed synthetic pretrained diagnostic scored 0/8 before and after
training and failed that gate. A reviewed dataset is still required for the
representative study.

## Large-model text SFT source identity

Muse Spark and frontier SFT capture each dataset split once. Strict chat validation,
overlap checks, file SHA-256 and normalized conversation SHA-256 all use those
captured bytes; tokenizer/model initialization cannot change the examples by
replacing a source path. Training and preview receipts include both raw file
identity (`dataset_sha256`) and normalized content identity
(`training_content_sha256`). Row IDs and whitespace affect the raw file hash,
while IDs do not create independent conversations for overlap checks.

Both paths check the Torch 2.7 load-time FSDP minimum before importing training
backends. The GLM FP8 contract loads locally serialized float8 attention and
per-expert checkpoint tensors, verifies exact BF16 dequantization and native
expert merging, then checks adapter backpropagation, optimizer updates, frozen
weights and export/reload. This is CPU mechanics evidence with generated weights;
released checkpoint and GPU/distributed conformance remain unverified.

## Distributed SFT input agreement

Muse and frontier SFT verify `stateset-sft-consensus-v1` after training arguments
initialize the process group and before constructing `SFTTrainer` or loading
model weights. Every rank must agree on the canonical model/revision, training
settings, captured dataset identities/counts, native tokenizer backend and chat
template, ordered encoded inputs/labels, runtime versions, and critical StateSet
implementation hashes. Local source/output paths are excluded so nodes can stage
identical files under different mount points. Raw file identities must match;
this is not a path for intentionally distinct per-rank input shards.

The guard uses a fixed three-int header and fixed-size SHA-256 tensors, with no
application text or serialized Python objects in the collective. Invalid identity
construction, missing fields, nonfinite metadata, protocol/world-size mismatches
and differing components stop participating ranks before trainer construction.
The adapter receipt's `rank_consensus` records the agreed identity/component
hashes, protocol, world size and backend. An initialized Gloo or NCCL group is
required; there is no local-success fallback.

Seventeen scenarios run through two actual local CPU/Gloo ranks, including
agreement after rejected calls. The same contract passes against the installed
wheel; [retained evidence](../benchmark_results/training_contracts/20261015/sft-rank-consensus/summary.json)
binds the wheel to all 398 packaged source files. This verifies consensus and rejection mechanics;
NCCL/GPU execution, FSDP/EP adapter synchronization, full-size weight loading and
multi-node training remain separate unverified gates. Local validation and
launcher/process failures outside the collective still use the launcher's normal
failure handling and process-group timeouts.
