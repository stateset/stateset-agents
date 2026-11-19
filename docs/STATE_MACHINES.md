# State machines and lifecycle ownership

Source audit: 2026-11-16. This inventory describes the current working tree,
including the in-progress private improvement workflow. Paths below are the
implementation owners, not alternate specifications of their behavior.

A state machine exists whenever retained information changes which operation
may happen next. That includes enum values, strings, flags, counters, registry
membership, task handles, leases, and the presence of files or receipts. An enum
alone does not make transitions explicit: most named states here still have
their edges and guards distributed across methods.

The audit covered the packaged Python source, the JavaScript client, the Rust
agent service and native extension, and their tests. Compatibility shims are
aliases of their canonical owners. Generated protobuf messages, immutable
configuration, pure numerical kernels, examples, archived documentation, and
benchmark outputs are not separate runtime lifecycle owners. User-supplied
tools, environments, callbacks, and provider internals can introduce additional
machines; a static source audit cannot prove those external graphs exhaustive.

## Decision

Make ownership and recovery explicit at effect boundaries, one owner at a time.
Keep existing wire values, persistence formats, return types, and lifecycle
owners. A common enum would erase meaningful differences between `completed`,
`succeeded`, cancellation requested, execution unconfirmed, and cleanup pending.

For every change, require all three benefits:

| Audience | Required benefit |
| --- | --- |
| User experience (UX) | Honest progress and completion, no duplicate effects, actionable recovery. |
| Developer experience (DX) | One owner of transitions, inspectable state, documented guards, deterministic failure-path tests. |
| Agent experience (AX) | Machine-readable state and retained handles; retries must not guess whether an earlier effect committed. |

The first implementation applies this decision to remote chat (R05). Repeated
`start()` previously overwrote the owned pod. `close()` previously cleared its
handle and exit hook even when termination failed. The explicit lifecycle now
rejects duplicate provisioning, preserves pending cleanup, and publishes local
state without changing the successful `start` / `ask` / `close` interface.

The next implementation applies it to async pools and pooled resources (C10/C11).
Checkout, validation, return, and retirement now share an ownership boundary;
cleanup is tracked separately from availability. The global HTTP factory owns
its shared connector, so retiring one session cannot close a sibling session.

The API circuit breaker (A07) now admits one recovery probe at a time and records
outcomes only in the generation that admitted them. Fallbacks run outside the
state lock, and cancelled probes release admission without asserting recovery.

Legacy GRPO jobs (A03) now commit transitions under the state manager's lock.
The first terminal outcome preserves its timestamps, progress, results, and
error; workers retain their job identity, and only the winning transition can
report cancellation or completion metrics.

GRPO shutdown (A09) now closes admission and joins actual worker ownership before
releasing engines. Concurrent shutdown callers share one result, cancellation
cannot detach cleanup, and failed cleanup retains its resource handle. One
monotonic deadline covers every phase and prevents later callbacks after expiry
while retaining ownership of the active callback until it unwinds.

State snapshots (C12) now detach nested values when capturing, restoring, and
returning history. Copy failures are explicit and cannot publish a partial
snapshot or clear live state before restoration is prepared.

Async task management (C13) now closes admission before draining, retains each
concurrency slot through cancellation cleanup, and shares one shutdown sequence.
Batch failure joins unfinished siblings; queued coroutine objects cannot be
silently abandoned when submission is rejected or cancelled.

State-service shutdown and global resource aggregation now retain one cleanup
sequence and its outcome. State maintenance/probes finish before cache cleanup;
managed tasks finish before their HTTP dependencies close. Failed cleanup keeps
its owner and prevents replacement or automatic replay.

The core circuit breaker (A07) now observes awaited outcomes, shares one recovery
slot across synchronous and asynchronous callers, and fences late completions.
Core retry (A08) snapshots each invocation's policy, respects an explicitly empty
exception list, and propagates cancellation without repeating the operation.
Existing core/API counter policies remain distinct.

API bulkheads (A08) now reserve idle capacity independently of the waiting queue,
retire cancelled/expired waiters, and retain active capacity through cancellation
cleanup. Timeouts distinguish their own deadlines from upstream errors and join
owned work before reporting expiry or caller cancellation.

API retry (A08) now validates and snapshots each invocation's policy, propagates
cancellation/process-control exceptions, and preserves its exclusion precedence
and upward jitter. API and core retry share bounded backoff arithmetic while
retaining their distinct policies. Intermediate overflow/underflow cannot force
an incorrect cap, zero delay, or unbounded integer calculation.

API health checks (A13) now fence observations by registration identity, share
in-flight probes, and return detached measurements. Request deadlines retire
observers without losing ownership of probe cleanup or synchronous workers.
Application shutdown drains those probes before their dependencies close.

Advanced monitoring (A13) now owns startup, collection, alerting, and retention
workers through shutdown. Concurrent callers share startup/cleanup, partial
startup rolls back, and unexpected worker termination remains inspectable
instead of being silently restarted. Health summaries include this lifecycle.

Advanced tracing (A14) now retains SDK span and exporter handles. Each span end
is attempted once, explicit parents reach the SDK, and pipeline cleanup is owned
by the tracer. Its private provider neither replaces nor shuts down the process
global provider. Monitoring drains workers before releasing tracing resources.

Utility monitoring (A13) now retains its automatic worker and listener handles.
Synchronous and asynchronous close share one cleanup outcome; global replacement
joins the previous owner before starting another listener. Metrics bind their
Prometheus labels, and local summaries copy labels under the collector lock.
GRPO shutdown closes monitoring after engine cleanup and rolls it back when
later startup fails.

Utility health checks (A13) now share an owned probe among simultaneous observers.
Deadlines and cancellation retire observations without abandoning sync calls or
async cleanup. Checks, checker registries, and monitoring services expose probe
cleanup state and drain it before dependencies close. Unconfigured database/cache
checks and unavailable system measurements no longer report fabricated health.

Continuous evaluation (A13) now retains one selected agent's worker and drains
it before replacement. The evaluator selects the actual agent by result position,
serializes foreground batches, and pauses background sampling while measuring.
Close drains foreground work and continuous monitoring. Cancellation, failed
cleanup, and an explicit restart after a successful stop are separate states.

Evaluation evidence (A13) now moves from `unavailable` to explicitly `measured`
or `inferred`, and back to `unavailable` when its value changes or evidence is
withdrawn. Numeric defaults remain compatible; rankings, monitoring selection,
and version 2 reports require evidence. Missing scores cannot establish safety,
recovery, or a winner, and response-time variation cannot establish a quality
confidence interval.

Automated rubric assessment (A13) now distinguishes assessed pass/fail outcomes
from unsupported criteria, abstentions, and execution/checker errors. Suites
capture their rubric registrations before agent calls, and comparison evidence
includes the applied checker methods. Cancellation retires the suite before
callback cleanup, preventing later attempts even if a callback suppresses
cancellation or raises during cleanup.

Backend initialization (C01) now records admission, phase, and failure. Repeated
initialization preserves stub progress and reuses the existing adapter. An
adapter call that fails after starting cannot replay on that same model, and
failed/in-progress initialization cannot enter native, streaming, or rollout
generation. Explicit retries can reuse completed loading steps when the failure
occurred before an uncertain adapter effect.

Enhanced memory now participates in agent readiness (C01/C14). A failed or
pending memory setup cannot serve responses, and completed custom setup is not
repeated. Vector memory publishes a prepared encoder and a fully reindexed entry
set together. Missing optional embeddings use deterministic lexical retrieval;
the backend is observable, and retrieved references reach enhanced response
prompts. Memory and reasoning disable flags now suppress their actual effects.

Enhanced generation (C14) now shares preparation across response, turn, batch,
and streaming calls. Completion records distinguish produced output from memory
write acknowledgement. Partial streams are not remembered; batch alternatives
require explicit selection, with one owned write per retained batch.

Enhanced reasoning (C14) now separates provider declarations from scoped checker
assessments. Default planning templates have unavailable confidence. Actual
proposals can come from a supplied async provider, including a separate model;
only a passed assessment at the captured threshold permits early stopping.
Completion, budget exhaustion, failure, and cancellation remain distinct facts.

Named conversations (C04) now own one turn per ID through callback cleanup and
publish prepared context only after revision checks. Explicit portable snapshots
restore settled history and turn receipts without replaying model/tool work.
The GRPO gateway records its first assistant answer once and reports turn
conflicts as HTTP 409.

Built-in long-term planning (E02) now stages changes and fences publication by
revision. Named turns consume one-shot planning commands before invoking the
planner, so later response failure cannot advance the plan again. Version 2
conversation snapshots restore settled planning state alongside the transcript.

Canonical API conversations (A05) now own each request through provider cleanup.
Overlapping requests receive HTTP 409, and deleted, expired, evicted, or replaced
records cannot accept late responses. Authorized status endpoints expose request
outcomes independently of whether a response was recorded.

The v1 hosted-inference gateway can now opt into a versioned SQLite journal.
History, principals, and request receipts survive restart. A cross-process
request lock distinguishes live ownership from abandoned work; uncertain
interrupted generation requires explicit acknowledgement before another turn.

Persistence (A12) now exposes explicit task-owned transactions across its four
repositories. Repeated connection preserves data; detached records prevent
edits from bypassing publication. Memory rollback and SQLite commit/rollback
share observable outcomes, with uncertain commit acknowledgements kept separate
from successful publication.

## Inventory

`I` means implicit states encoded in control flow or data. `H` means named states
with transitions distributed through the implementation. `E` means a locally
explicit, guarded lifecycle. Arrows describe the principal flow, not permission
to bypass the implementation's validation. Terminal outcomes and invariants
belong to the indicated owner.

### Agents, environments, and resources

| ID / representation | Owner | States, events, and important boundaries |
| --- | --- | --- |
| C01 · E/I | [Agent initialization](../stateset_agents/core/agent.py), [initialization state](../stateset_agents/core/agent_initialization.py), [checkpoint factories](../stateset_agents/core/agent_factories.py), [checkpoint restoration](../stateset_agents/core/agent_checkpoint.py), [backends](../stateset_agents/core/agent_backends.py), [enhanced agent](../stateset_agents/core/enhanced/enhanced_agent.py), [lazy presets](../stateset_agents/core/environment_presets.py) | Backend initialization moves `new` → `initializing` → `ready`/`failed`; phase and component identities distinguish loading from adapter attachment. Completed adapter calls are not repeated; uncertain attachment requires a fresh model. Agent snapshots stage and publish an explicit artifact kind; deferred restoration validates the captured publication before loading saved components. Enhanced memory setup joins readiness through an async extension hook; cancellation drains that hook before failure is final. Lazy preset materialization remains a separate owner. Optional dependencies and stub backends retain different initialization paths. |
| C02 · E/I | [Conversation ownership](../stateset_agents/core/conversation.py), [Generation mode](../stateset_agents/core/generation_mode.py), [agent streaming](../stateset_agents/core/agent.py) | Training mode → temporary inference → restored mode; streaming accumulates partial text, holds possible stop-marker suffixes, then finishes, errors, or is closed. Generator cleanup owns the model lock. Conversation reset advances an epoch; late results, chunks, and selections are rejected while already-admitted work retains cleanup ownership. |
| C03 · I | [Rollout synchronization](../stateset_agents/core/agent.py), [vLLM backends](../stateset_agents/training/vllm_backend.py), [GSPO generation](../stateset_agents/training/gspo_generation.py) | Detached/attached backend; uninitialized/initialized; current weights → stale → synchronized at a new publication version. Successful generation alone does not establish weight freshness. |
| C04 · E/I | [Multi-turn agent](../stateset_agents/core/multiturn_agent.py), [context](../stateset_agents/core/multiturn_context.py), [turn ownership](../stateset_agents/core/multiturn_lifecycle.py), [snapshots](../stateset_agents/core/conversation_snapshot.py), [memory](../stateset_agents/core/memory.py) | Named conversation absent → active → ended; one turn per ID owns preparing → analyzing/generating/tools → committing → completed/failed/cancelled. Accepted user input survives later failure; candidate changes publish only after ownership/revision checks. Explicit snapshots capture settled context and receipts without callback replay. Registry membership and metadata still carry goal, strategy, and sales-stage state; sales stage is caller supplied. |
| C05 · I | [ToolAgent](../stateset_agents/core/tool_agent.py), [function calling](../stateset_agents/core/function_calling.py), [structured output](../stateset_agents/core/structured_output.py) | Generate → parse/validate → execute tool → append result → regenerate or finish; bounded tool/retry counts and parse errors determine the next action. Tool execution and generated intent are distinct events. Complete JSON decoding preserves nested arguments; reset invalidates later rounds and parallel handlers settle before ownership is released. |
| C06 · H | [Episode state and driver](../stateset_agents/core/environment_base.py), [conversation](../stateset_agents/core/conversation_environment.py), [task](../stateset_agents/core/task_environment.py), [Gym](../stateset_agents/core/gym/adapter.py) | `ongoing` → `completed`, `failed`, or `timeout`; reset creates another episode. The driver accepts legacy and standard step tuple layouts, and examines both `done` and status. |
| C07 · H/I | [Conversation simulator](../stateset_agents/environments/conversation_simulator.py), [symbolic physics](../stateset_agents/environments/symbolic_physics.py), [multimodal environment](../stateset_agents/core/environments/multimodal_environment.py), [sampled workflow](../stateset_agents/training/sampled_workflow.py) | Turn/step budgets, user-simulator history, task completion, physics actions, and calibration flags refine the episode lifecycle. These are separate domain transition systems, not just another status enum. |
| C08 · H/I | [Refund environment](../stateset_agents/core/environments/refund_environment.py), [refund policy](../stateset_agents/core/environments/refund_policy_environment.py), [industry support](../stateset_agents/core/environments/industry_support_environment.py), [operations policies](../stateset_agents/core/environments/industry_operations_policies.py), [admin policies](../stateset_agents/core/environments/industry_admin_policies.py) | Lookup/observation → validated domain action → simulated order, inventory, subscription, or account mutation → finish/budget exhaustion. Monetary balances and trusted facts constrain legal actions. These environments simulate effects. |
| C09 · H/I | [Refund recovery](../stateset_agents/core/environments/refund_recovery_environment.py), [delegation](../stateset_agents/core/environments/refund_delegation_environment.py), [resolver](../stateset_agents/core/refund_tools.py) | Recovery phases include `ready`, `unknown`, `retryable`, `applied`, `unauthorized`, and `handoff`. Resolver flags distinguish no submission, uncertain submission, and confirmed result. Lost acknowledgments require reconciliation, not a new payment attempt. |
| C10 · E | [Async pool](../stateset_agents/core/async_pool.py) `AsyncResourcePool` | `initializing` → `ready` → `draining` → `closed`; failed initialization drains partial ownership. Concurrent initializers share one setup; drain rejects waiting/new borrowers; close waits for borrowed resources and pending cleanup. `state` exposes the lifecycle. `scaling` remains declared for compatibility but scaling leaves a ready pool ready. Global HTTP callers await initialization; a closed global pool is replaced only after confirmed cleanup. |
| C11 · E/H | [Pooled resources](../stateset_agents/core/async_pool.py) `PooledResource`, `HTTPSessionFactory` | `idle` → `active` during validation/use → `idle` or retirement (`failed`/`stale`). Pool membership, borrowed ownership, availability, and connection counts change under one condition lock. Retirement detaches before cleanup and consumes capacity until cleanup finishes. The global HTTP factory shares a connector across sessions and closes it only when its last owned session retires. Explicitly supplied connectors in the public factory retain the caller's configured ownership policy. |
| C12 · E/I | [State manager and service](../stateset_agents/core/enhanced_state_management.py), [state cache](../stateset_agents/core/enhanced_state_cache.py) | Local mutation → dirty keys → cache synchronization; snapshot copy → publish/version advance → copy before restore. Captured and inspected history is detached from live nested values. Service `new` → `running` → `closing` → `closed`/`failed`; shutdown joins maintenance/probes before final synchronization and local cache clearing. Watchers and conversations retain implicit registration/update/removal lifecycles. Cache entries are present, expired, invalidated, or evicted. |
| C13 · E/I | [Owned async calls](../stateset_agents/utils/async_calls.py), [computational engine](../stateset_agents/core/computational_engine.py), [task manager](../stateset_agents/core/async_pool.py) `AsyncTaskManager` | Task manager `new` → `running` → `draining` → `closed`; queued → admitted → execution/unwind → slot release. Condition-based admission closes before draining, and each admitted operation retains its slot until cleanup finishes. Timeout, caller cancellation, and shutdown converge on one cancellation request. Owned calls drain work before propagating cancellation; async generators have a separate next-item/close handshake. Engine semaphore permits gate trajectory generation. Global resource aggregation remains a separate ownership boundary. |
| C14 · E/I | [Enhanced memory](../stateset_agents/core/enhanced/vector_memory.py), [enhanced agent](../stateset_agents/core/enhanced/enhanced_agent.py), [generation receipts](../stateset_agents/core/enhanced/generation.py), [reasoning assessment](../stateset_agents/core/enhanced/reasoning.py) | Memory `new` → `initializing` → `ready`/`failed`; backend changes prepare all embeddings before publication, and stale completions cannot mutate the index or access counts. Response/turn/batch/stream share preparation; generation `running` → `completed`/`failed`/`cancelled`/`closed` is separate from memory `pending`/`deferred` → `writing` → `stored`/`unconfirmed`. Partial streams skip memory; batches require explicit selection. Reasoning `preparing` → `generating` → `assessing` separates declared confidence from `passed`/`failed`/`unavailable` checks; reaching a step limit does not establish correctness. [Memory snapshots](../stateset_agents/core/enhanced/memory_snapshot.py) separate capture, revision checks, replacement, durability acknowledgement, and fenced restore. |

### Training, checkpoints, and distributed execution

| ID / representation | Owner | States, events, and important boundaries |
| --- | --- | --- |
| T01 · I | [Base trainer](../stateset_agents/training/base_trainer.py), [single-turn](../stateset_agents/training/single_turn_trainer.py), [multi-turn](../stateset_agents/training/multi_turn_trainer.py), [runtime](../stateset_agents/training/trainer_runtime.py), [callbacks](../stateset_agents/training/callbacks.py) | Setup → collect → reward/advantage → update → evaluate/checkpoint → finish, early stop, cancellation, or error. Episode progress, optimizer progress, and successful completion are different observations. |
| T02 · I | [GSPO](../stateset_agents/training/gspo_trainer.py), [token GSPO](../stateset_agents/training/gspo_token_trainer.py), [DAPO](../stateset_agents/training/dapo_trainer.py), [GEPO](../stateset_agents/training/gepo_trainer.py), [VAPO](../stateset_agents/training/vapo_trainer.py), [TRL GRPO](../stateset_agents/training/trl_grpo_trainer.py) | Algorithm-specific sampling/filtering/update cycles. VAPO additionally gates policy updates on value warmup. Third-party trainer completion follows its own callback/control contract. |
| T03 · I | [PPO](../stateset_agents/training/ppo_trainer.py), [online PPO](../stateset_agents/training/ppo_online.py), [offline GRPO](../stateset_agents/training/offline_grpo_trainer.py), [offline RL](../stateset_agents/training/offline_rl_algorithms.py), [BCQ](../stateset_agents/training/offline_rl_bcq.py), [BEAR](../stateset_agents/training/offline_rl_bear.py), [decision transformer](../stateset_agents/training/decision_transformer.py), [RLAIF](../stateset_agents/training/rlaif_trainer.py) | On-policy collection/update ownership and `_ppo_update_in_flight`; offline value pretraining, critic/actor/target updates, and sequence history. Resume must preserve warmup and optimizer phase, not just model weights. |
| T04 · I | [Accumulation](../stateset_agents/training/grpo_accumulation.py), [trainer utilities](../stateset_agents/training/trainer_utils.py), [single-turn state](../stateset_agents/training/single_turn_state.py) | Empty accumulation → pending gradients/microsteps → normalize/finalize → optimizer step or skipped update → clear. Loss banks, counts, scaler, scheduler, and partial-window checkpoints form one consistency boundary. |
| T05 · I | [Curriculum](../stateset_agents/core/curriculum_learning.py), [continual learning](../stateset_agents/training/continual_learning.py), [KL controllers](../stateset_agents/training/kl_controllers.py), [EMA](../stateset_agents/training/ema.py) | Stage/task progression, replay and consolidation, difficulty adjustment, and step-dependent KL/EMA schedules. EMA also stores original weights, temporarily applies averaged weights, then restores originals. Thresholds and counters select phases; they are not terminal job status. |
| T06 · I | [Zero-signal guard](../stateset_agents/training/callbacks.py), [early abort](../stateset_agents/training/auto_research/early_abort.py), [evaluation ladder](../stateset_agents/training/eval_ladder.py) | Observing → patience/budget reached → abort or escalate evaluation. Counters, best metrics, and stop latches are retained control state. Reset behavior belongs to each callback. |
| T07 · I | [Checkpoint publication](../stateset_agents/training/checkpoint_publication.py), [checkpoint I/O](../stateset_agents/training/checkpoint_io.py), [model checkpoint](../stateset_agents/training/model_checkpoint.py), [core checkpoint I/O](../stateset_agents/core/checkpoint_io.py) | Stage files → inventory/durability checks → publish under lock; interrupted replacement → recover backup/staging. File presence and manifests encode the state. A model file alone is not a committed training checkpoint. |
| T08 · I | [Runtime state](../stateset_agents/training/runtime_state.py), [checkpoint state](../stateset_agents/training/checkpoint_state.py), [learning state](../stateset_agents/training/learning_state.py), [multi-turn checkpointing](../stateset_agents/training/multi_turn_checkpointing.py), [single-turn checkpointing](../stateset_agents/training/single_turn_checkpointing.py) | Validate/prepare → commit model, optimizer, learning state, and RNG; failed commit → quarantined; complete successful reload clears quarantine. Training entry points must respect the poisoned-runtime latch. |
| T09 · I | [DDP checkpoint](../stateset_agents/training/ddp_checkpoint.py), [DDP state](../stateset_agents/training/ddp_checkpoint_state.py), [continual DDP](../stateset_agents/training/ddp_continual.py), [learner preflight](../stateset_agents/training/learner_preflight.py), [replica consensus](../stateset_agents/training/replica_consensus.py) | Rank-local preparation → collective agreement → commit/publication → synchronized replicas. Pending consensus and restore-failed flags gate use. One rank completing does not prove a distributed commit. |
| T10 · I | [Distributed trainer](../stateset_agents/training/distributed_trainer.py), [distributed mixin](../stateset_agents/training/distributed.py), [DDP control](../stateset_agents/training/ddp_control.py) | Process group setup, device/model wrapping, collectives, coordinated stop, and cleanup. `is_initialized`, rank membership, and native-rollout flags encode ownership. |
| T11 · I | [Async rollout coordinator](../stateset_agents/training/async_rollouts.py) | Open → closed/drain; rollout offered → queued → consumed, rejected as stale, or recognized as duplicate. Policy versions, seen IDs, capacities, and condition variables govern admission. Restore validates these together. |
| T12 · I | [Async rollout runtime](../stateset_agents/training/async_runtime.py) | Never run → producers/learner active → stop/drain → consumed runtime. `_has_run` prevents replay; worker failures, learner failure, and cancellation converge on producer cleanup. |
| T13 · I | [Distributed rollout control plane](../stateset_agents/training/distributed_rollouts.py), [router](../stateset_agents/api/routers/distributed_rollouts.py), [policy artifacts](../stateset_agents/training/policy_artifacts.py) | Worker unregistered → leased → renewed/expired/unregistered; policy staged → published → acknowledged by worker. Lease identity, artifact identity, admission counts, and checkpoint exclusion jointly fence stale work. |
| T14 · H/I | [Training models](../stateset_agents/training/advanced_training_models.py), [orchestrator](../stateset_agents/training/advanced_training_orchestrator.py), [journal](../stateset_agents/training/job_journal.py), [tracking](../stateset_agents/training/experiment_tracking.py) | `pending` → `queued` → `running` → `completed`/`failed`/`cancelled`/`timed_out`; recovery may classify work as `interrupted`. Retry attempts, resource reservations, scheduler shutdown, journal commit, and cache projections are separate submachines. `paused` is declared; do not infer a complete pause/resume implementation from that declaration. |
| T15 · H/I | [Auto-research loop](../stateset_agents/training/auto_research/experiment_loop.py), [tracker](../stateset_agents/training/auto_research/experiment_tracker.py), [checkpoint manager](../stateset_agents/training/auto_research/checkpoint_manager.py), [proposer](../stateset_agents/training/auto_research/proposer.py) | Propose → train/evaluate → `keep`/`discard`/`crash` → update best checkpoint/history → next trial or stop. Proposal exploration phase, best-model publication, and stop requests have independent state. |
| T16 · H/I | [HPO base](../stateset_agents/training/hpo/base.py), [Optuna](../stateset_agents/training/hpo/optuna_backend.py), [Ray](../stateset_agents/training/hpo/ray_tune_backend.py), [W&B](../stateset_agents/training/hpo/wandb_backend.py) | Study/sweep creation → trial execution → success/failure/pruning → selection. External scheduler state and local result lists must not be conflated. |
| T17 · I | [Improvement pipeline](../stateset_agents/training/improvement_pipeline.py), [flywheel](../stateset_agents/flywheel.py), [industry pipeline](../stateset_agents/training/industry.py), [harvest](../stateset_agents/training/harvest.py), [SFT](../stateset_agents/training/sft.py) | Baseline/harvest/curation → SFT → RL → evaluation → reload/serve; generation directories and stage receipts encode progress. Failed evaluation can coexist with valuable saved artifacts. |
| T18 · I | [Promotion](../stateset_agents/training/promotion.py), [checkpoint selection](../stateset_agents/evaluation/checkpoint_selection.py), [judge selection](../stateset_agents/training/judge_selection.py), [acceptance](../stateset_agents/evaluation/acceptance.py), [serving artifacts](../stateset_agents/training/serving_artifacts.py), [lineage](../stateset_agents/training/lineage.py), [study stages](../stateset_agents/evaluation/agent_study.py), [judge evidence](../stateset_agents/evaluation/judge_evidence.py) | Candidate → evaluated → accepted/rejected → artifact-bound promotion receipt → verification before serving. A study plan describes dependencies without executing them; later artifacts and judge receipts establish completion. Artifact identity, quality evidence, and authorization to promote are separate from training success. |
| T19 · I | [Sim-to-real](../stateset_agents/training/sim_to_real.py), [domain randomization](../stateset_agents/training/domain_randomization.py), [reward calibration](../stateset_agents/training/reward_calibration.py), [neural reward training](../stateset_agents/training/neural_reward_trainer.py), [Bayesian rewards](../stateset_agents/rewards/bayesian_reward_model.py) | Uncalibrated → observed/fitted/calibrated; reward/model posterior, normalizers, replay buffers, and adaptation schedules evolve between updates. They are learning state needed for reproducible continuation. |

### API and service infrastructure

| ID / representation | Owner | States, events, and important boundaries |
| --- | --- | --- |
| A01 · H/I | [Training service](../stateset_agents/api/services/training_service.py), [training router](../stateset_agents/api/routers/training.py), [v1 adapter](../stateset_agents/api/routers/v1.py) | Admission → running → `cancelling` → terminal `completed`/`failed`/`cancelled`. Task ownership, cancellation events, progress callback latches, and service `_closing` state refine the public string. |
| A02 · H/I | [Durable queue](../stateset_agents/api/training_queue.py), [worker](../stateset_agents/api/training_worker.py) | Idempotent submit → `queued` → claimed `running` → finish; queued cancel is terminal, running cancel is a request. Live lease tokens fence heartbeat/finish; expiration fails unknown execution instead of replaying it. SQLite/PostgreSQL transactions own admission and state changes. |
| A03 · E/H | [GRPO state](../stateset_agents/api/grpo/state.py), [handlers](../stateset_agents/api/grpo/handlers.py), [service](../stateset_agents/api/grpo/service.py) | Local jobs move `starting` → `running` → `completed`/`failed`/`cancelled`, with direct terminal transitions from `starting` retained. The first terminal transition wins under the TTL lock. Workers claim execution once and fence updates by job identity, stopping at iteration boundaries after cancellation, expiry, or replacement. Engine/service registries, conversation membership, and TTL eviction remain separate lifecycles. This is not the durable queue transition authority. |
| A04 · H/I | [Training Lab](../stateset_agents/api/routers/training_lab.py) | `created` → `running` ↔ `paused` → `completed`/`failed`; clone creates a fresh experiment. Background tasks, WebSocket subscribers, capacity eviction, and stop behavior are local to this optional surface. |
| A05 · E/I | [Gateway conversations](../stateset_agents/api/conversation_lifecycle.py), [durable v1 journal](../stateset_agents/api/conversation_store.py), [agent service](../stateset_agents/api/services/agent_service.py), [v1 routes](../stateset_agents/api/routers/v1.py), [inference service](../stateset_agents/api/services/inference_service.py), [dependencies](../stateset_agents/api/dependencies.py) | Conversation records are `active` → `ended`; one request per ID owns preparation → generation → publication through cleanup. Receipts distinguish completed/failed/cancelled requests, callback return, and recorded messages. Optional v1 persistence journals admission before dispatch, retains principals, and recovers abandoned work as `interrupted` without replay. Retirement and record identity fence stale work; access rules precede admission/status/recovery. Local agent registration, model/planner restoration, and standalone inference metrics remain separate lifecycles. |
| A06 · H/I | [API streaming](../stateset_agents/api/streaming.py), [messages](../stateset_agents/api/routers/messages.py), [OpenAI compatibility](../stateset_agents/api/routers/openai.py), [message utilities](../stateset_agents/api/messages_utils.py) | Request accepted → stream start/metadata → token or tool blocks → stop/done; error/disconnect cancels or drains work. Block indices, tool IDs, partial arguments, WebSocket connection membership, and finish reasons form protocol state. |
| A07 · E | [API circuit breaker](../stateset_agents/api/resilience.py), [core circuit breaker](../stateset_agents/core/error_handling.py) | `closed` → failure threshold → `open` → cooldown → `half_open` → recovery or reopen. Both owners admit one recovery probe at a time, fence older outcomes after transitions/reset, use monotonic cooldowns, and expose `probe_in_flight`. The core breaker observes synchronous and awaited outcomes under a shared thread lock; closed successes preserve its cumulative failure count. The API breaker remains separate with a consecutive-failure policy and configurable sequential recovery successes. |
| A08 · E | [Retry, timeout, bulkhead](../stateset_agents/api/resilience.py), [core retry handling](../stateset_agents/core/error_handling.py) | Attempt → success or retry delay → next attempt/exhausted. Both retry owners snapshot validated per-invocation policies, respect empty filters, and propagate cancellation/process-control exceptions; their classification and jitter remain distinct. API bulkheads atomically reserve capacity or queue FIFO; cancelled/expired waiters retire without effects, and active cancellation retains capacity until cleanup finishes. API timeouts distinguish upstream errors from deadline expiry and drain owned work before returning cancellation/timeout. |
| A09 · E/H | [API lifespan](../stateset_agents/api/main.py), [application ownership](../stateset_agents/api/application_lifecycle.py), [shutdown](../stateset_agents/api/shutdown.py), [GRPO lifespan](../stateset_agents/api/grpo/service.py) | Canonical app `new` → `starting` → `serving` → `stopping` → `closed`/`failed`; only confirmed close permits a fresh generation. HTTP/stream/WebSocket ownership drains before ordered dependency cleanup. Native replacements preserve application history and model identity; cleanup failure blocks reuse. Separate GRPO handler admission is permanently closed before draining; worker ownership survives job eviction. Its shared monotonic deadline bounds phase admission and callback waits; expiry waits for active cancellation cleanup and prevents later phases. |
| A10 · I | [Rate limiter](../stateset_agents/api/rate_limiter.py), [GRPO limiter](../stateset_agents/api/grpo/rate_limiter.py), [middleware](../stateset_agents/api/middleware.py), [security](../stateset_agents/api/security.py) | Tokens/window available → throttled → refill/expiry; auth failures → blocked → cooldown/reset. Redis availability, fallback, retry timing, and local failure tracking are separate operational state. |
| A11 · E/I | [API cache](../stateset_agents/api/cache.py), [distributed cache](../stateset_agents/api/distributed_cache.py), [callback and entry ownership](../stateset_agents/api/cache_lifecycle.py), [shared resource ownership](../stateset_agents/api/resource_lifecycle.py), [utility cache](../stateset_agents/utils/cache.py) | Global ownership moves `empty` → `initializing` → `ready` → `replacing`/`closing`; application leases prevent premature close. Redis clients move `new` → `connecting` → `ready`/`unavailable` → `retiring`/`closing` → `closed`/`failed`. Connections and probes share observations; commands retain one client through cancellation and all scan pages. Hybrid entries move waiting → admitted → callback/publication → release; prefix deletion/clear form barriers. Refills retain bounded Redis expiry; mutation observations distinguish layer acknowledgements. Multiple processes and memory/Redis writes are not one atomic owner. |
| A12 · E/I | [Persistence](../stateset_agents/api/persistence.py), [transaction ownership](../stateset_agents/api/persistence_transactions.py), [shared resource ownership](../stateset_agents/api/resource_lifecycle.py), [auth](../stateset_agents/api/auth.py), [GRPO auth](../stateset_agents/api/grpo/auth.py) | `UnitOfWork.transaction()` owns begin → working → commit or rollback across all repositories, fences other/escaped tasks, and retains acknowledgement uncertainty. Connect is idempotent; close rejects active transactions and drains one retained cleanup. Global initialization serializes and matching configuration preserves records; application leases retain the database through overlapping lifespans. Resource context management, ordinary autocommit, persisted fields, and token validity remain separate. |
| A13 · E/H/I | [API health](../stateset_agents/api/resilience.py), [monitoring](../stateset_agents/utils/monitoring.py), [alerts](../stateset_agents/utils/alerts.py), [advanced monitoring](../stateset_agents/core/advanced_monitoring.py), [continuous evaluation](../stateset_agents/core/enhanced/advanced_evaluation.py), [metric evidence](../stateset_agents/core/enhanced/advanced_evaluation_metrics.py), [rubrics](../stateset_agents/core/enhanced/advanced_evaluation_rubric.py) | API checker `ready` → `closing` → `closed`; registration identity, shared probes, detached observations, and retained cleanup are explicit. Advanced monitoring workers move `new` → `starting` → `running` → `closing` → `closed`, with retained `failed` outcomes. Utility workers start automatically and move `running` → `closing` → `closed` or `failed`, retaining thread/listener cleanup. Utility probes/checker registries move `ready` → `closing` → `closed` or `failed`; observer deadlines and unfinished cleanup are separate. Continuous evaluation moves `idle` → `running` → `stopping` → `idle`, or through terminal `closing` → `closed`/`failed`; foreground evaluator admission closes before work drains. Evaluation evidence moves from `unavailable` to `measured`/`inferred`; changed values retire evidence before rankings or reports consume it. Rubric checks distinguish pass/fail, unavailable criteria, and errors; suite retirement prevents further attempts after cancellation. Component health remains healthy/degraded/unhealthy, with unavailable checks explicitly represented. Utility alerts trigger, repeat, resolve, or are silenced until expiry. |
| A14 · E/H/I | [Advanced tracing](../stateset_agents/core/advanced_monitoring_tracing.py), [Inference metrics](../stateset_agents/api/inference_metrics.py), [API tracing](../stateset_agents/api/observability.py), [utility tracing](../stateset_agents/utils/observability.py), [performance monitor](../stateset_agents/utils/performance_monitor.py), [profiler](../stateset_agents/utils/profiler.py), [W&B integration](../stateset_agents/utils/wandb_integration.py), [API versioning](../stateset_agents/api/versioning.py) | Advanced tracer `ready` → `closing` → `closed` or `failed`; SDK span ends, provider ownership, partial setup, and exporter cleanup are explicit. Other span/operation owners move open → success/error → finish; profiler and telemetry initialize/start/stop; version metadata transitions between supported/deprecated/sunset policy. Observations do not establish job completion or remote telemetry delivery. |

### Remote execution and serving

| ID / representation | Owner | States, events, and important boundaries |
| --- | --- | --- |
| R01 · H | [Job contract](../stateset_agents/remote/job.py), [executor](../stateset_agents/remote/executor.py) | `pending`/`running` → `succeeded`/`failed`/`cancelled`. `wait()` polls, then fetches artifacts; it also attempts artifact recovery on failure. A successful remote status is distinct from a successful artifact download. |
| R02 · H/I | [Local](../stateset_agents/remote/local.py), [Modal](../stateset_agents/remote/modal.py), [RunPod](../stateset_agents/remote/runpod.py), [CoreWeave](../stateset_agents/remote/coreweave.py), [Nebius](../stateset_agents/remote/nebius.py) | Provision/submit → process/workload/pod ready → execute → poll → fetch → release/cancel. Process exit codes, provider state, artifact availability, cleanup confirmation, and metering are independent dimensions. RunPod also tracks whether submission started and whether cleanup remains unconfirmed. |
| R03 · H/I | [River](../stateset_agents/remote/river.py), [River RL runner](../stateset_agents/remote/river_rl_runner.py), [Tinker](../stateset_agents/remote/tinker.py), [Prime](../stateset_agents/remote/prime.py), [Fireworks](../stateset_agents/remote/fireworks.py), [Together](../stateset_agents/remote/together.py), [Hugging Face](../stateset_agents/remote/huggingface.py) | Provider/SDK session → training batches or hosted job → checkpoint handle → export/deployment. Local cancellation flags, provider status mapping, and retained handles differ by provider. Adapter and hosted-pointer results are different artifact types. |
| R04 · I | [River batches](../stateset_agents/remote/river_batches.py), [River environment](../stateset_agents/remote/river_environment.py), [River runtime](../stateset_agents/remote/river_runtime.py), [rollout budget](../stateset_agents/remote/rollout_budget.py), [progress](../stateset_agents/training/river_progress.py) | SDK client/session acquired → operation/future → completion → cleanup; episode adapter tracks reset/step/done; durable reservations move budget from available to exhausted before paid rollouts. Observed optimizer progress may remain unknown. |
| R05 · E | [Remote chat](../stateset_agents/remote/chat_session.py), [remote REPL](../stateset_agents/remote/chat_repl.py) | Local `new` → `starting` → `ready` → `closing` → `closed` or `cleanup_pending`. REPL startup and prompt/response/history rollback are a nested protocol. The session retains its pod ID until termination succeeds; see the contract below. |
| R06 · I | [Remote serving](../stateset_agents/remote/serve_session.py) | Provision → networking → server boot → readiness/adapter probe → persistent endpoint; startup failure cleans up, network failure may reprovision, manual stop/remote timer terminates. A successful serving pod intentionally outlives the CLI. |
| R07 · I/H | [Deployment contract](../stateset_agents/remote/deployment.py), [provider registry](../stateset_agents/remote/deployment_registry.py), [provider implementations](../stateset_agents/remote/coreweave.py), [ledger](../stateset_agents/remote/ledger.py), [canary](../stateset_agents/remote/canary.py) | Deploy → provider-specific readiness → stop; promotion verification gates deploy. Cost recording and canary cleanup evidence have separate completion boundaries. Unknown price is not zero cost. |
| R08 · I | [Submission recorder](../stateset_agents/finetune_provider_submission.py), [artifact handling](../stateset_agents/remote/artifacts.py), [remote worker](../stateset_agents/remote/worker.py) | Submission intent/observer events → returned provider handle or uncertainty; artifact staging → validated extraction/publication; worker entry → dispatched job kind → result. Persist the handle before relying on later polling or cleanup. |

### Fine-tuning UI, review, evaluation, and private workflows

| ID / representation | Owner | States, events, and important boundaries |
| --- | --- | --- |
| U01 · H/I | [Workspace](../stateset_agents/finetune_workspace.py), [worker/supervisor](../stateset_agents/finetune_worker.py), [finetune API](../stateset_agents/api/routers/finetune.py), [CLI](../stateset_agents/cli_finetune.py) | Prepared plan → launch claim → starting/queued/running → succeeded/failed/cancelled. Supervisor ownership loss yields `interrupted` and `execution_unconfirmed`; `run_terminal_status()` returns `None`, not confirmed exit. Plan hash, lock, PID/start token, stop receipt, and artifact receipt refine status. |
| U02 · H/I | [Validation](../stateset_agents/finetune_validation.py), [inspection](../stateset_agents/finetune_inspection.py), [remote code selection](../stateset_agents/finetune_remote_code.py) | Editable settings → checked → reviewed snapshot → queue submission → submitted/uncertain. Settings/data/package drift invalidates a review. Preflight checks, provider submission, and real training are different activities. |
| U03 · H/I | [Evaluation](../stateset_agents/finetune_evaluation.py), [evaluation worker](../stateset_agents/finetune_evaluation_worker.py), [quality](../stateset_agents/finetune_quality.py), [nomination](../stateset_agents/finetune_nomination.py) | Prepared comparison → queued/running phases → succeeded/failed/cancelled/unconfirmed; baseline/candidate generations and judging produce evidence. A nominated checkpoint is not automatically a promoted or deployed checkpoint. |
| U04 · I | [Recovery](../stateset_agents/finetune_recovery.py), [workspace](../stateset_agents/finetune_workspace.py) | Discover checkpoint → verify completeness/code identity → prepare separate recovery plan → review → launch. Reading interrupted state does not restart a worker; a new launch has a new ownership boundary. |
| U05 · I | [TUI](../stateset_agents/tui.py), [actions](../stateset_agents/tui_actions.py), [rendering](../stateset_agents/tui_render.py), [evaluation panel](../stateset_agents/finetune_evaluation_tui.py) | Focus/overlay/selection, draft/reviewed settings, pending launch/stop/recovery, and independent busy flags form a product of UI states. Rendered status is a projection of run records, not an execution authority. |
| U06 · I | [Background TUI task](../stateset_agents/tui_tasks.py) | Idle → busy → result ready → consumed; cancellation marks a result for discard while work retains the single-flight slot. Cancelled UI intent does not prove a thread stopped. |
| U07 · I | [Private experiment](../stateset_agents/improvement/workflow.py), [storage](../stateset_agents/improvement/storage.py), [CLI](../stateset_agents/cli_workflow.py) — in-progress workspace source | Sealed plan/splits → training and evaluation attempt files → verified result receipts → paired comparison → reviewed approval. Attempt without a result is uncertain; reopening the workspace does not authorize replay. Approval is bound to evidence and checkpoint identity. |
| U08 · I | [Private deployments](../stateset_agents/improvement/deployments.py), [durable refund resolver](../stateset_agents/improvement/tools.py), [Tinker batches](../stateset_agents/remote/tinker_batches.py) — in-progress workspace source | Deployment creation intent → provider handle → verified ready route → stop requested → reconciled stop. Refund authorization/lookup → durable attempt → committed receipt or reconciliation. Batch future resolution and client cleanup retain SDK ownership. These files were inspected but not rewritten by this audit. |

### Experimental and other-language implementations

| ID / representation | Owner | States, events, and important boundaries |
| --- | --- | --- |
| E01 · H/I | [Experimental orchestrator](../stateset_agents/experimental/intelligent_orchestrator.py), [models](../stateset_agents/experimental/intelligent_orchestrator_models.py), [logic](../stateset_agents/experimental/intelligent_orchestrator_logic.py) | Orchestrator stopped/running; components inactive/initializing/active/error/optimizing/updating according to `ComponentStatus`; task assignment, recovery, and component metrics influence scheduling. Core-path imports are compatibility aliases. |
| E02 · E/H | [Long-term planning](../stateset_agents/experimental/long_term_planning.py), [planning snapshots](../stateset_agents/experimental/planning_snapshot.py) | Plan steps pending → active → done/blocked. Preparation owns one operation per ID, runs callbacks outside the state lock, and commits only to its admitted revision. End/clear and managed updates fence late results. One-shot commands are consumed separately from response completion. Version 2 conversation snapshots include canonical plans, counters, and settled preparation receipts. Plan completion remains independent of conversation completion. |
| E03 · I | [Multi-agent coordination](../stateset_agents/experimental/multi_agent_coordination.py) | Agent idle → assigned/active → result → idle; task allocation, team membership, communication buffers, and consensus progress govern collaboration. |
| E04 · I | [Adaptive learning](../stateset_agents/experimental/adaptive_learning_controller.py), [few-shot adaptation](../stateset_agents/experimental/few_shot_adaptation.py) | Curriculum/exploration/hyperparameter changes, detected domain, adaptation history, and meta-trained flags gate available behavior. |
| E05 · I | [Architecture search](../stateset_agents/experimental/neural_architecture_search.py), [multimodal processing](../stateset_agents/experimental/multimodal_processing.py), [performance optimizer](../stateset_agents/core/performance_optimizer.py) | Candidate population → evaluation/selection/mutation; lazy modality resources → processing; profiling inactive/active → measurements and optimization decisions. |
| X01 · I | [JavaScript client](../npm/src/index.js) | Request/stream pending → response/chunks → done/error/timeout/user abort; linked abort signals, timers, SSE buffers, and reader cleanup belong to the request. API lifecycle values pass through rather than being renamed. |
| X02 · I | [Rust agent manager](../src/agents/mod.rs), [base](../src/agents/base.rs), [agent loops](../src/agents/customer_service.rs), [main](../src/main.rs) | Spawn loops → scheduled process cycle → decision → approval-needed or execution → action event; event channel receive → shutdown/task completion. Requires-approval metadata is not itself an approval workflow implementation. |
| X03 · H/I | [Rust domain contracts](../src/clients/mod.rs), [order routing](../src/agents/order_routing.rs), [fulfillment](../src/agents/fulfillment.rs), [quality control](../src/agents/quality_control.rs), [supplier](../src/agents/supplier.rs), [demand forecast](../src/agents/demand_forecast.rs) | Order, return, shipment, and purchase-order status enums; agents select eligible records, issue updates, and maintain forecast/decision history. The external commerce API ultimately owns domain transition validity. Generated protobuf enums describe transport values. |
| X04 · I | [Rust REST client](../src/clients/rest.rs), [unified client](../src/clients/unified.rs), [LLM service](../src/services/llm.rs) | Request attempts/backoff/rate limiting, optional gRPC initialization with REST fallback, and LLM request/response completion. Most commerce operations currently delegate directly to REST. The native extension's [trajectory accumulators](../rust_core/src/trajectory.rs) and numerical kernels have no independent completion/cleanup lifecycle. |

## Remote chat contract implemented here

`ChatSessionState` is a string enum; `session.state` and `session.pod_id` are
read-only observations with no network request. `start()`, `ask()`, and `close()`
retain their signatures and successful return values. An operation lock
serializes access to the same session's provisioning and prompt/response stream.

| Current state | Event | Result |
| --- | --- | --- |
| `new` | `start()` | `starting`; resolve configuration, then attempt one pod creation. |
| `starting` | Configuration failure before creation | `new`; caller may correct configuration and try again. |
| `starting` | REPL reports ready | `ready`. |
| `starting` | Provisioning/boot failure | Attempt cleanup and propagate the original error. |
| `ready` | `ask()` | Send one prompt and consume its response; append only successful turns. |
| `new`, `starting`, `ready`, `cleanup_pending` | Owned cleanup | `closing`; close the local channel and attempt termination of the retained pod. |
| `closing` | No known pod, or termination returns successfully | `closed`; unregister exit hook; finalize known pod cost once. |
| `closing` | Termination raises | `cleanup_pending`; retain pod ID and exit hook, log recovery guidance. Ordinary provider errors do not mask the chat/start error. |
| `closed`, `closing` | Repeated/reentrant `close()` | No duplicate deletion or ledger entry. |
| Anything except `new` | `start()` | Reject before provisioning or changing transcript metadata. |
| Anything except `ready` | `ask()` | Reject before writing a prompt. |

```python
from stateset_agents.remote.chat_session import ChatSessionState

session.close()
if session.state is ChatSessionState.CLEANUP_PENDING:
    # Keep this handle for a later close() attempt or provider-side cleanup.
    pending_pod = session.pod_id
```

These states describe local ownership. A successful termination call means the
provider accepted it, not independently verified billing cessation. If creation
raises before returning a pod ID, the session has no handle to reconcile; inspect
the provider before creating another session. Exit hooks are best effort and do
not survive process crashes. `ready` records completion of startup; it is not a
continuous provider health check. This patch does not redesign chat wire-protocol
recovery after a partial or timed-out response.

Regression coverage in
[test_remote_chat_session.py](../tests/unit/test_remote_chat_session.py)
exercises duplicate/concurrent/reentrant starts, start after close, preflight
correction, uncertain submission, repeated cleanup failure followed by recovery,
exit-hook retention, interrupt paths, preserved errors/transcripts, and cost
finalization. CLI, remote-serving, and ledger tests protect adjacent contracts.

## Async pool ownership contract implemented here

`pool.state` is a read-only `PoolState`. Existing enum values, method signatures,
the `async with pool.acquire()` interface, and `PoolStats` fields are retained.
Pool sizes must satisfy `0 <= min_size <= max_size` before anything is allocated.

| Event | Ownership and outcome |
| --- | --- |
| `initialize()` | Create the minimum set once; simultaneous callers wait for the same setup. Repeating initialization in `ready` is a no-op. Failure or cancellation drains partial resources before reaching `closed`. |
| Checkout | Atomically remove availability and claim borrowed ownership before validation. Health checks, scale-down, and shutdown cannot close that borrowed resource. |
| Successful return | Requeue a healthy, unexpired resource only while the pool is `ready`. |
| Validation/use failure, cancellation, expiry, or return during drain | Detach the resource, then perform one tracked cleanup attempt. Failed and stale wrappers cannot become idle through `release()`. |
| Health check / scale-down | Select only unborrowed resources; detach them under the condition, then await cleanup outside the pool lock. Removed resources cannot remain available. |
| Growth | Reserve ownership through creation, even if the caller is cancelled. Resources still undergoing cleanup count against `max_size`; health checks replenish only after cleanup and only while ready. |
| `drain()` | Transition to `draining`, wake and reject waiting borrowers, and wait for validation/use to return. New acquisition and scaling are rejected. |
| `close()` | Serialize shutdown, stop the health task, drain borrowed ownership, and await all tracked cleanup before `closed`. Repeated close is harmless; closed/draining pools cannot be reinitialized. |
| Global HTTP sessions | Sessions share the original connector limit of 100, with the factory owning the connector. Retiring a session preserves its siblings; retiring the last session closes the connector. Later creation gets a fresh connector. |

Connection counts describe current usable membership: `total = active + idle`,
where active includes validation. Detached resources awaiting cleanup are no
longer counted as usable connections but still consume allocation capacity.
`failed_requests` counts an ordinary acquisition/use exception once; cancellation
does not count as a request failure. `total_requests` retains its meaning of
successful validation followed by entry into the borrow context.

Acquisition retains the existing wait-then-grow timeout behavior. Scaling is
best effort when borrowed resources, pending cleanup, or creation failures
prevent reaching the requested size. The unused `scaling` enum value remains
available for compatibility.

Creation and cleanup are drained before propagating caller cancellation; this
does not impose an I/O deadline. Factories must configure their own timeouts,
and graceful shutdown requires borrowers to finish. Cleanup exceptions are
logged and their handles retained in `cleanup_failures`, with no automatic retry
or reuse: `closed` establishes completion of
local cleanup attempts, not proof that a failing external close succeeded.
Factories own partial allocations that they never return. Resources that share
an underlying handle need a factory that owns that shared handle, as the global
HTTP factory now does. Its connector handle is retained until close succeeds,
including after session-construction failure. Factories can expose remaining
ownership through `cleanup_pending`. Global cleanup treats these observations
as unconfirmed cleanup, retains the pool, and refuses replacement. Failed cleanup
handles are outside usable membership and remain available for diagnosis; they
do not become available to borrowers again.

Regression coverage in
[test_async_pool_lifecycle.py](../tests/unit/test_async_pool_lifecycle.py)
uses event barriers for competing initialization, validation, drain, scale,
health, close, and repeated cancellation. It also checks failed resource reuse,
capacity during cleanup, honest counts, concurrent borrowers, initialization
rollback, and actual aiohttp connector ownership without network requests.

## API circuit breaker contract implemented here

The API breaker's public call/decorator/reset interfaces, configuration fields,
state strings, exception classification, and serialized `CircuitStats` keys
remain unchanged. The read-only `probe_in_flight` property distinguishes an
occupied recovery slot from a half-open circuit that can admit its next probe.
It is an observation, not a reservation; `call()` performs the admission check.

| State / event | Admission and outcome |
| --- | --- |
| `closed` / request | Admit concurrently. Current-generation successes reset the failure streak; counted failures open the circuit at the configured threshold. |
| `open` / cooldown pending | Reject with the existing `CircuitOpenError`, or run the configured fallback outside the state lock. |
| `open` / cooldown elapsed | The next request transitions to `half_open` and owns the single probe slot. |
| `half_open` / probe active | Reject competing requests or use the configured fallback. Count these as rejected requests; fallback results do not establish recovery. |
| `half_open` / current probe succeeds | Release the slot and count a recovery success. Close only after `success_threshold` sequential successful probes. |
| `half_open` / current probe has a counted failure | Release the slot and reopen with a fresh full cooldown. |
| Probe cancelled, excluded exception, or exception outside the existing counted group | Propagate the outcome and release the slot without changing recovery/failure counts. Previous successful probe evidence is retained. |
| Completion from before a state transition or manual reset | Deliver its result/error to the original caller without changing current counters, timestamps, or probe ownership. |
| `reset()` | Start a closed generation with fresh stats and admission. Previously admitted work continues, with its completion fenced from the new generation. |

Every transition and reset advances a local generation. Admission and completion
bookkeeping hold the same state lock; neither upstream calls nor fallbacks hold
it. Uncontended completion commits inline before another ready request can be
admitted; fast failures therefore preserve the configured opening threshold.
Contended bookkeeping is drained before caller cancellation propagates, including
repeated cancellation. If an upstream result is already known when cancellation
arrives during bookkeeping, that result can still supply current-generation
health evidence; cancellation is not itself recorded as a service failure.

Cooldown uses monotonic elapsed time. `last_failure_time` and `last_state_change`
remain wall-clock timestamps for existing monitoring consumers. Success/failure
counts are breaker control state, not lifetime totals for every upstream outcome;
late completions cannot extend a new cooldown or close a recovering circuit.
Request and rejection totals retain their existing reset-on-manual-reset policy.

The breaker performs no automatic retry or upstream cancellation on reset. A
cancelled call does not prove that a remote side effect was undone. Upstream
timeouts remain the caller's responsibility. As with its asyncio lock, an instance
belongs to one event loop; synchronous reset must run on that loop while calls
are active. The core breaker has a separate contract below and retains its
different counter and recovery policies.

[test_circuit_breaker_lifecycle.py](../tests/api/test_circuit_breaker_lifecycle.py)
uses controlled clocks and event barriers to exercise concurrent probes, stale
success/failure across transitions and reset, cancellation during completion,
fallback isolation, clock jumps, fresh cooldowns, and closed-state concurrency.
An immediate-failure burst checks that opening is committed before admitting
the next ready call.
The existing resilience tests protect thresholds, decorators, fallback behavior,
and adjacent resilience patterns; the existing breaker performance check guards
request overhead.

## Core circuit breaker and retry contract implemented here

The core breaker's enum values, configuration names, synchronous call interface,
and `await breaker.call(async_operation)` interface are retained. State reads
through `get_status()` return a consistent snapshot of `state`, `failure_count`,
`last_failure_time`, and `probe_in_flight` without reserving admission. Callers
must use `reset()` to reset a live breaker; direct field mutation bypasses its
ownership rules. Configuration should be established before concurrent use.

| Event | Admission and outcome |
| --- | --- |
| Closed call | Invoke outside the state lock. Count configured ordinary exceptions; preserve the existing cumulative failure count across successes. Open at the configured threshold. |
| Open call before cooldown | Reject with the existing `GRPOException` before invoking the operation. |
| Open call after cooldown | Move to `half_open`; reserve one probe shared by threads and async callers. Competing calls reject before invocation. |
| Probe succeeds | Close and reset the failure count only after the synchronous value or awaited result is known. |
| Probe has a counted failure | Release admission and reopen with a fresh full cooldown. |
| Cancellation, process-control exception, or excluded exception | Propagate the original outcome. Release probe admission after the call unwinds, retaining `half_open` without claiming recovery or adding failure evidence. |
| Old completion after a transition or reset | Return its outcome to its caller without changing current state, counters, timestamps, or probe ownership. |
| `reset()` | Begin a fresh closed generation. Do not cancel already admitted work. |

The state lock covers admission and bookkeeping only. Async upstream cleanup
retains probe ownership until it unwinds. Cooldowns use monotonic time, while
`last_failure_time` remains a wall-clock timestamp. Counts are control state,
not lifetime upstream-failure totals; completions from older generations do not
extend a newer cooldown.

Known async functions and async callable objects begin admission when awaited.
A synchronous factory returning an awaitable starts immediately under admission;
its returned wrapper must be awaited to settle ownership. `call_async(factory)`
defers the factory invocation until awaited. Coroutine, future, and custom
awaitable results are observed on completion; iterator/async-generator iteration
is outside this call-result boundary. Abandoning an awaitable is not proof that
upstream work stopped. This is local health/admission state, not a guarantee of
remote cancellation or a deadline for upstream work.

Core retry validates its policy and copies it at invocation entry. Attempt
budgets are positive integers; delays are finite and nonnegative, with finite
positive exponential bases. An explicit empty `retry_on` disables retrying;
`None` retains the network/resource defaults. Exception lists are copied, and
changes to configuration affect subsequent invocations rather than an in-flight
retry. Each invocation owns its own budget. Backoff remains capped with the
existing jitter behavior, including when exponentiation overflows. Exhaustion
raises the last exception without another sleep. Cancellation and process-control
exceptions propagate even under a broad exception filter. Both decorators retain
function metadata, signatures for introspection, and async-function detection.

Wrapping retry inside the breaker, as the production customer-service example
does, records only the final retry outcome as breaker health evidence. Retrying
an operation with uncertain remote effects still requires caller-owned
idempotency or reconciliation; neither wrapper establishes that an earlier
effect was undone.

[test_core_circuit_breaker_lifecycle.py](../tests/unit/test_core_circuit_breaker_lifecycle.py)
covers awaited outcomes, mixed thread/async probe exclusion, cancellation cleanup,
excluded errors, late outcomes/reset, monotonic cooldowns, cumulative counters,
concurrent closed calls, awaitable factories, and metadata.
[test_core_retry_lifecycle.py](../tests/unit/test_core_retry_lifecycle.py)
covers explicit retry intent, policy snapshots, concurrent budgets, cancellation
during backoff, overflow/caps, original exception identity, and breaker/retry
composition. Existing error-handling and API resilience tests protect adjacent
behavior and the distinct core/API policies.

## API retry contract implemented here

`RetryConfig`, `RetryStrategy.execute()`, the strategy decorator, and `retry()`
retain their successful interfaces, argument forwarding, function metadata,
and structured attempt logging. Every invocation copies and validates the
current configuration before invoking user work. Attempts must be positive
integers, delays finite and nonnegative, and the exponential base finite and
positive. Invalid configuration, including a template mutated after construction,
fails before an operation starts. Exception filters normalize to tuples of
exception classes; mutable input collections cannot change a captured policy.

| Retry event | Outcome |
| --- | --- |
| Start | Capture one policy and begin a local attempt budget. Concurrent calls do not share attempt counters. Later configuration edits affect later invocations. |
| Success | Return the original result immediately. |
| `CancelledError`, `KeyboardInterrupt`, `SystemExit`, or `GeneratorExit` | Propagate the original interruption without another attempt, even if a broad exception filter includes it. |
| Exception matches `non_retryable_exceptions` | Propagate immediately; exclusions take precedence over retryable matches. |
| Exception does not match `retryable_exceptions`, including an empty filter | Propagate without repeating the operation. |
| Retryable failure with budget remaining | Calculate delay from the captured policy, log the failed attempt, await backoff, then start the next attempt. Cancellation during backoff prevents that next invocation. |
| Retryable failure exhausts the budget | Log exhaustion and raise that final exception, preserving its identity; do not sleep again. |

The default API filter still retries ordinary `Exception` values; the core
default remains limited to its network/resource exceptions. The API still adds
0–50% upward jitter **after** capping the base delay, so `max_delay` is the cap
before jitter. The core's existing 50–100% multiplier remains separate. At the
floating-point limit, upward jitter saturates at the largest finite float.

Both policies use [bounded backoff arithmetic](../stateset_agents/utils/backoff.py).
Normal calculations retain their ordinary floating-point results. Range-limit
calculations account for the initial scale before clipping: a very large power
can still yield a delay below the cap after multiplying by a very small base,
and a very small power can still yield a representable delay after scaling up.
Integer configuration literals cannot trigger arbitrary-size exponentiation.
Zero base delay or zero cap remains zero; exponent overflow cannot replace the
operation's failure with an arithmetic error.

A policy snapshot is captured without suspending; configure templates before
concurrent use rather than mutating their fields from another thread. Retry owns
the attempt sequence, not detached work inside an operation. Use the operation's
ownership boundary for cleanup, and idempotency or reconciliation for uncertain
remote effects. A breaker around retry observes its final outcome; a bulkhead
around retry retains capacity across the whole retry sequence, while retry around
a bulkhead reacquires admission for each attempt. Neither order starts a retry
before an owning timeout/bulkhead finishes the earlier attempt's cleanup.

[test_retry_lifecycle.py](../tests/api/test_retry_lifecycle.py) covers policy
snapshots, independent concurrent budgets, exception precedence, cancellation
during backoff, invalid configuration, original outcomes/metadata, jitter,
exhaustion, and composition with breakers, timeouts, and bulkheads.
[test_backoff.py](../tests/unit/test_backoff.py) checks overflow, underflow,
subnormal intermediate powers, caps, zero delays, and very large exponents.
Both API and core retry tests exercise the shared calculation through real retry
sequences.

## API bulkhead and timeout contract implemented here

Bulkhead constructors, decorators, `execute()` results, `BulkheadFullError`, and
existing statistics keys are retained. A bulkhead captures validated limits at
construction: positive integer `max_concurrent`, nonnegative integer `max_waiting`,
and a nonnegative, non-NaN queue timeout. `None` or positive infinity allows
unbounded queue waiting. Changing the supplied config or `bulkhead.config` does
not resize a live instance; statistics report the actual enforced capacity.

| Bulkhead event | Ownership and outcome |
| --- | --- |
| Idle capacity available | Reserve one active slot without joining the queue. `max_waiting=0` admits immediately available capacity and rejects excess work. |
| Capacity occupied, queue has room | Join the FIFO queue without invoking user work. Queue timeout measures admission waiting, not execution duration. |
| Queue full | Raise `BulkheadFullError` before invoking user work. |
| Waiting caller cancelled or queue deadline expires | Remove that request and reclaim queue capacity. Do not invoke its operation or disturb an active owner. |
| Active owner finishes | Release its slot without suspending; reserve it directly for the oldest live waiter. Expired queue entries cannot win admission just because their timer callback has not run yet. |
| Waiting caller cancelled after a slot was handed to it | Return that reservation and pass it to the next live waiter; do not invoke the cancelled request's operation. |
| Active caller cancelled | Cancel its owned operation once, retain the slot through cleanup, and propagate caller cancellation after retrieving the outcome. Further caller cancellation cannot interrupt that cleanup. |

An instance belongs to one event loop. Admission, release, and statistics reads
do not suspend. `active` includes granted reservations, running calls, and calls
unwinding after cancellation; `waiting` counts requests still in the admission
queue. `available = max_concurrent - active` never reports an occupied slot as
free. Each admitted operation runs in an owned task, inheriting request context;
changes to task-local context variables remain in that task. Decorator metadata
and signature introspection are preserved.

The timeout decorator retains successful values, its public `TimeoutError`,
function metadata, and upstream exception identity. It now distinguishes the
following outcomes explicitly:

| Timeout event | Outcome |
| --- | --- |
| Nonpositive deadline, or deadline elapsed before the worker starts | Raise the wrapper's `TimeoutError` without invoking the function. |
| Operation finishes within its monotonic deadline | Return its result, even if its observer is scheduled later. |
| Upstream raises its own timeout or other exception | Propagate that original exception, rather than misreporting it as wrapper expiry. |
| Deadline expires during execution | Cancel once and drain the operation. A late successful return, including suppressed cancellation, cannot turn expiry into success. A cleanup exception retains its original type and identity. |
| Caller cancelled, including during expiry cleanup | Drain the owned operation before propagating caller cancellation, which takes precedence over its late result or ordinary cleanup error. |

`None` or positive infinity disables deadline expiry; NaN is rejected before
work. Timers cannot preempt blocking synchronous work, so monotonic checks before
invocation and on completion also reject execution that starts or finishes late.
Timeout and bulkhead compose in either order without releasing the bulkhead's
capacity before cleanup finishes.

These boundaries own the awaited operation. User code must join its own child
tasks and synchronous workers; detached work or a cancelled `to_thread` waiter
does not prove the underlying thread stopped. Cleanup can outlast a deadline or
wait indefinitely on a noncooperative operation. Configure upstream I/O deadlines;
neither wrapper establishes that a remote side effect was undone or safe to retry.

[test_bulkhead_lifecycle.py](../tests/api/test_bulkhead_lifecycle.py) covers idle
and queued admission, queue overflow, FIFO order, cancellation at permit handoff,
expired waiters, capacity through repeated cancellation, stable configuration,
request context, and exception identity.
[test_timeout_lifecycle.py](../tests/api/test_timeout_lifecycle.py) covers upstream
timeout identity, cancelled cleanup, late returns, cleanup-error precedence,
deadlines before invocation and at completion, and composition in both orders.
The existing resilience tests and performance checks protect adjacent behavior.

## Legacy GRPO job contract implemented here

The secondary GRPO app retains its status strings and response models. Initial
submission still responds with `started` while the stored job is `starting`;
that response acknowledges submission, not execution. Job state changes belong
to `StateManager`, using the same reentrant lock as job retention and expiry.

| Current state / event | Result |
| --- | --- |
| Absent ID / `create_job()` | Create a `starting` job. An ID still retained in the registry cannot be recreated, including a retained terminal job. |
| `starting` / worker start | Commit `running` and set `started_at` once. Duplicate workers cannot claim an already running job. |
| `starting` or `running` / terminal transition | First caller commits `completed`, `failed`, or `cancelled` and sets `completed_at`. Direct completion from `starting` remains supported. |
| Active job / progress update | Validate the whole update, then publish progress/results and refresh TTL under the same lock. |
| Terminal job / any update | Preserve status, timestamps, progress, results, error, and TTL. Repeated or conflicting outcomes do not rewrite completion evidence. |
| Invalid or regressive status update | Ignore the entire update, including attached metrics or results; return the unchanged record. |
| Worker result after expiry or replacement | Ignore it. The worker's retained job object must still own the registry entry; matching the ID alone is insufficient. |
| Cancellation during an iteration | Commit local `cancelled`; when the current iteration returns, discard its late result and start no further iteration. |
| Worker task cancellation after entry | Commit `cancelled` if still active and propagate cancellation. Unexpected ordinary exceptions commit `failed` when possible and still propagate; previously handled engine exceptions retain their existing handling. |
| Cancel request loses to completion or expiry | Return unsuccessful cancellation. Both batch API versions report the observed terminal/missing outcome instead of claiming cancellation. |

`transition_job()` returns whether this caller committed a new legal transition.
Handlers use that result for cancellation responses and completion/failure
metrics. `update_job()` retains its job-or-None interface and adds an optional
`expected_job` keyword for worker ownership. `TrainingJob.is_terminal` provides
a read-only observation. Ownership checks also prevent a cancellation authorized
for an old record from affecting a replacement owned by another user.

The distributed entry point still delegates to computational training, which
owns its single execution claim. Shutdown job cleanup uses the same transition
guard, so a completion that wins the race is preserved.

Cancellation remains a local terminal decision, not acknowledgment that engine
work has stopped. An in-flight iteration may finish and may already have changed
the engine; callers should not infer rollback or safe resubmission from the
label. These are in-memory guards, with existing TTL eviction and background
task scheduling. A callback cancelled before it ever enters the worker has no
worker completion callback in this legacy app; closing admission now classifies
those pending jobs as cancelled and prevents a delayed callback from using a
closed engine. Direct mutation of returned job objects or the public registry
bypasses the manager's guards; integrations must use manager operations.

[test_grpo_job_lifecycle.py](../tests/api/test_grpo_job_lifecycle.py) covers late
updates for every terminal status, threaded terminal races, duplicate workers,
cancel-before-start and cancel-during-iteration, propagated task cancellation,
unexpected failures, expiry/replacement, authorization races, batch responses,
shutdown cancellation, and preserved successful responses and metrics.

## GRPO shutdown ownership contract implemented here

The training handler tracks workers separately from their job records.
`active_workers` measures executions that still own engine access, including
jobs that were cancelled, expired, or evicted. `is_closing` means admission is
permanently closed; `is_closed` means the owned worker drain finished. These
observations belong to one handler on one event loop.
The handler retains worker tasks until their execution finishes, so dropping a
caller's task reference cannot allow garbage collection to abandon engine work.
Startup with a closing handler or a shutdown manager that has already run is
rejected before allocating engines; a new lifecycle requires fresh owners.

| Boundary | Ownership and outcome |
| --- | --- |
| Stop acceptance | `close_admission()` rejects new work and commits local cancellation for active/pending jobs. Both API versions return 503 for training submissions during shutdown, without creating jobs or incrementing submission metrics. |
| Pending callback | Callbacks that never entered a worker are covered by admission closure. Delayed callbacks and queued iterations check ownership before engine work begins. |
| Worker drain | `aclose()` joins all tracked workers, independently of TTL registry membership. Concurrent callers join the same drain; cancelling a waiter does not detach it. |
| Worker task cancellation | Interrupt the current iteration once, then wait for its coroutine to unwind before releasing worker ownership. Repeated cancellation cannot interrupt that unwind again. |
| Engine cleanup | Requires successful worker drain. Await async cleanup; run synchronous cleanup in an owned thread. Multiple registry aliases and the demo-engine reference trigger one cleanup per engine. Remove registry handles only after cleanup returns successfully. |
| Serving error / cancellation | Lifespan cleanup runs in `finally`. Caller cancellation propagates after the owned shutdown sequence settles. |
| Critical drain or engine cleanup failure | Record the failed task and abort subsequent phases. Failed engine handles remain available for diagnosis; repeated shutdown calls observe the same failure without automatically retrying cleanup. |
| Concurrent or repeated shutdown | `execute_shutdown()` joins one owned sequence, preserving phase order and executing each registered callback once. Registration is rejected after execution starts. |

`get_status()` retains its existing fields and adds `outcome`: `not_started`,
`running`, `completed`, `completed_with_errors`, or `failed`. A noncritical
handled callback failure, or expiry of the request-drain timeout, is visible as
`completed_with_errors`; an aborted sequence is `failed`. The admission gate
remains closed after either outcome. A failed worker drain does not authorize engine
cleanup, and a failed sequence is not automatically replayed.

`total_timeout` starts with the first `execute_shutdown()` call and supplies one
monotonic time budget for request draining and every callback. Callback waits use
the smaller of their configured timeout and the remaining total budget. A queued
callback checks the deadline again before invoking user code. After expiry, no
later callback starts, and the sequence fails with `TimeoutError` unless an
active callback fails with another exception first. `deadline_exceeded` is visible
while cancellation cleanup is still running and remains true after failure.
A callback that suppresses cancellation cannot authorize another phase or turn an
expired sequence into success. Completed callbacks remain in `completed_tasks`;
the sequence outcome separately reports failure. Repeated callers cannot reset
the budget or replay callbacks. A zero budget admits no callbacks; positive
infinity disables the total deadline. Negative values and NaN are rejected.

Shutdown remains cooperative. Engines must bound their I/O and own any lower-level
threads or remote operations; waiting for a coroutine cannot prove an unowned
external effect has stopped. Both timeouts can take longer while owned work
unwinds. Expiry does not issue another cancellation into cleanup already unwinding
from a shorter task timeout. Blocking code cannot be preempted, but the deadline
is checked again when it returns. The shorter request-drain timeout still allows
later phases to proceed within the total budget, while reporting incomplete drain.

[test_grpo_shutdown_ownership.py](../tests/api/test_grpo_shutdown_ownership.py)
uses event barriers and local engines to cover serving failures, concurrent and
cancelled shutdown callers, dropped task references, evicted job records, callbacks
cancelled before entry, repeated worker cancellation, queued iteration admission,
HTTP rejection,
async and threaded engine cleanup, engine aliases, critical failures/timeouts,
retained handles, and non-replayed outcomes.
[test_shutdown_deadline.py](../tests/api/test_shutdown_deadline.py) covers the
shared budget, delayed callback entry, request draining, cancellation suppression,
repeated waiter cancellation, expiry during unwind, terminal observations, and
zero/infinite/invalid budgets.

## State snapshot isolation contract implemented here

Snapshots use `copy.deepcopy` rather than serialization, preserving Python value
types, nested aliases within a captured graph, and cycles. Custom objects retain
their normal copy protocol, including `__deepcopy__`; hooks must return an
appropriately isolated value. Resources such as locks and live connections belong
outside snapshot data unless their type supplies meaningful copy semantics.

| Boundary | Ownership and outcome |
| --- | --- |
| Capture | Copy state before publishing history or advancing its version. A copy failure raises `TypeError` with the original exception as its cause; there is no shallow-copy fallback. |
| Write with `create_snapshot=True` | Copy the candidate state before the write changes live state, cache, dirty keys, change log, or watchers. The snapshot describes that write boundary and is published after the cache write returns; unrelated changes during that await do not enter the captured history. |
| Inspect | `get_snapshots()` returns copies of snapshot records, nested data, and metadata. Editing the returned records cannot change retained history. |
| Restore | Prepare a fresh copy before clearing live state. Cache reads and later nested mutations cannot change the saved snapshot, so repeated restore returns the same captured values. |
| Copy failure | Publish no snapshot and preserve the version. Failed restore preparation leaves live state and cache untouched. Ordinary writes and disabled versioning retain their existing behavior for arbitrary values. |

Snapshot IDs, record types, retention, and version advancement/restoration keep
their existing interfaces. Copying costs memory and CPU proportional to the
object graph. Ordinary `get()`/`set()` values and change-log entries still use
references. Direct edits to the public `state` and `snapshots` internals bypass
these boundaries. Cache writes and restore are still nontransactional; this
change does not promise rollback on cache errors or isolation from concurrent
writes while cache I/O is awaited.

[test_state_snapshot_isolation.py](../tests/unit/test_state_snapshot_isolation.py)
covers conversation updates, nested input/cache mutations, repeated restore,
inspection, cycles, aliases, custom types, copy failures, disabled versioning,
retention, and mutations during cache writes or watcher notification.

## Async task manager contract implemented here

`TaskManagerState` is available from `stateset_agents.core.async_pool`.
`state` and the additive `get_status()["state"]` expose `new`, `running`,
`draining`, or `closed`. The existing count fields remain; `waiting_tasks`
separates queued submissions from admitted work, and `available_slots` is zero
after admission closes. A newly constructed manager starts on explicit `start()`
or its first submission. Repeated start is idempotent; shutdown is terminal for
that instance.

| Boundary | Ownership and outcome |
| --- | --- |
| Admission | Waiting submissions claim capacity in queue order and wake when capacity becomes available or shutdown closes admission. Cancelling a queued submission removes its place without blocking followers. The state check and slot claim occur under one condition; a waiter cannot slip into execution after shutdown. |
| Queued cancellation/rejection | Close a fresh, unstarted coroutine so it cannot leak or perform effects. Leave pre-existing tasks/futures untouched until admitted. Custom awaitables retain their caller-defined ownership until admission. |
| Admitted work | Hold strong references and retain capacity until the operation has returned or its cancellation cleanup has unwound. Dropping the submitter's task reference cannot abandon the operation. |
| Task timeout | Begins after admission, excluding queue time. `None` uses the configured default; zero cancels before a fresh coroutine starts. Preserve `asyncio.wait_for` behavior for operations that suppress timeout cancellation and return a result or raise their own error. |
| Caller cancellation | Request cancellation once and wait for cleanup before returning cancellation to the caller. Repeated cancellation cannot detach cleanup or release capacity early. |
| Batch failure/cancellation | Request cancellation of unfinished siblings and join them before propagating the failure. `return_exceptions=True` preserves ordered result/error collection and lets successful siblings finish. |
| Shutdown | Close admission immediately and wake queued submissions. Share one drain across concurrent/repeated callers; cancelled waiters cannot interrupt it. Allow admitted work a grace period, then request cancellation once and await actual completion. |
| Global getter | Replace a closed task manager with a fresh instance. A still-draining instance rejects new access rather than being replaced while it owns work. |

`shutdown_timeout` is an optional keyword-only grace period, defaulting to the
existing 30 seconds. It is not a hard limit on cleanup: operations that suppress
cancellation or own blocking external work must provide their own I/O timeouts.
Shutdown must be called outside the manager's own operations, on the same event
loop; direct self-shutdown raises instead of deadlocking. Callers should submit
fresh coroutines for concurrency-limited effects: a task that was already started
cannot be retroactively stopped from running while it waits for admission.
Successful return values and ordered batch results are preserved. On batch
failure, siblings are now cancelled and joined instead of continuing after the
batch has raised. These guards apply to supported manager methods, not direct
mutation of private task registries or external cancellation of caller-held tasks.

[test_async_task_manager_lifecycle.py](../tests/unit/test_async_task_manager_lifecycle.py)
covers repeated start, admission closure, queue cancellation, zero timeout,
concurrent shutdown, overlapping cancellation sources, timeout suppression,
batch cleanup, arbitrary awaitables, capacity and queue-order observations, dropped task
references, self-shutdown, and global replacement.

## State-service shutdown contract implemented here

`StateServiceState` and `get_status()` expose maintenance ownership separately
from cache health: `new`, `running`, `closing`, `closed`, or `failed`. Construction
outside an event loop leaves maintenance unstarted; later `get_state_service()`
access on a running loop starts it once. An already closed singleton is replaced
on next access. Closing or failed owners reject replacement.

| Boundary | Ownership and outcome |
| --- | --- |
| Stop | Close maintenance/probe admission and create one shutdown task shared by concurrent and repeated callers. Cancelling a waiter cannot detach this task. |
| Join maintenance | Request cancellation once, then join both maintenance tasks before final synchronization. An unexpected maintenance failure is observable and prevents later cache cleanup. |
| Join probes | Wait for admitted health checks, including those whose callers were cancelled. New checks report an unhealthy lifecycle without touching the cache. |
| Final synchronization | Require every reported key synchronization to succeed. Exceptions or false acknowledgments fail shutdown and preserve the cache. |
| Local cleanup | Clear the in-memory cache only after joins and successful synchronization. A failed or unacknowledged clear is a failed lifecycle, never a successful close. Remote cache data is not cleared. |
| Repeated failure | Preserve the same exception and owner without retrying writes or clearing again. `get_status()` requires no cache I/O and remains available. |

Health probes use independent temporary keys, verify the returned value, and
remove their key after reading it. They cannot overwrite the former fixed
`health_check_test` user key or invalidate a concurrent probe. Their existing
healthy/unhealthy/error results retain cache-health meaning; an unexpected dead
maintenance task makes an otherwise healthy cache report `degraded`.

The service owns maintenance and health probes, not arbitrary calls made through
its public `state_manager` or `cache_backend` handles. Stop external writers
before shutdown. Cache I/O and cancellation cleanup remain cooperative; lower
layers must provide appropriate timeouts. Direct self-shutdown from a service
task is rejected to avoid waiting for itself. Failed owners intentionally remain
available for diagnosis; there is no automatic reset or cleanup retry.

[test_state_service_shutdown.py](../tests/unit/test_state_service_shutdown.py)
covers maintenance unwind, concurrent/repeated callers, waiter cancellation,
probe ownership and isolation, synchronization/clear failures, unexpected worker
failure, singleton replacement, and deferred startup.

## Global resource cleanup contract implemented here

`GlobalResourceState` and `get_global_resource_status()` in
`stateset_agents.core.async_pool` expose `ready` → `draining_tasks` →
`draining_http` → `closed`, with `failed` retaining incomplete ownership.
Observations include which owners remain and whether HTTP cleanup is unconfirmed.

| Boundary | Ownership and outcome |
| --- | --- |
| Begin cleanup | Close the global task manager's admission immediately. Concurrent/repeated callers join one owned sequence, including cancelled waiters. |
| Drain tasks | Join the manager before closing dependencies. HTTP getters remain available during this phase so admitted work can finish, including creating the first HTTP pool. New task-manager getters are rejected. |
| Drain HTTP | Capture the current pool after tasks settle. Reject new getters and wait for existing borrowers/initialization and resource cleanup. An in-progress getter cannot return a pool that global shutdown has closed. |
| Confirm release | Clear each global reference only after its cleanup succeeds, checking owner identity. Pool cleanup failures and factory-owned pending cleanup block HTTP release even when the pool's local state is `closed`. |
| Failure | Propagate the failure, retain incomplete owners, and reject new getters. Later cleanup calls observe the same failure without replaying effects. A failed task drain leaves the HTTP dependency untouched. |
| New generation | After successful cleanup, the first getter creates a fresh lifecycle. Repeated cleanup alone does not recreate or replay the previous owners. |

`managed_async_resources()` inherits this behavior, including errors from
incomplete cleanup. Call cleanup outside managed tasks on the owners' event loop;
direct self-cleanup is rejected. Directly retained handles and user-created
background work are outside global getter admission. Factory error reporting is
part of the ownership contract: a factory that hides its own failures cannot
provide confirmation to the pool. The shared HTTP factory retains failed session
and connector ownership; its status does not infer external release from a local
`closed` label.

[test_global_resource_cleanup.py](../tests/unit/test_global_resource_cleanup.py)
covers dependency ordering, late HTTP initialization, waiter cancellation,
concurrent callers, getter admission and initialization races, failed resource
and connector cleanup, retained handles, non-replay, and fresh generations.

## API health observation and cleanup contract implemented here

`HealthCheckerState` in `stateset_agents.api.resilience` separates admission and
cleanup (`ready` → `closing` → `closed`) from component health
(`healthy`, `degraded`, `unhealthy`). `get_status()` exposes the lifecycle,
registered checks, active/cancelling probe counts, and pending names without
starting I/O. `get_results()` returns a detached view of current observations.

| Boundary | Ownership and outcome |
| --- | --- |
| Register / replace | Every registration gets a fresh identity, even for the same callable. Previous health is invalidated. An old probe can finish for its existing observers but cannot publish into a replacement. |
| Remove | Exclude the check from current snapshots immediately. Keep its already admitted probes owned until they finish; a late result cannot recreate the registration. |
| Observe | Concurrent observers of the same registration share one probe and receive separate result objects. Starting a fresh probe invalidates the previous live observation. Missing, unobserved, or closing checks cannot establish health. |
| Cancel one observer | Propagate that observer's cancellation while preserving work needed by other observers. Probe self-cancellation reports an unhealthy observation instead of cancelling the observer. |
| Cancel the last observer | Retire publication, request cancellation once, and retain cleanup. Further checks report unhealthy while that registration's old probe is still cleaning up; they do not start duplicate work. Suppressed cancellation cannot publish success. |
| Run synchronous work | Execute the callable off the event loop and retain its worker through completion. A cancelled factory's unstarted coroutine is closed; a returned running awaitable is cancelled and drained. Cancelling a request does not imply its thread stopped. |
| Finish a batch | Preserve this batch's completed measurements for unchanged registrations, even if a later request has already started another probe. Exclude removed registrations and use current observations or explicit unknown/unhealthy results for new or replaced registrations. An unexpected batch failure retires unfinished sibling observers while the checker retains their probes. |
| Publish a snapshot | Copy provider results before recording latency and copy observations before returning them, including nested details. Copy failures cannot expose an owner alias or leave a previous success as evidence of the failed probe. |
| Close | Close admission before awaiting cleanup. Concurrent/repeated callers share one shutdown operation; cancellation cannot detach it. Drain every admitted probe, including removed/replaced checks and synchronous workers, before reporting `closed`. |

The API's `/health` summary and components now describe the same returned
snapshot. `/ready` reads the current application's checker and uses the same
response deadline. Missing checks, failures, or deadline expiry report HTTP 503;
the existing degraded policy is preserved (`/ready` accepts degraded health,
while `/health` requires healthy). Liveness at `/healthz` remains independent.
Request expiry releases its observer promptly; cleanup continues under the
checker. Application lifespan shutdown closes the checker before training,
cache, database, and inference dependencies.

Use each checker on one event loop, and close it before closing that loop or
its dependencies. A closed checker is terminal; create a new checker for a new
lifecycle. Direct self-observation, a batch containing the calling probe, and
self-shutdown are rejected, including with eager task scheduling. Arbitrary
cycles across probes are not detected. A hung synchronous callable or
cancellation-resistant probe can delay shutdown; configure timeouts in its
underlying SDK. Unreturned, user-created background work remains outside the
checker's ownership. Replacing an application's entire checker directly also
requires explicitly closing the old owner.

[test_health_lifecycle.py](../tests/api/test_health_lifecycle.py) covers late
completion, registration changes, nested snapshot isolation, concurrent polls,
observer cancellation, cancellation suppression, synchronous factories, probe
self-cancellation, eager scheduling, shutdown ordering, and cancelled shutdown
waiters. [test_health_integrity.py](../tests/api/test_health_integrity.py) covers
application isolation, measured health, unavailable checks, and HTTP deadlines.
The API checker is separate from the monitoring, alerting, and evaluation owners
in A13; the advanced monitoring worker contract follows below.

## Advanced monitoring worker contract implemented here

`MetricsCollector`, `AlertManager`, and `AdvancedMonitoringService` in
`stateset_agents.core.advanced_monitoring` expose `MonitoringState` and
`get_status()`. Their background lifecycle is `new` → `starting` → `running` →
`closing` → `closed`; unexpected worker termination or startup/cleanup failures
produce `failed`. Failure does not mean cleanup has completed: inspect local
`active_workers` and each child's status, and call `aclose()` to drain ownership.

| Boundary | Ownership and outcome |
| --- | --- |
| Construct / record | Construction remains lazy for async workers. Passive metric recording and history access do not start workers and remain usable after shutdown. |
| Start | Concurrent callers join one startup operation; repeated start while running does not duplicate workers. Worker errors, unexpected return, and unexpected cancellation are retained. Failed or closing/closed instances cannot be silently restarted. |
| Cancel a startup observer | Finish admitted startup before propagating observer cancellation. Successful startup leaves an inspectable, running owner that the caller must close; another startup observer is unaffected. |
| Fail during startup | Drain all workers and children already admitted before returning the original failure. Startup and rollback outcomes remain owned; later close observes the same failure without replaying cleanup. |
| Report health | Health summaries include detached lifecycle diagnostics. An unstarted, stopped, or failed monitoring service cannot claim healthy background workers. Existing alert severity still determines health while workers are running. |
| Begin shutdown | Close admission immediately and join any admitted startup. Multiple close callers share the same operation; cancellation of a waiter cannot detach or repeat cleanup. |
| Drain | Stop the service's collection loop, then its alert manager, then metric retention, then release tracing resources. Join each worker's unwind. A child's failure does not skip independent cleanup of later children or tracing; repeated close reports the retained failure. A standalone alert manager borrows its collector and does not close it. |
| Stop polling | A collection/evaluation callback that suppresses cancellation cannot start another polling cycle after close. Cleanup errors propagate instead of entering a retry sleep. Blocking CPU sampling runs off-loop and is joined before its monitor can finish shutdown. |

Use one event loop and close monitors before that loop exits. Direct self-start
and self-close are rejected, including during startup and eager task scheduling;
arbitrary dependency cycles are not detected. User callbacks must eventually
finish cleanup. Synchronous custom collectors and notification handlers retain
their existing calling convention; they must avoid blocking the event loop.
The service owns its three async workers and its tracer's pipeline. Arbitrary
external calls to collection methods and background work detached by user
callbacks remain outside this boundary. `get_status()["tracing"]` exposes the
separate tracer lifecycle and retained SDK ownership.

```python
from stateset_agents.core.advanced_monitoring import AdvancedMonitoringService

monitor = AdvancedMonitoringService()
try:
    await monitor.start()
    await run_application()
finally:
    await monitor.aclose()
```

[test_advanced_monitoring_lifecycle.py](../tests/unit/test_advanced_monitoring_lifecycle.py)
covers all three owners, shared/cancelled startup and close observers, partial
startup rollback, dependency ordering, cleanup failures, stopped workers,
cooperative shutdown, blocking CPU sampling, and eager scheduling.

## Advanced tracing span and exporter contract implemented here

`DistributedTracer` in `stateset_agents.core.advanced_monitoring_tracing` exposes
`TracerState` (`ready`, `closing`, `closed`, `failed`) and `get_status()`. Local
admission, SDK availability, active observations, unconfirmed SDK span ends,
setup errors, and retained provider/processor/exporter handles are separate
facts. Local tracing remains available when the optional SDK setup fails.

| Boundary | Ownership and outcome |
| --- | --- |
| Configure SDK | Build a private provider and retain each returned resource as setup advances. Do not set the global provider or attach processors to another service's provider. Partial setup keeps local tracing available and retains cleanup ownership. |
| Start span | Copy caller tags before admitting the observation. Retain the SDK span before adding attributes or reading its context, so metadata errors cannot lose the handle. Explicit local parents propagate through the SDK context. Existing local UUIDs remain unchanged; tags expose SDK IDs separately. |
| Finish span | Preserve the first local completion time. Attempt the SDK end once, including after metadata failure and under reentrant finish calls. Ordinary SDK end errors do not replace an application outcome; their handles/errors remain inspectable and are reported by close. Control-flow exceptions propagate without making the end eligible for replay. |
| Exit trace scope | Record handled application errors and SDK error metadata; record cancellation separately. Finish on success, failure, or cancellation. Summary tags are detached from stored history. |
| Close admission | Reject new spans and share one close operation among concurrent/repeated callers. Finish still-open observations with `interrupted=True`; this does not cancel or declare completion of the application operations they observe. Later scope exit cannot rewrite that terminal observation. |
| Release SDK | After ending observations, run blocking SDK shutdown off-loop and retain ownership until it returns. Close the provider after confirmed processor attachment, otherwise the processor or exporter that actually owns cleanup. Uncertain attachment does not cause both provider and processor to repeat the same effect. |
| Cleanup failure | Retain failed span ends and incomplete resource handles. A provider cleanup failure or unconfirmed span end yields `failed`; later close observes the same outcome without replay. Cleanup of export resources still runs after span-end failure. |
| Process exit | Preserve a synchronous exit fallback for callers that have not begun explicit cleanup. Unregister it when explicit close begins. The fallback neither detaches nor repeats an explicit/failed shutdown; its outcome is also retained. |

Use one event loop for an instance's async lifecycle and close it before that
loop exits. `AdvancedMonitoringService.aclose()` closes its tracer after all
workers, including when worker cleanup fails. Prefer explicit close to the exit
fallback so failures can be handled while the application is running. A hung
exporter can delay cleanup; configure its own transport timeout. Resource
factories must clean up resources they allocate but fail to return. The tracer
cannot own an unreturned handle.

The SDK contracts are documented by the
[OpenTelemetry trace API](https://opentelemetry-python.readthedocs.io/en/latest/api/trace.html)
and [SDK tracing API](https://opentelemetry-python.readthedocs.io/en/latest/sdk/trace.html).
The synchronous exit fallback leaves the exit-handler registry unchanged,
avoiding the undefined behavior documented by the
[Python exit-handler contract](https://docs.python.org/3/library/atexit.html).
`closed` confirms local cleanup returned; it does not confirm delivery to a
remote telemetry backend. Exporters may report delivery failures internally.

[test_advanced_tracing_lifecycle.py](../tests/unit/test_advanced_tracing_lifecycle.py)
covers span completion, parent propagation, partial setup, metadata errors,
retained cleanup failures, cancellation, exit fallback, provider isolation, and
monitoring shutdown order.
[test_advanced_tracing_sdk.py](../tests/integration/test_advanced_tracing_sdk.py)
checks actual SDK export, parent/error metadata, and automatic process-exit
cleanup with local exporters and no network services.

## Utility monitoring worker and listener contract implemented here

`MetricsCollector` and `MonitoringService` in `stateset_agents.utils.monitoring`
preserve automatic constructor startup. Both expose `ThreadMonitorState`
(`running`, `closing`, `closed`, `failed`), `get_status()`, blocking `close()`, and
`aclose()`. The dashboard includes lifecycle diagnostics separately from health
probe results; neither observation silently restarts a failed worker.

| Boundary | Ownership and outcome |
| --- | --- |
| Start worker | Retain a named daemon thread before starting it. Repeated start cannot duplicate the worker; closing, closed, and failed owners reject restart. Unexpected termination records the failure and explicit target completion. |
| Stop admission | Signal stop and wake polling waits immediately. Concurrent sync/async close callers share one retained outcome. A worker, listener initializer, or active closer cannot close or replace its own owner. |
| Join work | Wait for the current sample/callback and confirm target completion before releasing dependencies. `aclose()` runs blocking cleanup off-loop and retains it through repeated caller cancellation. An interrupted join cannot authorize dependency cleanup. |
| Release dependencies | Join the service worker, wait for listener initialization, drain health probes, close the metric collector, then shut down the listener. Unconfirmed probe cleanup retains dependencies. Collector failure still permits a listener cleanup attempt. Listener shutdown, socket close, and serving-thread join follow the [Prometheus client contract](https://prometheus.github.io/client_python/exporting/http/). Shared metric collectors stay registered for sibling services. |
| Startup rollback | Close resources already returned before propagating construction failure. If rollback fails, `MonitoringStartupError.owner` retains the owner; process-control exceptions keep their type and retain it as `monitoring_owner`. A resource factory must clean up allocations it never returns. |
| Cleanup failure | Retain incomplete handles and errors. Further close calls observe the same failure without replaying effects. Older Prometheus clients returning no handles preserve startup but report unconfirmed cleanup. A known listener bind error remains distinct from an owned listener. |
| Global initialization/replacement | Reserve the transition under the registry lock, then construct/join outside that lock so callbacks cannot deadlock on it. Concurrent getters/setup receive an explicit transition error. Initialization failure is retained until explicit `setup_monitoring()` retries; replacement cannot discard an owner with failed cleanup. |
| Record/observe metrics | Bind declared Prometheus labels before updating a collector. Preserve metric names and summary fields. Copy incoming metrics and summary labels; protect local data updates and snapshots with the collector lock. Passive recording/history remains usable after close. |
| GRPO lifespan | Run monitoring cleanup in the final shutdown phase after engine cleanup. Remove the registered owner only after successful close. If later startup fails, drain monitoring rollback before propagating the original exception. Failed rollback retains the owner. |

Prefer `await monitor.aclose()` inside an event loop and `monitor.close()` in
synchronous teardown. Thread shutdown cannot forcibly interrupt a hung sampler
or callback; those operations need their own timeouts. Listener ownership covers
the returned server and serving thread, not arbitrary SDK request callbacks.
Utility `HealthCheck` executor probes follow the owned cleanup contract below.
Direct mutation of public metric collections is outside the synchronized
recording/snapshot methods.

GRPO still respects its shared shutdown deadline. A prior critical failure or
exhausted budget can prevent the final phase from running; the monitor stays
registered in that case. The startup rollback above covers monitoring, not all
other GRPO resources allocated before startup completes.

[test_monitoring_thread_lifecycle.py](../tests/unit/test_monitoring_thread_lifecycle.py)
covers repeated start/close, concurrent cleanup, cancellation, worker failure,
partial startup, listener initialization races, failed global transitions, and
metric snapshot isolation.
[test_monitoring_prometheus_lifecycle.py](../tests/integration/test_monitoring_prometheus_lifecycle.py)
checks actual scrapes, shared labeled collectors, listener-thread exit, and port
reuse through direct close and global replacement.
[test_grpo_shutdown_ownership.py](../tests/api/test_grpo_shutdown_ownership.py)
checks monitoring teardown order and retained startup rollback under cancellation.

## Utility health probe contract implemented here

`HealthCheck` and `HealthChecker` in `stateset_agents.utils.monitoring` expose
`UtilityHealthState` (`ready`, `closing`, `closed`, `failed`), `get_status()`,
`close()`, and `aclose()`. These utility owners remain separate from the API
health checker. Existing result keys and measured healthy/unhealthy statuses
remain; `unavailable` explicitly describes missing configuration, retired probe
cleanup, or stopped admission, and `cancelled` describes a retired execution.

| Boundary | Ownership and outcome |
| --- | --- |
| Admit observation | Capture the caller's finite, nonnegative timeout before effects. Concurrent calls on the same loop share one probe. Name, criticality, and callback belong to that probe; later configuration changes apply to a subsequent probe. Each caller receives its own result dictionary. |
| Invoke dependency | Support sync functions, async callable objects, and sync functions returning awaitables. Retain the dedicated sync executor and join it off-loop. Use monotonic time for durations. A dependency's own `TimeoutError` is an error observation, separate from the observer deadline. |
| Retire observer | A deadline returns `timeout` promptly; caller cancellation propagates. Retire the probe when its last observer leaves, but keep the active slot through sync execution and async cleanup. New observers receive `unavailable` while retired cleanup is pending, preventing overlapping replacements. A remaining observer keeps the shared probe alive. |
| Handle late awaitables | Retrieve sync results even after cancellation. Close an unstarted returned coroutine or cancel/drain an already-started awaitable before releasing ownership. Cancellation during executor shutdown cannot discard that returned object. |
| Close ownership | Stop admission, request async cancellation once, and wait for actual probe completion and executor shutdown. Concurrent sync/async close calls share one outcome; repeated caller cancellation cannot detach close. A probe cannot await or close its own check, registry, or monitoring service. Sync close on a probe's running event loop directs callers to `aclose()` before changing admission. |
| Handle cleanup failure | Retain the executor/owner and failed outcome without automatically replaying cleanup. A stopped loop with pending work cannot establish successful cleanup. Monitoring retains dependent resources when health cleanup is unconfirmed. |
| Snapshot registration list | Batches capture their registration list. The checker retains ownership of registered or observed checks even when the public list later removes them; close drains these retired owners too. Registration through `add_health_check()` is rejected after admission closes. |
| Report availability | A `None` callback produces `unavailable` without executing a placeholder. Default database/cache checks require real callbacks; missing `psutil` makes system measurements unavailable. Empty or closed checkers cannot establish health. An unavailable critical check keeps the existing aggregate `critical` policy; noncritical failures produce `degraded`. |

Configure the existing public definitions with actual dependency callbacks, then
close the service before closing their event loop or dependencies:

```python
from stateset_agents.utils.monitoring import MonitoringService

monitor = MonitoringService(enable_prometheus=False)
for check in monitor.health_checker.health_checks:
    if check.name == "database":
        check.check_func = check_database_connection
    elif check.name == "cache":
        check.check_func = check_cache_connection
try:
    health = await monitor.get_health_status()
finally:
    await monitor.aclose()
```

An observer timeout does not impose an execution deadline on a blocking callback.
Use timeouts in the dependency itself: a hung callable or cancellation-resistant
async cleanup can delay shutdown indefinitely. Keep the owning loop alive until
`aclose()` returns. Closing a short-lived loop is not an alternative to draining
timed-out work. Idle checks may subsequently run on another loop; an active probe
cannot be observed from a different loop.

[test_utility_health_lifecycle.py](../tests/unit/test_utility_health_lifecycle.py)
covers coalescing, distinct observer/probe timeouts, concurrent cancellation,
sync/async cleanup ordering, returned awaitables, executor cleanup failures,
registration snapshots, missing dependencies, invalid configuration, task
creation failure, eager scheduling, and stopped admission.

## Continuous evaluation lifecycle contract implemented here

`ContinuousMonitor` exposes `ContinuousMonitorState` (`idle`, `running`,
`stopping`, `closing`, `closed`, `failed`) and `get_status()`.
`AdvancedEvaluator` exposes `EvaluatorState` (`ready`, `closing`, `closed`,
`failed`) and reports its foreground and monitor ownership separately. The
states are exported from `stateset_agents.core.enhanced` as well as the source
module. Use one event loop for each owner.

| Boundary | Ownership and outcome |
| --- | --- |
| Managed start | `await monitor.start(agent, environment)` retains one worker before callbacks run. Repeating the same object identities shares that worker. A different agent or environment drains the previous generation before starting the next. Invalid intervals/agent interfaces fail before starting work. |
| Long-running compatibility | `start_monitoring()` remains a long-running coroutine. Repeated calls share a worker; cancelling a runner stops and drains its selected generation. Another runner for that generation also finishes. A late cancellation from an old generation cannot stop its replacement. Use `start()` for a lifetime controlled independently of a waiting caller. |
| Replace/stop | A committed transition finishes before caller cancellation propagates. Cancellation of a replacement observer can therefore leave the replacement running under monitor ownership. `stop_monitoring()` joins the current worker and returns to `idle`; a later explicit start is allowed. A queued start cancelled before it obtains admission has no effect. |
| Close | `aclose()` closes admission immediately, then shares one retained cleanup operation. Close racing with replacement prevents the new worker from starting. Repeated cancellation of close observers cannot detach callback cleanup. Closed/failed owners do not silently restart. |
| Worker outcome | Unexpected termination is retained as `failed`, including external cancellation. Errors from cancellation cleanup cannot be converted into ordinary error samples. Existing handled sample/alert errors keep their retry policy while the worker is accepting work. |
| Observation publication | Stop fences late callback results before they reach history or alerts. Stored metric/alert data is copied, history retains its 1,000-entry limit, and each entry includes the selected generation. Durations use monotonic time; timestamps retain their existing wall-clock format. |
| Foreground batch | Snapshot agents/configuration and serialize comprehensive batches. Drain the previous monitor before invoking the suite. On success, `include_monitoring=True` selects an agent by result position only when every candidate has comparable quality evidence; ties choose the first candidate deterministically without claiming a unique winner. Duplicate names remain distinguishable. False or unavailable evidence leaves monitoring stopped. Empty batches return an empty list. Cancelled/failed batches leave monitoring stopped. |
| Evaluator shutdown | Stop admission and cancel active/queued batches once, drain their cleanup, then close the monitor. Report cleanup failure without replay. Returned results and captured test cases are detached from history; completed per-agent history remains available if a later stage fails. |
| Reentrant callbacks | Callback context propagates into child tasks. A callback cannot start/stop/close its own monitor or recursively evaluate/close its evaluator, avoiding cycles in which cleanup waits for its own caller. |

Managed usage keeps the evaluator alive for the desired monitoring lifetime:

```python
from stateset_agents.core.enhanced import AdvancedEvaluator

evaluator = AdvancedEvaluator()
try:
    results = await evaluator.comprehensive_evaluation(
        agents, environment, {"include_monitoring": True}
    )
    # Use the results and continue serving while the selected agent is monitored.
finally:
    await evaluator.aclose()
```

These owners do not close agents or environments they borrow. Close evaluation
before releasing those dependencies. A cancellation-resistant callback can delay
shutdown; the owner cannot forcibly interrupt arbitrary agent code. Ownership
also cannot cover work a callback starts and detaches without returning a handle.
Passive history access does not restart execution.

[test_continuous_evaluation_lifecycle.py](../tests/unit/test_continuous_evaluation_lifecycle.py)
covers selection, repeated starts, replacement/close races, legacy cancellation,
late publication, retained failures, foreground serialization and cleanup,
configuration snapshots, reentrant callbacks, rejected task creation, and eager
scheduling. The shared drain helper also closes an unstarted coroutine when task
creation fails, covered by [test_async_calls.py](../tests/unit/test_async_calls.py).

## Evaluation evidence contract implemented here

[EvaluationMetrics](../stateset_agents/core/enhanced/advanced_evaluation_metrics.py)
keeps its numeric fields, defaults, constructor, and existing import paths.
`MetricEvidenceStatus` and immutable `MetricEvidence` are exported from
`stateset_agents.core.enhanced`. Evidence records the producer's declaration;
it does not independently validate the measurement method.

| Boundary | Evidence and outcome |
| --- | --- |
| Construct or assign | Numeric values alone remain `unavailable`. A default safety/recovery score of `1.0` is not an observation. |
| Record | `record(name, value, source=..., status=..., sample_size=..., scope=...)` validates the value and metadata before replacing either. Status is `measured` or `inferred`; the method, optional support size, and scope travel with the captured value. |
| Change or retire | Changing a recorded value invalidates its evidence. Restoring the old numeric value does not restore evidence; explicitly record it again. `mark_unavailable(name, reason)` withdraws evidence while preserving the numeric attribute. |
| Capture cases | Each suite captures its definitions before agent callbacks run, copies conversation inputs, and validates positive run counts and unique nonempty names. Quick comparison honors its positive `num_tests` case limit. |
| Aggregate | Raw runs take precedence over supplied summaries. Complete recorded latencies are `measured`; supplied summaries and heuristic response-quality/session-count proxies are `inferred`. Missing, invalid, or partial observations remain unavailable. A real zero is valid evidence. Latency moments avoid intermediate overflow. |
| Compare | A metric participates only when every candidate has evidence with matching status, method, support, scope, and captured cases. Normalize existing weights over those shared metrics; expose included/excluded fields. No shared evidence means no composite score or rank. This is a descriptive score, not a statistical superiority test. |
| Select | Report/monitor quality selection requires comparable quality evidence across the entire cohort. Preserve all tied result indices and duplicate labels. Reports do not claim a unique winner for a tie. |
| Publish | Reports use `schema_version: 2`, with JSON `null` for unsupported values, unavailable duration, and no fabricated quality interval. Serialize before opening the destination so invalid metadata cannot truncate an existing report. File writes themselves are not transactional. |

For callers supplying measurements, record the method alongside the value:

```python
from stateset_agents.core.enhanced import EvaluationMetrics

metrics = EvaluationMetrics()
metrics.record(
    "response_time", 0.25,
    source="request_latency_seconds/v1", sample_size=10, scope="benchmark-v1",
)
assert metrics.evidence("response_time").status.value == "measured"
assert not metrics.evidence("safety_score").available
```

`metrics.to_dict()` retains legacy numeric groups and adds `provenance`.
`metrics.to_dict(include_unavailable_values=False)` produces the evidence-only
projection used by reports and comparisons. Existing code reading numeric
attributes continues to work; consumers of generated reports must handle the
version 2 nullable values and explicit `selection_status`. Constructor-supplied
scores must be recorded with their method to participate in comparisons.
Unique algorithm lookup keys remain available; the `candidates` list is the
authoritative view when labels repeat or collide with metadata keys.

The built-in response rubric remains a heuristic. Its inferred quality score
does not establish measured safety, reasoning correctness, or human preference.
`standard_deviation` describes case-average latency, not uncertainty in quality;
the quality confidence interval stays unavailable until a suitable estimator
supplies it. This change is scoped to the legacy enhanced evaluator, separate
from the training/evaluation evidence pipelines.

[test_evaluation_metric_evidence.py](../tests/unit/test_evaluation_metric_evidence.py)
covers defaults, stale evidence, invalid values and metadata, incomplete data,
raw/summary conflicts, comparability, ties, duplicate labels, report serialization,
zero/extreme latencies, bounded quick comparison, and case snapshot isolation.

## Automated rubric assessment contract implemented here

[EvaluationRubric](../stateset_agents/core/enhanced/advanced_evaluation_rubric.py)
is exported from `stateset_agents.core.enhanced`. Use
`suite.rubric.capabilities()` to inspect registered characteristic checks and
their method identifiers. The defaults assess greeting-word and polite-phrase
presence plus case-insensitive expected-answer substrings. Word count is no
longer presented as a factuality check. These are heuristics, not semantic,
safety, or human-preference judges.

| Boundary | Assessment and outcome |
| --- | --- |
| Prepare | Validate all expected-characteristic lists and expected answers before the first agent call. Capture cases and rubric registrations for each suite invocation; concurrent invocations retain independent captures. |
| Check | Each criterion publishes `passed`, `failed`, `unavailable`, or `error`, with a nullable verdict and method/reason. Missing registrations and checker abstentions are unavailable. Invalid checker outputs and ordinary checker exceptions are errors. Process-control exceptions propagate. |
| Aggregate a response | An assessment is `assessed` only when every configured criterion has a verdict. Empty or incomplete rubrics do not establish quality. Preserve the numeric characteristic-ratio/answer-floor score for existing readers, but mark its availability; the case verdict is `null` when unsupported or errored. Greeting/polite checks use word boundaries to avoid accidental matches inside other words. |
| Summarize runs | `pass_rate` is `null` unless all runs were assessed. Separate passed, failed, unavailable, and error counts. `assessed_pass_rate` describes only the assessed subset and cannot establish a whole-suite quality score. Generation failures are execution errors, not completed quality assessments. |
| Compare and select | Quality evidence includes the applied criterion methods and aggregation policy. Different captured methods cannot establish comparable quality even with identical case definitions. Incomplete quality leaves report/monitor winner selection unavailable; latency remains independently usable. |
| Publish | `EvaluationResult.test_results` retains detached per-case summaries and per-run assessments. Version 2 reports include these reasons and replace unsupported rubric scores with `null`, as they do unsupported metric values. |
| Cancel | Each public suite invocation owns its worker. Retire admission before cancellation, drain callback cleanup, and prevent later runs or publication even when a callback returns after cancellation. Cleanup errors escape the ordinary per-case error path and remain visible to the owning evaluator. Repeated caller cancellation cannot detach cleanup. |

Register fast, synchronous, deterministic checks for additional criteria:

```python
from stateset_agents.core.enhanced import AutomatedTestSuite

suite = AutomatedTestSuite()
suite.rubric.register_criterion(
    "asks_question", lambda response: "?" in response,
    source="question_mark_presence/v1",
)
suite.test_cases = [{
    "name": "follow_up", "input": "Help me plan a trip.",
    "expected_characteristics": ["asks_question"],
}]
```

A checker returns `True`/`False`, or `None` to abstain. It must not perform I/O,
start background work, or change behavior during a run; registrations capture
the callable and its declared source, not the callable's mutable internals.
Async checkers are rejected. Use the separate evaluation/judge pipelines for
model-based judging rather than hiding those operations inside a predicate.

The shipped broad suite still requests reasoning, privacy, and other criteria
without default checkers. Its complete quality score is therefore unavailable
until callers supply appropriate checkers or select a fully supported case set.
Consumers of raw suite results must handle nullable `passed` and `pass_rate`
and use the explicit status/counts to distinguish errors from failed checks.
The original case definitions and numeric metric attributes remain available.

[test_evaluation_rubric.py](../tests/unit/test_evaluation_rubric.py) covers coverage,
real failed checks, unsupported factuality, checker errors/abstentions, validation
before effects, captured methods under concurrency, comparison eligibility,
detached report diagnostics, custom complete rubrics, and cancellation cleanup.

## Backend initialization contract implemented here

`BackendInitializationState` is exported from `stateset_agents.core.agent` and
`stateset_agents.core`. `agent.get_initialization_status()` describes the last
managed backend attempt, its phase/error type, the adapter request/outcome, and
whether current component identities match the completed attempt. Observation
does not load, attach, retry, or reset resources.

| Boundary | Ownership and outcome |
| --- | --- |
| Initialize | Serialize synchronous initialization on the same agent, including separate calling threads. Reject reentrant initialization before changing the active attempt. Backend loading and the multi-turn adapter hook complete inside one boundary. Async subclass dependencies join through `_initialize_extensions`; overlapping initializers are rejected while that hook runs. |
| Publish readiness | Model, tokenizer, and generation configuration must be present and synchronous before `ready`. Missing components fail explicitly. Async custom loader results are rejected; unstarted owned coroutines are closed. Injected backend getters are read and validated before assigning their components. |
| Repeat | Identical stub configuration retains the model, tokenizer, generation configuration, and response cursor. A changed stub request can explicitly replace the stub on a later `initialize()`. The same injected backend does not overwrite its attached wrapper or an explicitly supplied replacement model. |
| Resume setup | Completed base loading steps remain available after a later setup failure. Missing adapter files/dependencies or invalid adapter configuration fail before attachment; after repair, an explicit retry can continue without reloading completed components. |
| Attach | Capture the adapter request separately from mutable SDK arguments. Retain input/output identities and attach once. Repeated initialization restores/reuses the recorded output, including when the input model was mutated in place. A different adapter request on that model requires a fresh model or agent. |
| Uncertain effect | Once the SDK attachment call begins, any exception, cancellation, or missing output leaves a retained failure. Never repeat attachment on that recorded input. Recovery requires a caller-supplied fresh model or agent; the caller owns freshness and compatibility of replacements. |
| Async extension | Release thread locks before awaiting dependency setup. Keep the agent initializing until the owned extension finishes. Cancellation requests cleanup once, drains it despite repeated cancellation, then returns the original failure; suppressed cancellation cannot publish readiness. Backend component replacement during the hook is rejected. |
| Generate | After a managed attempt has failed, or while it is initializing, reject prompt rendering, native/batched/streaming generation, and rollout calls before inference effects. Enhanced generation checks the backend before invoking memory/reasoning helpers. |
| Compatibility | `initialize()` remains an async method with synchronous loader hooks. Existing manual model/tokenizer/configuration assignment remains supported; `new` means no managed attempt, not an independently verified ready backend. Copies/pickles of completed agents recreate the initialization lock and preserve component identity relationships. |

```python
await agent.initialize()
status = agent.get_initialization_status()
assert status["state"] == "ready"
assert status["backend_matches"]
await agent.initialize()  # Reuses the loaded components and attached adapter.
```

The status covers model, tokenizer, generation configuration, the owned
adapter step, and registered async prerequisites. The enhanced agent's memory
setup uses that hook. Other work performed after a subclass's super-call and
unrelated services retain their own readiness. `backend_matches` compares handles,
not weight contents or
configuration compatibility. Direct training/checkpoint replacement remains a
caller-owned operation; stop concurrent use before changing components. The
initialization lock serializes initializers, not arbitrary model mutation or
training. No automatic cleanup of borrowed models or backend resources is added.

Loader calls retain their existing synchronous behavior and can block the event
loop. Cancellation cannot forcibly interrupt a synchronous SDK call; if a hook
raises cancellation after attachment begins, its uncertain effect is retained.
This contract does not turn a failed status into permission to retry externally
owned effects or to reuse an already-mutated model as a fresh replacement.

[test_agent_initialization_lifecycle.py](../tests/unit/test_agent_initialization_lifecycle.py)
covers repeat calls, stub progress, injected getter failures, partial setup,
missing components, adapter validation/attachment failures, request snapshots,
fresh-model recovery, generation fences, parallel/reentrant initialization,
copying/pickling, async loader rejection, and a real local LoRA model whose
parameter objects remain unchanged across repeated initialization.

## Conversation reset and operation ownership

`Agent.reset()` now advances a local conversation epoch and clears history, turn
count, and the last reasoning trace under one lock. A generation started before
that reset cannot append its response to the new history. Responses, turns,
batches, streaming, `process_turn`, enhanced preparation/selection, tool loops,
and structured-output retries retain the epoch of their original invocation.
Nested calls share an owner; concurrent independent requests remain supported.

```python
from stateset_agents.core.agent import ConversationResetError

await agent.reset()
try:
    old_response = await pending_response
except ConversationResetError:
    # Display the new conversation instead of inserting a retired answer.
    pass
fresh_response = await agent.generate_response("Start again")
status = agent.get_conversation_status()
```

| Boundary | Behavior |
| --- | --- |
| Reset | Advance the epoch immediately; it does not wait for or claim cancellation of provider/tool work already admitted. |
| Late response | Raise `ConversationResetError` before changing history, turn count, or the current last reasoning trace. Do not turn invalidation into an empty successful answer or a fallback model request. |
| Batch continuation | Reject later sequential samples and candidate selections after reset. An old candidate that was already stored retains its historical receipt but cannot be selected in the new conversation. |
| Stream | Check the captured epoch before delivering another chunk and before history publication. Join source/worker cleanup on error or close; caller cancellation remains distinct from explicit close. |
| Enhanced memory write | A write admitted before reset may finish and report `stored`. That acknowledgement is retained even when the enclosing generation fails due to reset; subsequent bookkeeping cannot mutate the new conversation. |
| Tool loop | Already-started handlers may finish. Do not start a later tool or model round after reset. Parallel handlers are joined before the owning round exits, including cancellation cleanup. |
| Structured output | A reset error is not a schema-validation retry. Even a custom generator's late output is checked before validation or another generation attempt. |
| Custom hooks | Persona previews run outside owner locks and publish only into their original conversation. A hook can inspect status from a helper thread without deadlocking. Arbitrary external effects performed by hooks retain their own ownership obligations. |

`get_conversation_status()` returns detached `epoch`, `active`, and
`last_finished` data. Each operation has an ID, mode, captured epoch, terminal
state/error, and an `invalidated` flag relative to the current epoch. An active
operation can remain `running` and invalidated while its previously admitted work
settles. Historical completion remains completion after reset; invalidation
prevents reuse and does not rewrite that outcome. Enhanced status includes this
view under `conversation`, and generation/candidate metadata includes the captured
`conversation_epoch`. These are process-local receipts, not durable session IDs.

Reset preserves existing long-term memory, persona, aggregate metrics, and
reasoning history. It does not roll back an external action or erase already
stored knowledge. Keep awaiting or explicitly closing old work to finish cleanup;
an uncooperative provider can delay its own termination, while new conversation
requests can proceed. Copying/pickling an agent with an active conversation
operation is rejected; idle copies and older serialized objects remain supported.
`get_history()` and `add_to_history()` detach nested history data from callers.

Built-in wrappers use `conversation_call` for coroutines and `conversation_stream`
for async generators. Custom overrides that perform work outside a `super()` call
can use these decorators to retain the same boundary. Streaming scopes live in an
owned producer task so they do not leak into the consumer or fail when another
task closes the stream. A detached child inheriting a finished operation cannot
reuse its authorization to start more conversation work. Standalone structured
output/function-call mixins without an Agent owner retain their existing usage.

Function-call responses are decoded as complete JSON objects, including nested
objects and escaped argument strings, instead of regex fragments. Fenced JSON,
prose prefixes, and standalone function objects remain supported. Invalid
envelopes and duplicate keys/IDs cannot dispatch partial calls; an incomplete
enclosing object never authorizes an inner function fragment. Argument validation
still runs before each handler. The simple `ToolAgent` parser also uses the JSON
decoder's absolute end offset so a prose prefix cannot hide trailing text.

[Conversation tests](../tests/unit/test_conversation_lifecycle.py) cover reset
during generation, streaming, tool execution, memory writes, and persona hooks;
parallel work, retained acknowledgements, cancellation, serialization, and fresh
requests after reset. [Parser tests](../tests/unit/test_function_call_parsing.py)
exercise actual nested and escaped envelopes. Run the
[offline reset example](../examples/conversation_reset.py) for a small UI flow.
Coordinated durable conversation checkpoints remain a separate requirement.

## Named conversation turns and recovery

The named API in [multiturn_agent.py](../stateset_agents/core/multiturn_agent.py)
owns both user and assistant turns in `ConversationContext`. It is distinct from
the model-oriented `MultiTurnAgent` in `core.agent` and the enhanced-agent session.
GRPO's first-message handler now records the assistant reply once; generation
already owns that append.

`continue_conversation` and `generate_multiturn_response` admit one turn per ID.
Another turn on that ID raises `ConversationConflict` before accepting its input;
different IDs can proceed independently. Both GRPO HTTP chat routes return 409
for these conflicts and for work invalidated by end/reset. Clients should inspect
the conversation before submitting another turn, rather than blindly resubmitting
an operation that might have invoked tools.

Each admitted turn prepares a detached candidate. Context updates, optional
compression, and the user message publish before generation; if generation fails,
the accepted user input remains. Analysis, strategy changes, and the assistant
reply publish together only while the same context, operation, and revision remain
current. Managed `add_turn`/context updates during generation cause a conflict
instead of being overwritten. Concurrent direct edits to public nested fields or
the public registry bypass this guard and are unsupported. Returned histories and
status dictionaries are detached.

Ending a conversation removes registry membership and retires its context. Already
started callbacks may settle, but cannot publish a late assistant turn or start the
next managed tool call. Agent `reset()` fences all existing work while retaining
the named registry; use `end_conversation(id)` to end an individual named context.
Cancellation joins callback cleanup, retains ownership through repeated cancellation,
and fences nested generation even if a callback suppresses cancellation. Custom
callbacks remain responsible for their own external effects and background tasks.

`get_conversation_status(id)` or `context.get_status()` reports the context revision,
active/latest turn, and snapshot receipts. A turn's `generation_status` and
`tool_status` distinguish `not_started`, `unconfirmed`, and `returned`. A callback
return does not establish that every external effect succeeded. Failed/cancelled
receipts preserve the phase, exception type, and whether user input was recorded;
they do not authorize automatic replay. Copying a context or agent with owned work
in progress is rejected.

```python
from stateset_agents.core.multiturn_agent import MultiTurnAgent

agent = MultiTurnAgent({}, response_backend=lambda prompt: "Hello again.")
context = await agent.start_conversation(user_id="example")
await agent.continue_conversation(context.conversation_id, "Remember order 42")
saved = await agent.save_conversation(context.conversation_id, "conversation.json")

recovered = MultiTurnAgent({}, response_backend=lambda prompt: "Order 42 is in context.")
context = await recovered.restore_conversation(
    "conversation.json", expected_revision=saved["revision"]
)
await recovered.continue_conversation(context.conversation_id, "What was my order?")
```

Snapshots are explicit point-in-time saves, not an automatic per-message journal.
An active turn cannot be captured. Once capture finishes, later turns may proceed
while the detached payload is written. Version 1 includes all canonical context
fields, original timestamps, the context revision, and the last settled turn
receipt. Agents with a built-in planning manager write version 2, adding its
settled plan, step statuses/notes/timestamps, counters, revision, and last planning
operation receipt. Version 1 files remain readable and agents without planning
continue writing that format. Replacing a version 1 file with a version 2 save
requires its current file revision, like any other update. Context data must be
finite JSON values; unsupported/custom context
representations fail explicitly. The bounded reader rejects unknown schemas,
duplicate keys, invalid receipts, and checksum mismatches before registry mutation.
The SHA-256 checksum detects accidental corruption; it does not authenticate data.

`save_conversation` is create-only by default. To replace a file, supply its current
`expected_revision`. It shares the [snapshot publisher](../stateset_agents/utils/snapshot_files.py)
with vector memory: cooperating writers lock a stable sibling file, sync a same-directory
stage, replace the target, then sync the directory where supported. See the
[memory snapshot durability contract](#durable-memory-snapshots) for filesystem,
directory trust, and advisory-lock limits. File replacement, file sync, and
directory sync remain separate facts in `get_conversation_snapshot_status()`.
Cancellation drains file work and may leave a published snapshot. Inspect its
revision and receipt before retrying; never assume cancellation means no file change.
Snapshot receipts include local paths and should not be exposed unfiltered to
remote clients.

Restore reads and validates off-loop, checks the optional requested file revision,
and attaches one fresh context only if its ID is absent. Reset/cancellation while
reading prevents attachment. Restore never replaces live context or invokes models,
strategies, or tools; a following turn uses the destination agent's configuration.
Failed/cancelled turns remain historical receipts, not pending jobs. The source
model name is informational. `inspect_conversation_snapshot` validates and returns
metadata without attaching a context.

Planning state is captured under the manager's lock alongside the context; an
active planner operation rejects capture. Restore validates and prepares both
owners before attaching either under their locks. Existing planning state at the
same ID, a changed destination manager during reading, or a planning snapshot with
no destination manager causes an explicit failure. Restoring a version 1 file into
a built-in planning manager creates no plan and invokes no planner. Destination
configuration, planner, and goal extractor remain unchanged; callbacks are never
loaded from JSON.

Backend weights, custom strategy/tool implementations, enhanced memory/reasoning/
generation receipts, and GRPO's separate user/access-control registry are outside
this format. Custom planning-manager, plan, and step subclasses are rejected
instead of silently dropping additional state. This API does not provide automatic
gateway restart recovery or a coordinated enhanced-session checkpoint.

The [offline example](../examples/durable_conversation.py),
[turn tests](../tests/unit/test_named_conversation_lifecycle.py), and
[snapshot tests](../tests/unit/test_conversation_snapshot.py) cover continuing from
recovered prompts, failed/cancelled receipts without replay, overlap, end/reset,
revision conflicts, invalid files, cancellation during file work, and process death
on either side of replacement. The shared memory tests retain filesystem failure
and cleanup coverage.

## Planning commands and preparation ownership

The built-in `PlanningManager` admits one preparation per conversation ID.
`get_status(id)` reports its revision, counters, and active/latest preparation
receipt. Inputs, policy, and the existing plan are copied before callbacks run;
callbacks do not hold the state lock. A candidate plan, counters, and completion
receipt publish together only if the admitted revision and original plan remain
current. Managed updates advance the revision; clear/end invalidates old work
without dropping ownership of a callback still in progress. Other IDs can proceed.
When called through an agent, admission, subsequent planner callbacks, and plan
publication also check that agent operation's reset epoch. Named turns additionally
check their context revision. Reset or a conflicting context edit therefore rejects
late planning results as well as late responses; reset preserves existing plans.
The existing `get_plan` return remains mutable; concurrent direct mutations are
unsupported, and callers should use `apply_update` for guarded updates.

`callback_status` distinguishes `not_started`, `unconfirmed`, and `returned`.
A returned planner result does not establish external-effect success or completion
of a real-world task. Preparation failure preserves the previous plan and counters;
copying an active manager is rejected. Planner and goal-extractor callbacks remain
synchronous, and custom callbacks own any external effects or background work.

Named conversation turns consume `plan_update` and `plan_goal` from the accepted
context before invoking planning. If planning succeeds and response generation
later fails or is cancelled, the next turn retains the updated plan without
advancing it again. If planning itself fails, the previous plan remains and the
command is not silently queued for retry. Inspect the planning receipt and submit
a new explicit update only after deciding how to handle the earlier outcome.
Planning state and response completion are deliberately separate facts.

The [planning recovery example](../examples/durable_planning.py),
[planning lifecycle tests](../tests/unit/test_planning_lifecycle.py), and
[planning snapshot tests](../tests/unit/test_planning_snapshot.py) cover failed
response/command replay, clear during callbacks, competing updates, fresh-process
recovery, preserving destination callbacks/configuration, version upgrades, and
rejecting partial attachment on malformed or interrupted reads.

## Canonical gateway conversation ownership

The [gateway owner](../stateset_agents/api/conversation_lifecycle.py) is shared by
the agent service and `/api/v1/conversations`. Their stores and existing history
policies remain separate. Each owner admits one request per conversation key;
different IDs can progress concurrently. Authorization precedes admission and
inspection. The agent service binds planning context to the actual conversation
ID and authenticated principal rather than accepting an alternate ID in context.

| Event | Record and request outcome |
| --- | --- |
| Admit | Capture detached input/history and the current record, agent, and principal. Reject a competing request with HTTP 409 before invoking generation. |
| Generate | Mark `generation_status=unconfirmed` before invoking the callback; only a returned callback sets `returned`. |
| Publish | Check record identity, lifetime, ownership, and unchanged baseline history. Prepare response validation and copies before publishing the assistant message and metadata. |
| Cancel | Fence subsequent owned agent effects, cancel the callback once, and retain admission through cleanup even if the caller cancels repeatedly. A callback that suppresses cancellation can finish; its answer is not published. |
| Delete or expire | End/remove the record without releasing unfinished request ownership. Late work cannot append an answer, refresh v1 access time, or publish an owned built-in plan. Recreating an ID does not transfer the old request to the new record. |
| Service cleanup fails | Retain the ended conversation or retiring agent for explicit cleanup retry; reject new generation. Concurrent conversation deletions cannot repeat cleanup while an earlier deletion owns it. |
| Read history/status | Return detached data under the owner's lock and apply the existing access rules. |

The agent service records accepted user messages before generation, preserving
them if generation fails. The v1 route records incoming messages and the answer
together after success, preserving its previous success-only history policy.
An already-recorded response is not rolled back if subsequent work fails;
`response_recorded` remains the independent publication fact.

Inspect `/conversations/{id}/status` for agent-service conversations, or
`/api/v1/conversations/{id}/status` for v1 conversations. Each returns
`conversation_id`, record `state`, `revision`, `active`, and `last_finished`.
An active/latest receipt includes:

- `state`: `running`, `completed`, `failed`, or `cancelled`.
- `phase`: `preparing`, `generating`, `committing`, or `finished`.
- `user_recorded`, `response_recorded`, and `cancellation_requested`.
- `generation_status`: `not_started`, `unconfirmed`, or `returned`.
- `error_type` and `work_error_type`: exception class names, never provider error
  text or conversation content.

Status does not replay work. After HTTP 409, inspect the owning request and
history before deciding what to submit. `unconfirmed` does not establish that a
provider performed no work; `returned` does not independently verify external
effects. Deleted or expired records return HTTP 404 rather than retaining a
globally queryable tombstone. The agent-service status route preserves existing
administrator visibility; v1 preserves its existing principal rules.

Core agent checks inherit the host guard, including in built-in planning worker
threads and child tasks. A finished request's escaped task cannot retire a newer
request. Lock order is agent → gateway → planning; deletion releases the gateway
lock before calling planning cleanup. Custom callbacks remain responsible for
their own external effects and background work. Cancellation-resistant callbacks
can delay cleanup, so configure deadlines in their underlying clients.

Default ownership is in memory within a process. Settled `_lifecycle` receipts
are plain data attached to records. The v1 journal below adds explicit durable
history and principal recovery; the agent-service store remains in memory.
Mutating public backing dictionaries directly is unsupported. Existing
named-conversation snapshots do not restore these gateway stores. Coverage lives in
[test_conversation_ownership.py](../tests/api/test_conversation_ownership.py).

## Durable v1 gateway conversations

`API_V1_CONVERSATION_DB` opts `/api/v1/conversations` into the
[versioned journal](../stateset_agents/api/conversation_store.py). Default behavior
stays in memory. The native OpenAI-compatible `/v1/chat/completions` route remains
stateless, and the local `AgentService` retains its separate history/planner owner.

The journal uses short SQLite transactions with full synchronous acknowledgement.
A stable advisory lock per conversation owns the request across provider waits
and cancellation cleanup. Workers sharing one canonical local path cooperate
without holding a SQLite write transaction during generation. Database revisions
fence publication after deletion, expiry, eviction, or replacement. Unrelated
conversation IDs can progress concurrently; brief SQLite lock contention surfaces
as HTTP 503 rather than blocking the event loop or replaying writes.

| Boundary | Durable fact and recovery behavior |
| --- | --- |
| Admit | Authorize before exposing request ownership. Persist the principal and active receipt before generation is eligible to start. |
| Dispatch | Persist `generation_status=unconfirmed` before calling the provider. Failure to acknowledge this write prevents dispatch. |
| Provider returns | Persist `returned` independently of answer publication. A later interruption may still have no recoverable answer. |
| Publish answer | Write trimmed history and `response_recorded=true` together; only then finish the request receipt. |
| Publication acknowledgement fails | Stop further writes from that attempt and report failure. Later reads reconcile committed data; neither retries nor fabricated successful acknowledgements are introduced. |
| Cancel/shutdown | Persist cancellation when possible, cancel each owned request once, and drain cleanup before releasing request locks or closing inference dependencies. Persistence failure does not bypass cancellation cleanup. |
| Observe a live owner | Keep its active receipt. A second worker cannot dispatch for the same ID. |
| Observe abandoned work | Acquire the released process lock and move the receipt to `interrupted`. Keep its last observed generation/publication facts; do not invent a provider exception or replay the request. |
| Interrupted before dispatch | No generation was durably admitted; a caller may submit a new turn. |
| Interrupted with an answer already recorded | Preserve the answer and permit later turns. The original client may not have received that answer. |
| Interrupted after possible dispatch, without a recorded answer | Set `recovery_required=true`; reject new turns with HTTP 409 until the owner acknowledges the exact interrupted operation ID. |
| Acknowledge | Clear the recovery requirement under a revision check. Repeating the same acknowledgement is harmless while that remains the latest receipt; stale IDs conflict. No provider request is made. |

History, status, deletion, and acknowledgement apply the persisted principal's
existing access rules. Keep authentication identities stable across restart.
With authentication disabled, the existing shared anonymous principal remains
shared; persistence does not invent user separation. Schema/version mismatch or
invalid stored records fail closed, and explicitly configured journal startup
failure never falls back to an empty in-memory store.

The journal retains the v1 TTL and capacity policy. Reading status/history
refreshes access; deleted, expired, or evicted records return HTTP 404. Lock files
are stable coordination objects and are not removed during live retention
cleanup. Keep the database and adjacent lock directory in an application-owned
directory, use the same canonical path in every worker, and do not rename or
replace them while workers are running. SQLite-aware backups and maintenance
must respect those ownership boundaries. This is local filesystem cooperation,
not a distributed lease service or exactly-once provider execution.

The journal contains messages, principal IDs, and content-free receipts, not
provider keys or local model weights. SQLite operations remain synchronous and
short. Existing inference-client retry settings remain separate; the recovery
layer never resubmits abandoned work. Stored request receipts describe observed
local outcomes and cannot prove remote billing or tool-effect completion.

[Durable gateway tests](../tests/api/test_durable_conversations.py) exercise HTTP
authorization, application restart, competing workers, real process exit before
dispatch/after dispatch/after provider return/after answer publication, live
process ownership, corruption, retention, deletion, uncertain acknowledgements,
cancelled shutdown, self-close rejection, and stale operation handles. See the
[API workflow](API_REFERENCE.md#durable-v1-conversation-recovery) for configuration
and acknowledgement requests. Coordinated local-agent/planner recovery remains
a separate integration task.

## Persistence transactions and resource ownership

`UnitOfWork.transaction()` provides atomic writes across agents, conversations,
training jobs, and API keys for the memory and SQLite backends. The existing
`async with UnitOfWork(config)` remains a resource scope; repository calls outside
an explicit transaction still publish individually. `database_session()` borrows
the global database without implicitly opening a transaction.

```python
from stateset_agents.api.persistence import Agent, Conversation, DatabaseConfig, UnitOfWork

async with UnitOfWork(DatabaseConfig()) as database:
    async with database.transaction():
        await database.agents.create(Agent(id="support"))
        await database.conversations.create(
            Conversation(id="order-42", agent_id="support", user_id="customer")
        )
    receipt = database.get_status()["last_transaction"]
    assert receipt["commit_status"] == "acknowledged"
```

| Event | Ownership and result |
| --- | --- |
| Connect | Create repositories once. Repeated calls preserve handles and records. A resource scope whose setup fails closes partial connections before re-raising. |
| Begin transaction | One task owns all repositories until exit. Memory retains a detached rollback snapshot; SQLite starts one `BEGIN IMMEDIATE` transaction on the shared connection. |
| Read/write | Reads and returned entities are detached. Only explicit writes change storage. Competing tasks, inherited child tasks, and finished scopes are rejected before effects with `TransactionConflict`. |
| Nest a scope | Reject without changing the active transaction. Use one scope for the related operations; nested savepoints are not implemented. |
| Operation fails | Mark the transaction rollback-only. Even if the body catches the repository exception, later operations and normal scope exit raise `TransactionAborted`, and previous writes roll back. |
| Body raises or propagates cancellation | Roll back before releasing ownership. Preserve the original exception; retain rollback failure separately. Ordinary application errors caught inside the body still permit normal commit. |
| Commit returns | Mark `commit_status=acknowledged` and `state=committed`. Caller-owned entity objects are not themselves durable receipts. |
| Commit raises | Keep `commit_status=unconfirmed`; attempt rollback without replay. If no active transaction can be rolled back, close admission and retain `failed` rather than asserting that nothing committed. Reopen SQLite separately to reconcile stored records. |
| Rollback raises | Retain the original error and `rollback_error_type`, close admission, and keep the connection available for cleanup. |
| Close | Reject while a transaction is active; callers must finish/cancel and join it first. Otherwise share one cleanup, drain every repository through repeated caller cancellation, and retain cleanup failure without replay. A closed owner is terminal. |

`get_status()` returns detached `state`, `active_transaction`, and
`last_transaction`. Receipts contain an operation ID, state/phase, commit and
rollback status, and exception class names; they contain no entity data or
exception messages. They describe this process's observations and are not
persisted alongside database records. An escaped old scope cannot close or
mutate a later transaction. An old global close cannot detach a newly registered
database. Global initialization and application leases have the separate contract
below.

SQLite retains one connection per `UnitOfWork`, also supporting `:memory:` across
its repositories. Cleanup attempts that shared connection once, including after
failure; repository aliases do not repeat the close. Use an instance on one
event loop and its creating thread.
SQLite work remains synchronous; lock contention raises promptly rather than
blocking the event loop while another task holds the write lock. There is no
automatic retry, connection pool, PostgreSQL implementation, or schema migration
framework. Existing on-disk tables remain readable. Avoid long awaits or external
provider calls while holding a transaction: database rollback cannot undo their
effects.

Both backends reject duplicate creates and missing updates. Filtering precedes
pagination, and typed optional timestamps round-trip without mutating input
dictionaries. Memory records are isolated from input/get/list aliases; SQLite
JSON serialization rejects non-finite numbers. Direct mutation of private
repository storage or connection handles bypasses these contracts.

The [offline example](../examples/transactional_persistence.py) commits related
records, rolls back a rejected change, and reopens SQLite. Tests in
[test_persistence_transactions.py](../tests/api/test_persistence_transactions.py)
cover both backends, cancellation, failed acknowledgements, competing/escaped
tasks, connection isolation, cleanup, and real process termination during a
transaction. This establishes persistence primitives; canonical gateway stores,
authorization restoration, and coordinated session recovery still require their
own integration and schema.

## Canonical API application generations

`ApplicationLifecycle` captures one lifespan and its exact dependencies. The
canonical `create_app()` also installs an ASGI admission guard. The owner has
`new`, `starting`, `serving`, `stopping`, `closed`, and `failed` states; there is
no automatic restart or replay. A later explicit lifespan can start only after
the previous generation's retained cleanup task has completed successfully.

| Event | Ownership and result |
| --- | --- |
| Begin | Reject overlapping startup, serving, or shutdown before touching dependencies. Reject an active request from unmanaged embedding. Each generation records its event loop and startup task. |
| Prepare | Capture cleanup immediately after each dependency is returned. Inherited startup callbacks cannot close or restart their owner. Only the owning startup task can initiate rollback while startup is active. |
| Serve | Open admission after startup finishes. Retain each complete ASGI callback through HTTP response/stream, WebSocket disconnect, and cancellation cleanup. Cross-loop managed requests are rejected. |
| Stop | Close admission before waiting. New HTTP requests receive 503; WebSocket handshakes close with 1013. `/ready` reports unavailable without probing dependencies; passive `/live`, `/healthz`, and `/lifecycle` remain accessible. |
| Drain | Wait for accepted callbacks, then close health probes, durable conversations, training, cache lease, database lease, and inference. Caller cancellation does not abandon the shared shutdown task. Request/startup/cleanup callbacks and inherited child tasks cannot close or restart their own application. |
| Cleanup fails | Record exception class separately from the initiating startup/serving error. Health, journal, or training failure stops dependent cleanup. Cache/database/inference failure allows independent later cleanup but still fails the generation. Retain outcomes and handles without retry, including failure to create the shutdown task. |
| Restart | Replace successfully retired native health, inference, training, and journal owners. Preserve `AgentService`, models, in-memory conversations, detached local training history, configured queue/journal records, and custom health registrations. Rebind default health checks to current services. |
| Custom replacement | A retired custom service/subclass requires a fresh `app.state` instance or synchronous `app.state.lifecycle_factories` entry. Validate replacement requirements before allocating dependencies. A factory returning an awaitable, `None`, or the retired instance cannot publish a replacement. |
| Late exit | A previous lifespan joins its own cleanup result and cannot close a successor. A generation handle from another application is rejected. Cleanly closed applications may start on a new event loop. |

`GET /lifecycle` and `app.state.lifecycle.get_status()` return detached generation,
phase, active request count, completed/pending cleanup components, and primary
and cleanup exception class names. They perform no dependency I/O and omit
request identities, content, configuration, credentials, and exception messages.
`closed` confirms cleanup; it can still accompany a failed startup or serving
error. `failed` never establishes successful release and prevents a new lifespan.
Resolve the external cleanup failure before restarting the process.

A successful default checkpoint registration is retained when its requested
path/base model and registered model identity are unchanged. Removing/replacing
the model or changing that request permits explicit registration again. Changing
files at the same checkpoint path does not hot-reload an existing model.
Failed registration retains the existing optional-startup behavior and may be
attempted by a later explicit lifespan. Restart never resubmits training jobs or
replays journaled provider requests.

Native replacement retains the old inference configuration, training capacity
and queue location, and journal TTL/capacity. Explicitly replace/remove services
between clean lifespans to reconfigure them. Health callback registrations remain
caller-owned: callbacks should read current `app.state` dependencies rather than
capture retired services. Factories must clean up allocations they fail to return.
The same application object's in-memory history/model retention is separate from
durable process recovery.

The ASGI guard is installed by `create_app()`; calling `lifespan()` on a bare
FastAPI app owns dependencies but requires explicit middleware installation for
request admission/drain. Before the first lifespan, the guard permits unmanaged
ASGI embedding for compatibility. Request drain has no invented hard deadline:
open connections, unresponsive callbacks, or cancellation cleanup can delay it.
Custom escaped background tasks, direct service calls, and private live binding
mutation remain outside this ownership boundary. The separate GRPO shutdown
deadline contract is unchanged.

[Application tests](../tests/api/test_application_lifecycle.py) cover native
restart, conversation/model/job preservation, durable queue idempotency, HTTP and
stream drain, WebSocket admission/disconnect, concurrent close observers,
repeated cancellation, startup rollback, callback reentry, failed cleanup,
factory validation, foreign generation handles, and successive event loops.
Existing startup, health, training, conversation, and API compatibility tests
exercise integration with the component owners.

## Shared API cache and database ownership

The canonical API lifespan acquires a lease on each process-wide cache and
database. Separate applications with matching configuration share these resources;
one application's shutdown or failed startup releases only its own leases.
Health probes, hosted conversations, and training cleanup finish before their
dependencies are released. The final lease closes its captured resource.

`init_cache()` and `init_database()` retain their return types. Matching
configuration now preserves the existing usable instance and its records.
Changing configuration prepares a new instance before retiring the old one,
and is rejected while any application holds a lease. A directly closed unleased
instance is replaced with a fresh owner, not reopened. To explicitly reset an
unleased default, call its global close helper before initialization.

| Event | Ownership and result |
| --- | --- |
| Initialize/acquire | Capture configuration before waiting. Serialize transitions; concurrent matching callers share one prepared instance. Acquire adds its lease before releasing the transition boundary. |
| Preparation fails or is cancelled | Cancel the owned connection callback once and wait for it, then close the unpublished candidate. Clean rollback leaves the previous instance available. Preserve the initiating error. |
| Replace | Keep the previous instance available during preparation. Fence getters while retiring it; publish the candidate only after cleanup succeeds. Cancellation during retirement finishes old cleanup and closes the unpublished candidate. |
| Release | Drop exactly one lease. Other applications continue using the same instance. The final release closes once; concurrent/repeated observers join its retained outcome. |
| Global close | Reject while applications hold leases. Otherwise serialize behind initialization, close the captured instance once, and detach only that identity. Repeated cancellation cannot abandon cleanup. |
| Database close admission rejects | An active transaction raises the existing `TransactionConflict` before cleanup starts. Keep the published database available; close/replacement can be explicitly requested after the transaction ends. The same exception raised after cleanup starts is retained as a cleanup failure. |
| Cleanup fails | Retain the resource, exception class, and `failed` state. Disable getters, replacement, and automatic cleanup retry. Development startup cannot silently skip this failure. |
| Callback reentry | Reject callbacks and inherited child tasks that try to wait on their own resource transition. Active transitions and leases reject a different event loop before effects. |

`get_cache_status()` and `get_database_status()` return detached state, phase,
generation, lease count, initiating/cleanup exception class names, and the count
of unreleased resources. They perform no backend I/O and expose no configuration
or credentials. `empty` means no published instance after confirmed cleanup;
`failed` does not imply that a backend was released. Reconcile failed cleanup
and restart the process rather than repeatedly invoking an uncertain SDK close.

Embedded applications can use `acquire_cache()` and `acquire_database()` directly:
their returned lease exposes `.resource` and asynchronous `.aclose()`. Keep the
lease until all dependent work has stopped. Plain getters and `database_session()`
borrow a handle without acquiring lifetime ownership; their callers still need
an enclosing application or explicit lease. `create_cache()` and directly
constructed `UnitOfWork` objects remain independent owners.

Redis retains one SDK close task and the client when cleanup fails. Repeated close
observes that outcome; a successful close removes the client. The backend's own
operation boundary now drains standalone calls as described below. Use one event
loop for each resource's lifetime, and do not overwrite private module instances
or close a resource behind another application's lease.
The canonical application generation owner coordinates replacement of terminal
health/training and other dependencies as described above.

[Shared ownership tests](../tests/api/test_shared_resource_lifecycle.py) cover
real memory/SQLite preservation, overlapping application lifespans, failed
startup, configuration conflicts, queued/cancelled initialization, replacement,
foreign loops, repeated cancellation, inherited callbacks, and Redis SDK cleanup
failures without network calls. Existing startup, health, training, and durable
conversation tests continue to verify dependency shutdown ordering.

## Redis and hybrid cache lifetimes

`RedisCache` owns connection attempts, health probes, admitted commands, and pool
cleanup separately. `get_status()` returns its state, client generation, active
command count, shared-observer counts, cancellation requests, and initiating/
cleanup exception class names. It performs no Redis I/O and includes no keys,
values, URLs, or credentials. `HybridCache.get_status()` adds its own lifetime
and active operation count around the Redis observation.

| Event | Ownership and result |
| --- | --- |
| Connect | Concurrent callers share one client construction and initial ping. Snapshot connection configuration before dispatch. A negative acknowledgement, exception, expired deadline, or retired attempt cannot publish readiness. Missing Redis configuration remains an offline unavailable result. |
| Cancel an observer | One cancelled observer leaves other observers' work running. When the final observer cancels, request cancellation once and drain the callback, including suppressed or repeated cancellation, before releasing the attempt. New callers cannot overlap a retired attempt's cleanup. |
| Probe | Share one active health probe rather than returning cached health while a newer probe is pending. Cache settled observations using monotonic time. Connection/probe deadlines wait for callback cancellation cleanup before reporting unavailability. |
| Run a command | Capture the client inside an admitted operation. Retain ownership through SDK return and cancellation cleanup; paginated scan/delete operations keep that same client through every page. Concurrent ordinary commands remain supported. |
| Retire an unhealthy client | Stop new commands, drain admitted commands, then close its pool once. A new connection cannot replace the client until disposal settles. An old probe's final observation cannot overwrite health for a later client generation. |
| Close Redis | Stop admission immediately, cancel/drain shared connection/probe attempts, wait for admitted commands, and join pool cleanup. Close does not cancel accepted writes or replay them. Repeated close observers share the retained outcome even under cancellation. |
| Close hybrid | Stop hybrid admission and wait for the entire operation, including Redis lookup, memory refill/invalidation, fallback, and writes, before closing Redis. An already accepted operation can finish while close waits; none can publish after close returns. |
| Reject reentry | SDK callbacks and inherited child tasks cannot close or start an observation that waits for their own callback. Calls from another event loop fail before using the client. |
| Cleanup fails | Keep the client and `failed` state, record the cleanup exception class, and prohibit automatic reconnect or cleanup retry. The same close failure remains observable. |

After close starts, ordinary cache calls retain their unavailable return shapes
(`None`, `False`, or `0`), health is false, and an explicit connection request
raises `ResourceOwnershipError`. Closed hybrid caches cannot silently resume
memory-only service. An unavailable but open hybrid cache still uses memory as
before; an authoritative Redis miss still invalidates stale memory.

The optional Redis SDK's own exception hierarchy participates in availability
handling. Logs identify operation and exception class without SDK error text.
There is no automatic mutation replay in this layer; SDK transport retry policy
remains independent. A cancelled/failed write does not prove that Redis performed
no write. Hybrid entry ordering and fallback policy are described below. Direct
use of private client/memory handles bypasses these boundaries. A callback that
never finishes cleanup can delay close; underlying socket timeouts still matter.

[Deterministic lifecycle tests](../tests/api/test_redis_cache_lifecycle.py) exercise
shared observers, sole-observer cancellation, retirement during commands,
paginated deletions, deadlines and suppressed cancellation, eager task execution,
callback reentry, generation fencing, monotonic health caching, and hybrid memory
publication. [Optional real SDK tests](../tests/integration/test_redis_cache_sdk.py)
start an isolated local Redis server using a Unix socket with TCP disabled. They
verify concurrent setup/commands, pool disposal, server-loss recovery, and release
of a borrowed connection after command cancellation.

## Hybrid cache entries and fallback consistency

The hybrid cache retains its Redis-primary availability policy. Successful Redis
reads, including misses, are authoritative. An unavailable Redis read can use an
unexpired local copy. A local-only write is not replicated: when Redis recovers,
its answer can replace that value, or reveal a key whose remote deletion was not
acknowledged. Recovery does not replay mutations or install implicit tombstones.
These are cache observations, not a cross-process transaction or durable record.

`CacheEntryOrdering` owns each operation from entry admission through callback
unwind. It retains only active/waiting keys, with no unbounded history of keys.

| Event | Entry ownership and result |
| --- | --- |
| Read/write/delete/exists | Serialize operations on the same key within one hybrid instance. A delayed Redis answer cannot refill after a later local mutation has completed. Different keys remain concurrent. |
| Pattern deletion / clear | Admit an exclusive barrier after earlier entry operations settle. Later operations cannot bypass the barrier; all scan pages and local deletion remain inside it. Pattern deletion conservatively blocks all keys, including nonmatching keys. |
| Cancel while waiting | Remove the queued operation without running its callback. Wake operations whose dependencies are now clear; cancellation of a queued barrier cannot strand later keys. |
| Cancel during a mutation | Keep ordering and cache lifetime ownership through SDK unwind and local invalidation. Discard the affected fallback value(s), preserve the cancellation/error, and never resubmit Redis writes. If local invalidation fails, disable all fallback reads until an explicit successful local clear. |
| Callback reentry | Reject same-instance entry calls from a callback or inherited child before waiting, including calls for another key or a barrier. This prevents callbacks from constructing a cycle through pending barriers. |
| Capture values | Copy a submitted value before waiting for its key. Copy fallback read results and remote refill values so ordinary nested caller containers cannot mutate the retained copy. Custom values must support `deepcopy`; failure before admission performs no write. |
| Refill | Read Redis value and PTTL in one read-only transaction. Use a monotonic deadline beginning before dispatch, deducting response latency; retain that absolute deadline through local publication. A persistent Redis entry gets at most the configured fallback TTL. An expired/missing entry cannot seed fallback. |
| TTL unavailable | Return a readable Redis value without refilling memory and discard the previous local copy. Server rejection of transaction/TTL commands can fall back to a plain GET without refill authority; network errors retain the ordinary availability fallback. |
| Expire local memory | New memory entries use monotonic expiry, including standalone `MemoryCache`. Wall-clock changes cannot extend or prematurely expire them. Legacy manually constructed `MemoryCacheEntry` objects retain their wall-clock representation. |

The Redis read snapshot relies on its documented
[transaction isolation](https://redis.io/docs/latest/develop/using-commands/transactions/)
and [PTTL results](https://redis.io/docs/latest/commands/pttl/). Transactions do not
make the later local copy atomic with Redis changes from another process. Another
writer can still replace or delete the remote entry during a local outage.

Existing return shapes remain: `set()` succeeds when either layer acknowledges
storage, `delete()` reports whether either layer removed a key, and pattern/clear
counts sum the two layers. A zero-deletion Redis reply is an acknowledged miss,
not an unavailable command. A negative SETEX acknowledgement is not write success.

`HybridCache.get_status()` adds redacted active-key/barrier/waiter counts,
`fallback_safe`, a cumulative `unconfirmed_mutations` count, and `last_mutation`.
The last observation distinguishes memory/Redis acknowledgements, ordinary
completion, and escaping exception classes. Redis acknowledgement is `None` in
explicit memory-only mode. An unavailable configured Redis layer is `False` even
when the public operation succeeds locally. These detached diagnostics expose no
keys, values, or exception text. The last observation is whichever mutation
completed most recently across all keys; it is not a request-correlated receipt.
The cumulative count records past incomplete acknowledgements, not a retry queue
or the number of currently divergent keys. Do not infer write replay authority
or global linearizability from these observations.

[Entry tests](../tests/api/test_hybrid_cache_consistency.py) cover delayed
refills, write ordering, independent keys, barrier fairness/cancellation,
repeated cancellation through SDK unwind, callback reentry, detached values,
remote expiry, publication delay, clock jumps, partial acknowledgements, and
fallback recovery policy. [Real Redis tests](../tests/integration/test_redis_cache_sdk.py)
also verify short-lived entries during an outage and readable caches with
MULTI, EXEC, or PTTL denied by ACLs, plus transactional-read cancellation through
connection release and pool shutdown.

## Enhanced memory initialization and retrieval

[Enhanced agents](../stateset_agents/core/enhanced/enhanced_agent.py) keep memory
setup inside the agent's `initializing` attempt. Failure/cancellation leaves the
agent `failed`, and generation or self-improvement cannot use it. Completed custom
memory setup is reused by identity. A custom initializer that raises after it
starts may have changed external resources, so retry requires a fresh memory
system or agent. Built-in `VectorMemory` can retry on the same object because it
prepares a detached model and embeddings before publication.

Async setup runs without holding thread locks. An overlapping initializer fails
before invoking memory again. During cancellation cleanup, status remains
`initializing` at `extensions_cleanup`; it becomes `failed` only after owned work
settles. Repeated cancellation cannot interrupt that cleanup. An uncooperative
hook can delay cancellation indefinitely; the framework does not forcibly stop
external operations. If the hook finishes successfully after suppressing
cancellation, the original caller still receives cancellation. A later explicit
retry can reuse that known completed setup without repeating its effects.

[Vector memory](../stateset_agents/core/enhanced/vector_memory.py) exposes
`get_initialization_status()` with its state, backend, generation, captured
model/device request, identity match, and error type. Repeated initialization
reuses a ready model. Changing the embedding model/device requires initialization
before further insertion or retrieval; all existing entries are reencoded before
the new backend becomes ready. Failed preparation retains the previous model and
entry embeddings, while rejecting further use until an explicit successful retry.
Late embeddings from an earlier backend generation cannot be stored or counted
as a successful retrieval.

When `sentence-transformers` is absent, initialization records the `lexical`
backend. Retrieval ranks positive word-count cosine similarity; it does not
produce random embeddings or claim semantic similarity. Empty queries, zero
vectors, and a zero result limit return no matches without incrementing access
counts. Broken installed SDKs and model/device errors remain failures rather
than being mislabeled as missing optional dependencies. `initialize(force=True)`
explicitly retries dependency discovery and can promote a lexical index to a
dense one after installation.

```python
from stateset_agents.core.enhanced import create_enhanced_agent

agent = create_enhanced_agent(
    model_name="stub://memory-demo",
    memory_enabled=False,
    reasoning_enabled=False,
)
await agent.initialize()
assert agent.get_agent_status()["memory_entries"] == 0
```

Disabled components are not constructed by the factory. Disabled memory performs
no setup, retrieval, or writes, including self-improvement notes; disabled
reasoning does not invoke the reasoning engine. With memory enabled, all four
generation forms place retrieved entries in the latest user message as
labeled reference data, preserving system instructions, the current request, and
caller-owned inputs. Its context metadata remains available to custom backends.
Status snapshots are detached from live configuration, metrics, and persona
values. Existing `VectorMemory`/`MemoryEntry` import paths and older serialized
agents remain readable. These in-memory entries are still outside the portable
agent checkpoint contract below.

[Agent lifecycle tests](../tests/unit/test_enhanced_agent_lifecycle.py) exercise
readiness, optional flags, actual prompt contents, replacement, task cancellation,
suppressed/repeated cancellation, helper threads, and serialization.
[Memory lifecycle tests](../tests/unit/test_vector_memory_lifecycle.py) cover
fallback retrieval, atomic reindexing, failed setup, late embeddings, and async
SDK misuse. The optional [real SDK test](../tests/integration/test_vector_memory_sdk.py)
builds and reloads tiny local sentence-transformers encoders, verifies retrieval
without word overlap, switches embedding dimensions, and exercises enhanced batch
selection against that memory without model downloads.

## Durable memory snapshots

`VectorMemory.save_snapshot(path)` captures entry data at one point in time.
`restore_snapshot(path)` prepares a detached index with the destination's current
backend, then replaces the live entries once. Both return receipts; inspect
`get_snapshot_status()` after failures or cancellation. Enhanced agent status
includes those receipts as `memory_snapshot`.

```python
from stateset_agents.core.enhanced import VectorMemory

memory = VectorMemory()  # Existing lexical behavior; no encoder is loaded here.
await memory.add_entry("Refund window: 30 days", {"policy_version": 1})
saved = await memory.save_snapshot("memory.json")  # Parent directory must exist.

restored = VectorMemory()
await restored.restore_snapshot("memory.json", expected_revision=saved["revision"])
matches = await restored.retrieve_relevant("refund")
updated = await restored.save_snapshot(
    "memory.json", expected_revision=saved["revision"]
)
```

For dense retrieval, configure and initialize the destination before restoring.
The source model name is metadata only: restore never loads it, downloads weights,
or changes destination configuration. All embeddings are rebuilt with the current
destination encoder. A snapshot exceeding destination capacity is rejected without
trimming entries. A failed backend can still save entry data for recovery into a
fresh memory object.

Version 1 is strict JSON with a unique revision and a SHA-256 integrity checksum.
It preserves entry order, text, nested JSON context, timestamps, importance,
access counts, and last-access times. Embeddings, model/tokenizer instances,
subclass-specific state, conversation history, reasoning runs, and generation
receipts are excluded. Canonical `MemoryEntry` objects are required; context values
must be finite JSON types with string dictionary keys. Cycles, arbitrary Python
objects, duplicate JSON keys, unknown fields/versions, and corrupt checksums fail
explicitly. The checksum detects corruption; it does not authenticate authorship.
Reads and writes default to a 64 MiB limit, configurable through `max_bytes`.
Unknown versions require an explicit future migration; no implicit coercion occurs.

| Operation boundary | Contract |
| --- | --- |
| Save admission | One save/restore operation per memory object. Capture canonical data under its owner lock; later live updates do not change the captured snapshot. |
| File revision | Omitting `expected_revision` is create-only. Replacement requires the exact current revision, checked under an OS advisory writer lock. Invalid existing files are never silently overwritten. |
| Publication | Write a private temporary file in the same directory, flush and fsync it, replace the destination, then fsync the directory on POSIX. The parent directory must already exist. |
| Uncertain replacement | `publication` becomes `unconfirmed` before replacement and `published` only after acknowledgement. A later durability failure does not roll back the published file. |
| Save cancellation | Drain owned file work before releasing operation ownership. The caller receives cancellation even if the snapshot was published; inspect the terminal receipt and file revision before retrying. |
| Restore preparation | Read and validate a complete file image, then build all candidate entries and embeddings before changing the index. Encoder failure or inconsistent dimensions leaves live entries intact. |
| Concurrent memory change | Restore checks backend generation, mutation revision, entry identities, and canonical data. Managed insertion/retrieval, configuration changes, and direct canonical-data edits cause a conflict instead of discarding newer work. |
| Restore cancellation | Cancel embedding preparation once and join cleanup. Suppressing or repeatedly requesting cancellation cannot publish the candidate index or start a later embedding. |
| Restore commit | Replace the entry list and advance the backend generation. Earlier retrieval/insertion and deferred enhanced candidates cannot commit against the restored index. |
| Process restart | Validate the canonical file through `inspect_snapshot` or `restore_snapshot`. Never promote orphan staging files. OS writer locks are released when their process dies. |

Receipts distinguish terminal `state` (`completed`, `failed`, `cancelled`, or
`conflict`) from effect acknowledgement (`publication`), `file_synced`, and
`directory_synced`. For restore, publication refers to the **in-memory index**;
both file-sync fields stay false because restore does not write the snapshot.
Receipts include the phase, revision, observed revision, captured local mutation
revision, entry count, and error types. `inspect_snapshot(path)` validates and
returns file metadata without entry content or encoder execution; it establishes
the current readable revision, not whether an earlier directory sync succeeded.

Cross-process safety assumes cooperating writers in a trusted application-owned
directory. Stable sibling `.stateset-memory.lock` files are intentionally retained;
do not unlink them while writers may run. Avoid direct public-entry mutation from
other threads during capture/commit; the owner lock coordinates managed methods,
not arbitrary Python writes. Derived embeddings and custom subclass state are
outside the canonical-data comparison. A hung worker or encoder cleanup can delay
cancellation indefinitely. No automatic saves or full conversation recovery are
implied by these explicit memory APIs.

The file replacement follows [Python's same-filesystem replacement contract](https://docs.python.org/3.10/library/os.html#os.replace).
On Linux, [file and directory fsync acknowledge different persistence boundaries](https://man7.org/linux/man-pages/man2/fsync.2.html).
Receipts report actual acknowledgements; non-POSIX platforms do not claim directory
fsync, and filesystem/device guarantees still govern power-loss behavior.
[Snapshot tests](../tests/unit/test_memory_snapshot.py) cover corruption,
concurrent mutation, stale candidates, repeated cancellation, injected failures,
real process termination before/after replacement, and fresh-process recovery.
The [SDK integration](../tests/integration/test_vector_memory_sdk.py) restores
with a different local encoder dimension and verifies semantic retrieval.
Run the [offline example](../examples/durable_memory.py) for a complete small flow.

## Enhanced generation completion and candidate selection

`generate_response()`, `generate_turn()`, `generate_turns()`, and
`generate_response_stream()` use one captured memory/reasoning context per call.
Batch samples share this context, and native sampler/token metadata is preserved.
The base pipeline still owns conversation history and turn counts; enhancement
bookkeeping does not repeat those effects. Batch history retains the base
behavior of recording every returned candidate, while vector memory stores only
the selected candidate.

| Event | Memory and completion behavior |
| --- | --- |
| Single response or turn completes | Remember the generated response once, then record enhanced metrics and persona/reasoning updates. |
| Stream exhausts normally | Remember the cleaned assembled output once. The consumer must request exhaustion, not just receive the last nonempty chunk. |
| Stream closes, fails, or is cancelled | Retain partial text for inspection without remembering partial output or applying enhanced completion bookkeeping. Drain the producer and its cleanup before returning; cleanup errors do not replace an initiating consumer error or cancellation. |
| Batch completes | Record metrics for each output and one shared reasoning/persona update; defer memory until `remember_candidate(turn)`. |
| Selected candidate write succeeds | Retain `stored`; selecting that same candidate again is a no-op. A different selection is rejected. |
| Memory write raises after invocation | Retain `unconfirmed`, the error type, and generated outputs; the effect may have happened. A candidate write cannot be replayed automatically. |
| Caller cancels a memory write | Join the owned write through repeated cancellation. Its receipt can still report `stored` if the write succeeded; caller cancellation is preserved. |
| Memory source changes before the write | Reject before invoking it. Selection preflight failures leave an unwritten batch `deferred`; source compatibility must be restored before selection can succeed. |

Preparation and sampling callbacks receive cancellation once and are joined
before cancellation returns. Suppressing cancellation cannot continue to a later
stage or memory write. These ownership boundaries have no forced timeout: a hung
callback or stream producer can delay cancellation until its own timeout/cleanup
settles. Caller cancellation during cleanup takes precedence over an earlier
ordinary error. Stream consumers should close an abandoned generator explicitly, for
example using `contextlib.aclosing`.

```python
turns = await agent.generate_turns("Explain the refund policy", n=3)
selected = turns[1]  # chosen by the caller's user or ranking policy
await agent.remember_candidate(selected)
generation_id = selected.metadata["stateset_enhancement"]["generation_id"]
receipt = agent.get_generation_status(generation_id)
assert receipt["memory_status"] == "stored"
```

`get_generation_status()` returns detached active receipts, the latest finished
receipt, and retained batch IDs. Each receipt includes its phase, outputs,
partial text, terminal state, selected index, and memory/error outcome. A failed
call can therefore expose a generated answer even when remembering it failed.
The initial turn metadata is a snapshot; inspect the receipt for current status
after later candidate selection. Receipts can contain conversation text.

The agent retains the latest 32 idle batches plus all writes still in progress.
Expired, foreign, edited, or incompletely generated candidates cannot be selected.
Candidate selection rights belong to the live agent and are excluded from deep
copies and pickle restoration; the latest output receipt remains available as
evidence. Deep-copying or pickling an agent during generation or selection is
rejected. Ordinary shallow copies continue to share components and receipts.
Neither these receipts nor enhanced memory are part of portable model checkpoints.

[Generation tests](../tests/unit/test_enhanced_generation.py) cover shared prompts,
explicit selection, concurrent and uncertain writes, partial streams, suppressed
and repeated cancellation, memory replacement, detached concurrent receipts,
retention, serialization, and real local Transformers sampler metadata.

## Reasoning providers, assessments, and stopping

[ReasoningEngine](../stateset_agents/core/enhanced/reasoning.py) retains the
`reason(query, context) -> list[ReasoningStep]` interface and existing import paths.
Its default backend emits at most five planning instructions, labeled `template`.
Those instructions neither claim to have analyzed the question nor manufacture
evidence. Their legacy numeric `confidence` field is zero, with
`confidence_status="unavailable"`. A template run stops at the end of its plan or
the configured step limit. The old increasing confidence schedule is removed.

Supply an async `step_generator(ReasoningRequest)` to produce actual proposals.
The request contains the query, reasoning category, step index, context, and
previous proposals with their assessments. Each callback receives detached data.
`ModelReasoningProvider(separate_planner.generate_response)` adapts an initialized
text backend to this contract using a concise JSON proposal, action, declared
confidence, and reference list. It rejects malformed JSON, duplicate/unknown
fields, invalid scores, and incorrect field types. Invalid output fails the run;
it does not silently fall back to templates or invoke the model again.

An optional async `assessor(request, step)` returns `ReasoningAssessment`:

| Assessment | Stopping behavior |
| --- | --- |
| `unavailable` | Requires an abstention reason and has no score. Continue within the captured step limit. |
| `failed` | Requires a named method, supporting evidence, and finite score in `[0, 1]`. Pass the rejection to the next proposal; even a high score cannot establish completion. |
| `passed` | Requires the same provenance. Stop early only when its score reaches the captured `confidence_threshold`. |
| Checker raises or returns an invalid result | Retain the proposal as unassessed, mark the run failed, and propagate the error before final answer generation. |

The legacy `confidence` attribute remains a provider declaration and never
controls the stopping condition. An assessor's score records that checker's
scoped judgment; it is not automatically a calibrated probability, independent
attestation, or a measure of general reasoning quality. The checker owns the
meaning and coverage of its method. Assessments cover preparatory proposals;
final responses require their own evaluation. Changing a step's content, evidence, declared
confidence, origin, or run identity makes its bound assessment unavailable.
Assessments supplied by a proposal generator are discarded before assessment.

Run the [offline example](../examples/enhanced_reasoning.py):

```bash
python examples/enhanced_reasoning.py
```

It uses a scripted proposal backend and a checker against supplied policy data.
A declared-confidence `0.99` proposal with the wrong refund window fails; a
corrected proposal with declared confidence `0.5` passes that exact policy check.
The example demonstrates the contract, not learned model quality. For a model
backend, initialize a separate planner and supply its text generation method:

```python
from stateset_agents.core.enhanced import ModelReasoningProvider, ReasoningEngine

engine = ReasoningEngine(
    max_steps=3,
    step_generator=ModelReasoningProvider(planner.generate_response),
    assessor=check_step,  # async, with a declared method and supporting evidence
)
steps = await engine.reason(query, context)
status = engine.get_status()
```

Pass the configured engine to `EnhancedMultiTurnAgent(..., reasoning_engine=engine)`
or `create_enhanced_agent(..., reasoning_engine=engine)`. The factory also accepts
an explicit `memory_system`; disable flags still suppress those components' effects.

Each model proposal is an additional model call. Recursive use of the same
reasoning engine is rejected; use a separate backend. Providers and checkers
must finish their owned cleanup on cancellation, even if they suppress the
initial cancellation or the caller cancels repeatedly. No subsequent proposal,
assessment, answer generation, or memory write starts after cancelled preparation.
Callbacks have no framework-enforced deadline; their own timeouts must bound
external work. Completed work is not rolled back.

`get_status()` exposes detached active/latest runs, captured limits, query, mode,
step evidence, failure phase/type, and stop reason. `completed` means preparation
finished; `step_limit`, `template_complete`, and `assessment_threshold` have
different meanings. Concurrent calls have separate run IDs. Step IDs connect
these runs to enhanced generation's `reasoning_steps` receipt and
`reasoning_evidence` context. Custom/provider proposals and their scoped assessments
reach native generation prompts as labeled reference data. Templates remain
context metadata and are not inserted as apparent findings.

`get_agent_status()["reasoning_backend"]` reports the configured mode and its
available run status. The legacy `chain_of_thought` capability string remains an
enabled-pipeline alias; it is not evidence of reasoning quality. The more precise
`reasoning_preparation` capability is also exposed. Existing step-hook subclasses,
duck-typed engines, and completed legacy pickles remain supported. Deep-copying
or pickling an engine during a run is rejected; ordinary shallow copies share
references. These receipts are process-local and may contain conversation text.

[Reasoning tests](../tests/unit/test_reasoning_evidence.py) verify scoped policy
checks, revision, abstention, changed evidence, captured configuration, concurrent
and recursive calls, malformed model output, helper threads, serialization,
cancellation through nested agent/provider cleanup, and actual proposal text in
all enhanced generation surfaces.

## Agent checkpoint restoration contract

The [checkpoint helpers](../stateset_agents/core/agent_factories.py) publish
inference components through [explicit artifact metadata](../stateset_agents/core/agent_checkpoint.py).
They distinguish full-model, adapter, stub, and configuration-only snapshots.
Full-model restoration selects the saved local weights and tokenizer. Adapter
restoration selects the saved adapters and tokenizer, with their recorded base
model as an external dependency. It does not create a fresh randomly initialized
LoRA adapter from the old training configuration.

```python
from stateset_agents.core.agent import (
    load_agent_from_checkpoint,
    save_agent_checkpoint,
)

await save_agent_checkpoint(agent, "outputs/agent")
restored = await load_agent_from_checkpoint("outputs/agent")

# Inspect configuration without allocating a model or reading its weight payload.
pending = await load_agent_from_checkpoint("outputs/agent", load_model=False)
await pending.initialize()  # Verify and restore the captured publication.
```

New publications include the effective generation configuration. Stub snapshots
also retain actual response values, response position, model configuration,
tokenizer attributes, and training flag. Repeated initialization does not reset
these restored components. Conversations, optimizer state, external rollout
engines, arbitrary subclass extensions, and enhanced memory are not included.
Stop concurrent training and generation before saving; the filesystem lock does
not freeze in-memory model mutation.

The writer creates a separate staged directory. Only a complete, synced inventory
is published; failures retain the previous checkpoint, and cooperating reads
recover interrupted replacement. Agent publications use a distinct versioned
marker and journal. Training publications retain their original schema and
require `training_state.pt`; neither writer may erase the other checkpoint kind.
Unrelated destination directories are refused. All cooperating reads/writes hold
the existing parent-directory advisory lock; external Transformers readers must
not race replacement.

`load_model=False` verifies the new publication's metadata and file layout without
reading weight payloads or loading Transformers. Initialization verifies all
artifact hashes under the lock and rejects a changed publication before loading
components. A failed restore remains in the agent's failed initialization state;
generation cannot proceed. Legacy checkpoints without publication receipts use a
full file inventory even during configuration inspection. Their full-model and
adapter weights are selected locally, while legacy configuration-only snapshots
retain their original source-initialization behavior.

`save_model=False` explicitly creates a configuration-only snapshot. Loading it
initializes the original configured source when requested. It cannot overwrite a
snapshot containing model or stub state; choose a separate path. A model save
requires initialized model, tokenizer, and generation components. Configuration
must be JSON data: unsupported runtime objects fail before publication rather
than being stringified into values that cannot be restored.

LoRA snapshots include every installed adapter, including named adapters in PEFT's
subdirectories. [Adapter metadata and restoration](../stateset_agents/core/agent_adapter_checkpoint.py)
preserve each name and its weights, the primary wrapper adapter, the ordered
active selection (including an empty selection), enabled/disabled state, and
merge order. Layer receipts retain the merge order that PEFT's sorted model
summary loses; this matters for DoRA. Restoration freezes parameters for inference.
Repeated initialization reuses the loaded model and preserves later caller changes
to its selection, even after the checkpoint directory has been removed.

Adapter collections use descriptor version 2. Version 1 publications and legacy
single-default adapter directories remain readable. Metadata inspection validates
the declared adapter layout and shared base source before loading PEFT. Unsafe or
colliding names, missing weights, undeclared adapter directories, inconsistent
layer state, and conflicting base sources fail explicitly. The adapter snapshot
cannot replace or contain its own external base model.

All attachment, selection, enablement, and merge operations belong to one retained
initialization attempt. A failure or cancellation after attachment starts requires
a fresh base model before retry; no partial wrapper becomes the ready agent.
Completed initialization does not replay these operations. Failed staged writes
retain every adapter from the previous published snapshot.

Named collections currently require LoRA tuners; other single-default PEFT tuners
retain their prior save/load behavior. Custom stub subclasses still require their
own serializer. The base model must remain available and reproduce the saved
backbone; adapter artifacts do not include arbitrary changes to backbone weights.
Base revision stability remains a caller responsibility. This is inference
restoration, not optimizer/training resume or a promise of bitwise-identical
generated text across devices and SDKs.

[Round-trip tests](../tests/unit/test_agent_checkpoint_roundtrip.py) use tiny local
Transformers/PEFT models with changed parameter values, remove the original source
for full-model restores, verify saved tokenizer and generation settings, exercise
deferred loading and corruption, and terminate real subprocesses during
publication. The existing [publication tests](../tests/unit/test_checkpoint_publication.py)
continue to cover training compatibility, rollback, and interrupted cleanup.
[Adapter collection tests](../tests/unit/test_agent_adapter_checkpoint.py) compare
every adapter's tensors and model outputs, cover multiple/empty selections,
disabled and merged execution, DoRA merge order, older formats, malformed
publications, partial saves, and errors/cancellation after attachment or merge.

## Remaining review findings

These are source-level findings and follow-up priorities, not claims that every
listed lifecycle has been repaired or exhaustively race-tested.

| Priority | Finding | Next compatibility-preserving step |
| --- | --- | --- |
| C01 · Extended checkpoint representations | Named LoRA collections now retain weights and activation/merge state. Named non-LoRA tuners and custom stub subclasses remain unsupported; enhanced memory, conversations, and rollout engines are separate owners. | Define representation and activation receipts before extending the supported snapshot surface; prove restoration against each actual backend rather than silently discarding state. |
| C14 · Reasoning quality | Providers and scoped assessments now replace fabricated confidence-based completion. A passed checker result does not establish general reasoning quality or calibrated confidence. | Measure each provider/checker pair on held-out tasks with explicit coverage and provenance before making quality claims. |
| C04/C14/A05/A12 · Coordinated session recovery | Vector-memory snapshots preserve canonical entries. Named-conversation snapshots preserve transcript/context, built-in planning state, and settled receipts without replay. The optional v1 journal restores hosted conversation history, principal ownership, and interrupted-request receipts. Local AgentService model/planner state, enhanced reasoning/generation, and custom subclasses remain separate owners. | Integrate local agent identity, planner/model restoration, and versioned service records before claiming complete local-session recovery; preserve destination backend choice and separate publication/durability acknowledgements. |
| A09 · Custom application work | Canonical application generations now drain ASGI callbacks, retain cleanup outcomes, and replace native dependencies after confirmed close. Escaped custom background tasks and direct service calls remain independently owned. | Give custom work an explicit join/close contract before extending application shutdown guarantees to it. |
| A11 · Cross-process cache consistency | One hybrid instance now orders conflicting entries and barriers, retains TTLs, and exposes partial acknowledgements. Redis remains authoritative after outages; independent application instances and two-layer mutations remain separate owners. | Introduce an explicit distributed version/invalidation protocol only when stronger consistency is required; preserve availability and acknowledgement distinctions instead of claiming atomic replication. |
| Cross-cutting | Cancellation, interrupted ownership, terminal execution, artifact publication, and resource deletion are separate facts. | Preserve their existing distinctions in TUI/API/MCP projections; never turn uncertainty into success or authorize automatic resubmission from a status label alone. |

## Keeping the inventory current

When changing an owner, update its row and test sequences at the effect boundary:
normal completion, failure before/after an effect, repeated calls, cancellation,
late completion, and restart/restore where supported. Use behavior fakes for
providers and clocks; no paid jobs are needed to verify local lifecycle rules.

Start a renewed survey with both named-state and implicit-state searches:

```bash
rg -n 'class .*State|class .*Status|class .*Phase|TERMINAL|transition' stateset_agents
rg -n 'status|phase|stage|_closed|_closing|_initialized|_has_run|should_abort' stateset_agents --glob '*.py'
rg -n 'lease|receipt|attempt|exists\(|cancel|done\(|is_set\(|dirty_keys|stale|poisoned' stateset_agents --glob '*.py'
rg -n 'status|state|abort|retry|loop|shutdown' npm/src src rust_core/src
```

Then inspect the writers and readers together. Constant assignments and enum
declarations find candidates; flags that change both ways, SQL predicates,
`finally` blocks, durable receipts, and collection membership reveal the hidden
guards. Similar names are not evidence of shared ownership.
