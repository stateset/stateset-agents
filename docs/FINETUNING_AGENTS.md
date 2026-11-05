# Fine-tuning from AI agents

Agents can use the studio's workspace through a noninteractive JSON CLI. It
requires no terminal, web service, extra protocol server or training imports for
schema discovery. Use the Python environment where StateSet Agents is installed:

```bash
stateset-agents finetune schema
stateset-agents finetune models
# From this checkout:
.venv/bin/python -m stateset_agents.cli finetune schema
```

Every command emits one JSON object on stdout:

```json
{"schema_version":1,"ok":true,"command":"status","result":{"runs":[]}}
```

Command failures return `ok: false`, a controlled `error.code`/`error.message`,
and exit 1. CLI usage errors (such as unknown flags) use normal Typer help and
exit 2. Library diagnostics go to stderr. Parse stdout as JSON and check both
`ok` and the exit code. A successful status request describes the run; it does
not imply that training succeeded. For readiness, inspect `result.failed` and
`result.warnings`; `ok` says the inspection completed, not that training is ready.

| Command | Effect |
| --- | --- |
| `schema` | Config schema, defaults, supported methods and command effects |
| `models` | Catalog model IDs and maturity; no backend/hardware certification |
| `inspect-data --dataset PATH --method sft` | Dataset statistics; private excerpts require `--include-samples` |
| `readiness --config config.json` | Local file/package/hardware/queue inspection; no provider authentication |
| `validate --config config.json` | All field-addressable configuration issues; no dataset read, workspace write or provider request |
| `prepare --config config.json` | Validate data and create a private prepared run; no training |
| `launch RUN_ID --expected-plan-sha256 SHA` | Start exactly that reviewed plan once |
| `status [RUN_ID]` | Read status, latest metrics, known failure guidance and whitelisted receipts |
| `logs RUN_ID --source auto` | Explicitly read bounded private training/supervisor logs |
| `compare REFERENCE_ID CANDIDATE_ID` | Compare saved settings, dataset hashes and latest training metrics |
| `workflow --config config.json` | Guided setup checks without preparation, launch or provider requests |
| `quality RUN_ID [--evidence reports.json]` | Rehash completed local weights and inspect/import paired test reports |
| `checkpoints RUN_ID` | Inspect plan-bound local recovery checkpoints without deserializing optimizer state |
| `prepare-resume RUN_ID --checkpoint NAME --expected-recovery-sha256 SHA` | Privately snapshot exact recovery inputs into a new prepared run; no launch |
| `stop RUN_ID --expected-plan-sha256 SHA` | Request a cooperative local stop bound to the reviewed plan |

Pass `--workspace /absolute/path` to share a workspace with the TUI. The default
is `outputs/finetune`. Runs prepared by an agent appear in the TUI's Runs screen;
TUI run details include a quoted status command for agent access.
For a human handoff, select the prepared run and press `r` in the TUI. It shows
the retained plan checksum and resources; Enter launches the same ID, without
copying the run or replacing the human's Configure draft. Esc starts no work.

## Guided setup, quality and local stop

`workflow --config PATH --workspace WORKSPACE` accepts incomplete configuration
objects and returns five `steps` with names, statuses, details and action IDs.
Repair blocked stages through validation, dataset inspection and readiness.
This report never prepares a run; launch still requires a separate review.
Runtime checks do not certify GPU fit, provider authentication or prices.

`quality RUN_ID --evidence PATH --workspace WORKSPACE` accepts a regular JSON
file within 8 MiB. The object has `schema_version: 1`, `baselines` and `candidates`
arrays (1–64 paired reports). Reports follow
[`evaluation_report`](../stateset_agents/evaluation/agent_runs.py), including
`split: "test"`, environment, base model, seed, exact `case_hashes`, protocol
`settings`, `selected_checkpoint` and exactly one outcome per case. Outcomes
include success, reward, policy violations, truncation, generated tokens, tool
calls and elapsed seconds. Cases, seeds and protocols must pair exactly.
Validation-split reports are rejected.

Exactly one candidate's `selected_checkpoint.path` and `inventory_sha256` must
match this successful run's completed weights (`artifact_digest` from
`training.promotion`). Hosted pointers require provider-bound evaluation support
and are rejected here. Imports atomically retain private reports bound to plan,
completion and artifact hashes. Retained pretty-printed receipts must fit within
8 MiB plus 128 KiB; an import whose retained representation exceeds that limit
is rejected. Invalid imports preserve earlier receipts.
`quality RUN_ID` rehashes weights and recomputes comparisons; changed bindings
invalidate evidence. Neither operation runs a model or judge. Responses hide
case IDs/transcripts; up to 50 success regressions include seed, a case-ID SHA256
handle (first 16 hex characters) and recorded reward values.

`recorded_study_passed` requires the paired six-seed significance/safety gate,
distinct candidate inventory hashes and at least 32 distinct case hashes per seed.
The bound report must match this run's recorded training seed; missing historical
training seeds cannot establish independence or pass this gate.
It also requires `settings.model_revision` matching the reviewed nonzero full
model commit, plus nonempty `judge_model`, nonzero full `judge_revision` and
64-hex `rubric_revision`; include envelope
`review: {"source": "reviewed", "reviewer": ...}`. One-seed results remain
descriptive. Only this run's weights are rehashed; other artifacts, judge
identity, independence, reviewer attestations and actual train/test isolation
require external review. This does not select checkpoints, promote artifacts
or set `quality_gain_verified`.

`stop RUN_ID --expected-plan-sha256 SHA` is idempotent for a matching local plan.
It writes a private receipt without signals or Buildr cancellation. Prepared
runs become `cancelled`; queued workers check before loading; active SFT requests
save/stop at a step boundary, and native RL saves a checkpoint after a committed
update. Status includes `stop_requested` and `recovery_checkpoint` when recorded.
Changed package code and remote backends reject requests. Remote cleanup needs
provider-specific controls; local checkpoint recovery is described below.

## Local paired evaluations

`j` in the TUI and the following commands share one local evaluation protocol:

```bash
stateset-agents finetune evaluation-plan RUN --config evaluation.json
stateset-agents finetune evaluation-prepare RUN --config evaluation.json \
  --expected-plan-sha256 REVIEWED_EVALUATION_SHA
stateset-agents finetune evaluation-launch RUN EVALUATION_ID \
  --expected-plan-sha256 REVIEWED_EVALUATION_SHA
stateset-agents finetune evaluations RUN
stateset-agents finetune evaluation-targets RUN
stateset-agents finetune evaluation-status RUN EVALUATION_ID
stateset-agents finetune evaluation-status RUN EVALUATION_ID --verify
stateset-agents finetune evaluation-stop RUN EVALUATION_ID \
  --expected-plan-sha256 REVIEWED_EVALUATION_SHA
```

All accept `--workspace`. Planning is read-only: it hashes completed local
adapter bytes and captures cases without importing inference libraries. Prepare
requires the exact returned checksum and snapshots cases privately. Launch
checks that checksum again, verifies current inputs/code, takes an exclusive
submission receipt and starts a detached supervisor. A second submission is
refused; failed or uncertain submissions are retained for inspection, never
automatically retried. Status reads small metadata; `--verify` performs a fresh
artifact hash and judge replay after success. Stop requests are evaluation-only,
cooperative and bound to the evaluation plan, not its parent training plan.

`evaluation-targets` lists receipt metadata for terminal local checkpoints,
plus the final adapter on a successful run. `bytes_verified: false` is explicit:
discovery does not hash model/optimizer bytes. Set `checkpoint` to a listed name
and `split` to `validation` in the evaluation configuration to capture a full
receipt/weight review. A checkpoint must contain a native safetensors adapter;
the parent worker and supervisor must have exited. Exhausted training budgets
and older trainer code do not prevent inference; exact-state recovery retains
its stricter constraints. Linux process ownership is required for checkpoints.

Use `"split": "validation"` in the cases document and declare
`provenance.validation_is_fresh: true` with the reviewed source and reviewer.
The config, report, judge recording and replay all retain this partition.
Validation results never pass the test study gate or import through `quality`.
`checkpoint: ""` selects the successful final artifact; its default partition
remains `test` with `test_is_fresh` provenance. Published checkpoints are currently
validation-only; fresh-test evaluation uses the completed final artifact.
No automatic checkpoint ranking, artifact replacement or promotion occurs.

New local native RL configurations can set `checkpoint_every` to a positive
optimizer-update interval (zero preserves the final-only default). The plan
records the expected number of full-state checkpoints, capped at 100. Budget
workspace/scratch storage explicitly. SFT and remote backends reject a positive
interval; SFT continues to save epoch checkpoints.

Protocol errors expose controlled `error.code` values and a diagnostic containing
the expected partition, without echoing case values:

| Code | Client action |
| --- | --- |
| `evaluation_checkpoint_partition` | Choose validation for a published checkpoint |
| `evaluation_case_partition` | Supply schema 1 cases for the configured partition |
| `evaluation_provenance` | Supply reviewed source, reviewer and the partition freshness flag |

Stopping a prepared, unlaunched evaluation makes it terminal `cancelled` without
submitting a job. Launched evaluations continue to use cooperative stop.

For keyboard automation, `f` in the evaluation form opens the held-out JSON
case picker. Enter chooses only a path and returns to the captured form; Escape
preserves its values. Selection never changes training settings, reads cases or
starts inference. `r` still captures a separate review before submission.

Example `evaluation.json` (replace the judge commit with a real cached commit):

```json
{
  "cases": "/absolute/path/reviewed-test.json",
  "judge_model": "organization/independent-judge",
  "judge_revision": "REPLACE_WITH_FULL_40_CHARACTER_COMMIT",
  "max_new_tokens": 192,
  "judge_max_new_tokens": 1024,
  "context_length": 8192,
  "seed": 42,
  "timeout_seconds": 3600,
  "cpu_slots": 2,
  "memory_gb": 4.0,
  "scratch_gb": 5.0
}
```

Held-out cases use this format. The example is synthetic documentation, not
reviewed study evidence; supply your actual reviewer and fresh cases:

```json
{
  "schema_version": 1,
  "split": "test",
  "provenance": {
    "source": "reviewed",
    "reviewer": "YOUR_REVIEWER",
    "test_is_fresh": true
  },
  "cases": [{
    "id": "damaged-delivery-1",
    "context": "You help customers with delivery issues.",
    "user_turns": ["My delivery arrived damaged. What are my options?"],
    "policy_facts": ["Damaged deliveries are eligible for a free replacement."],
    "expected_resolution": "Offer a free replacement and explain the next step.",
    "must_avoid": ["A replacement fee applies"]
  }]
}
```

Limits are 2 MiB of case JSON, 1–64 cases, 1–8 scripted user turns per case,
32 policy facts/forbidden phrases, 4,096 characters per list entry and 16,384
characters per text field. IDs and initial questions must be distinct. Exact
training user-turn overlap is rejected; RL case IDs are also checked. Declared
review and freshness are not proof against semantic leakage. This format
exercises support conversations, not arbitrary tools or adaptive customers.

Final-artifact evaluation requires a successful local run with a full nonzero base-model commit
and a safetensors adapter bound to that base. Judge model/revision must differ
from the base identity. Three sequential subprocesses generate baseline replies,
candidate replies and then two-pass judge scores. Models load from the existing
local cache with native code only. Exact commits resolve through the Hub cache
before loading, avoiding relative-directory revision overrides. Missing cache entries or context overflow
fail without downloads, truncating facts or falling back to a dry run. Buildr
is required and owns the complete timeout. CPU/RAM/scratch budgets do not reserve
a GPU. There are no provider requests or paid-resource provisioning.

`evaluation-status` and `evaluations` include a `queue` object when complete
Buildr lifecycle events are available. It exposes only the job handle, runner
state, sampled wait, safe admission category, disk/RAM measurements and sample
time/age. It reads at most 64 KiB from the private event tail without querying
shared state or loading models. Missing, partial or invalid events produce an
empty object; an old sample never proves success, failure or a dead worker.
The evaluation's own `state.status` remains authoritative. Use the handle with
`buildr show --json JOB` or `buildr logs --follow JOB` for independent read-only
inspection. Metadata omits commands, log paths, other repositories and raw queue
messages. It grants no cancellation or automatic resubmission.

The private evaluation directory retains `plan.json`, `cases.json`, progress,
`queue-events.jsonl` (including raw queue diagnostics), worker logs, both
transcripts, raw judge requests/replies, `paired-reports.json` and a
content-free `result.json`. Judge replay uses recorded replies and calls no
model. Parse failures or self-disagreement above 0.2 reject completion; a case
passes only at score ≥0.8 with every resolution, accuracy, policy and
non-repetition check true in both passes and no token-limit truncation. These
critical checks are included in the plan; aggregate score cannot compensate
for a wrong resolution, invented facts or asking for information already given.
Successful diagnostics preserve existing `quality.json`; use an explicit
`quality RUN --evidence RUN/evaluations/ID/paired-reports.json` import if desired.
One diagnostic does not meet the six-independent-training-seed study gate,
automatically rank checkpoints or approve deployment. HTTP execution remains unsupported.

### Validate a published checkpoint

`evaluation-targets RUN_ID --workspace WORKSPACE` lists receipt metadata without
hashing model bytes, loading weights or starting inference. Choose an available
name in the evaluation config:

```json
{
  "cases": "/absolute/path/validation.json",
  "judge_model": "YOUR_INDEPENDENT_JUDGE",
  "judge_revision": "YOUR_FULL_COMMIT",
  "checkpoint": "checkpoint-20",
  "split": "validation"
}
```

The case document must declare `split: "validation"` and reviewer provenance
with `validation_is_fresh: true`. Supply genuinely separate validation cases;
declared provenance and exact overlap checks cannot prove semantic separation.
Checkpoint selection is validation-only. Keep fresh test cases for final quality
assessment; validation reports cannot be imported as test quality evidence.
Final test mode currently evaluates the completed artifact, so a validation win
does not automatically nominate or export an intermediate checkpoint.

The local parent must be terminal and its worker must have exited. Review hashes
the published checkpoint, receipt, training inputs and current parent state.
Safetensors adapters from stopped or failed runs and exhausted checkpoints can
be evaluated without restoring or deserializing optimizer state. Evaluation can
use an older trainer's published checkpoint; resume retains stricter trainer
implementation checks. Live workers and unbound adapters are rejected.

For native local GRPO/GSPO/DAPO, training config `checkpoint_every` sets the
committed-update interval. Zero preserves final-only saving; positive intervals
retain at most 100 checkpoints, including the final checkpoint. Reserve storage
for full training state. SFT retains its existing epoch cadence; provider-backed
checkpoint evaluation is not supported.

Agent errors `evaluation_checkpoint_partition`, `evaluation_case_partition` and
`evaluation_provenance` include a safe `diagnostic.expected_partition` without
echoing private case values. Stopping a prepared, unlaunched evaluation marks it
cancelled immediately. Submitted jobs retain cooperative stop behavior.

## Queued validation comparison

```bash
stateset-agents finetune validation-compare RUN_ID ANCHOR_EVALUATION_ID --workspace WORKSPACE
stateset-agents finetune validation-comparison-status RUN_ID COMPARISON_ID --workspace WORKSPACE
# Optional bounded cohort: repeat this flag (the anchor is always included).
stateset-agents finetune validation-compare RUN_ID ANCHOR_EVALUATION_ID \
  --evaluation-id OTHER_EVALUATION_ID --workspace WORKSPACE
```

The anchor must be a completed validation of a worker-published named
checkpoint in a terminal local Linux run. Submission privately retains a frozen
list of completed checkpoint validations and their plan hashes, then uses a
detached local Buildr worker for byte verification and judge replay. There is no
inference, weight download, provider call or training. The worker reserves 1 CPU,
2 GiB RAM and 1 GiB scratch for at most one hour. These disk reads may be large;
do not wrap a submitting command inside another queue allocation.

The JSON result includes `comparison_id`, `job_id`, `state`, and `result` after
success. Submission success means a request was retained; inspect status.
States include `queued`, `running`, `succeeded`, `failed`, `interrupted` and
`submission_uncertain`. Pending/uncertain submissions for the same anchor
reconnect rather than duplicate work. The status command reads only retained
metadata and result checksums, with no Buildr database request, replay, model
hashing, cancellation or resubmission. A missing owner or uncertain queue
receipt is not a completed verification. Inspect the job using `buildr show
--json JOB` and `buildr logs --follow JOB`; ambiguous submission diagnostics are
private in `RUN/validation-comparisons/ID/submission-diagnostic.json`.

Results contain `rows`, `excluded`, `baseline`, `comparison_complete`,
`verified_at` and `recommendation`. Rows rank by success rate, judge score and
earliest saved step; eligibility requires no new case-level success, policy,
truncation or required-check regressions in either judge pass and improvement
over the base. All cases/provenance and evaluation settings must match the
anchor. Baseline recordings must agree; consistent repeats merge into one row.
Incompatible protocols are excluded; unverifiable or conflicting evidence blocks
a recommendation. `recommendation.kind` is `checkpoint`, `base` or `none`.
Every result has `quality_gain_verified: false`.

The scan is bounded to 100 directory entries and never silently truncates. Use
explicit IDs if evaluation history exceeds that bound. Only captured completed
validations are ranked; later or unscored checkpoints are not included. Results
record verification at execution time, not perpetual byte validity. A further
evaluation review rechecks the selected bytes. No export, artifact promotion,
fresh test of a nominated intermediate or independent quality certification is
performed by these commands.

## Nominate a recommended checkpoint for fresh test

`nomination-plan RUN_ID COMPARISON_ID` returns the content-free selection plan
and `plan_sha256` without writing a receipt or hashing model weights.
`nomination-prepare RUN_ID COMPARISON_ID --expected-plan-sha256 SHA256` retains
that exact nomination and returns its `nomination_id` and checkpoint. Both accept
`--workspace WORKSPACE`. Neither starts inference or changes training completion.

For the next `evaluation-plan`, set `checkpoint` to the nominated name,
`nomination_id` to the returned ID, `split` to `test`, and `cases` to a separate
reviewed fresh-test document. Preserve the original evaluation settings when
assessing the validation recommendation. Planning rehashes the selected weights
and verifies the nomination lineage. Preparation and launch retain the existing
exact-checksum protocol and explicit inference boundary.

Test IDs and normalized user turns cannot overlap the run's retained validation
cases, including prepared/failed validation jobs. Training text checks and test
provenance still apply. Validation history is bounded to 100 entries and fails
closed if it is unreadable. Exact overlap checks do not establish external or
semantic independence; reviewer provenance remains an attestation.

Keep `RUN/nominations/ID/receipt.json` with its referenced comparison and
evaluation. It references existing checkpoint bytes, copies no model weights,
replaces no training/quality receipt, and grants no deployment approval. A named
checkpoint without a valid nomination remains validation-only. Nomination
receipts and evaluated results keep `quality_gain_verified: false`; plans do
not certify quality.

To resume a saved nomination without creating another:

```bash
stateset-agents finetune nominations RUN_ID --workspace WORKSPACE
stateset-agents finetune nomination-status RUN_ID NOMINATION_ID --workspace WORKSPACE
stateset-agents finetune nomination-config RUN_ID NOMINATION_ID \
  --expected-plan-sha256 NOMINATION_PLAN_SHA256 --workspace WORKSPACE
```

List rows expose `state: available` or `unavailable`. Available means saved
selection metadata matches its references; `bytes_verified` stays false. The
bounded list includes unavailable records and fails above 100 directory entries;
specific-ID status/config inspection remains possible. Config restoration
requires the selected nomination checksum, clears `cases`, and returns
`prepared: false`. These commands write nothing, read no weights or private case
documents, and submit no queue job. Existing evaluation-plan/prepare/launch
boundaries still apply. The TUI exposes the same saved selections with **h**.

## Training seeds


Training JSON accepts `seed` (integer 0–4294967295, default 42). It is retained
in drafts/plans, reaches local initialization, trainer/data randomness and the
RunPod/Modal/River SFT contracts, and remains unchanged during exact recovery.
Old plans with no seed remain unknown in historical inspection. Seed selection
does not promise bit-identical GPU execution.

`train-remote --seed INTEGER` exposes the shared explicit SFT seed. The remote
executor capability `sft_seed` states support. Local, RunPod, Modal, River,
Hugging Face Jobs, CoreWeave and Nebius transport it; other executors reject an
explicit value before allocation rather than silently ignore it. Omitting it
preserves legacy provider defaults (`RemoteJobSpec.seed=None`: packaged SFT 42,
River 0). River's legacy provider `seed` option remains accepted, but conflicting
top-level/provider seeds are rejected before client construction. Harvest/RL
retain their existing mode-specific seed protocols. Remote worker packages must
contain these changes; unreleased local edits do not update a provider image.

An evaluation's `config.seed` controls evaluation randomness. Its paired reports
use the parent's recorded training seed as `seed`, and
`settings.evaluation_seed` retains the separate evaluation setting. A known
training seed must match the bound quality report. Missing historical training
seeds remain unknown and cannot pass the recorded study gate; an evaluation
seed label never establishes training independence. Old-run evaluation remains
available with its legacy pair label and this limitation.

## Checkpoint recovery protocol

Use `finetune checkpoints RUN_ID --workspace WORKSPACE` to inspect eligible
worker-published checkpoints. Each available choice includes `checkpoint`,
`checkpoint_sha256`, `recovery_sha256`, saved `step`, `epoch`, copy bytes and
remaining original budget. The report never prepares a run or loads a model.
Individual invalid choices have `available: false`; an ineligible parent returns
`recovery_unavailable`. Inspection is limited to 100 receipts, 4,096 regular files
per checkpoint and 1 TiB of bytes.

Pass the chosen name and exact recovery checksum to `prepare-resume`. It copies
the retained dataset and complete checkpoint into a new private run, verifies the
copy and atomically publishes it as prepared. It neither launches training nor
modifies the parent. A failed copy is not exposed as a prepared run. Use the
returned child `run.run_id` and `plan_sha256` with the existing separate
`finetune launch --expected-plan-sha256` protocol. Status includes bounded resume
lineage; the child owns its snapshot independently of later parent/source edits.

Recovery requires a pinned nonzero full model commit, identical framework code,
Linux, a single local training process and stopped parent supervisor/executor
processes. The checkpoint needs a receipt published by its own worker after a
complete save. SFT requires model, trainer, optimizer, scheduler and RNG files;
native RL requires a complete atomic checkpoint publication. The receipt binds
the original plan, training settings, dataset and checkpoint bytes. These checks
repeat before child launch and training. Changed settings or bytes fail; an
explicit checkpoint never silently falls back to fresh training.

The original total epochs/updates stay fixed, with no extension when resuming.
An exhausted checkpoint is unavailable. Copying needs checkpoint bytes, dataset
bytes and 64 MiB overhead in addition to launch resource needs. Remote recovery,
automatic quality promotion and migration from older unbound checkpoints are not
supported by this protocol. SFT's separate legacy boolean `--resume` behavior in
other commands remains unchanged.

## HTTP operator API

The FastAPI gateway mounts `/finetune` with strict authentication and the `admin`
role, including when general routes allow anonymous development access. Configure
server-side `STATESET_FINETUNE_WORKSPACE` to an existing absolute workspace for
retained-run access. These are single-operator controls, not tenant ownership
grants. Clients cannot choose arbitrary server workspaces or dataset/report file
paths. Point the TUI at the same operator workspace.

| Request | Effect |
| --- | --- |
| `GET /finetune/capabilities` | Shared config schema, backends, payload limits and command effects |
| `POST /finetune/validate` | JSON draft configuration; no files read |
| `GET /finetune/runs?limit=20` | Bounded IDs, statuses and model/backend metadata |
| `GET /finetune/runs/{id}` | Review status, model/backend/method and the exact plan checksum for a stop request |
| `GET /finetune/runs/{id}/workflow` | Inspect retained dataset and local setup |
| `GET /finetune/runs/{id}/quality` | Recompute comparisons and artifact hashes |
| `GET /finetune/runs/{id}/checkpoints` | Inspect local published recovery choices; no preparation |
| `POST /finetune/runs/{id}/quality` | Upload paired JSON reports and retain verified evidence |
| `POST /finetune/runs/{id}/stop` | JSON `{"expected_plan_sha256": "..."}`; request local stop |

An HTTP client can review `GET /finetune/runs/{id}` and pass its
`result.plan_sha256` to the stop endpoint. This metadata read works for queued
and running jobs without inspecting datasets, logs, weights or quality reports.
It remains available during another run's artifact inspection. Re-read the
metadata after a stale-checksum refusal; a stop receipt requests cooperative
local shutdown and does not mean an active worker has already exited.

Success uses `schema_version: 1`, `ok: true`, `command` and `result`. Failures use
the gateway's error envelope, HTTP status and specific `error.details[].code`.
Streaming body limits are 64 KiB for configuration, 8 MiB for quality and 4 KiB
for stop requests. Invalid encoding/nonfinite values and malformed JSON yield
controlled errors without reflecting payload contents. Known environment
credential values are redacted from responses.

One setup/artifact inspection runs per application; overlaps return 429 with
`inspection_busy`. Cancellation drains owned background work before capacity is
released. Capabilities, run metadata and local stop remain responsive during hashing. These
routes never launch training, call a judge or claim remote cleanup. HTTP launch,
HTTP recovery preparation, HTTP launch and live provider conformance remain
separate work. Recovery preparation and reviewed launch are available through
the CLI and TUI.

Agents should discover capabilities, list retained runs, then inspect a chosen
run. A successful validation response can still contain `result.valid: false`;
inspect its field issues before preparing through the CLI. A successful quality
request can likewise contain `recorded_study_passed: false`. HTTP success means
the operation completed, not that the configuration or trained model passed.

| HTTP status / detail code | Client action |
| --- | --- |
| `400 / invalid_json`, `invalid_stop_review` | Correct the body; do not retry unchanged |
| `400 / invalid_evidence_or_run` | Check report format, completed artifacts and their hashes |
| `400 / stop_unavailable` | Check backend support and the reviewed checksum; inspect status before another request |
| `404 / run_not_found` | Refresh retained run IDs |
| `413 / request_too_large` | Reduce the payload to the advertised operation limit |
| `429 / inspection_busy` | Retry with backoff after the active inspection; metadata remains available |
| `503 / workspace_unavailable` | Ask the server operator to configure an existing workspace |

Do not treat a disconnected inspection as a released worker slot: its owned
thread finishes before another inspection is admitted. Stop requests are
idempotent for the same reviewed local plan; inspect `status` and
`stop_requested` rather than assuming receipt means the worker has already
stopped. A queued stop remains queued until Buildr admits that worker.

## Inspect datasets

`finetune inspect-data --dataset PATH --method sft` returns counts, captured byte
hash/size, character and turn distributions, message roles and duplicate checks.
Native scenarios use `--method grpo`, `gspo` or `dapo`. It reads one regular file
within 64 MiB, including a regular-file symlink, and shares the preparation
reader/parser. Fixed read limits apply during concurrent appends. Invalid data
returns a controlled error without echoing customer text. It writes no workspace,
prepares no run, loads no model/tokenizer and makes no provider request.

For format failures, `error.code` remains `invalid_dataset` and the additive
`error.diagnostic` contains a rule `code`, safe `message` and `location` object.
SFT locations use physical `line` numbers (including blank lines); JSON syntax
errors also include `column`. RL schema failures identify `example` and, when
applicable, `field`; duplicate IDs identify the later example without its ID
value. Encoding errors identify `byte`. All offsets are one-based; unavailable
locations are omitted. The first failure is reported; fix it and inspect again.
No record, unknown key, source path or offending value is reflected by these
diagnostics. File access errors retain the existing `io_error` contract.

For example, an RL schema failure can return this `error.diagnostic`:

```json
{
  "code": "invalid_scenario",
  "message": "Scenario needs nonempty text list user_responses",
  "location": {"example": 7, "field": "user_responses"}
}
```

Use the location to repair the source, then rerun `inspect-data`. Invalid rows
are never skipped or rewritten automatically; successful inspection remains
separate from preparing and authorizing a training run.

Lengths count characters, not tokens; p95 is nearest rank. Turns count assistant
replies (SFT) or scripted user responses (RL). Duplicates use parsed messages
without IDs for SFT and documented core fields without IDs/extra metadata for
RL. Warnings identify observed duplicates and dominant exact assistant replies;
no records are removed and data quality/model fit is not certified.

No text samples are returned by default. Explicit `--include-samples` adds at
most three excerpts, each at most 4,096 characters, with example index and
truncation flag. These share the report's captured hash; known environment
credentials are redacted. The excerpt limit counts source characters before
credential redaction. This is an explicit private-data read. Preparation
captures data again; an inspection report is not a launch review. Other training
settings are outside this command's scope; use `validate` separately.
`schema.result.dataset_inspection` exposes the sample flag, excerpt limits and
length unit; `effects.inspect-data` is `reads_dataset`.

## Compare experiments

`finetune compare REFERENCE_ID CANDIDATE_ID --workspace outputs/finetune`
returns reference/candidate snapshots, `config_differences`, `dataset_identity`
(`same`, `different` or `unknown`), metric rows and interpretation notes.
Each configuration difference includes presence flags so a missing setting is
distinct from a recorded null. Metric `delta` is candidate minus reference;
missing or nonfinite deltas are null. Steps/timestamps remain separate: these
latest `metrics.json` snapshots are not aligned evaluation samples.

Only bounded regular `plan.json`, `state.json` and `metrics.json` records are
read. Each is limited to 128 KiB; symlinks, special files and damaged records
are unavailable. The response lists unreadable records and caps metrics at 64
finite numeric fields per run with keys up to 128 characters. Invalid/over-limit
metric fields are counted as omitted. It whitelists configuration fields,
excludes nested/oversized values, and applies known-credential redaction.
The command reads neither transcripts nor logs, rechecks no dataset hash,
polls no provider/process and modifies no run. Status is the persisted value.
Same-run comparisons, traversal, missing runs and linked directories are errors.

`quality_gain_verified` is always false: training reward and completion are not
independent quality evidence, and no winner is selected. The TUI shares this
reader through `a` (pin reference) and `x` (compare candidate).

## Prepare, review, launch, monitor

Before preparation, `finetune validate --config config.json` returns
`result.valid` and `result.issues`, with `field`, `code` and a controlled
`message` per problem. It reports independent problems together and does not
echo invalid values or unknown field names. Missing keys use schema defaults;
JSON numeric strings and booleans are rejected for numeric fields. A valid
nonempty dataset path is only a setting here: file availability and contents
still require readiness and preparation.

The command exits 0 with `ok: true` when it completes a report, including an
invalid configuration (`result.valid: false`); inspect that field before
preparing. Unreadable, oversized or malformed config JSON returns `ok: false`
and exits 1. The TUI's `v` checklist uses the same rules and supports jumping
to a field for repair. Neither interface starts training by validating settings.

For private diagnostics, `finetune logs RUN_ID --source supervisor` selects
`supervisor.log`; use `training` for `run.log` or `auto` for nonempty training
output with supervisor fallback. JSON includes `requested_source`, actual
`source`/`filename`, `bytes_read`, `window_truncated`, `available` and controlled
`error` (`missing` or `unreadable`). Missing/unreadable logs return an empty
`lines` list; a successful read request does not imply training succeeded.
Reads stay within the final 128 KiB, even during appends. Limits/offsets address
lines within that window; they do not page back through the full file. Explicit
sources never fallback, and symlinks/nonregular files are rejected.

Create a JSON config using fields returned by `schema`. For example:

```json
{
  "model": "Qwen/Qwen3.5-0.8B",
  "method": "sft",
  "provider": "local",
  "dataset": "/data/reviewed-chat.jsonl",
  "epochs": 3,
  "learning_rate": 0.00001,
  "timeout_seconds": 3600
}
```

SFT uses chat JSONL; native GRPO/GSPO/DAPO use support scenarios. See the
[studio guide](FINETUNING_TUI.md) for both data formats and training semantics.
Local training requires CUDA. RunPod currently supports SFT through this
interface. Catalog maturity is separate from actual loader/hardware support.

`provider` also accepts `modal` and `river` for SFT. These require
`"cost_policy": "provider_billed"` explicitly: this integration does not enforce
a dollar ceiling for them. The RunPod `max_cost_usd` setting is not passed to
these backends. All remote launches require `--allow-paid`. Modal resolves its
SDK credentials/profile; River requires `RIVER_API_KEY`. Use `gpu`/`gpu_count`
for RunPod/Modal; River manages hardware and rejects overrides. The JSON schema
includes backend modes/result kinds and conditional configuration constraints.
River completion reports a validated hosted checkpoint pointer, rather than
local weight files. See [backend selection](FINETUNING_TUI.md#select-a-training-backend).

```bash
stateset-agents finetune readiness --config config.json --workspace outputs/finetune
stateset-agents finetune prepare --config config.json --workspace outputs/finetune
```

`prepare` returns the reviewed configuration/resources, retained dataset hash,
run ID, `plan_sha256`, and `launch_argv`. The input data is copied into a private
snapshot; changing the original dataset afterwards does not change the review.
Preparing again creates a new local run, so retain the returned ID when handling
client retries. Preparation is not a readiness or independent quality certificate.

Before launching, inspect the returned review within your authorized compute and
spending budget. Launch with the exact returned checksum. The worker separately
checks the data hash, package identity and submitted plan checksum after queue
wait; changed plan bytes, snapshots or code require a
fresh review. Execute `launch_argv` as an argument array to preserve paths with
spaces and use the same Python environment, rather than joining it into shell
code. For example, an agent with an already-authorized local training task can
use this pattern:

```python
import json
import subprocess
import sys

prefix = [sys.executable, "-m", "stateset_agents.cli", "finetune"]
workspace = "/absolute/path/to/outputs/finetune"

def request(args):
    process = subprocess.run(args, capture_output=True, text=True, check=False)
    try:
        response = json.loads(process.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("CLI usage or startup error; inspect stderr") from None
    if process.returncode or not response["ok"]:
        raise RuntimeError(response.get("error", {"code": "cli_usage_error"}))
    return response["result"]

prepared = request(prefix + ["prepare", "--config", "config.json",
                             "--workspace", workspace])
# Inspect prepared["review"] before the authorized launch.
submitted = request(prepared["launch_argv"])
run_id = prepared["run"]["run_id"]
status = request(prefix + ["status", run_id, "--workspace", workspace])
```

Launch is detached; the JSON response confirms submission rather than completion.
Poll status with a reasonable backoff. `terminal` is true for succeeded, failed
or cancelled, false for known active states, and null for interrupted supervision
or unreadable/unknown execution. An interrupted run reports
`state.execution_unconfirmed: true`: its detached worker may still be queued or
running. A reviewed local stop remains available, but its receipt does not
confirm worker exit. HTTP run metadata and stop responses expose the same
`terminal` and `execution_unconfirmed` fields. Inspect `state.status`,
`failure`, `unreadable_records`, metrics and completion metadata. Provider-reported
costs remain provider reports. Training reward and completion do not establish
independent policy quality; `quality_gain_verified` remains false.

An owned worker commits success after publishing its checked completion receipt,
even if supervision has disappeared. It does not fabricate an observed process
exit code. An unfinished direct run with missing or changed worker ownership is
unconfirmed, just as a run with lost supervision is. A completion file alone
never promotes a run to success. Recovery still verifies that owners have exited.
Worker and supervisor state updates are serialized; a later worker completion
cannot replace an already recorded failure or cancellation.
Config, plan and optional status receipts must be bounded regular JSON files;
special files and linked records are refused without waiting for their contents.

## Retries and paid compute

Launch requests share the studio's exclusive launch lock. Repeating a launch
with the same reviewed checksum returns `already_submitted: true` and starts
no additional worker, including after failure. This is submission idempotence;
it does not restart training or resume a checkpoint. If a launch response is
lost, retry that exact request or inspect status before preparing another run.

RunPod launch additionally requires `--allow-paid` and `RUNPOD_API_KEY` in the
launching environment. The returned `launch_argv` intentionally omits the paid
flag. Add it only when paid execution is authorized within the reviewed budget.
The existing quoted pod-lifetime ceiling, timeout and provider lifecycle apply;
ancillary costs still need separate review. After remote failure/interruption,
inspect retained provider handles and the provider account for active resources
or charges before a fresh paid run. No command silently repairs or retries a job.

These commands use your OS account and workspace permissions. Keep workspaces
private and authorize which agents can launch. Put credentials in the environment,
not config JSON. Known credential values are redacted from JSON responses.
`logs` is an explicit read of private training output and may contain customer
text; `prepare` and `status` do not include dataset contents. Logs read at most
the last 128 KiB; `--limit` is 1..200 and `--offset` is 0..1000000. Status lists
at most 100 retained runs; request a specific ID for older runs. Run IDs cannot
traverse outside the selected workspace or point through a symlink directory.

## Verification

[Buildr job `fdedd41e32ba`](../benchmark_results/training_contracts/20261016/finetune-studio-agents/README.md)
passed 570 targeted source tests, 459 isolated installed-wheel tests and typing
for all 423 package modules. Source and wheel suites both include all 163
studio/inspection/failure/agent CLI cases. Counts overlap. Repository gates,
formatting and workflow lint passed. The wheel's real CLI returned schema and
model JSON; fresh-process tests verify schema discovery imports neither torch
nor transformers. Package/test Python stayed fixed during acceptance; wheel
Python, recovery profile and rubric bytes match. The live Herdr split was
refreshed with its private draft preserved. Training/provider and fault tests
use tiny CPU fixtures and mocks; no paid run or GPU quality gain is certified.

[Backend selection acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-backends/README.md)
adds Modal/River discovery, explicit provider-billed policy, paid opt-in,
provider routing and hosted-result checks. It passed 786 targeted source tests,
675 installed-wheel tests and typing for all 423 modules in Buildr job
`9bb286e32d88`. Provider calls are mocked; this is not live training certification.

[Shared settings validation acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-validation/README.md)
passed 816 targeted source tests, 705 installed-wheel tests and typing for all
423 modules in Buildr job `8d8e95da3e2e`. It verifies strict JSON numeric types,
all primary field diagnostics, no input-value reflection, no dataset/provider
side effects, root CLI registration without training imports, and the TUI's
matching repair workflow. Counts overlap; no paid training was performed.

[Source-aware log acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-logs/README.md)
passed 831 targeted source tests, 720 installed-wheel tests and typing for all
423 modules in Buildr job `71c5cce5bd41`. Both suites cover explicit sources,
fixed read bounds during appends, missing/unreadable source metadata,
symlink/special-file rejection, credential redaction and the matching TUI
viewer. Real terminal checks exercise pause, filter and source switching.
Counts overlap; provider calls remain mocked and no paid training was launched.

[Experiment comparison acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-comparison/README.md)
verified 861 source tests, 750 installed-wheel tests and typing for all 423
modules. It covers comparison JSON, missing-setting presence, dataset hash
states, finite deltas, metric limits, record boundaries, credential redaction,
invalid run IDs, read-only behavior and root CLI use without training imports.
The matching TUI pin/compare workflow is covered in real terminals. Queue
interruptions and recovered stage evidence are recorded separately; the final
checks passed in job `e3db43feb339`. Source/wheel counts overlap; no paid run or
independent quality gain was verified.

[Dataset inspector acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-dataset/README.md)
passed 887 source tests, 776 installed-wheel tests and typing for all 423 modules.
It covers default statistics without text, explicit bounded excerpts, credential
redaction, schema capability metadata, parser/read bounds and root CLI use
without training imports. The installed wheel's real CLI runs with and without
sample opt-in; real terminals exercise the shared inspector at two pane sizes.
Final checks passed in Buildr job `ac3daeb94a5f`. Counts overlap; no paid training
or independent quality gain was verified.

[Quality, setup and operator API acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-quality-api/README.md)
passed 1,025 targeted source tests, 914 installed-wheel tests and typing for all
425 package modules. It verifies shared schema discovery, incomplete-draft
workflow inspection, private paired-evidence import/reverification and
checksum-bound cooperative local stop. HTTP tests cover strict admin access,
bounded finite JSON, workspace restrictions, secret redaction, OpenAPI bodies,
inspection concurrency and cancellation ownership. Real installed CLI checks
reject a stale stop checksum and expose 32 synthetic outcome regressions despite
higher recorded reward. Three wheel groups match selected source case identities
without skips or failures; counts overlap. Final checks passed in Buildr job
`07bc3da201b9`. Earlier failures and the repaired shared-fixture harness are
retained. No paid run, live provider certification or model-quality gain was
verified.

[Local checkpoint recovery acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-recovery/README.md)
passed 1,183 targeted source tests, 1,072 installed-wheel tests and typing for all
426 package modules. It covers content-free checkpoint discovery, checksum-bound
private preparation, separate launch authorization, fixed original budgets and
recovery lineage. HTTP provides authenticated inspection; preparation and launch
remain CLI/TUI operations. The installed CLI rejects a stale checksum without
creating a child and prepares an exact private snapshot without starting training.
Tiny real CPU SFT/native tests verify state recovery, and real terminal tests
exercise the two reviews. Final job `2c6e30457995` passed with frozen package/test
Python and matching wheel case identities; source/wheel counts overlap. No paid
run, remote recovery, GPU recovery or model-quality gain was certified.

[Paired-evaluation acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-evaluation/README.md)
passed 1,265 source tests, 1,154 installed-wheel tests and typing for all 429
package modules in Buildr job `16e925fa033f`. The frozen source, actual wheel
bytes and selected test identities match; counts overlap. Installed CLI checks
verify stale-review refusal, private preparation, progress, evaluation-only stop
and refusal to verify unfinished results without starting inference.

The suites cover strict judge success gates, recording replay, bounded queue
metadata, real tiny CPU model/adapter generation and captured-editor case
selection. Actual installed terminals choose files without creating an
evaluation or changing the draft. Positive judge transport is injected. No paid
run, live judge model, GPU, Windows, remote evaluation or independent quality
gain was certified. One diagnostic does not select a production checkpoint.

[Published-checkpoint acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-checkpoint-evaluation/README.md)
passed 1,572 targeted source tests, 1,461 installed-wheel tests and typing for
429 modules in Buildr job `4987a3a77fc0`. Source/wheel counts overlap; frozen
package/test inventories and selected wheel case identities match. Installed
JSON commands discover targets without model-byte hashing, prepare a bound
validation evaluation without launching, reject test-partition selection with
a controlled diagnostic, and cancel an unlaunched evaluation immediately.
Real tiny CPU and terminal tests cover save cadence and checkpoint selection.
No paid run, live judge/GPU certification or independent quality gain occurred.

[Final training-seed acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-seed-zero/README.md)
passed 613 current-source and 613 matching wheel tests, including 63 seed cases,
and typing for 429 modules in job `d808523574bf`. Source/wheel counts overlap.
The broader seed baseline passed 1,839 source and 1,728 wheel tests; only the
zero-value renderer and its tests changed afterward. Evidence covers retained
seed settings, provider rejection before allocation, real CPU adapter seed
identity, recovery, distinct training/evaluation seed bindings and truthful
unknown historical seeds. The refreshed Herdr draft is preserved. Provider and
judge boundaries are injected; no paid run or GPU quality gain was certified.
