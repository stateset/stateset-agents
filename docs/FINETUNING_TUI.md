# Fine-tuning in Ghostty and Herdr

Open the studio in a Ghostty or Herdr terminal pane, using the environment where StateSet
Agents is installed:

```bash
stateset-agents tui --workspace outputs/finetune
# From this checkout:
.venv/bin/python -m stateset_agents.cli tui --workspace outputs/finetune
```

The studio uses POSIX curses and the package's existing dependencies. No web
server or extra UI installation is needed. Phone-sized panes are supported from
32 columns by 16 rows. A pane of 72 columns by 22 rows or larger shows the full
desktop controls. Larger panes show more settings and logs. WSL is required on Windows.

## Desktop navigation

Press **Ctrl+P** on Configure or Runs to search available commands by words or
shortcut. Arrows and Home/End move through matches; Enter opens the selected
workflow, and Esc closes the palette. Backspace edits the search and Ctrl+U
clears it. Digits remain search text on compact screens. Launch and stop commands
still open their existing review screens; choosing a command never confirms them.
The **:** Actions menu remains available without typing a search.

On Configure, **PageUp/PageDown** move through a page of settings and **Home/End**
jump to the first/last field. These keys continue to move the cursor while editing
text. On Runs, Home selects the first matching run; End continues to follow logs.
Desktop headers show the workspace, retained active-run count and current setting
position. Taller Runs views show more retained rows.

The interface uses the terminal's own palette and background. Headings, selected
rows, separators and validation states have distinct weight or color, including
a monochrome fallback. Unchanged rows are left intact during refresh; resized
views redraw completely. Escape responds promptly by default. An explicit
`ESCDELAY` environment setting is respected for terminals that need a longer
escape-sequence timeout.

## Phone-sized terminals

Below 72 columns or 22 rows, the studio uses a shorter header and two rows of controls to
leave room for the selected setting. Outside text editors and search/filter
fields, **2** moves down, **8** moves up, **3** pages down, **9** pages up, and
**0** goes back. **4** switches Configure/Runs on the main screens. Arrow keys and PageUp/PageDown still work. Use **Enter** to edit,
select or explicitly confirm the displayed review. No number shortcut launches
training or inference. **?** opens the complete keyboard guide.

All digits remain literal while editing numeric settings, paths, model searches,
dataset filename filters, log filters and evaluation values. Use the terminal's
Esc control to leave a text editor. Backend selection follows the highlighted
row on small screens; page shortcuts scroll its longer details.

If a resize or phone rotation leaves fewer than 32 columns or 16 rows, the
interface pauses input actions. Hidden Enter presses cannot confirm a launch
or commit an edit. If the viewport changes while a key arrives, the studio
repaints before accepting actions. Resize to resume the same draft/editor, or press **q** or
Ctrl+C to close the UI. Training jobs continue independently. These controls
are for the terminal UI; no native mobile app is required.

Preview a phone-sized screen without opening a terminal, writing a draft or
launching work:

```bash
stateset-agents tui --snapshot --columns 40 --rows 24
```

Explicit dimensions require `--snapshot`; interactive rendering follows the
actual terminal size, including rotations and resizes.

## Configure and train

Press **:** on Configure or Runs to open **Actions**. Use arrows (or **2/8**
on a phone), page keys and Enter to choose a named workflow. The menu keeps the
selected action visible at every supported terminal size. Its launch, stop and
recovery actions open the existing review screens; they do not confirm work.
**Guided setup** is the first Configure action and is also available with **w**.

Preview (**p**) and launch review (**r**) validate the captured dataset in the
background. The interface remains responsive while parsing larger files.
**Esc**, or **0** on a phone, discards the pending preview; its current read may
finish in the background. A cancelled or outdated result cannot reopen a launch
review. A second preview waits until the previous read finishes. Once the review
is displayed, a separate Enter is required to launch.

The Runs search and status filters include **all retained runs**, including older
active runs beyond the first 100. Interactive history refreshes run in the
background, preserve the selected run, and show an explicit loading or failure
message rather than presenting an incomplete load as an empty workspace.

1. Press `b` to select local CUDA, RunPod, Modal or River, or cycle the Compute
   field. Choose SFT, GRPO, GSPO or DAPO locally; remote studio backends support SFT.
2. Press `m` to search the model catalog, or edit the Model field directly.
   Catalog maturity is shown; it does not certify that every model fits your
   hardware or works with the chosen training path. Specialized multimodal and
   frontier loaders remain available through their dedicated CLI commands.
3. Enter the dataset path and training settings. `Enter` edits text or cycles
   choices; `f` opens a JSON/JSONL file picker with directory navigation and
   filename filtering. Selection updates the draft without reading file contents.
   Generated industry `train.jsonl` files keep their synthetic provenance in
   inspection and launch notes. Declared validation/test rows are rejected as
   training inputs; metadata is not passed into model messages. Use
   `industry audit-datasets` to verify the full generated bundle separately.
   Arrows move between fields or move the cursor within an edit. Home/End jump
   to either end, Delete removes the next character, `Ctrl+U` clears an edit, and `Esc`
   discards it. Only settings used by the selected method are shown.
4. Press `i` for background local readiness checks (details below).
   Press `v` for a configuration checklist; Enter jumps to a selected problem.
   Press `p` for free validation and resource planning, then `r` to review the
   exact dataset snapshot and launch settings. `Enter` in this review starts
   training; `Esc` returns without starting anything.
5. `Tab` opens retained runs. Arrows select a run, `PageUp`/`PageDown` scroll
   its log, and `End` follows the tail. `q` closes the UI while jobs continue.
   Open the same workspace to reconnect. `?` shows the keyboard guide.

Narrow split panes use a single-column form with the complete model ID above
the settings. Wider panes also show a resource preview. Long identifiers and
details wrap using terminal cell widths, including wide Unicode characters.
Launch review, help and run details scroll through their complete content.

In the dataset picker, type to filter filenames, Enter opens a directory or
selects a file, and Backspace removes filter text or goes up a directory when
the filter is empty. Esc leaves the draft unchanged. Hidden entries are omitted;
each directory listing inspects at most 2,000 entries.

On the Runs screen, Enter (or `d`) opens full model/configuration details,
elapsed time including queue wait, retained output paths, latest metrics,
provider handles and provider-reported cost when available. Automatic refresh
preserves the selected run when newer runs appear. Press `c` to reuse settings
for a **new** run using its retained dataset snapshot; its hash is checked
before reuse. The original run and artifacts remain intact. Reusing settings
starts no work: validate and confirm a fresh launch review. Recorded completion
and provider cost are not independent quality or billing verification.

For an agent-prepared run, press `r` on Runs (or in run details) to review its
saved plan, checksum, dataset snapshot and resource reservations. Enter launches
that **same run ID**; Esc leaves it prepared. This preserves the Configure draft
and does not depend on the original dataset file still existing. Changed plan
bytes, snapshots or package code prevent submission; an already-submitted run
cannot launch twice. Use `c` when you want to edit settings for a new run.

`s` saves settings in `workspace/draft.json`, which is loaded on the next open.
Incomplete drafts can be saved before choosing a dataset; launch still requires
successful validation. Changing the model clears a previous model's commit pin
so it cannot accidentally be applied to a different repository. Ctrl+C exits
from any screen, including an unfinished edit, without stopping detached jobs.
Use `--config /path/to/draft.json` to load a different draft. A static view works
without a terminal and never launches training:

```bash
stateset-agents tui --workspace outputs/finetune --snapshot
```

## Guided setup, quality and local stopping

Press `w` on Configure for guided setup: Settings, Dataset, Runtime, Review
launch and Evaluate quality. Checks run in the background without preparing a
run or calling providers. Up/Down selects a step; Enter opens its existing tool,
and PageUp/Down scrolls details. Editing settings marks the snapshot stale;
`w` rechecks. Opening Review captures inputs and shows a separate confirmation;
training starts only with Enter inside that launch review.

On Runs or in run details, `e` opens quality evidence and `k` reviews a local
stop request. Quality inspection hashes completed weights in the background;
Esc returns while it finishes. Press `f` inside the quality view to enter a
paired-report path, then Enter to validate and retain private `quality.json`.
Failed imports preserve earlier evidence. Reopening or pressing `e` recomputes
comparisons and rechecks weight/plan bytes.

The view shows success rates, gains, wins, regressions and individual gate
results. Regression examples use hashed case handles; source IDs/transcripts
remain hidden. A one-seed diagnostic stays separate from the recorded study
gate: six seeds, significance, distinct checkpoint hashes, at least 32 distinct
cases per seed, pinned model/judge/rubric metadata, reviewer provenance and
paired safety gates. Only this run's artifact is rehashed; other artifacts,
independence and evaluator identity remain recorded claims. Neither view selects
a checkpoint or approves production deployment. See the [agent protocol](FINETUNING_AGENTS.md).

The local stop review shows the run and checksum. Enter writes a cooperative
request; Esc changes nothing. Prepared runs become cancelled. Queued entries
remain in Buildr and exit before model loading when admitted. Active SFT
requests save/stop at a step boundary; native RL saves `stopped-step-N` after a
committed update before cancellation. Stops may wait for loading or an active
update; they never kill PIDs or cancel queue jobs. Runs show pending requests;
agent status includes a recovery checkpoint when recorded. Changed package
implementations reject new requests rather than promising control of older code.
Remote stop needs provider-specific cleanup controls. Quitting the TUI
continues to leave launched workers running.

## Run a paired evaluation

Press `j` on Runs, in run details, or inside the quality view to open evaluations
for that captured run. Press `n` to configure a new evaluation. Up/Down selects
fields and Enter edits them. Choose a held-out cases JSON file, an independent
judge model and its full commit, response/context limits and queue resources.
Press `r` to capture a review in the background. It shows the base/adapter/cases
identities, maximum generation workload, resource requests and timeout. Enter
in that review prepares and submits the evaluation; Escape starts nothing.

Press `f` from the evaluation form to browse held-out JSON files. Type to filter,
use arrows or PageUp/Down to select, Enter to open a directory or choose a file,
and Backspace to go up. Escape returns to the same form without changing it.
The case picker lists regular `.json` files and directories, omitting JSONL,
hidden files, file symlinks and special files. It examines at most 2,000 directory
entries. Choosing a file reads no contents, changes no training settings or
saved draft, and starts no work. It clears any old evaluation review; press `r`
to validate the selected cases and capture a new review. A changed or busy
evaluation editor rejects a late selection.

Final-artifact evaluation requires a successful local run with a pinned Hub base model and a
local safetensors adapter. Both policy and judge must already be in the local
Hugging Face cache. The loader resolves the exact cached commit before loading,
so a matching relative directory cannot override the requested revision.
Model loading uses `local_files_only=True`, native model code
and safetensors; missing weights fail instead of downloading or substituting a
dry run. Buildr bounds the entire job. Its CPU/RAM/scratch reservations do not
reserve a GPU; use the local GPU placement configured for your workstation.

Baseline, candidate and judge run sequentially in separate processes. Each
policy sees the same fixed user turns and supplied policy facts with greedy
decoding. Oversized prompts fail rather than discard context. The judge scores
each full transcript twice in reversed checklist order, with at most one parse
retry. Parsing failures or disagreement above 0.2 reject the evaluation. A
successful case requires score at least 0.8 and every resolution, accuracy,
policy and non-repetition check to pass in both judge passes, with no token
limit truncation. A high mean cannot hide invented facts, a wrong resolution,
redundant requests or repetition. These fixed thresholds are included in the
reviewed protocol.

The evaluation list retains progress after leaving the view or restarting the
TUI. Open a prepared row and press `r` to review and submit that same ID. Open
any row to see its phase and status; `l` reads its bounded log tail,
and details show the Buildr handle, sampled queue wait, admission constraint
and disk/RAM headroom when available. The sample age is explicit: old progress
does not prove that a runner is alive or has failed. Lifecycle metadata is
retained separately from worker output, survives reopening, and can be followed
with the displayed read-only `buildr show --json JOB` command.

From details,
`s` reviews a cooperative evaluation-only stop, and `v` rehashes the completed
adapter and replays judge recordings without model calls. A stop takes effect
between generations; current inference and queue admission may finish first.
Training and other jobs are unaffected. Raw cases, transcripts and judge
requests/replies stay in the private `RUN/evaluations/ID` directory. Results do
not replace an existing `quality.json`; explicitly import `paired-reports.json`
through the quality view when desired.

This is a one-run diagnostic; independent multi-seed quality certification and
automatic checkpoint ranking remain separate. Scripted customer turns do not exercise tools or
adaptive customers. Case review and freshness are declared provenance, with
exact training-text overlap checks; semantic leakage still needs human review.
See [the case format and agent commands](FINETUNING_AGENTS.md#local-paired-evaluations).

## Choose reproducible training seeds

Set **Training seed** in Configure (default 42, range 0–4294967295). It is saved
in the private draft, shown in launch review and run details, and included in
the immutable plan, experiment comparison and exact recovery settings. Old
drafts use 42; old runs without recorded seeds remain explicitly unknown.

Local SFT seeds model/adapter initialization before loading, trainer randomness
and the data sampler; adapter manifests retain both training and data seeds.
Native RL seeds before agent initialization and passes the same seed to the
trainer. RunPod and Modal transport the value to packaged SFT; River passes it
to LoRA initialization and epoch shuffling. Remote worker packages must include
this unreleased feature. For a development RunPod run, set
`STATESET_AGENTS_WHEEL` to a built wheel containing these changes; Modal uses
its installed framework version, not local source edits. No live provider
certification is implied. Identical seeds do not guarantee identical GPU results.

Evaluation review shows training and evaluation seeds separately. Paired report
labels use the recorded training seed; evaluation randomness is a separate
protocol setting. Re-evaluating one checkpoint with different evaluation seeds
cannot create independent training evidence. Quality import rejects a bound
report with a mismatched training seed, and historical runs with unknown seeds
cannot pass the recorded study gate. Other seed artifacts still require review.

## Validate a retained checkpoint

For new local GRPO/GSPO/DAPO runs, set **Checkpoint interval** to the number of
committed optimizer updates between saves. Zero preserves the final-save-only
default. Positive intervals retain at most 100 full-state checkpoints, including
the final save. The launch review reports the expected count; reserve workspace
disk for weights, optimizer and RNG state before launch. SFT keeps its existing
epoch checkpoint policy; a positive interval is rejected for SFT and remote
backends rather than silently ignored.

Open a terminal local run with `j`, then `n`, then `c` to choose its final adapter
or a worker-published checkpoint. Discovery reads receipt metadata and checks
adapter-file presence, without hashing model or optimizer bytes. Arrows,
PageUp/Down and Home/End navigate; Escape preserves the form. Selecting a
published checkpoint sets the partition to **validation** and clears any old
evaluation review. Use `f` to choose matching reviewed validation JSON, pin the
judge, then `r` to hash the complete checkpoint and capture its exact receipt.
Enter in that review remains the only submission action.

Checkpoint evaluation requires Linux single-process ownership, a terminal local
parent, exited worker/supervisor processes, intact publication metadata and a
safetensors adapter. It can inspect stopped/failed runs and checkpoints that
exhausted their original training budget. Optimizer state is hashed without
deserialization. Inference can use a checkpoint from an older trainer build;
`z` recovery still requires matching trainer code and remaining budget.

Validation findings never pass the held-out test study gate or import as final
quality evidence. Each retained evaluation contains exactly one replayable
base/candidate pair. Published checkpoint targets accept validation cases;
fresh-test mode currently evaluates the completed final artifact. Individual
evaluations do not change the completed artifact or promote weights. Use the
explicit comparison below for a validation recommendation. Review/freshness are declared provenance; separate validation
and test cases still require reviewer control, including semantic overlap.

Stopping an unlaunched prepared evaluation marks it cancelled immediately.
Queued/running evaluations retain cooperative stopping at safe boundaries;
training and other queue jobs remain unchanged.

## Compare validated checkpoints

Open `j`, select a completed validation of a named published checkpoint, and
press **b**. The studio captures the completed checkpoint-validation history
and submits a detached local Buildr verification (1 CPU slot, 2 GiB RAM, 1 GiB
scratch reservation, one-hour timeout). It rehashes checkpoint bytes and replays
retained judge recordings. It generates no replies, downloads no models and
starts no training or provider work. Large checkpoints can take time to read.

The comparison survives leaving the view or exiting the TUI. Reopen `j` then
**g** for the latest retained comparison. **r** refreshes metadata only; **b**
requests another verification after completion, or reconnects to the same
pending request. The Buildr handle supports independent queue/log inspection.
Uncertain submission or interrupted verification never triggers automatic
retry. Submission diagnostics remain private in
`RUN/validation-comparisons/ID/submission-diagnostic.json`.

Only records matching the anchor's exact cases/provenance, parent/model,
training/evaluation seeds, judge/rubric, generation limits, decoding and code
protocol belong to its cohort. Resource reservations may differ. Different
protocols are explicitly excluded. Changed bytes, invalid evidence, inconsistent
baseline replies/scores or conflicting repeated evaluations prevent a
recommendation. Consistent repeats count as one checkpoint, not independent
training seeds. Metadata scans fail above 100 entries rather than silently
truncate; the agent CLI supports explicit evaluation IDs for a bounded subset.

Rows rank by successful outcomes, judge mean and earliest saved step on ties.
A checkpoint is eligible only if it improves over the base and introduces no
case-level success, policy, truncation or required-check regression in either
judge pass. A high average score cannot hide a new failure. The recommendation
keeps the base if no eligible checkpoint improves. It covers the captured
completed cohort, not unscored checkpoints or later evaluations. To include the
final saved checkpoint, validate its named `checkpoint-N` target explicitly.

Use arrows/Home/End to inspect rows, Enter for the original evaluation, or **c**
to configure validation of the recommended checkpoint. This restores the
anchor's retained case snapshot and protocol without changing the training
draft. **r** in the form captures a new byte-bound review; Enter in that review
is still required to submit an evaluation. The base recommendation does not
select the completed final adapter. These are recorded verification results;
bytes are rechecked before a new evaluation. Comparison alone does not nominate,
export or promote a checkpoint, and does not certify an independent quality gain.

### Nominate a checkpoint for fresh test assessment

On a completed comparison with a checkpoint recommendation, press **n** to review
the nomination. **Enter** retains a private, hash-bound receipt and opens a test
form for that exact published checkpoint. **Esc** returns without retaining a
receipt. The form clears the validation case path: select a separate reviewed
test JSON, then **r** captures the complete evaluation workload and rehashes the
checkpoint. **Enter** in that evaluation review is still required for inference.

The receipt binds the parent plan, comparison result, selected evaluation and
checkpoint bytes. Changed selection evidence, receipts or checkpoint bytes are
rejected. Test IDs and normalized user turns must be distinct from retained
validation history, including prepared and failed validation jobs; an unreadable
history or more than 100 entries fails closed. Existing training-overlap checks
and reviewer provenance requirements also apply. These are exact overlap checks;
reviewer freshness remains an attestation, not proof of semantic or external
dataset independence.

Receipts live at `RUN/nominations/ID/receipt.json`. They reference the existing
checkpoint; they copy no weights, change no completion/quality receipts, start
no training, and approve no deployment. Fresh-test results retain their own
evaluation and judge recordings. The original validation-only rule still applies
to any named checkpoint without a valid nomination. Selecting another checkpoint
through the target picker clears the nomination.

Press **h** inside the evaluation list, details, comparison or form to reopen
saved nominations. Choose a row and press **Enter** to restore that exact
selection into a fresh-test form with no cases selected. Reopening creates no
nomination, evaluation or queue job. **Esc** returns to the originating screen
and preserves an existing unsaved form. Leaving during a background restore
also preserves the current editor. **r** in the browser refreshes its metadata.

The browser lists up to 100 entries, newest first. Broken or stale records stay
visible as unavailable; they cannot replace the form. A larger history fails
closed and can be inspected by a specific ID through the agent CLI. “Metadata
ready” checks nomination/selection lineage, not current weight bytes or case
freshness. The later evaluation review rehashes weights and checks fresh cases.

## Recover a local checkpoint

Press `z` on Runs or run details to inspect worker-published checkpoints in the
background. Select a checkpoint with Up/Down, then Enter to review its identity,
remaining budget and copy size. Enter in that review copies the checkpoint and
retained dataset into a **new private run**. A separate prepared-run review
follows; only Enter in that launch review starts training. Esc before preparation
writes nothing. Esc during copying returns to the studio while the authorized
copy finishes. Exiting waits for an authorized snapshot or launch submission to
finish; read-only inspection can be abandoned.

Recovery currently requires Linux, one local training process, a pinned nonzero
model commit, matching framework code, a terminal parent and stopped supervisor
and executor processes. New workers publish plan-bound receipts after complete
SFT or native checkpoint saves. Older model-only exports and checkpoints without
these receipts are unavailable. A checkpoint that has exhausted its original
budget is also unavailable; recovery preserves the original total epochs or
updates and all training settings.

The report hashes checkpoint bytes and checks complete optimizer/scheduler/RNG
state files. Preparation copies regular files into a private snapshot without
links to the source. Launch and execution verify the copied bytes, dataset and
training settings again. Altered inputs fail instead of starting fresh. Native
recovery restores the trainer's saved cursor and full state; SFT passes an exact
checkpoint path to HF Trainer. Parent logs, stop requests and artifacts remain
unchanged, and the new run starts its own metrics history.

Inspection permits up to 100 recovery receipts, 4,096 files per checkpoint and
1 TiB of checkpoint bytes. Copying needs enough free workspace space for the
checkpoint, dataset and 64 MiB overhead; queue scratch and model-cache needs
still apply at launch. Recovery does not select a quality winner or certify GPU
or provider behavior. Remote recovery remains provider-specific. `c` continues
to reuse settings for fresh training and does not resume optimizer state.

## Inspect the dataset before training

Local SFT checks its captured input against the reviewed dataset hash before
model loading, then carries that source identity through adapter publication.
Later edits or deletion cannot relabel the adapter with a different file's hash.
The manifest also records a hash of normalized training rows and separate source,
parsed-input and effective training-row counts. These records describe the input
used by the worker; they do not establish independent data review or model quality.
Fresh local SFT also verifies its safe LoRA export and persisted lineage before
returning successfully. Structural export checks do not run inference or certify
quality, and older/provider-specific artifacts retain their separate checks.

Press `u` on Configure to read dataset statistics in a background thread. Esc
returns while the scan finishes; `u` in the report rescans after completion.
Other draft settings need not be launch-valid. The scan validates only the
selected dataset format and never prepares a run, loads a model/tokenizer,
contacts providers or writes a workspace.

The report includes the captured byte hash/size, example count, character and
turn distributions (minimum, median, nearest-rank p95, maximum), message roles
for SFT, and duplicate counts. Characters are Unicode code points, not tokens.
SFT turns count assistant replies; RL turns count scripted user responses,
not completed rollouts. Duplicate checks do not remove or alter any examples.
SFT compares parsed messages with IDs ignored; RL compares the documented core
scenario fields with IDs and extra metadata ignored. Repeated assistant replies
are flagged when at least four replies exist and one exact reply makes up half
or more. These observations do not certify data quality or model fit.

Private text is hidden by default. Press `e` to reveal/hide a sample; Left/Right
selects among the first three examples. Each excerpt is limited to 4,096
characters and clipping is labelled. The UI retains these bounded excerpts in
memory from the same captured bytes as the statistics. Use PageUp/Down to
scroll the complete report. Rechecking or closing hides samples again.
Changed path/method settings mark a report stale and hide its contents until
rescanned. External file edits are not watched; preparation reads and validates
again, so this report is not an immutable launch review.

Inspection and preparation share a fixed 64 MiB read limit, including when a
file grows while being read. Regular-file symlinks remain supported; special
files, including FIFOs, are rejected without reading them. Invalid formats yield
a controlled diagnostic without echoing dataset contents. AI agents can use
`finetune inspect-data --dataset PATH --method sft` (or `grpo`, `gspo`, `dapo`);
private excerpts require `--include-samples` explicitly.

Format errors show the first failing physical SFT line or RL example/field;
syntax errors include a column and encoding errors a byte offset when known.
All positions are one-based, including blank SFT lines. Fix the source and press
`u` to retry. Reports never echo offending values or unknown field names.
The footer updates when a scan completes or fails; results from old settings
request a rescan, and a later unrelated status message is preserved.

## Find and fix configuration problems

Press `v` on Configure to list all settings issues together. Up/Down selects an
issue, PageUp/PageDown moves by eight, and Enter returns to that setting.
Each field gets a primary diagnostic; recheck after edits to refresh constraints.
The form reveals an invalid setting even when the selected backend normally
hides it. Edit it, then press `v` again; the refreshed checklist hides unused
fields again. Esc closes the report without changing values or the launch preview.

The checklist covers numeric types/ranges, required dataset path, model/revision
format, backend/method compatibility, explicit uncapped billing policy, River
hardware overrides and LoRA limits, and native RL generation requirements. It
uses the same rules that block invalid training launches. Invalid input values
are not repeated in diagnostic messages. Text edits accept numeric strings;
booleans are never converted into numeric settings. Incomplete drafts remain
saveable using `s` when their types are valid.

This check does not read dataset contents, inspect hardware, authenticate a
provider, prepare a run or start training. A passing report is not readiness:
use `p` for strict dataset validation, `i` for local readiness, and `r` for a
separate launch review. AI agents can use the same rules through
`stateset-agents finetune validate --config config.json`.

## Select a training backend

`b` opens the backend picker from Configure. Up/Down selects; Enter applies;
Esc preserves the draft. Selection never contacts a provider or starts a job.
The picker explains supported studio methods, setup, billing and result type.
Changing backends preserves training settings, so incompatible settings must be
fixed before validation. River clears GPU overrides because it manages hardware.

| Backend | Studio methods | Result | Setup and billing |
| --- | --- | --- | --- |
| Local CUDA | SFT, GRPO, GSPO, DAPO | Local weights | Training extra, CUDA, local Buildr CPU/RAM/scratch reservations |
| RunPod | SFT | Downloaded weights | `RUNPOD_API_KEY`; enforced quoted pod-lifetime ceiling, ancillary costs separate |
| Modal | SFT | Downloaded weights | Modal extra and configured authentication; provider billed, no enforced dollar ceiling |
| River | SFT | Hosted checkpoint pointer | Python 3.12+, River extra, tokenizer dependencies, `RIVER_API_KEY`; provider billed, no enforced dollar ceiling |

For Modal/River, explicitly change **Cost policy** to `provider_billed` before
preview. The launch review repeats that there is no enforced dollar ceiling;
the RunPod ceiling is hidden and is not passed to these executors. Set provider
account limits and review current pricing yourself. The agent CLI also requires
`--allow-paid` for every remote backend. All launches still require confirmation.

RunPod and Modal expose Remote GPU and Remote GPUs. Blank GPU uses the RunPod
catalog recommendation or Modal's A10 default; a GPU count of 0 uses the
recommendation (one GPU for Modal). These are resource requests, not
evidence that a model fits. Remote backends do not reserve local Buildr slots.
Modal's timeout bounds remote function execution; image building, artifact
download and cleanup are additional. River uses provider request timeout settings
and has no whole-run deadline in this integration. Closing the TUI preserves the
supervisor; it does not cancel remote resources or charges.

Studio remote job specifications include the reviewed framework release version.
For a reviewed release install, the worker clears an ambient
`STATESET_AGENTS_WHEEL` override. A reviewed RunPod development wheel is copied
into the private run snapshot and bound by SHA256 and byte count. The executor
checks that regular archive before allocation and checks the uploaded archive
again before installation. A changed archive fails before pip runs. This binds
the reviewed package archive; it does not attest installed dependencies or the
remote runtime's source bytes. A release pin identifies a release rather than
proving a particular build or dependency environment. RunPod renders model
names, dataset paths and installation requirements as quoted shell arguments.

Modal installs the matching published StateSet Agents version remotely; it does
not upload uncommitted source. Configure a published version before live use.
[Modal authentication](https://modal.com/docs/sdk/py/latest/config) can use
environment credentials or the active SDK profile; readiness does not read stored
tokens. River's executor checks [account model access](https://docs.river.ai/guides/models/)
before opening a training session. Choosing a model in the catalog does not
establish access. Local readiness makes no authenticated provider request.

This River path retains `river_checkpoint.json`, a `river://` URI and provenance;
it does not download weights. Completion requires a valid pointer matching the
reviewed model, LoRA rank, epochs and learning rate, plus a positive step count.
Other backends still require nonempty weight files. A result is not independent
quality evidence. River RL remains available through its
[dedicated workflow](RIVER_PROVIDER.md); native studio RL is not silently
translated into a different River objective or data format.

Remote run details show **Provider submission / retained observations**. RunPod
and Modal workers retain a private, plan-bound `provider-submission.json` while
their synchronous submission is still running. Allocation intent records the
logical resource name before creation; returned pod, hydrated volume and Modal
function-call IDs are retained before subsequent work. An interrupted request
may have allocated a resource before returning its ID, so unresolved intent is
shown explicitly. River's managed path does not claim the same allocation
receipt capability.

AI agents can read these receipts with
`stateset-agents finetune provider-status RUN_ID --workspace WORKSPACE`.
The authenticated admin API exposes the same bounded local inspection through
`GET /finetune/runs/{run_id}/provider`. Missing receipts are reported as
unavailable; malformed, changed-plan, linked or oversized receipts return a
controlled error. These views make no provider request, reconnect no job and
never retry, cancel or resume training. Known IDs support explicit provider
reconciliation; retaining an ID does not establish a live status or successful
artifact recovery. Modal uses an ephemeral app, so a retained call ID does not
promise work continues after client disconnection.

A cleanup acknowledgement records the provider SDK accepting a termination,
deletion or cancellation request. It does not independently prove provider
terminal state or that billing stopped. Check the provider account before a
new paid launch whenever cleanup is required or unconfirmed.

## Readiness, run search and progress

`i` checks the current configuration draft in a background thread. You can
leave the report while it runs; press `i` in the report to check again. Editing
settings makes the earlier report visibly stale. This advisory snapshot checks
file availability/size, local disk, installed training packages, NVIDIA hardware
when `nvidia-smi` exists, and configured Buildr CPU/RAM limits via read-only
`status --summary --json`, including reported disk-blocking warnings. Missing
optional inspection tools yield warnings.
Dataset contents still require `p`. Installed packages and detected hardware do
not certify CUDA compatibility, model fit or future queue admission. Worker
checks remain authoritative when training starts.

For remote backends, readiness checks required credential presence with values
hidden, optional backend package presence, and River's Python prerequisite.
Modal credential profiles remain unchecked. It makes no provider request: authentication,
availability, compatibility and current full-cost pricing remain unchecked.
Inspection never provisions hardware or changes jobs or queue settings.

On Runs, `/` searches IDs, models, methods, providers and statuses. All search
words must match; search ignores case. Enter applies, Esc cancels. `t` cycles
all, active, failed and succeeded; Backspace clears both filters. Search applies
to the 100 most recent retained runs, including interrupted runs in the failed
filter. Active includes prepared, starting, queued and running states.

`g` opens scrollable metric trends. Native RL progress uses committed optimizer
updates; SFT progress uses reported fractional epochs. Trainer log steps are
shown separately and are never substituted for committed updates. Freshness
shows when metrics were reported, without inferring that a quiet job has failed.
The private append-only `metrics.jsonl` preserves numeric finite callback values;
the UI reads at most its final 128 KiB and charts up to 120 recent events.
Partial/malformed records are skipped. Older runs fall back to `metrics.json`
with no invented history. Charts use actual reported samples, sample when space
is limited, and do not assume uniform time intervals. No ETA is inferred.
Training reward is not independent quality evidence.

## Compare experiments

On Runs, press `a` to pin the selected run as a reference. Select another run
and press `x` to compare it. Both shortcuts also work in run details. Press `a`
again on the reference to unpin it. The pin lasts for this UI session and stays
bound to its run ID across new runs and search/status filters; it is not saved
in the training draft.

The comparison shows changed recorded configuration fields, saved dataset
hashes, examples, framework/package identity, plan checksums and the latest
`metrics.json` values. Deltas are **candidate minus reference**; missing values
and nonfinite differences are unavailable, never zero. Up/Down or PageUp/Down
scroll the report. Esc returns; reopen with `x` to capture fresh snapshots.
An open comparison stays fixed during refresh and filtering.

These are training snapshots, which can differ in step, time, data and
objective. The comparison does not rank experiments or certify quality.
Dataset identity compares valid recorded hashes without rereading data;
unknown hashes stay unknown. Persisted statuses may lag running jobs. No
provider/process polling, dataset/log reads, job launch or record changes occur.
Each record is limited to 128 KiB; malformed, oversized, linked or nonregular
records are unavailable. Metrics are limited to 64 finite numeric fields per
run with keys at most 128 characters; omissions are labelled. Numeric booleans
are excluded. AI agents can read the same report with
`stateset-agents finetune compare REFERENCE_ID CANDIDATE_ID --workspace outputs/finetune`.

## Inspect full training and supervisor logs

On Runs or in run details, press `l` for the full log viewer. Long lines wrap;
the viewer shows the last 128 KiB and at most 2,000 lines from one source.
The header labels the source, follow/pause state, filter and tail scope.
Truncated windows are marked; their first line may be partial. Earlier output
is still retained on disk and can be inspected separately.

- `s` cycles auto, training (`run.log`) and supervisor (`supervisor.log`). Auto
  chooses nonempty training output, falling back to supervisor output. Explicit
  source selection never silently switches to another file.
- Space pauses or resumes following. Arrows/PageUp/PageDown scroll and pause;
  Home moves to the oldest displayed content. End resumes following the tail.
  A paused snapshot remains fixed when new output arrives or the pane resizes.
- `/` enters a case-insensitive literal filter and pauses output. Enter applies,
  Esc discards the edit, and Backspace edits text. Outside the editor, Backspace
  clears the filter. Filters search only the retained tail window, not the entire
  log. End follows new matching output with the filter still applied.
- Esc returns to Runs. Ctrl+C exits the interface while detached jobs continue.

The viewer stays bound to the selected run when newer runs appear. If that run
leaves the active Runs filter, the viewer closes instead of showing a different
job. Reading, switching sources and filtering do not change run files or worker
state. Logs can contain private prompts or customer text; opening the viewer is
an explicit read of that output.

The reader caps bytes even during concurrent writes, removes terminal control
characters and rejects log symlinks and nonregular files, including FIFOs. Missing
and unreadable sources are distinguished. AI agents can select the same source
using `finetune logs RUN_ID --source supervisor`; JSON reports the actual source,
byte count, truncation and availability alongside the bounded output.

## Data and training behavior

SFT takes text-only chat JSONL. Each row has `messages` (and optionally `id`),
with an optional initial system message, alternating user/assistant messages,
and a final assistant reply. Every content value must be nonempty text.
Malformed rows are rejected before launch, rather than silently discarded.

```json
{"messages":[{"role":"user","content":"Can I return it?"},{"role":"assistant","content":"Returns are accepted within 30 days."}]}
```

Native RL takes a JSON list of support scenarios:

```json
[
  {
    "id": "damaged-item",
    "context": "The customer says their item arrived damaged.",
    "policy_facts": ["Damaged items may be replaced at no charge."],
    "expected_resolution": "Offer a replacement using the stated policy.",
    "user_responses": ["Please send a replacement."],
    "must_avoid": ["Your refund has already been processed"]
  }
]
```

Policy facts are supplied to the agent as context. RL uses support reward v3,
LoRA with zero dropout, a frozen reference and the configured positive KL
coefficient. The requested budget is committed optimizer updates; early exit
with fewer updates is a failure. The studio retains the final checkpoint and
does **not** declare it independently judge-selected or production-approved.
For baseline-preserving judge selection with sealed partitions and replayable
judge evidence, use the [recorded recovery study](RUNPOD_REWARD_RECOVERY.md).

Local SFT uses the packaged SFT trainer and reports live Trainer log metrics;
native RL reports its learner metrics and committed update count. Local
execution requires CUDA and the `training` extra, and fails explicitly when
they are unavailable. A CPU plan is never labeled successful training. Local
model commits are optional; pin a full commit for reproducible loading.

## Share the studio with AI agents

AI agents can use the same workspace without terminal keystrokes. The
`stateset-agents finetune` commands return versioned JSON for config/model
discovery, readiness, immutable preparation, exact-plan launch, status and logs.
Run details show the status command using the studio's Python environment.
Repeated launch requests start no additional worker. See the
[agent fine-tuning protocol](FINETUNING_AGENTS.md) for examples and paid-launch
controls. Human and agent workflows retain the same data snapshots, queue,
failure guidance and artifact requirements.

New launch records bind the worker to the reviewed plan checksum, including
after queue wait. Older empty launch locks remain readable for compatibility;
they have no saved plan checksum and retain the earlier data/code checks.

## Understanding a failed run

Open a failed run with Enter to see **Failure and next steps**. New workers
retain a private `failure.json` with a versioned reason code and the stage that
failed. These records contain controlled codes, not exception messages, prompts
or credentials. The original private logs remain available for debugging.

The worker distinguishes unverified review inputs, unavailable CUDA/dependencies,
GPU or host memory exhaustion, storage/quota exhaustion, timeouts, missing
weights, incomplete RL optimizer updates and unsuccessful provider results.
Failures before process launch are also recorded as terminal failures. When the
training child records a cause, its supervisor preserves that cause instead of
replacing it with a generic child-exit message. If storage prevents writing any
diagnosis, the worker still exits unsuccessfully; retained logs and interrupted
status remain the fallback.

Guidance recommends checking readiness, changing context length/generations,
reviewing budgets or checking artifact paths as appropriate. It never executes
a repair or retries a job. Older, malformed, oversized or unknown diagnosis
records receive generic guidance; the UI does not guess a cause from log text
or display arbitrary action text from those records.

For failed or interrupted RunPod jobs, inspect the retained provider handle and
your provider account for active resources or charges before another paid
launch. `c` reuses the intact input snapshot and settings for a **new** reviewed
run. It does not resume a checkpoint, reconnect a crashed executor or silently
retry a provider job.

## Compute and retained runs

On this workstation, local training uses the installed Buildr queue. Elsewhere
the studio looks for `buildr` on `PATH`, then the workstation install. Set
`STATESET_BUILDR` to an explicit executable if needed. With no queue installed,
it uses a detached local worker. Queue CPU/RAM/scratch reservations are editable;
they do not reserve GPU memory. Timeouts exclude queue wait. Each run uses its
own output resource lock, and OMP/MKL worker limits match the CPU reservation.

RunPod SFT uses the existing provider's resource planner and quoted pod-lifetime
ceiling. Set `RUNPOD_API_KEY` in the launching environment; the studio never asks
for it or saves environment values. Launching is billable. The review shows
estimated versus measured resource recommendations, the ceiling and timeout.
Storage and other ancillary costs need separate review. The current remote SFT
contract cannot enforce a model revision, so an explicit remote revision is
rejected. Native RunPod RL remains outside the studio; River SFT returns a hosted
checkpoint pointer as described in the backend table.
Remote waits do not hold local Buildr CPU/RAM reservations.

Every exclusive run directory retains the reviewed plan, private dataset
snapshot, state, command, logs, latest metrics and produced artifacts. The
worker checks the package version/source identity and input hash before running,
including after queue wait. A successful run requires actual nonempty weight
files and a completion receipt. The owned worker then commits success, including
when its supervisor has disappeared. This outcome does not establish that its
process has exited; recovery checks process ownership separately. An unfinished
run with an unconfirmed supervisor or direct worker is shown as interrupted;
inspect logs and any retained provider handle before starting another paid job.
Reconnection is monitoring, not recovery of a crashed provider executor. This
studio supports reviewed cooperative local stop requests; remote cancellation
needs provider-specific cleanup, and quitting never cancels work.

An interrupted supervisor leaves detached execution unconfirmed: its worker may
still be queued or running. Such runs remain visible in the Active filter, and
`k` can review a cooperative stop for a local run. A saved stop receipt does not
confirm worker exit; the worker consumes it at a safe boundary. Agent CLI and
HTTP metadata report `terminal: null` and `execution_unconfirmed: true` for this
case. Confirmed succeeded, failed and cancelled runs remain terminal.

Each run admits one execution. Repeated supervisor or worker entry refuses
without restarting training or replacing the winning run's state and artifacts.
Uncertain ownership after a crash is retained; prepare a new reviewed run or
recover a published checkpoint instead of reusing that run directory.

Workspace runs contain customer transcripts. Keep them in protected storage.
Opening the studio, previewing and saving a draft allocate no compute. Tests
use tiny CPU fixtures and mock providers; no GPU quality gain or paid launch is
claimed by this interface's verification.

## Verification

The initial studio acceptance,
[Buildr job `bb7581c10563`](../benchmark_results/training_contracts/20261016/finetune-studio/README.md),
passed 2,442 targeted source regressions (seven skips), 1,080 isolated
installed-wheel regressions and typing for all 421 package modules. The source
and wheel checks overlap. They include real terminal startup, model search,
run-view switching and clean exit, as well as synthetic training/provider
contracts. Lint, formatting, hygiene, release governance, public API compatibility
and workflow lint passed. Package/test Python remained fixed during acceptance;
the wheel matches the source, recovery profile and rubric bytes. No paid run
or full-size GPU training was performed by this verification.

The previous [responsive studio acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-responsive/README.md),
Buildr job `04b9e08842a6`, passed 467 targeted source regressions, 356 isolated
installed-wheel regressions and typing for all 421 package modules. Both source
and wheel include all 60 studio tests, with actual terminal editing/navigation
at 74x44 and 110x32, file-picker boundaries, Unicode widths, complete dialog
scrolling, run selection and snapshot-checked settings reuse. Source/wheel
counts overlap. Source-tree allowlist governance runs against the source tree;
the corrected wheel harness and its earlier failed attempt are retained.
The live Herdr split was refreshed and its compact screen captured. This update
does not repeat the earlier broad training/distributed matrix or certify new
GPU/provider quality results. No training job was started by the update.

The [readiness and monitoring acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-monitoring/README.md)
passed 506 targeted source tests, 395 installed-wheel tests and typing for all
422 package modules in Buildr job `411c4af837e7`. All 99 studio/inspection tests
ran in both suites, including real terminal search at both pane sizes. The live
split was refreshed with its draft preserved and local readiness exercised.
Source/wheel Python, recovery profile and rubric bytes match. The earlier
[pending record](../benchmark_results/training_contracts/20261016/finetune-studio-monitoring-pending/README.md)
is historical; that queue wait has been resolved. No paid training or GPU
quality certification was performed.

The [failure guidance acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-failures/README.md)
passed 537 targeted source tests, 426 installed-wheel tests and typing for all
422 package modules in Buildr job `18ec9773998c`. It covers controlled failure
codes, preserving the original child cause, terminal failures before startup,
full-storage fallback and bounded diagnosis reads. Synthetic faults and mocked
process/provider boundaries do not certify live GPU/provider behavior.

The [shared agent CLI acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-agents/README.md)
passed 570 source tests, 459 installed-wheel tests and typing for all 423 package
modules in Buildr job `fdedd41e32ba`. Both suites include 163 focused studio,
inspection, failure and agent CLI tests. Immutable plan binding, submission
idempotence, paid opt-in, known-credential redaction, workspace boundaries and
shared TUI visibility are covered. Source/wheel counts overlap. The final live
split was refreshed with its draft intact. No paid training was launched.

The [prepared-run handoff acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-handoff/README.md)
passed 595 targeted source tests, 484 installed-wheel tests and typing for all
423 modules in Buildr job `777c09774483`. Both suites include 188 focused studio,
inspection, failure, agent CLI and handoff cases. Real terminals cover prepared
review and Escape at 74x44 and 110x32. Tests cover same-ID submission, preserved
drafts, saved resources, missing source datasets, changes during review/queue
wait, duplicate submissions and legacy lock compatibility. Package/test Python
was frozen; wheel Python, recovery profile and rubric bytes match. Source and
wheel counts overlap. The live split was refreshed with the draft hash unchanged;
no paid or GPU training was launched.

The [backend selection acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-backends/README.md)
passed 786 targeted source tests, 675 installed-wheel tests and typing for all
423 modules in Buildr job `9bb286e32d88`. Both suites cover backend selection,
cost-policy and paid-launch controls, remote routing, saved GPU reservations,
River checkpoint receipts and Modal/River executor/runtime contracts. Real
terminals select Modal at 74x44 and 110x32; the picker also scrolls at 72x22.
Frozen package/test Python and wheel Python/profile/rubric bytes match. Source
and wheel counts overlap. The live Herdr picker was inspected, then returned
to Configure with the saved draft hash unchanged. Provider calls are mocked;
no authenticated provider request or paid training was performed.

The [settings checklist acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-validation/README.md)
passed 816 targeted source tests, 705 installed-wheel tests and typing for all
423 modules in Buildr job `8d8e95da3e2e`. Both suites cover shared field
diagnostics, numeric text/type rules, hidden-field repair, report refresh,
incomplete drafts and JSON validation without dataset/provider side effects.
Real terminals open the checklist and jump to Dataset at 74x44 and 110x32;
checklist navigation and complete shortcuts are also covered at 72x22.
The wheel's real CLI reports both valid and invalid configurations, and fresh
processes verify validation imports neither torch nor transformers. Frozen
package/test Python and wheel Python/profile/rubric bytes match; counts overlap.
The live Herdr checklist was inspected, then returned to Configure with the
saved draft hash unchanged across refresh. No paid training was launched.

The [full log viewer acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-logs/README.md)
passed 831 targeted source tests, 720 installed-wheel tests and typing for all
423 modules in Buildr job `71c5cce5bd41`. Both suites cover fixed byte windows
during concurrent writes, source selection, special-file rejection, paused
snapshots, literal filtering, wrapping, run binding and agent log metadata.
Real terminals pause, filter, switch sources and return to prepared-run review
at 74x44 and 110x32 without launching training. Frozen package/test Python and
wheel Python/profile/rubric bytes match; source and wheel counts overlap.
The live split was refreshed with its private draft preserved. Its empty
workspace exposes the new shortcut; viewer behavior uses temporary fixtures.
No paid training or independent quality gain was verified.

The [experiment comparison acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-comparison/README.md)
verified 861 targeted source tests, 750 installed-wheel tests and typing for all
423 modules. It covers recorded settings and dataset identity, missing/finite
metrics, explicit step/time context, bounded records, safe files, stable pins,
fixed reports, draft preservation and agent JSON. Real terminals compare runs
at 74x44 and 110x32; scrolling and complete controls are also covered at 72x22.
Queue database errors interrupted earlier attempts. Completed source evidence
was recovered after byte checks; three queued wheel groups cover exactly the
expected cases, and job `e3db43feb339` completed the remaining acceptance checks.
Counts overlap. Package/test Python stayed frozen and wheel Python/profile/rubric
bytes match. The live split exposes the new controls with its private draft
preserved; its empty workspace uses no fabricated runs. No paid training or
quality gain was certified.

The [dataset inspector acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-dataset/README.md)
passed 887 targeted source tests, 776 installed-wheel tests and typing for all
423 modules. It covers SFT/native RL statistics, duplicate definitions, bounded
private excerpts, hidden-by-default output, background responsiveness, stale
reports, safe file reads during appends and preparation error compatibility.
Real terminals inspect statistics, reveal a fixture excerpt and Escape at 74x44
and 110x32; scrolling and controls are covered at 72x22. Three queued wheel groups
match the selected source cases exactly; final job `ac3daeb94a5f` completed real
CLI sample opt-in checks and remaining acceptance. Source/wheel counts overlap.
Package/test Python stayed frozen and wheel Python/profile/rubric bytes match.
The live split was refreshed with its private draft preserved; its empty dataset
setting exercises controlled missing-input guidance. No paid training or GPU
quality gain was verified.

The [dataset diagnostics acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-diagnostics/README.md)
passed 913 targeted source tests, 802 installed-wheel tests and typing for all
423 modules. It covers one-based physical SFT lines, syntax columns, encoding
offsets, RL example/field locations, duplicate IDs and content-free CLI errors.
Existing preparation line markers remain compatible. Background scan outcomes
replace pending status without overwriting later actions. Three wheel groups
match the selected source cases exactly; final job `e99e61c8870b` verified the
installed CLI's malformed-data exit code and structured diagnostic. Counts
overlap. Frozen source/test Python and packaged assets match the wheel. The
live Herdr split verifies the corrected scan footer with its private draft
preserved. Earlier typing and compatibility failures are retained with their
fixes; no paid training or quality gain was certified.

The [quality, setup and operator API acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-quality-api/README.md)
passed 1,025 targeted source tests, 914 installed-wheel tests and typing for all
425 package modules. It covers guided setup, artifact-bound paired test reports,
explicit local stop reviews, committed RL recovery checkpoints and authenticated
operator routes. Real terminals open the guide and quality view and Escape a
stop review without requesting cancellation. The wheel's CLI imports and
rechecks a synthetic reward-up/outcomes-down report, refuses a stale stop
checksum and stops only an unlaunched synthetic run idempotently. Wheel case
identities match the source selection; counts overlap. Earlier failures and the
isolated harness's shared-fixture repair are retained. Final job `07bc3da201b9`
passed; package/test Python stayed frozen. The live split shows the guide's
missing-dataset guidance with private draft bytes preserved. No judge execution,
checkpoint resume, remote stop, paid run or GPU quality gain was verified.

The [local checkpoint recovery acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-recovery/README.md)
passed 1,183 targeted source tests, 1,072 installed-wheel tests and typing for all
426 package modules. It verifies published local checkpoint receipts, private
snapshot preparation, fixed original training budgets and separate launch
review. A tiny CPU SFT recovery produces exactly the same final adapter tensors
as uninterrupted training; native CPU recovery restores optimizer, RNG and
cursor state and matches subsequent updates. Real terminals exercise inspection,
Escape and both reviews at 74x44 and 110x32. The installed CLI rejects stale
checksums, prepares without launching and returns recovery lineage. Final job
`2c6e30457995` passed; wheel cases match the selected frozen source identities
without skips or failures. Counts overlap. The live split exposes `z recover`
with its private draft preserved. Remote recovery, GPU recovery, judge execution
and independent quality gain remain unverified; no paid training was launched.

The [paired-evaluation acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-evaluation/README.md)
passed 1,265 source tests, 1,154 installed-wheel tests and typing for all 429
package modules in Buildr job `16e925fa033f`. Package/test Python stayed frozen;
wheel bytes and selected case identities match the source. Repository lint,
hygiene, release governance, stable API compatibility, serial formatting,
workflow lint and installed CLI checks passed. Counts overlap.

Coverage includes a real tiny CPU model/adapter, exact cached commits, replayable
two-pass judging with injected replies, critical success gates, private queue
progress and the captured-editor file picker. Installed terminals at 74x44 and
110x32 choose a case file and exit without changing the draft or preparing work.
The live split is refreshed with its private draft preserved. No paid run, live
judge model, GPU, Windows, remote evaluation or independent quality gain was
certified; a single diagnostic does not select a production checkpoint.

The [published-checkpoint acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-checkpoint-evaluation/README.md)
passed 1,572 targeted source tests, 1,461 installed-wheel tests and typing for
all 429 package modules in Buildr job `4987a3a77fc0`. Counts overlap. Frozen
package/test inventories, actual wheel bytes and selected test identities match.
Real tiny CPU tests verify native save cadence and selected safetensors loading;
real terminals choose checkpoints and review without submission at 74x44 and
110x32. Installed CLI checks metadata discovery, validation-only preparation,
controlled partition errors and immediate cancellation before launch. The live
Herdr split is refreshed with its private draft preserved. Automatic ranking,
fresh test of a nominated intermediate, live judge/GPU certification and
independent quality gain remain separate work; no paid run was launched.

[Training-seed acceptance](../benchmark_results/training_contracts/20261016/finetune-studio-seeds/README.md)
passed 1,839 targeted source tests and 1,728 wheel tests. A subsequent
[zero-value display correction](../benchmark_results/training_contracts/20261016/finetune-studio-seed-zero/README.md)
changed only the renderer and its tests. The final code passed 613 source and
613 matching wheel tests, including 63 seed cases, plus typing for 429 modules
in Buildr job `d808523574bf`. Counts overlap. Twelve cases cover valid zero
values in rows/help at 72x22, 74x44, 110x32 and 140x40. Real CPU SFT repeats
verify equal weights for the same seed and different weights for a different
seed; provider boundaries use fakes. The live split is refreshed with Training
seed visible and private draft bytes preserved. No paid run, GPU certification
or independent quality gain was established.
