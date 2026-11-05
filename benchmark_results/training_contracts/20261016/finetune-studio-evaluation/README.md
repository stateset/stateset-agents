# Local paired evaluation: acceptance

1265 targeted source tests and
1154 installed-wheel tests passed with no
skips or failures; counts overlap. All 429 package modules passed typing.
Repository Ruff/hygiene, stable API compatibility, release governance, serial
format checks for ten changed Python files and workflow lint passed.

Pipeline job `16e925fa033f` used one shared allocation for serial formatting,
preparation, three wheel groups and finish. Preparation froze package/test
Python and packaged assets, ran source regressions and built/installed a
byte-matching wheel. Wheel groups use matching pytest/root fixtures and cover
exactly the selected source case identities. Finish completed installed CLI and
offline audit checks while verifying frozen identities. Stage and queue receipts and
earlier fixture/typing/lint corrections are retained; these were code/harness
failures, not queue failures.

An earlier candidate passed its source and wheel groups and was superseded
after inspection found that a relative model directory could override a Hub
commit argument. Current acceptance resolves the exact cached snapshot before
loading and tests directory collisions, invalid repository paths and missing or
mismatched cached commits. Earlier jobs finished normally; their evidence is
retained separately and is not counted as current acceptance.

## Operator and agent behavior

Press `f` in the evaluation form to browse regular JSON case files using the
shared directory picker. Selection returns to that exact editor; Escape keeps
its values. Browsing reads no case contents and writes no training settings.
JSONL, hidden files, symlink files and special files are omitted for cases.
PageUp/Down navigates both evaluation and training file lists. Changed or busy
evaluation editors reject a late selection. Review remains a separate action.

Press `j` on Runs/details/quality, then `n` to configure held-out cases, a pinned
judge and resource/token limits. `r` captures a background review. Only Enter in
that review prepares and submits through Buildr. Existing prepared evaluations
can be reopened, reviewed and submitted using their same ID. The editor does
not alter training configuration or its private draft. Escape before submission
creates no evaluation or worker. Owned submission and stop work finishes even
after leaving its view; read-only work never steals later focus.

Plans bind base commit, adapter inventory, parent plan/completion/dataset,
installed implementation, rubric, cases and fixed workload/thresholds.
Preparation atomically publishes private case snapshots; changed inputs,
oversized serialized snapshots and failed writes expose no partial evaluation.
Launch requires an exact checksum and an exclusive receipt. Execution claims
its pipeline once. No automatic retry, provider provisioning, weight download,
model-code execution or training modification is performed.

Three sequential subprocess commands generate baseline replies, candidate
replies and judge scores. Model access is cache-only and safetensors-only with
native code. Policies use the same scripted turns, supplied facts and greedy
decoding. Prompt overflow fails instead of losing facts; token-limit responses
cannot pass. The two-pass checklist judge records requests/replies/retries;
parsing errors and excessive self-disagreement reject the diagnostic.
Success additionally requires every resolution, accuracy and policy check,
plus no repeated replies in both passes. A 0.96 aggregate score cannot hide
wrong resolution, invented facts or a redundant request.

Progress survives reopening the TUI. Bounded, allowlisted retained Buildr events
expose the job handle, sampled queue wait, admission category and disk/RAM
headroom without shared-state calls or raw commands/messages. Incomplete,
malformed, nonfinite, special-file and foreign-handle events cannot claim a new
evaluation state; the sample age stays explicit. Reopening preserves inspection
handles without granting cancellation or automatic resubmission.
Evaluation-only cooperative stop is bound
to its reviewed plan and does not stop training or unrelated jobs. Result
verification rehashes the adapter and replays judge recordings without models,
checking protocol, case hashes, transcripts, outcomes and displayed score means.
Existing multi-seed quality evidence is preserved; paired-reports.json is an
explicit import. A one-run diagnostic never certifies independent quality gain
or chooses a production checkpoint. CPU/RAM/scratch reservations do not reserve
a GPU; placement remains the operator's local configuration.

## Evidence and limits

Tests exercise a real tiny CPU HF model with a real PEFT safetensors adapter,
an actual pinned Hub-cache snapshot despite a conflicting relative directory,
cache/pin arguments, deterministic equal outputs for a zero-initialized adapter
and hard context bounds. Positive end-to-end scoring uses injected synthetic
judge replies through the real rubric, findings, two-pass parser, recorder and
replay. Queue and phase submission are checked with controlled process mocks.
These do not certify a live judge model, GPU fit, speed or improvement.

Real PTYs at 74x44 and 110x32 edit all evaluation fields, capture a review and
Escape without creating a job or changing the draft. Narrow layout is also
covered at 72x22. Installed CLI checks plan, stale prepare/launch/stop refusal,
private preparation without inference, progress/listing, refusal to verify an
unfinished evaluation and an isolated synthetic evaluation stop. Training is
unchanged and no supervisor/model/provider is launched by those CLI checks.

The live Herdr split was refreshed, its private draft preserved, and `j` gives
controlled guidance in its empty workspace. No synthetic live runs were added.
No paid training, remote evaluation, GPU certification, multi-seed study or
independent quality gain occurred. Existing free preflight planning remains
blocked from paid execution. Historical distributed matrices were not repeated.
