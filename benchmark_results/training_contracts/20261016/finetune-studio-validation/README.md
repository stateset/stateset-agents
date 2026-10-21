# Fine-tuning settings checklist and agent validation: acceptance

Buildr job `8d8e95da3e2e` verified frozen package/test Python and an isolated installed
wheel. Package Python, packaged recovery profile and rubric bytes match.

- 816 targeted source regressions passed (0 skips).
- 705 installed-wheel regressions passed (0 skips).
- All 423 package modules passed typing. Lint, serial formatting, hygiene,
  release governance, stable API compatibility and workflow lint passed.
- Real terminal regressions cover editing, run search and prepared-plan review
  with Escape at 74x44 and 110x32; no training is launched by these PTY checks.
- Installed-wheel CLI schema/model commands run without a terminal or training.

Shared configuration contracts cover independent problems reported together,
strict JSON versus terminal numeric text, rejection of numeric booleans, bounded
config input, finite diagnostics, huge/nonfinite numeric values, no raw-input
reflection, no data/provider/workspace side effects, hidden-field repair,
responsive checklist navigation, rechecking stale reports and preserved drafts.
Real PTYs open the checklist, jump to Dataset and edit it at 74x44 and 110x32.
Installed-wheel CLI validation reports both valid and invalid settings without
training imports, dataset reads or preparation. Invalid reports use exit 0 and
result.valid=false; malformed/unreadable config input uses the error envelope.
The initial focused check found a 72-column footer wrap regression. That was
corrected before the final frozen source and wheel verification. An unused
test import caught by the first gate attempt was removed before final acceptance.

Backend contracts cover the four-provider picker, escape without mutation,
Compute-field cycling, explicit uncapped billing policy, exact GPU overrides,
remote executor routing, paid opt-in for Modal/River, credential presence,
remote readiness without local GPU requirements, River pointer metadata,
rejection of missing/malformed/wrong-model artifacts, local queue bypass for
remote lifecycle waits and machine-readable backend capabilities. Modal and
River executor/runtime/registry regressions are included in both suites.
Real PTYs navigate the backend picker and select Modal at both pane sizes.
No authenticated provider request, allocation or paid training was performed.

Handoff contracts cover agent preparation to human review, same-ID submission,
retained data with a removed source file, preserved Configure draft and preview,
saved CPU/RAM/scratch and RunPod resource display, checksum binding through queue
wait, duplicate submission across UI/CLI, selection changes during refresh,
changed/corrupt/symlinked records, legacy empty launch locks, missing remote
credentials, receipt-write failure and narrow scrollable reviews. CUDA and
provider allocation are mocked; paid or GPU training was not launched.

Agent CLI contracts cover versioned JSON, schema/model discovery, free readiness
and immutable preparation, shared TUI visibility, checksum/data/code mismatch
rejection, submission idempotence, explicit paid opt-in and credential presence,
bounded private logs, environment credential redaction, protocol-key stability,
whitelisted status metadata, unknown/corrupt records, run-ID traversal/symlinks,
SFT and native RL preparation, stderr isolation, same-interpreter argv and root
CLI registration without importing torch/transformers for schema discovery.

Existing readiness, metric history, failure guidance, process lifecycle, tiny
CPU SFT, remote executor, reward/judge and recovery tests provide regressions.
Deeply nested diagnosis/history JSON is rejected without crashing readers.
Fault/provider tests use synthetic exceptions and mocked boundaries; no actual
OOM, disk exhaustion or paid training was induced. Catalog maturity is not
backend certification. Training completion/reward is not independent quality.
Broader historical distributed/GPU matrices were not repeated. Source and wheel
counts overlap. The installed recovery planner still blocks paid execution.
The live Herdr pane was refreshed with its draft preserved:
True.
