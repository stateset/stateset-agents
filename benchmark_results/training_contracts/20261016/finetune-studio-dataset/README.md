# Fine-tuning dataset inspector: acceptance

887 targeted source regressions and
776 installed-wheel regressions passed
(counts overlap). All 423 package modules passed typing. Repository lint,
serial formatting for changed Python, hygiene, release governance, stable API compatibility and
workflow lint passed. Preparation job `0ca969da4151` froze package/test Python
and packaged assets, ran source checks and built/installed the matching wheel.
Three queued wheel groups cover exactly the expected source-case identities,
without errors, skips, failures or duplicates. Final job `ac3daeb94a5f` combined
the reports and completed offline audit, recovery-planner and real CLI checks.
Queue IDs, argv, outcomes and timings are retained separately. The initial
preparation attempt found an unused test import; it was removed before the
final frozen acceptance. Its failure report is retained under verification-attempts.

Dataset contracts cover strict SFT and native GRPO/GSPO/DAPO data, captured
hashes/bytes, Unicode character counts, nearest-rank distributions, explicit
turn definitions, duplicate definitions, observed dominant replies, immutable
report bytes, opt-in first-three excerpts, 4096-character clipping, no text in
default JSON, known-credential redaction and fresh root CLI use without Torch
or Transformers imports. Regular-file symlinks remain supported. Shared reads
reject missing/directory/FIFO/oversized inputs, enforce bounds during appends
and preserve preparation's existing invalid-dataset error contract.

TUI checks cover background responsiveness, leaving an active scan, changed
path/method reports, controlled unexpected failures and retry, hidden samples,
sample switching, scrolling at 72x22/74x44/110x32, unchanged drafts and no
workspace/run creation. Real PTYs open the inspector, wait for statistics,
reveal a private fixture excerpt and Escape at 74x44 and 110x32 before exercising
the existing log/review/comparison flow, without launching workers.

The installed wheel's real CLI inspects a synthetic chat with and without
explicit sample opt-in. Existing backend, validation, handoff, comparison,
logging, worker, tiny CPU SFT, provider and reward/recovery contracts provide
regressions. Provider/worker boundaries are mocked or use tiny CPU fixtures;
no paid training, provider authentication or GPU quality certification occurred.
The installed recovery planner still blocks paid execution. Historical broad
distributed matrices were not repeated. Dataset statistics do not certify
quality, token fit or launch readiness; data is captured again at preparation.

The live Herdr split was refreshed: True; private draft
preserved: True. It has no retained runs or configured
dataset; only the new shortcut and controlled missing-input report are checked
live. Private sample behavior uses temporary test fixtures.
