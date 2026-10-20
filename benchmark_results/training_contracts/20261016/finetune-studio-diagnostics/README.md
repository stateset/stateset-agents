# Fine-tuning dataset diagnostics: acceptance

913 targeted source regressions and
802 installed-wheel regressions passed
(counts overlap). All 423 package modules passed typing. Repository lint,
serial formatting for changed Python, hygiene, release governance, stable API compatibility and
workflow lint passed. Preparation job `ff3756feb032` froze package/test Python
and packaged assets, ran source checks and built/installed the matching wheel.
Three queued wheel groups cover exactly the expected source-case identities,
without errors, skips, failures or duplicates. Final job `e99e61c8870b` combined
the reports and completed offline audit, recovery-planner and real CLI checks.
Queue IDs, argv, outcomes and timings are retained separately. Source and wheel checks use the same frozen dataset diagnostics and parser regressions.
The initial preparation failed on a missing CLI envelope type annotation;
that was corrected before the final frozen acceptance. A subsequent regression
required preserving the existing `:line:` syntax-error marker; the production
parser was fixed without weakening that test. Both reports are retained under
verification-attempts.

Diagnostics cover SFT physical line/JSON column/encoding offsets, RL one-based
example/field locations and duplicate IDs, safe structured CLI failures, shared
strict parsers, controlled background completion and unrelated-message preservation.
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
explicit sample opt-in, and reports a malformed second physical line with
exit code 1 and a content-free structured diagnostic. Existing backend, validation, handoff, comparison,
logging, worker, tiny CPU SFT, provider and reward/recovery contracts provide
regressions. Provider/worker boundaries are mocked or use tiny CPU fixtures;
no paid training, provider authentication or GPU quality certification occurred.
The installed recovery planner still blocks paid execution. Historical broad
distributed matrices were not repeated. Dataset statistics do not certify
quality, token fit or launch readiness; data is captured again at preparation.

The live Herdr split was refreshed: True; private draft
preserved: True. It has no retained runs or configured
dataset; the dataset shortcut, controlled missing-input report and corrected
completed-scan footer are checked live. Private sample behavior uses temporary test fixtures.
