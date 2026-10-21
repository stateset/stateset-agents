# Fine-tuning experiment comparison: acceptance

Acceptance combines 861 completed source
regressions from interrupted Buildr job `fe4af9f6c1c4` with
750 isolated installed-wheel regressions from
three queued groups and remaining checks from successful job `e3db43feb339` (counts
overlap). Wheel queue jobs: `3ef9fff0fa93`, `bf8ba571fca6`, `ba607693f84f`.
The merged wheel case identities exactly match the selected source cases. All 423 package modules passed typing. Repository
lint, serial formatting, hygiene, release governance, stable API compatibility
and workflow lint passed. Package/test Python stayed frozen; wheel Python,
recovery profile and rubric match source bytes.

Comparison contracts cover changed settings and missing-field presence, valid
recorded dataset identity, independent reported steps, missing metric values,
finite numeric filtering and overflow deltas, metric bounds and omissions,
malformed/oversized/linked/directory/FIFO records, no transcript/log/completion
reads, credential redaction, traversal/unknown/self-comparison errors and
read-only schema discovery. Quality gain remains false; no winner is inferred.

TUI contracts cover pin/unpin, a reference outside current filters, new runs,
captured reports during refresh, explicit reopen to refresh, unchanged drafts,
scrolling and controls at 72x22, 74x44 and 110x32. Real PTYs pin a reference,
select a new candidate, open comparison, then Escape and quit at 74x44 and
110x32, with no launch locks. The source and wheel suites include these checks
alongside the existing log, backend, validation, handoff, worker, tiny CPU SFT,
provider and reward/recovery contracts.

Initial focused job `7d02b8c54257` was interrupted by a queue SQLite lock error
before pytest could produce its summary. Its two PTY failures were reproduced
in `f12fc4124664`: curses retained a shared message prefix, so the literal-byte
assertion missed a successful pin. Assertions now wait for unique redraw text;
focused comparison/handoff retry `da9c2b64cb32` passed all 53 tests. Queue receipt
files retain these outcomes separately from the final acceptance run. The first
full attempt `a94c94b9cf6d` was also interrupted by the queue database lock after
repository gates emitted successful results, before a manifest was completed.
Job `fe4af9f6c1c4` completed gates, all typing, 861 source tests and wheel creation
before the same queue error interrupted wheel testing. Recovery retained its
completed stage evidence only after matching current package bytes to its wheel,
unit test bytes to its copied fixtures, packaged profile/rubric and regression
fixture bytes. It then reran wheel tests and remaining checks with frozen current
inventories. Job `b163151a6298` was also interrupted by the same queue error.
Wheel verification was then split into three queued groups, whose complete
JUnit reports are merged only after matching expected case identities and
checking for errors, failures, skips and duplicates. See recovered-stages.json,
frozen-inventory.json, shard reports and the archived recovery helpers. The first
shard client `44e8dc769dfd` exited on a database error and became lost; detached
retry `3ef9fff0fa93` replaces it. The remaining groups were allowed to finish.

The live Herdr split was refreshed: True; draft
preserved: True. Its workspace has no runs, so only the
new shortcuts are inspected live; comparison tests use temporary fixtures.
No paid training, authenticated provider request or GPU quality certification
was performed. Historical broad distributed matrices were not repeated. The
installed recovery planner continues to block paid execution.
