# Responsive fine-tuning studio acceptance

Buildr job `04b9e08842a6` verified frozen package/test Python and an isolated installed
wheel. Source/wheel Python, packaged recovery profile and rubric bytes match.

- 467 targeted source regressions passed (0 skips).
- 356 targeted installed-wheel regressions passed (0 skips).
- All 421 package modules passed typing. Lint, serial formatting, hygiene,
  release governance, stable API compatibility and workflow lint passed.
- Real terminals at 74x44 and 110x32 exercise cursor editing, Home/End,
  incomplete draft saving, model search, run-view switching and clean exit.

The studio contract covers compact/wide layouts and complete shortcuts,
terminal cell widths for Unicode, model pin clearing, cursor insertion/deletion,
quitting from any screen, complete dialog scrolling, stable selected run identity,
JSON/JSONL browsing with no file-content reads, directory/permission handling,
full run details, whitelisted provider fields, recorded costs and artifact paths,
reusing settings from an intact retained snapshot without starting work, rejecting
changed snapshots and structurally corrupt records, and reopening incomplete drafts.

Existing worker, SFT, remote job/executor, reward/judge and recovery contracts
provide regression coverage. Tiny learners, canned trajectories and synthetic
provider/judge replies are fixtures; these are not live GPU/provider quality or
cost measurements. Broader prior training/distributed evidence is historical and
was not repeated for these UI changes. Source/wheel regression counts overlap.
The first wheel harness accidentally included a source-tree AST/allowlist test;
its failure is retained under verification-attempts. That governance test remains
required and passing against the source tree. All studio tests still run against
both source and wheel. The live Herdr pane was refreshed and its visible frame
retained without starting a training job.
The installed-wheel planner still blocks paid execution; no paid run was launched.
