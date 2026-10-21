# Fine-tuning studio readiness and monitoring acceptance

Buildr job `411c4af837e7` verified frozen package/test Python and an isolated installed
wheel. Source/wheel Python, packaged recovery profile and rubric bytes match.

- 506 targeted source regressions passed (0 skips).
- 395 installed-wheel regressions passed (0 skips).
- All 422 package modules passed typing; lint, serial formatting, hygiene,
  release governance, stable API compatibility and workflow lint passed.
- Real terminals at 74x44 and 110x32 exercise editing, model selection, run
  search/application/reset, switching screens and clean exit.

Readiness contracts cover non-blocking inspection, duplicate-probe prevention,
stale drafts, sanitized failures, missing dependencies, provider key presence
without authentication or retained values, bounded read-only GPU/queue commands,
oversized queue requests and disk-blocking warnings. Metadata readiness does not
validate dataset contents, CUDA/model compatibility or provider pricing.

Metric contracts cover private append-only events, finite scalar filtering,
partial/corrupt history, bounded tail reads, legacy snapshot fallback, avoiding
stale snapshot duplication, truthful SFT epochs versus committed RL updates,
reported-sample charts, missing data, malformed counters and terminal clipping.
Run search/status filters and selected identity preservation are covered.

Existing worker, SFT, remote executor, reward/judge and recovery regressions use
tiny CPU fixtures and synthetic provider replies. These are not GPU/provider
quality or billing measurements. Broader historical training/distributed matrices
were not repeated. Source/wheel counts overlap. The installed-wheel recovery
planner still blocks paid execution. No paid run or training job was launched.
Earlier acceptance attempts caught Ruff dictionary-literal style in the new
tests and a missing TypeGuard narrowing annotation in the freshness check.
Their unmodified reports/logs are retained under verification-attempts. The
corrected source and wheel were independently verified in the final job.
Live Herdr refresh/capture: True.
