# Fine-tuning quality, setup and operator API: acceptance

1025 targeted source regressions and
914 installed-wheel regressions passed
(counts overlap). All 425 package modules passed typing. Repository Ruff,
serial Black/isort checks for 13 changed Python files, hygiene, release governance,
stable API compatibility and workflow lint passed.

Preparation job `74b783ca242d` froze package/test Python and three assets,
ran source contracts and built/installed the matching wheel. Three queued wheel
groups cover exactly the selected source-case identities, without skips, errors,
failures or duplicates. Final job `07bc3da201b9` combined reports, checked frozen
identities again, and ran real installed CLI checks, the three-pair historical
reward audit and free recovery planner. Sanitized queue receipts retain commands,
IDs, timings and outcomes. Earlier Ruff, type-check, metadata-expectation and type-floor failures are retained
under verification-attempts. The first isolated wheel harness omitted the shared
root fixtures, causing an existing API test to attempt loading its placeholder
model; offline mode rejected that call. Earlier wheel results and sanitized
receipts are retained under wheel-attempts. The harness was corrected to copy
the exact frozen root fixtures, including API development mode, module restoration
and cost-ledger isolation, then all wheel groups were rerun. Source Python and
package bytes did not change. The fixture hashes, repair record and originally
executed preparation helper are retained. These were code/fixture errors, not
queue failures.

## Behavior covered

- Guided setup (`w`) checks settings, dataset, runtime, review and evaluation
  in the background. Stale results require refreshing; inspection cannot launch
  a worker. Real PTYs exercise the guide at 74x44 and 110x32.
- Quality (`e`, then `f` to import) rehashes completed local weights and binds
  one candidate to the reviewed model, plan, completion and artifact inventory.
  Paired test reports expose successes, regressions, recorded rewards, declared
  study gates and significance information. Raw cases are not displayed; failure
  handles are hashes. Private receipts reject malformed input and changed
  artifacts/bindings. The synthetic regression fixture has higher recorded reward
  yet 32 regressions and a -1 mean success gain, and remains a failed study.
  Tests cover one/six seeds, malformed reports, file limits, symlinks/FIFOs,
  no overwrite on invalid import, concurrent UI work and narrow rendering.
- Local stop (`k`) reviews an exact run and plan checksum. Prepared runs become
  cancelled, queued workers check before loading models, SFT stops at a callback
  boundary and native RL saves a recovery checkpoint at a committed optimizer
  boundary. Stop requests are idempotent; remote requests, stale code/reviews and
  malformed receipts are refused. Real PTYs Escape the stop review without
  creating a request. The installed CLI stops only an unlaunched synthetic run;
  no queue jobs or real training processes are cancelled.
- The `/finetune` HTTP router uses strict authentication, admin authorization and
  an explicitly configured server workspace. Capabilities/config validation,
  bounded run metadata, retained setup inspection, paired-evidence upload/recheck
  and local stop have controlled structured failures. Tests cover request limits,
  finite JSON, path restrictions, known-secret redaction, OpenAPI bodies and
  cancellation ownership. Metadata and stop stay responsive while an inspection
  owns the single artifact-hashing slot. No HTTP launch or arbitrary server path
  submission is offered.

The real installed CLI also validates drafts, exposes the shared configuration
schema, inspects a synthetic dataset with opt-in samples, returns a safe malformed
second-line diagnostic, and produces wide/narrow studio snapshots. Existing
backend, comparison, logging, dataset, worker, tiny CPU SFT, provider, reward and
recovery contracts provide regression coverage. Provider/worker boundaries use
CPU fixtures and mocks. Source and wheel counts overlap.

## Live pane and limits

The live Herdr split was refreshed: True; its private
draft was preserved: True. It has no retained runs or
configured dataset. The new guide shows the real blocked/unchecked setup; quality
and stop panels use temporary test runs in automated terminal acceptance.

This verifies software behavior, not an A+ market ranking or a model-quality
improvement. Reviewed provenance, judge identity, other seeds' artifacts and study
independence are recorded claims requiring external review. The TUI does not
run a judge, select the best checkpoint or resume training. Remote stop and
provider-hosted artifact evaluation remain unsupported. No paid run, live provider
authentication or GPU certification occurred. The free recovery planner still
blocks paid execution pending the missing study evidence. Broad historical
distributed matrices were not repeated.
