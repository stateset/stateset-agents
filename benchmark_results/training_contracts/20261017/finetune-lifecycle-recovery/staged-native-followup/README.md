# Consolidated native checkpoint follow-ups

Staged only, **untested**. No canonical files edited and no tests, checks, builds,
or formatters run. Parent applies the seven files from `manifest.json` under the
source lock after its current verification finishes, then formats and verifies
through Buildr. Use this manifest only: the older native/RNG/metadata staging
directories are historical and overlap these full files.

- **Learner RNG restored last:** native and shared runtime loaders now perform
  rollout-state/publication before restoring validated RNG, while quarantine is
  still active. A legitimate publication consuming Python/NumPy/Torch RNG cannot
  advance the resumed learner stream. Existing explicit sync False/exception and
  late RNG restore failures still leave quarantine.
- **Version new payloads explicitly:** saves now write `native_checkpoint_schema=1`.
  This exact integer marker requires publication1 with a valid published
  inventory, all existing native counters, accumulation/RNG/config/learning/
  rollout records, and component identities. The old `checkpoint_publication=1`
  existed before this safety format and alone continues the legacy structural
  path. Missing global-mean baseline still fails through existing learning-state
  validation; group-mean legacy reset is retained. Legacy missing RNG/identities
  provides no exact stochastic/binding proof.
- **Scheduler fault fixture repairs ownership:** its late commit failure now
  patches the scheduler class after staging (only raises for the live target).
  Monkeypatch teardown removes/restores the actual class method rather than
  leaving an inherited bound method inside instance state. Production admission
  stays strict; no workaround for opaque scheduler fields was added.
- **Metadata-only regression:** current canonical source already restores prior
  quarantine when an incomplete state has no model artifact; no source change
  was necessary. Dedicated actual single/multi CPU cases preserve populated
  optimizer slots, pending gradients, counters/RNG/tokenizer identity and prior
  healthy/quarantined status, with no rollout publication.
- **Legacy fixtures describe actual legacy:** learning-state legacy cases strip
  the new schema/identity records while retaining old publication1. Unmanifested
  accumulation fixtures strip those newer records too. Resigned current-schema
  invalid-state tests retain markers, and now explicitly cover missing learning,
  missing component identities and a bool schema version.

New modules: `tests/unit/test_checkpoint_rng_publication.py` (shared runtime and
both native trainers, RNG-consuming publication plus actual stochastic-input
next optimizer update) and `tests/unit/test_native_checkpoint_metadata.py` (four
prior-status/actual-trainer cases). Existing native safety, learning-state and
accumulation modules must remain in verification. Actual collective DDP dispatch
and normal optimizer-step publication semantics are untouched.
