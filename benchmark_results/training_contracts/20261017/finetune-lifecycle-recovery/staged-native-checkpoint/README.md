# Native local checkpoint safety handoff

Status: staged only. No canonical source/tests modified, and no tests, builds,
formatters, or checks executed. Parent must apply under the repository source
lock, format, and verify through Buildr. The manifest lists baseline/staged SHA256
for the eight authorized repository files.

## Behavior

- Native single-/multi-turn local checkpoint saves finite-check materialized
  policy parameters and validate supported optimizer slots/groups, built-in
  scheduler/scaler state, counters, RNG, and pending accumulation before opening
  checkpoint publication. Existing publication/trust defaults remain in force.
- Loads preflight counters, accumulation/AMP, RNG, configuration, learning state,
  registered optimizer/scheduler/scaler records and any newly recorded component
  implementation/binding identities before the first model-loader mutation.
  Prepared CPU optimizer slots and pending gradients own their storage.
- Before a possible model mutation the learner is quarantined. Late loader,
  tokenizer, optimizer/scheduler, RNG or rollout publication failure leaves it
  quarantined; training, optimizer helpers and publication are blocked by the
  existing shared runtime guards. This is failure containment, **not rollback**.
  Policy finite checks after model/component load prevent NaN weights from
  returning successful resume.
- Only a complete successful reload, including final rollout publication and
  exit from the checkpoint read context, clears quarantine. A missing/invalid
  candidate cannot clear an earlier failure. Explicitly call `load_checkpoint`
  with a valid full checkpoint to recover a quarantined trainer.
- Publication occurs once within the local restore helper. Explicit sync
  `False` and exceptions fail the restore. `rollout_sync=False` deliberately opts
  out; missing sync and legacy sync returning `None` remain accepted without an
  acknowledgement (`last_rollout_sync=None`). Normal optimizer-step sync behavior
  is untouched. Actual DDP collective dispatch/protocol is untouched.

## Compatibility boundaries

- New published native records require all existing native learner counters,
  accumulation, RNG, configuration, learning and rollout records; optimizer and
  scheduler availability must match the destination learner.
- Older structurally complete checkpoints still recover at a completed legacy
  accumulation boundary; existing pending-gradient/AMP records remain supported.
  Legacy missing RNG emits a warning and provides **no exact stochastic recovery
  guarantee**. Legacy absent implementation identities retain structural
  admission, rather than proving old callable bindings.
- Partial metadata returns `False` without installing counters or changing a
  learner when no model artifact was loaded. `trusted=True` permits pickle
  parsing but does not make partial metadata a complete resume.
- Model/tokenizer-only imports retain imported weights and return `False`; any
  possibly mutated learner stays quarantined. Imported agent weights can be used
  for inference or passed to a **new trainer** for deliberate fresh warm start.
  Reusing stale optimizer/counters is not certified as resume.
- Supported component state is the existing shared runtime surface: exact
  standard SGD/Adam/AdamW, admitted built-in scheduler graphs/bindings and
  supported Torch scalers. Unsupported custom hooks/state and unverifiable
  quantized/opaque parameters fail explicitly for full recovery. This handoff
  does not certify new GPU/distributed/back-end combinations.

## Verification selection

New `tests/unit/test_native_checkpoint_safety.py` exercises actual single/multi
CPU learners, malformed preflight state, late commit failures/recovery, rollout
acknowledgements/opt-out, complete and pending AMP continuation, RNG continuation,
owned cached slots/gradients, model-only imports and nonfinite policy publication.

Existing tests updated: `test_resume_rollout_sync.py` now restores actual native
checkpoints; `test_checkpoint_trust.py` retains unsafe pickle rejection and
separates trusted parsing/import from full resume; incomplete mock-only resume
assertions in `test_multi_turn_trainer.py` and `test_training.py` now assert
truthful unchanged-state refusal. The multi-turn save-before-optimizer test uses
a real CPU model and artifact, preserving that positive case.

Parent should also include existing native accumulation, checkpoint
publication/trust, scheduler identity, learning/continual state, native
evaluation/lifecycle, and DDP regressions. No passing results are claimed here.
