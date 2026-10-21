# Collective DDP checkpoint publication and learner RNG

Staged only. No canonical source/tests changed and no tests, builds, checks or
formatters executed. Parent should apply `manifest.json` under the source lock,
format, then verify through Buildr using the approved two-CPU/3-GB budget.

## Bounded source change

`stateset_agents/training/ddp_checkpoint.py` now runs the existing strict
checkpoint `_sync_restored_policy` inside the collective `rollout-sync` phase.
An agent returning explicit `False` now fails coherently on every live peer,
just as a raised publication error does. The old void optimizer-step wrapper
discarded this result, allowing the checkpoint path to clear failure flags.
Missing sync, legacy `None` and explicit `rollout_sync=False` remain compatible;
legacy `None` has no positive publication acknowledgement. Normal optimizer-step
publication behavior is unchanged.

After successful publication, a new collective `rng-restore` phase reapplies
the already validated Python/NumPy/Torch RNG from each peer's own local shard.
That phase uses status coordination without cross-peer value agreement because
rank-local RNG streams intentionally differ. Flags clear only after the phase
succeeds on every live worker. A final restore failure therefore retains DDP
quarantine and requires a full coordinated reload. This is containment, not
rollback. The existing low-level `apply_payload` RNG behavior and the v2 payload
format remain unchanged.

## New real-worker regression module

`tests/integration/test_ddp_checkpoint_publication.py` uses the existing native
checkpoint subprocess fixtures with two actual CPU/Gloo workers. Both checkpoint
APIs run seven scenarios each: acknowledged publication, legacy `None`, no sync,
disabled sync, asymmetric `False`, asymmetric exception and asymmetric final RNG
restore failure. Checkpoints start with real optimizer momentum slots. Distinct
Python/NumPy/Torch streams are seeded per rank; backend publication consumes all
three with differing rank counts. After load, saved streams and draws match,
and random rewards drive an actual next update whose policy/reference,
optimizer/scheduler, pending state, baseline, counters and RNG match uninterrupted
training exactly.

Failure scenarios assert coordinated rejection, training/save refusal on both
peers and successful healthy reload before continuing. Compatibility scenarios
assert no disabled/absent publication calls and retained legacy acknowledgement
semantics. Each API subprocess contract has a 90-second deadline, Gloo's existing
30-second collective timeout, worker cleanup and optional JSON report export.

Parent should run this module plus the existing `test_native_ddp_checkpoint.py`
and `test_ddp_checkpoint_state.py` regressions against source and installed wheel.
GPU/AMP, other collectives, remote providers and worker/process loss remain outside
this CPU acceptance scope. No passing result is claimed in this handoff.
