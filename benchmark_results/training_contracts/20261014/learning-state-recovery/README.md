# Learning-state checkpoint recovery

Queue job `57a0fe3b75ff` reproduced 17 failures and six passing controls.
Global-baseline trainers restored different reward statistics and made different
next policy updates. Replay snapshots lost private sampling/reservoir and
fractional-replay streams; LwF snapshots lost the frozen task reference weights.
`before.json` retains the output and baseline wheel identity.

The fix records a versioned learning-state payload in both checkpoint APIs,
stages validation before policy mutation, and restores reward mean/count/sum,
private RNGs, replay contents, frozen references, EWC statistics and task identity.
Legacy global-baseline/LwF snapshots lacking required history are rejected.

Shared queue job `b47f5d9c8d5a` passed **8,488 tests**, with **57 optional skips**,
and typing for **385 package modules**. All **524 focused tests** passed. The
**87 new recovery cases** also passed against the rebuilt wheel, with loaded
module paths and hashes verified against that wheel. Lint, formatting, hygiene,
release governance and API compatibility checks passed.

Coverage includes both trainers and checkpoint APIs, future sampling/reservoir
and fractional replay decisions, exact next policy updates after restoring the
global baseline, frozen reference weights and trainer aliases, EWC precision,
malformed-state rejection before live mutation, legacy-state handling, and safe
`weights_only=True` loading when strategy enums are supplied. Legacy replay
snapshots retain the target loader's streams without claiming historical replay
recovery. Global-baseline/LwF snapshots missing required history fail explicitly.

The wheel also performed SFT/IQL updates, export, reload, inference and exact CPU
checkpoint recovery. Its sampled native Transformer workflow completed 160
rollout batches and 80 optimizer updates for each of seeds 42, 43 and 44, with
verified checkpoint reloads. The toy success-rate improvement is retained in
`native_workflow/summary.json`; it is a synthetic regression check.

`summary.json`, `verification.json`, `wheel_proof.json`, `learning_state_contract.json`,
`cpu_contract.json` and `native_workflow/` identify the same wheel and unchanged
package inventory. `cpu_artifact_inventory.json` records revalidated policy
bytes. `verification_runners.json` retains the helper source and hashes. The
initial 236-test result and the first full run's corrected typing failure are
retained separately.

These checks do not establish representative quality, GPU parity or production
approval. CUDA was not rerun, and no paid resources were allocated for this change.
