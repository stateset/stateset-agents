# Online accumulation weighting regression

Queue job `91d5410e2e30` reproduced 40 unequal-batch weight-update failures in
both native GRPO trainers, across five objectives, complete/partial accumulation
windows, and zero/nonzero entropy coefficients. Both equal-batch controls passed.
`before.json` retains the measured output and baseline wheel identity.

The independent reference computes causal-token losses and applies a real SGD
update to the combined rollout batch. It preserves each prompt group's reward
baseline. This checks gradient weighting, rather than changing the scope of
advantage estimation across independently sampled rollout batches.

The implementation accumulates policy sums and, for sequence objectives with
entropy, a separate token-weighted entropy gradient bank. It releases each
microbatch graph. Checkpoints retain pending counts, scales and auxiliary
gradients. Legacy scalar windows finish under their original estimator, then
subsequent native token windows use count weighting. Loss configuration changes
are rejected before loading checkpoint policy weights. The shared runtime API
also restores both banks and initializes AMP state for an immediate flush.

Shared queue job `a2440a64359b` passed **8,401 tests**, with **57 optional skips**,
and typing for **384 modules**. All **430 focused checks** passed, including
**110 new cases** covering both trainers, partial windows, real CPU GradScaler
recovery/overflow, malformed checkpoints, clipping order, regularization and
legacy-window transition. Lint, formatting, hygiene, release governance and API
compatibility checks passed.

The immutable rebuilt wheel performed SFT/IQL updates, export, reload, inference
and exact CPU checkpoint recovery. Its sampled native Transformer workflow ran
160 rollout batches with accumulation of two, committing 80 optimizer updates
for each of seeds 42, 43 and 44. All checkpoint reloads recovered identical
weights and evaluation outcomes. The toy workflow's mean success-rate improvement
was 0.490 (seed bootstrap interval 0.469–0.508). No comparison with the previous
workflow run was performed; the rollout-batch budget differs.

`summary.json`, `verification.json`, `wheel_proof.json`, `cpu_contract.json` and
`native_workflow/` bind these results to the same wheel and unchanged package
inventory. `cpu_artifact_inventory.json` records revalidated policy bytes;
`verification_runners.json` retains the helper source and hashes.

These are synthetic engineering checks. They do not establish representative
quality, GPU parity or production approval. CUDA was not rerun, and no paid
resources were allocated for this change.
