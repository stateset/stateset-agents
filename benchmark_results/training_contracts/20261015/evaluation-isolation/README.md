# Quality-study isolation and evaluation integrity

Queue job `1751f228201b` reproduced 15 failures: native evaluation advanced
training RNGs, shifted later cases when case order or earlier consumption
changed, leaked module modes, and changed a subsequent dropout optimizer update.
`before.json` retains the failures and preceding verified wheel identity.

Native evaluation now restores Python, NumPy, CPU Torch and available CUDA RNGs,
mixed module modes, gradient mode and cache settings on success, error and
cancellation. Per-case seeds bind the run seed to exact case content. Reports
record the protocol and seeds. Overlapping isolated task/thread contexts reject
before reseeding; other RNG consumers must be serialized with evaluation.

Reviewed split checks ignore annotations when detecting renamed duplicates.
Validation reports are explicitly labelled and rejected by test comparisons.
Initial and scheduled validation reports feed a hashed selection receipt with
earliest-step ties. Zero or inconsistent optimizer-update counts stop candidate
test evaluation. Unit controller fixtures exercise selection, not model quality.

The shared trainer evaluator now aborts on errors by default, routes required
rewards to final executed state, rejects nonfinite scores, propagates required
reward and OOM failures, and drains concurrent work before returning failure.
Diagnostic mode reports coverage and retains failures in the success denominator;
reward/length means describe valid samples only. Its legacy reward-threshold
success rate remains a heuristic, not verified task success.

Queue job `3c1600dfacec` passed the full suite (8,615 passed, 57 skipped in
403.93 seconds), 831 focused tests, typing for 386 modules, repository gates, SFT/IQL wheel smoke
checks and exact CPU checkpoint recovery. All 217 combined evaluation, recovery
and Fisher cases passed against the immutable rebuilt wheel, including the 87
new regression cases. Checks include real local Transformer sampling, preserved
pending gradients and an unchanged dropout optimizer update. CUDA RNG handling
was simulated; no live GPU or paid compute was used for this change.

Three native RL seeds each completed 160 rollout batches with accumulation of
two (80 committed optimizer updates), followed by verified checkpoint reload.
`verification.json` binds stage commands and results to unchanged hashes of all
386 packaged Python sources. `after.json` retains test output and the full log's
SHA-256. Wheel, CPU policy-artifact and loaded-module inventories retain the
source and artifact identities used by each execution.

The wheel SHA-256 is
`9d7ed1c1042608f27428a171408a4808a217162099057594482eab5518617f9c`.
`verification_runners.json` retains exact execution helpers.
`a_plus_readiness.json` records the remaining representative quality, matched
backend, distributed soak and independent evidence requirements. These local
engineering checks do not establish an A+ or production approval.
