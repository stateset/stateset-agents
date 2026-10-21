# Empirical Fisher estimation regression

Queue job `2fcf90c97124` reproduced 25 failures and one valid-example control.
Failed examples diluted or contaminated the Fisher estimate; no-data updates
changed previous task memory. Module modes leaked on exits, low-precision
squaring overflowed, and finite means could overflow their intermediate sums.
`before.json` retains the output and preceding wheel identity.

The fix averages only accepted trajectory gradients, skips unscored targets
before forward, preserves previous statistics and anchors when none are accepted,
and restores each module's train/eval flag. It promotes FP16/BF16 estimation to
FP32, retains FP64, uses a running mean, validates consolidation before commit,
and propagates OOM/interruption errors. Accepted/skipped counts remain inspectable.

Initial verification passed 120 tests. Shared queue job `5ecfc0d23af4` then passed
564 focused tests, the full suite (8,528 passed, 57 skipped), typing for 385
modules, lint, formatting, hygiene, governance and API compatibility checks.
All 127 combined learning-state recovery and Fisher-estimation cases passed
against the immutable rebuilt wheel, including 40 new estimation cases.
Independent references include a real local Transformer with explicit causal
cross-entropy gradients and promoted-statistics checkpoint recovery.

The wheel also completed SFT, IQL updates, export, reload and local inference.
The CPU device contract verified exact offline checkpoint recovery. Three
native RL seeds each completed 160 rollout batches with accumulation of two
(80 optimizer updates), followed by verified checkpoint reload.

`verification.json` records stage commands, exit codes and unchanged hashes for
all 385 packaged Python files. `after.json` retains test output and the full
local log's SHA-256. `summary.json` links the measured results to wheel SHA-256
`dbc9e0cacc149d1cc3e3444e37a45bf1010d762f7d7c76c4a425806365c37acb`.
These are synthetic engineering checks, not representative quality evidence.
No paid compute was allocated, and CUDA was not retested for this wheel.
