# Native DDP implementation identity

Native DDP now rejects differing package, model and learner implementations
before scoring, and rejects checkpoint restores from a different implementation
before mutating learner state. The current protocols are learner preflight v5,
implementation identity v1, and native DDP checkpoint v3.

The [summary](summary.json) binds all 439 packaged Python sources and the wheel
hash. Source validation passed 76 checks; installed-wheel validation passed 77,
including an explicit import-origin and package-source comparison. Each actual
CPU/Gloo worker completed 47 admission rejection/recovery scenarios and 99
checkpoint scenarios. The latter include 84 exact continuation cases across two
checkpoint APIs, two optimizers and seven objective configurations, plus 15
rejection/recovery cases. Both peers reject code mismatches without learner
mutation and can recover after the mismatch is corrected.

The complete CPU unit selection (`tests/unit`, excluding `performance`, `slow`
and `gpu`) passed 10,027 tests with 54 skips. Four pytest workers used
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. Follow-up regression runs passed
269 and 362 checks. The full run exposed and drove fixes for experiment metric
serialization, NumPy checkpoint diagnostics, eager Torch imports and terminal
tests that acted before asynchronous history loading completed. Documentation
example counts and the 439-module type coverage baseline were also updated.
Checkpoint validation and nonfinite-metric guards remain strict.

Source reads do not trust file timestamps: same-size rewrites with restored
mtime are covered by regressions. Model identities include submodule source
and loaded Python method bodies. Runtime identities include dependency versions
and numerical precision controls. Distribution metadata is captured once per
process; changing dependencies requires restarting workers. Source paths do
not enter the identity, allowing equivalent checkouts and unpacked wheels.

The checkpoint harness has a 180-second internal bound and a 210-second pytest
timeout to accommodate the additional source reads across 84 roundtrips.
Numerical equality and state-preservation assertions are unchanged. Superseded
timestamp-cache failures and the earlier harness timeout remain under
`development-failures/`, alongside the interrupted initial unit run and the
subsequent complete run that identified six remaining regressions.

These results use existing local dependencies and certify the stated CPU
contracts only. They do not certify GPU/NCCL execution, model quality, reviewed
datasets, live providers, multi-node recovery, a long-running soak, a clean
release installation, or an overall A+ grade. Native DDP v2 and older learner
checkpoints must be resumed with their original implementation; this format
does not silently upgrade them. Hardware/native binaries, arbitrary mutable
globals, hooks and external services are outside the identity contract.
