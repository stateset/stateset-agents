# RL and fine-tuning operator hardening

228 focused source tests and 8 release-governance tests
passed through shared Buildr jobs `66077d002339` and `783c846d1757`. Package-wide
typing passed for 431 modules. Checks include repository Ruff, hygiene,
API/release governance, serial formatting for changed Python and workflow lint.

The fixes prevent fabricated high-level PPO runs, reject invalid reference
updates before the affected optimizer commit, preserve adaptive KL state on
rejection and require a reference for positive KL penalties. All 32 PPO safety
cases exercise a tiny local PyTorch policy, including successful updates;
the objective's existing numeric contracts remain covered. Online PPO is
explicitly unavailable. Exact sampled BPE boundaries/on-policy provenance,
rollback of prior inner updates and atomic dual-optimizer exceptions remain
outside the reference text path's guarantees.

HTTP clients can obtain a safe plan checksum and review queued/running status
before a cooperative local stop, including during unrelated weight inspection.
Tests use HTTP for the review and cover stale checksums, auth and privacy.
Both metrics files reject symlinks, FIFOs and directories without blocking;
bounded reads preserve fallback, partial-tail and concurrent-append behavior.
Studio remote jobs carry the reviewed release version; RunPod commands preserve
literal model/path/install arguments and harvest rewrites only path values.
Command tests inspect inert strings and injected transports, never a provider.
Release pinning does not attest exact remote source bytes or dependencies.

The idle Herdr split was refreshed without changing draft bytes, mode or run
history. Its now-small viewport correctly shows the resize pause. Source/test
hashes match the refreshed code; their mtimes precede completion of focused
checks. No pane was resized and no training, paid compute or provider call
was started. Phone hardware and model quality gains are not certified.

Broader source/installed-wheel/CLI acceptance remains separate, under
`1e204ba66630` (state at capture: **queued**). This receipt does not
claim that pending run passed. The initial mobile acceptance stopped on a type
error in an optional selection variable; it was renamed before the successful
431-module type check. Its failed receipt is retained rather than discarded.
