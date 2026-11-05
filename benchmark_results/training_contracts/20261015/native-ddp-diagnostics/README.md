# Native DDP diagnostic recovery acceptance

Buildr job `67ceb2473dbf` passed source and isolated installed-wheel verification.
[summary.json](summary.json) records runtime, semantics and boundaries.
[verification.json](verification.json) pins every package/test Python file.
The wheel's packaged Python matches the accepted checkout.

- 2,034 targeted source regressions passed (7 skips).
- All 411 package modules passed typing.
- 767 targeted installed-wheel regressions passed.
- Eleven real two-worker CPU/Gloo contracts passed against source and wheel.
- Eight new diagnostic scenarios per peer have matching peer/source/wheel reports.
  Scenarios are not separate pytest cases; targeted source/wheel tests overlap.

The guard retains observation counters, zero-signal streaks, patience, last-step
identity, bounded recent rewards, finite latest metrics and retained failures.
Checkpoint staging validates this history before applying weights. Healthy
reloads repair stale target failures, preserve monitor references held by
callbacks and clear transient rollout telemetry. Failed snapshots retain their
failure. Process-local clocks restart. Serial version-one learning state remains
readable with fresh guard history; native DDP checkpoint v1 fails closed.

Real Gloo cases cover a pending-window roundtrip, a global empty no-op, threshold
failure and healthy reload, a nonfinite metric on one peer, local profiling
failure, divergent finite signal decisions, three pre-update control mismatches,
and complete training-loop error cleanup. Post-update errors poison all peers
until a complete reload; the next real optimizer update matches an unwrapped
combined-population learner. Differing patience, streak or observation count
rejects coherently before learner mutation and supports corrected inputs.

Preflight v4 keeps the 194-int64 header and rejects older protocol versions
before the added post-step collectives. Native checkpoint v2 agrees on guard
controls while retaining each rank's diagnostic logs. All live peers must enter
the same operation; process loss, backend collective failure and incompatible
callback collectives are outside this protocol. Evidence is CPU/Gloo with tiny
models and synthetic rewards. GPU, multi-node and quality gains are unverified.
Task-boundary/final-flush event reporting remains open.

The isolated wheel harness uses the repository's pytest.ini, including async
test mode. The configuration and its hash are retained. The executed verifier
and configuration-preparation helper document this run. The additional
stateset-ddp-diagnostics-verify-next.py.txt reproduction driver includes that
configuration setup inline for future runs. No package or test Python changes
during verification.

Workflow lint passed through Buildr job `b911909ff941` using actionlint 1.7.12.
