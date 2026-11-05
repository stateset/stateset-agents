# River SDK cleanup: sandbox wakeup restriction resolved

The apparent native SDK cleanup hang was caused by the execution sandbox denying worker-thread writes to asyncio's local wakeup socket: `PermissionError`, errno 1. Checkpoint writes could finish while their parent stayed asleep until a timer fired. The captured `selfpipe-probe.txt` records those denied writes. The standalone evaluator closed normally. Replacing the StateSet iterator wrapper, eliding filesystem sync, changing batching windows, and adding artificial wakeup timers were diagnostic experiments, not production fixes.

The unmodified native SDK suite passed **19/19 tests in 10.424 seconds** outside the sandbox, with normal filesystem sync, scheduling, scripted transport, and asyncio debug settings. This includes cancelled validation writes, trainer cleanup after persistence failure, selected-checkpoint activity publication, holdout seals, skipped-update stopping, recovery and SDK client lifecycle. The tests block provider clients and RPCs; no paid model calls were made. `sdk-standard-result.txt` is an explicitly labeled summary of the observed tool result, not a raw output transcript.

The packaged `stateset_agents` sources are unchanged from the prior selection-hardening review. River 0.11.0 source files matched the installed wheel RECORD, and loaded checkpoint/shield bytecode matched the inspected source. No SDK or Python installation was modified. This resolves the open SDK-regression blocker recorded in [the prior review](../river-selection-hardening/README.md).

## Prevent recurrence

`scripts/check_asyncio_wakeup.py` checks worker-thread socket writes before the River CI test command. It uses a process-local socket pair without contacting an external service or loading the SDK. Under the restricted sandbox it exits 2 immediately with the local-IPC permission error; outside the sandbox it passes. Main-thread socket checks alone would miss this restriction. Its eight regression cases include worker-only denial, incomplete delivery, resource cleanup, machine-readable failures and lightweight imports. Combined with the twelve async ownership checks, **20 tests passed** (see `guard-tests.xml`). Ruff, Black, isort, script type checks, repository hygiene and whitespace checks passed.

When a restricted environment blocks local worker-thread IPC, use an execution environment that permits it. Keep provider-client/RPC guards and normal checkpoint sync enabled. The precheck reports the restriction; it does not change permissions or bypass security controls.

## Evidence limits

This is offline real-SDK integration evidence with scripted model transport. It does not verify remote model execution, live crash durability, billing, GPU backend parity or independently reviewed production quality. The six-seed model results and safety regressions remain unchanged: base 325/384, SFT 338/384, SFT+RL selection 339/384, with no accepted quality comparison. Model safety and fresh independently reviewed data remain required for A+ readiness.

Diagnostic logs and runners retain the unsuccessful restricted-sandbox probes, including experiments with modified harness settings. They are not passing test receipts. The source archive contains the new precheck, its tests, and updated CI/documentation. `review-summary.json` binds the unchanged packaged implementation. Artifact hashes cover this evidence bundle.
