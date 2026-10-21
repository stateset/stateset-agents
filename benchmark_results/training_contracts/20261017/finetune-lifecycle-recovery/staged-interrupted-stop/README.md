# Interrupted supervision and cooperative local stop

Staged only. No canonical files changed and no tests, checks, jobs, provider calls
or Herdr actions ran. Parent owns integration, queued verification and any live
refresh. `finetune_worker.py` remains owned by the separate workflow agent.

## Fix

The supervisor launches its queue/worker in a separate session. A dead or reused
supervisor PID therefore does not establish that detached work has exited.
Previously `request_stop` treated `interrupted` as terminal and silently withheld
the receipt the surviving worker needs to stop.

Interrupted local runs now accept the same private plan-bound `stop.json` receipt
as queued/running local work. Expected plan checksum, local-backend scope and
matching package implementation remain required. Nothing signals a process,
cancels a queue job, restarts work, reassigns ownership or contacts a provider.
Known succeeded/failed/cancelled runs remain terminal no-ops for stop requests.

`run_status` adds `execution_unconfirmed`, derived from interrupted supervision.
It does not inspect execution records or pretend to establish child liveness.
The shared metadata helper `run_terminal_status` returns `None` for interrupted
or unreadable/unknown execution, `False` for known active states, and `True` for
succeeded/failed/cancelled. CLI status and HTTP metadata expose this distinction;
HTTP stop acknowledges the saved receipt without claiming worker exit.

The TUI keeps interrupted runs visible in Active as potentially surviving work
(and still visible in the failed filter). Details explain uncertainty; explicit
stop confirmation reports the receipt and unconfirmed exit. Queue admission and
worker cancellation remain the existing worker's responsibility.

## Artifacts

- Four staged full source files at their repository-relative paths.
- `tests/unit/test_finetune_interrupted_stop.py`: 25 deterministic cases.
- `interrupted-stop.patch`: source and new-test diff against baseline hashes.
- `manifest.json`: exact original/staged SHA-256 values for safe parent handoff.

Coverage includes dead/reused supervisor tokens with queued/running/persisted
interrupted states; a synthetic surviving execution record; idempotent private
stop receipts; queued-worker rejection before model loading; unchanged plan/code/
backend guards; confirmed terminal behavior; TUI review/escape/confirmation and
draft preservation; CLI terminal metadata; and authenticated in-memory HTTP review,
stop and listing. No actual PID is inspected or signaled. HTTP extras are gated
with `importorskip`; full verification should install the API dependencies.

After applying under the parent's source lock, queue formatting, typing and tests
including the new module plus existing CLI, quality/stop, studio, failures,
recovery and HTTP operator coverage. The parent should also include the new module
in the installed-wheel acceptance list and relevant governance selection.

## Limits

A stop receipt is a request consumed at a worker boundary. If no surviving worker
remains, execution stays unconfirmed rather than being invented as cancelled.
This fix does not reconcile successful orphaned workers, reconstruct provider
handles, restore a dead supervisor or guarantee acknowledgement while a worker
is stuck. Remote stops still require provider-specific cleanup. Historical
`TERMINAL_STATES` remains unchanged for recovery callers, which separately check
worker lifecycle; execution-facing metadata uses the new explicit helper.
