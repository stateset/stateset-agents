# Staged worker terminal reporting

Canonical files remain unchanged. No checks, formatters, tests, builds, queue
jobs, real training, provider calls or model loads were run. The full-file copies
and exact baseline/staged hashes are listed in manifest.json.

The owned execute path now records succeeded, finished_at and cleared error only
after training returns and the existing artifact/completion checks publish their
receipt. It records no observed process exit_code. A supervisor can still record
its observed child exit or override the worker outcome with a later queue/child
failure. Existing stop boundaries and owned failure handling remain unchanged.

All worker/supervisor state merges now use one stable private .state.lock file.
Its descriptor is opened no-follow/nonblocking, checked regular/private, and
locked with bounded nonblocking POSIX flock or Windows byte-lock attempts.
The 1-second deadline prevents indefinite contention, and the lock path is
never replaced or removed. Failed/cancelled outcomes refuse later running,
queued, starting or success transitions without changing their diagnosis or
observed exit_code. Plain progress merges remain available; a later supervisor
failure can still override a worker success. Refused worker transitions return
nonzero directly instead of recording a replacement failure.

run_status retains committed terminal outcomes. For unfinished runs, an existing
supervisor receipt remains authoritative for ownership inspection. Only when no
supervisor receipt exists does it inspect the execution owner (or a running
state with no execution receipt). Owner reads are regular-file/no-follow and
bounded to 4 KiB. Missing, malformed, inaccessible, reused or absent owner
identity produces interrupted with execution_unconfirmed; inspection writes
nothing and establishes no terminal execution claim. Completion files are never
read to infer success. A crash after completion publication but before the
terminal-state commit remains uncertain. Recovery now rejects interrupted or
unconfirmed parents when both ownership records are absent: missing evidence
cannot prove detached work has exited. Existing confirmed legacy failed,
succeeded and cancelled parents remain recoverable without ownership records.

The shared owner probe preserves Linux PID/start-token checks and existing
non-Linux signal-zero existence-probe semantics, including PermissionError.
Invalid/oversized/special owner records cannot turn uncertainty into a reported
terminal failure. Interrupted guidance now covers either supervisor or direct
worker uncertainty.

The new test module has 47 intended deterministic cases, using fake local
training and an injected Modal executor. It covers direct/orphaned success,
CLI/TUI terminal visibility, Linux owner loss/reuse, non-Linux probe outcomes,
missing/partial/oversized/symlink/FIFO owner records, fresh prepared runs,
completion files without committed success, terminal-state write failures and
a simulated process-death boundary, late supervisor failure, cooperative stop,
quality evidence availability without a certified quality gain, and recovery's
separate requirement that owners actually exit. Added regressions cover earlier
failure/cancellation both before running and during training, actual concurrent
descriptor contention in both outcome orders, progress merging after failure,
malformed/symlink/FIFO/public lock refusal, bounded busy-lock refusal and stable
private lock identity. Recovery regressions reuse the existing make_parent
fixture to prove ownerless running/interrupted work cannot be inspected or
resumed and creates no child, while legacy terminal paths remain available.
Tests remain staged and unexecuted.

The additional recovery.py lifecycle guard is included with its baseline hash;
no existing test files needed fixture changes during static review.
Parent should apply under the existing source lock, then queue formatting,
typing, source/wheel regression and repository gates. Suggested focused suite:

```bash
/home/dom/agent-builds/buildr run --lane quick -- .venv/bin/python -m pytest -q tests/unit/test_finetune_worker_terminal.py tests/unit/test_finetune_worker_ownership.py tests/unit/test_finetune_interrupted_stop.py tests/unit/test_finetune_quality.py tests/unit/test_finetune_recovery.py tests/unit/test_finetune_failures.py tests/unit/test_finetune_studio.py
```
