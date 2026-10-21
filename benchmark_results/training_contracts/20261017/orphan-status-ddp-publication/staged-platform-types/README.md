# Staged publication platform typing adjustments

Canonical files remain unchanged while verification 7f466030cf3c owns the source
lock. No checks, tests, builds, queue jobs or provider calls were run.

The Windows state-lock implementation uses the existing project convention
sys.platform == "win32", allowing static platform pruning of msvcrt attributes.
The non-Linux ownership existence probe is inside an explicit else, preserving
runtime semantics while avoiding a statically unreachable trailing statement.

Two existing test fixtures now distinguish missing ownership from confirmed
activity. The stop-boundary fixture keeps its deliberately absent owners and
expects interrupted/unconfirmed with unknown terminal status for raw running;
prepared cancellation and queued refusal assertions remain explicit. The HTTP
active-review fixture supplies a synthetic execution receipt and bounded fake
owner confirmation for running, preserving its exact active-state assertions
without probing real PIDs or inspecting training artifacts.

The full-file copies and exact current postformatter baseline hashes are listed
in manifest.json. Existing terminal regressions cover the non-Linux probe.
Parent applies under the source lock after the current verification completes
and queues the next checks. No new test cases or broader changes are included.
