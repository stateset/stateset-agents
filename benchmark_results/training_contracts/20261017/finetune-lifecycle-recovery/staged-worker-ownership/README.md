# Staged studio worker ownership fix

Canonical source/tests remain unchanged. No formatters, checks, tests, builds,
queue jobs, provider calls or model loads were run for this staging task.

Owned repository-relative files:

- `stateset_agents/finetune_worker.py`
- `tests/unit/test_finetune_worker_ownership.py` (new)
- `tests/unit/test_finetune_quality.py` (existing child/stop fixtures)

Canonical worker baseline SHA-256:

```text
f3d20026bfd668d3ae96adea9e0863ea8dbd59b43521c91ab0e495287ee987f6
```

Canonical quality-test baseline SHA-256:

```text
8495293f6fd01c39aab6c4f927838eeb8296732917b1aa59d4db5140e20d732f
```

Both modes first reserve permanent private admission.json with O_CREAT|O_EXCL,
selecting direct or supervised execution before lifecycle mutations or work.
Separate supervisor.json/execution.json claims retain per-role ownership. Existing
regular, partial, malformed, crashed, symlinked, FIFO or directory claims refuse
without reading/replacing the claim, diagnosing the shared run, or starting work.
Permission failures also leave the unowned run unchanged. Publication failures
retain their uncertain claim and start no work; there is no automatic reclamation.

Supervision admits prepared metadata without existing owners and reserves its
mode before settings or queue preparation. Direct execution admits only prepared
metadata and must reserve direct mode. A supervised child must explicitly prove
authority; queued/starting status never grants it. Terminal or uncertain running
states without a receipt cannot start fresh. Shared admission closes mixed-mode
entry races, with further phase rechecks. The private `_supervise` loop requires
current supervisor ownership and matching admission. An account rewriting private
metadata remains outside this coordination guarantee.

The random child capability stays in private admission metadata and the child's
STATESET_FINETUNE_SUPERVISOR_TOKEN environment. Command argv/command.json/Buildr
logs carry only a nonsecret --supervised marker. The CLI consumes/removes the
environment value. Direct execution never reads ambient authority, and execute
removes ambient copies before validation/model/provider setup. Missing/invalid
child capability refuses without falling back to direct mode. Bounded admission
reads validate schema/owner and compare the capability in constant time. No
provider credential, dataset or reviewed training setting transports this value.

Existing valid preflight errors remain diagnosed after a successful claim.
No launch.lock requirement, remote cancellation, budget or queue-policy change
was added. No receipt is removed after completion/crash. Unexpected CLI dispatch
errors log a constant message and return nonzero without shared-state writes;
owned public entrypoints retain their existing controlled failure diagnoses.

The dedicated test module has 71 intended cases. It exercises actual execute
entrypoints with injected local training and Modal submit, concurrent supervisors
with a fake queue wait, duplicate/completed entry, separate role claims, malformed
or crashed owners, FIFO safety, permission/publication failures, terminal metadata
without owners, phase rechecks, private-loop ownership and review diagnostics.
It also injects direct entry before supervisor ownership, at queue inspection
and before Popen, plus supervisor entry before a direct execution owner. The
losing mode cannot mutate the winner or queue/train. Tests cover token-less
queued refusal, invalid/malformed admissions, private environment transport and
consumption, and unowned CLI exceptions.

No existing test directly calling worker._supervise was found in the current
tests/unit/test_finetune*.py search. Any additional private-loop fixtures must
provide a truthful current supervisor receipt or use public supervise(), rather
than weakening ownership. Existing stopped-run fixtures now expect an unowned
queued/running entry to leave state unchanged with its stop retained. Prepared
stop already cancels through request_stop. The real fake-supervisor fixture
forwards its child's environment capability and verifies owned cancellation.

Suggested parent verification after source-lock handoff and formatting:

```bash
/home/dom/agent-builds/buildr run --lane quick -- .venv/bin/python -m pytest -q tests/unit/test_finetune_worker_ownership.py tests/unit/test_finetune_studio.py tests/unit/test_finetune_failures.py tests/unit/test_finetune_quality.py tests/unit/test_finetune_backends.py tests/unit/test_finetune_recovery.py
```

Use the parent's combined source/wheel, lint/type and governance checks.
