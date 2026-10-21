# Fine-tuning failure guidance acceptance

Buildr job `18ec9773998c` verified frozen package/test Python and an isolated installed
wheel. Package Python, packaged recovery profile and rubric bytes match.

- 537 targeted source regressions passed (0 skips).
- 426 installed-wheel regressions passed (0 skips).
- All 422 package modules passed typing. Lint, serial formatting, hygiene,
  release governance, stable API compatibility and workflow lint passed.
- Existing real terminal tests cover editing and run search at 74x44/110x32.

Failure contracts cover known CUDA/dependency/memory/storage/timeout failures,
invalid review inputs before worker startup, missing artifacts and completion
receipts, preserving the child's original diagnosis/exit code, corrupt state,
inability to persist a diagnosis, bounded/whitelisted diagnosis reads and
recovery guidance in terminal run details. Arbitrary exception messages and
recorded action/title fields are not copied into structured diagnoses or used
as recovery instructions. Generic failures are not guessed from log strings.

Recovery guidance starts no work. Copying settings remains a fresh reviewed
launch, not checkpoint resume or automatic provider retry. Remote failure
views direct users to inspect active resources/charges before a new paid run.

Failure tests use synthetic exceptions and mocked process/provider boundaries;
no actual OOM, disk exhaustion or paid training was induced. Existing tiny CPU
worker, SFT, reward/judge and recovery tests provide regression coverage. These
are not GPU/provider quality or billing measurements. Broader historical
training/distributed matrices were not repeated. Source/wheel counts overlap.
The installed-wheel recovery planner still blocks paid execution.
Live Herdr refresh/capture: True.
