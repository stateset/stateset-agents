# Release security and runtime validation

The checkpoint loader now requires an actual boolean trust decision. Previously,
`trusted="false"` was truthy and selected the unsafe unpickler. Invalid values
now reject before loading. Unsafe Torch-load scanning is enabled repository-wide,
with one documented annotation at the typed trust gateway. Tests scan the
imported package in both a checkout and an installed wheel, and reject an empty
source inventory.

Queue pagination selects complete literal SQL statements and binds owner and
status values separately. Hostile quoted/SQL values, visibility, ordering,
pagination and snapshot consistency were checked on SQLite and a disposable
PostgreSQL 12 instance. The instance was stopped after verification. Nemotron's
processor, configuration and model now bind both weight and remote-code revisions.

The development lock and training dependency floor use Accelerate 1.15.0.
`requirements-training-lock.txt` resolves the training/API runtime on Python
3.12/Linux. Its audit checked **98 packages with zero known vulnerabilities**.
The development lock still contains **CVE-2026-59951 in MCP 1.29.0**, pinned by
Semgrep. The strict A+ security validator rejects that report; no exception or
threshold reduction was added. See the [upstream advisory](https://github.com/modelcontextprotocol/python-sdk/security/advisories/GHSA-84m7-p3x7-pcfv).

Make targets regenerate and compare all three dependency locks. A separate CI
job audits the training/API lock, and the wheel CI matrix includes a clean
installation using that lock. Those CI changes were parsed locally; their remote
execution has not been observed.

## Verification

- Full CPU unit selection: **10,078 passed, 20 skipped**. This excludes the
  `performance`, `slow` and `gpu` markers and uses four workers with one
  OpenMP/MKL thread each, existing local dependencies and isolated Accelerate
  1.15.0. PostgreSQL queue cases ran on the disposable instance.
- Focused source/API checks: **297 passed, one SQLite skip** for a PostgreSQL
  clock contract. Subsequent test-only checkpoint path corrections passed
  **35 source checks** and **36 installed-wheel checks**, including import origin.
- The initial wheel run retained **175 passes, two path-audit failures and one
  skip**. Its two real CPU/Gloo distributed tests passed: each peer completed
  **47 admission rejection/recovery cases**, **99 checkpoint cases** and **84
  exact continuation cases**. The two path failures were corrected and rerun;
  their original reports remain under `development-history/`.
- Lock-comparison and extras checks: **13 passed** after adding the third lock.
- Typing: **439 package modules**. Lint, API compatibility, hygiene and maturity
  governance checks passed. Wheel and source distribution metadata checks passed.
- All 439 installed package source files match the source inventory. Training
  smoke validation performed a real tiny-model gradient and adapter save.

The [summary](summary.json) contains artifact hashes, raw report counts and the
remaining acceptance failures. The original environment audit, earlier lock
audit and unconfigured source scan are retained as development history. The
training/API audit is not an audit of that original development environment.

These results do not establish an A+ grade. Reviewed train/validation/fresh-test
data, representative quality gains, live GPU/backend comparisons, multi-node
recovery, the required soak and clean-release evidence remain open. Local GPU
readiness records zero CUDA devices. No paid jobs were launched. An audit with
zero findings is a point-in-time result and does not prove absence of security
defects.
