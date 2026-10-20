# Native prompt-group weighting regression

The previous token path averaged prompt-group losses equally. That changed the
configured trajectory/token weighting when groups had different sizes or response
lengths. It also multiplied the entropy bonus by assistant-turn counts for
sequence objectives, averaged diagnostics by group, and computed named batch
reward standard deviations separately within each group.

The valid baseline reproduction failed **20 cases** and passed **three controls**.
`before.json` retains its output; `reproduction.json` identifies the package
source and reference method. An initial integer-reward fixture error was corrected
before that baseline, and is not counted as evidence of a package defect.

The fix reduces policy terms using actual trajectory or sampled-token counts,
keeps entropy on its own global token denominator, weights KL/ratio/clipping
metrics by sampled tokens, and applies named batch advantage statistics across
all prompt groups after reward clipping. Legacy baseline callbacks are preserved.
Independent reference losses and gradients cover seven objectives, uneven group
sizes, variable assistant-turn counts, forward chunk sizes, KL and entropy.

Shared queue job `5a3ccf42636d` passed **8,291 tests**, with **57 optional skips**,
and type checking for **383 package modules**. All 137 focused tests passed,
including 28 new cases. Lint, formatting, hygiene, release governance and API
compatibility checks passed. The earlier attempt stopped on a test fixture's
Ruff style error after its focused tests and typing passed; that failure and
correction are retained in `verification-attempt1.json`.

The rebuilt wheel performed real SFT/IQL updates, export, reload, inference,
and exact CPU checkpoint recovery. Its native sampled workflow completed
80 RL optimizer updates for each of seeds 42, 43 and 44. All three policies
reloaded with identical weights and evaluation outcomes. The toy workflow's
mean success-rate improvement was 0.448 (seed bootstrap interval 0.430–0.477).
This measures the synthetic workflow regression only.

`summary.json`, `verification.json`, `wheel_proof.json`, `cpu_contract.json`,
and `native_workflow/` retain the measured results. Their package inventories
match the unchanged source verified throughout the run, and the native reports
bind loaded modules to the same wheel. `cpu_artifact_inventory.json` retains the
revalidated CPU policy inventory. `verification_runners.json` retains the helper
sources and their hashes; scratch paths in those helpers refer to this run.

These are synthetic engineering checks, not representative quality or market
performance evidence. CUDA was not rerun, and no paid compute was allocated for
this change. The earlier CUDA wheel remains identified in
`../native-offline-device/`.
