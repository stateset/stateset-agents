# Local improvement-loop verification

This directory retains CPU engineering evidence for an unreleased checkout
based on package version 0.66.1. The wheel content hash and complete package
source inventory identify the executed code; this is not the published 0.66.1
distribution's certification.

The final shared-queue verification passed **8,249 tests**, with **57 optional
skips**, and type checking for **383 package modules**. Formatting, lint, hygiene,
release governance and stable API compatibility also passed. `verification.json`
records every stage and confirms the package sources stayed unchanged. Its wheel
contained exactly those package sources.

The local transformer example performed **10 SFT optimizer updates** followed
by **2 IQL policy updates** on actual logged assistant tokens. Both baseline and
candidate were exported and reloaded for predictions. The candidate weights were
revalidated after local inference. Raw reports, split fingerprints, optimizer
metrics, wheel hash and source inventory are retained here. Policy exports remain
at `/tmp/stateset-platform-final-wheel-proof-20261014/run`.

These are tiny synthetic fixtures. They do not demonstrate representative agent
quality, independent generalization, an RL gain separate from SFT, GPU backend
parity, or production readiness. `production_approved` remains false. No RunPod
resources were allocated for this evidence.

`soak_cost_plan.json` is an estimate, not an executed benchmark or a provider
invoice. It identifies the minimum 73.5 GPU-hours for the required distributed
matrix, its cached catalog assumptions, excluded costs and remaining prerequisites.
Reviewed-case quality studies, live backend comparisons, and the multi-node
fault/soak matrix remain open.
