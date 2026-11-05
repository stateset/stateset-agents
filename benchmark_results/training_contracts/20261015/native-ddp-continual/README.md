# Native DDP continual-learning acceptance

Buildr job `179066406bdd` passed frozen source-and-wheel verification with two
real CPU/Gloo workers. [summary.json](summary.json) records runtime, counts,
wheel SHA-256 and boundaries. [verification.json](verification.json) pins every
package and test Python source file; the wheel matches the accepted source.

- 1,965 targeted source regressions passed, with 7 skips.
- Typing passed for all 409 source modules.
- 674 targeted installed-wheel regressions passed.
- Seven real distributed pytest contracts passed against source and wheel.
- The new contract contains 46 scenarios per worker: 36 population cases,
  eight rejection/recovery cases and two complete training/resume cases.
  These are not extra pytest cases; source/wheel regressions overlap.

Population cases cover LwF, EWC and combined replay/LwF/EWC with buffer
synchronization enabled/disabled, and uneven, empty-peer, all-empty, masked,
capped and privately sampled replay populations. Fisher applies in 24 cases,
and LwF teachers in 24. Real offline HF tokenization and assistant masks feed
causal cross-entropy gradients. Global Fisher matches an independent combined
population within `rtol=2e-6, atol=1e-8`; resumed parameters/optimizers match
exactly. Teachers remain plain frozen models without DDP forward collectives.

Direct consolidation preserves policy, optimizer, pending policy/entropy banks,
local buffers, module modes and global RNGs. Staged replay RNG/memory commits
only after peer validation. Rejected snapshot/estimation/configuration/task/
window changes recover on the same group. Partial memory/optimizer application
blocks further learning and saving until full native checkpoint reload.

Task boundaries flush the old pending optimizer window before capturing its
anchor. A completed flush remains committed if later estimation fails; retry
does not duplicate it. Complete five-segment training commits five updates with
one inner update or ten with two, then resumes without repeating final task
consolidation or changing task memory/RNG state. Serial schedule tests also
check exact learning rates across task windows and warmup.

Regressions cover temporary fast-tokenizer padding/truncation in checkpoint
identity, while vocabulary/template changes remain detectable; malformed task
receipts; failed teacher snapshots; and clearing stale teachers when restoring
a generation without a reference. Earlier failures and fixes are retained in
[failures/context.json](failures/context.json).

Per-worker reports are in [source-distributed](source-distributed/) and
[wheel-distributed](wheel-distributed/). Existing 294 update/report, 48 population,
40 admission, 32 startup, 98 checkpoint and 28 evaluation scenarios also pass.
Exact runner inputs are archived as `.py.txt` data beside the logs.

Every live peer must enter the same transition, using the standard continual
manager. These generated tiny FP32 CPU cases do not establish GPU/AMP,
multi-node, FSDP/DeepSpeed, multi-rank Accelerate or model-quality improvement.
General generation/callback/empty-rollout failure orchestration, episode cursors,
external state and lost-process recovery remain separate requirements.
