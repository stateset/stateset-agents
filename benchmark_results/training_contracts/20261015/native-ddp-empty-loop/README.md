# Native DDP empty-worker and loop-recovery acceptance

Buildr job `1ada9fbaddd8` passed source and isolated-wheel verification. The
[summary](summary.json) records runtime, counts, wheel SHA-256 and boundaries;
[verification](verification.json) pins every package/test Python source file.
The wheel's packaged Python matches the accepted checkout.

- 1,990 targeted source regressions passed, with 7 skips.
- All 410 package source modules passed typing.
- 717 targeted installed-wheel regressions passed.
- Nine actual two-worker CPU/Gloo pytest contracts passed against source and wheel.
- The new population contract contains 131 scenarios per peer; the loop contract contains 17.
  These are scenarios within pytest contracts, and source/wheel tests overlap.

The 128-case update matrix crosses eight objective settings, entropy enabled
or disabled, one or two inner updates, permanently empty or alternating workers,
and SGD or AdamW. Updates and weighted population metrics match an unwrapped
combined-population reference within `rtol=2e-5, atol=2e-6`. Empty contributors
perform no actor scoring forwards. Globally unused parameters retain absent
gradients, preserving weight decay and momentum semantics. Global empty batches
preserve pending banks, optimizer, counters, RNGs and model buffers.

Checkpoints with empty local pending banks resume exact next parameters and
optimizer slots. A real CPU GradScaler also supports direct post-resume flush
before another backward. Two overflow cases start nonfinite gradients on the
only populated worker: every peer skips the update, halves its scale, preserves
weights without poisoning the learner. Empty no-ops preserve diagnostic patience without
creating a monitor; real signal-free batches still consume patience.

Loop cases cover a permanently empty worker, a globally empty middle episode,
initial checkpoints that must resume episode zero, and completed resumes that
must not repeat updates. An interrupted task transition saves A, rejects B's
teacher snapshot, then reloads A and executes B. Final policy, optimizer and
teacher match an uninterrupted two-task run exactly.

Fifteen failure cases cover local preparation, scheduler, training-start
callback, rollout exception/cancellation, actor forward, backward, optimizer,
episode callback, training log, evaluation log/callback, divergent stopping,
training-end callback and tracking cleanup. Live peers reject coherently;
failed native loops block learning/saving until complete checkpoint reload.
Mode/wrapper restoration and tracking cleanup are checked. Batch reporting
detects committed update boundary crossings, including multiple inner updates.

Existing 294 update, 48 advantage, 40 admission, 32 startup, 98 checkpoint,
28 evaluation and 46 task-memory scenarios also remain passing. Peer reports
are retained under [source-distributed](source-distributed/) and
[wheel-distributed](wheel-distributed/). Earlier failures and fixes are in
[failures/context.json](failures/context.json). Runner inputs are `.py.txt`
data; the CI workflow now includes these regressions and its actionlint log
is retained separately.

These generated tiny FP32 CPU/Gloo cases do not certify GPU/NCCL, multi-node,
FSDP/DeepSpeed, multi-rank Accelerate or model-quality improvement. Every live
peer must enter the same operation. Empty/positive legacy sequence fallback
rejects before scoring; this guarantee targets exact token training. Lost
processes, failures inside backend collectives, callbacks with incompatible
collectives and external environment state remain outside this evidence.
Checkpointed/coordinated diagnostic history and event reporting for separate
task-boundary/final flushes remain lifecycle work. Episode callback rewards and
training-log rollout summaries still use worker-local groups; population-weighted
event rewards remain separate from the accepted global token-loss metrics.
