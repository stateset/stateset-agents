# Native DDP evaluation acceptance

Buildr job `b124f30f88ae` passed frozen source-and-wheel verification on CPU/Gloo
with two real workers. [summary.json](summary.json) records runtime versions,
counts, protocol, wheel SHA-256 and evidence boundaries.
[verification.json](verification.json) pins every package and test Python source
file. The accepted wheel matches every package source file.

- 1,917 targeted source regressions passed, with 7 skips.
- Typing passed for all 408 source modules.
- 586 targeted installed-wheel regressions passed.
- Six real distributed pytest contracts passed against source and wheel.
- The new contract contains 28 evaluation scenarios per worker, including
  18 rejection/recovery cases. Scenario counts are not extra pytest cases;
  source and wheel regressions overlap.

The tests run unequal local populations, ties/nonimprovement, one empty worker,
an empty global evaluation, local errors/cancellation, policy/settings/counter
disagreements, malformed statistics, a failed best-checkpoint write, failed
buffer restoration and different operations across workers. Buffer broadcasting
is tested both enabled and disabled. Local forwards use the unwrapped policy.

Population means and reward variance match the combined samples, with failed
attempts retained in success/error denominators. Native metrics retain an exact
integer success count privately, without changing public JSON fields. Numerical
unit cases include extreme finite rewards and counts beyond the range safely
reconstructed from ordinary dictionaries.

Both workers make the same best-checkpoint and patience decisions. Evaluation
preserves training RNGs, mixed module modes, KV-cache preferences, local buffers
and pending policy/entropy banks. A failed best save preserves the completed
generation and prior selection. Buffer-restoration failure blocks training and
evaluation until successful native reload. Repaired operations run on the same
process group, and best checkpoints resume identical next parameter/optimizer
updates.

Actual evaluation collectives use 66-element int64 status/identity headers,
four int64 counts and three float64 moments. Per-worker reports are retained in
[source-distributed](source-distributed/) and [wheel-distributed](wheel-distributed/).
The existing 294 update/report, 48 population, 40 admission, 32 checkpoint-startup
and 98 native-checkpoint scenarios also pass. Earlier failures and fixes are in
[failures/context.json](failures/context.json). Exact runner inputs are archived
as `.py.txt` data alongside the logs.

Every live native DDP participant must enter evaluation. Scenario lists are
local partitions, with a per-worker episode limit; inputs are not automatically
sharded or deduplicated. Counts describe attempts rather than unique reviewed
cases. Collective timeouts must cover the slowest partition. Serialize with
training and other process-global RNG consumers. Environment-owned RNG/state,
arbitrary callback/remote-provider collectives and process failure recovery are
outside this protocol. Complete training-loop task transitions, GPUs,
multi-node execution, FSDP/DeepSpeed, multi-rank Accelerate and model-quality
improvement remain separate evidence requirements.
