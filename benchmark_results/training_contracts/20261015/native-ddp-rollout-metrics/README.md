# Native DDP global rollout reporting acceptance

Buildr job `13cf424b8977` passed source and isolated-wheel verification.
[summary.json](summary.json) records runtime and boundaries;
[verification.json](verification.json) pins every package/test Python source.
The wheel's packaged Python matches the accepted checkout.

- 2,014 targeted source regressions passed (7 skips).
- All 411 package modules passed typing.
- 741 targeted installed-wheel regressions passed.
- Ten real two-worker CPU/Gloo contracts passed against source and wheel.
- The new contract contains 21 scenarios per worker, with matching peer/source/wheel reports.
  Scenarios are not separate pytest cases, and targeted tests overlap.

Callbacks observe global newly generated raw rewards. Training logs summarize
all samples used by the learner, including replay. A complete training case
puts new rewards [-2, 5] on worker 1 and replay rewards [-10, 1, 9] on worker 0.
Both callback observers report mean 1.5 and count 2; the main worker's logger
reports training mean 0.6, count 5 and exact median 1. Reward clipping for the
objective does not change these observed raw reward summaries. Real optimizer
updates still match the unwrapped combined-population learner.

Twelve population cases cover unequal/empty workers, globally empty batches,
empty groups, singleton/tied/signed-zero rewards, float64 extrema, subnormals
and skewed local medians. Independent high-precision Decimal references check
sample-weighted reward moments and group-weighted diversity. Native counters,
optimizer, pending banks and RNG remain unchanged by direct observation.

Five rejection/recovery cases cover missing/nonfinite rewards, invalid wire
counts and divergent detail controls. Four complete loop cases exercise the
W&B logger, an older logger signature, a logger body TypeError called once,
and callbacks that mutate their input metrics. Failed logging is coherent;
complete checkpoint reload clears learner poison. Callback mutations do not
contaminate subsequent logs.

The wire carries four int64 counts and eight float64 summaries after bounded
status/protocol agreement on the model's own process group. Exact global median
selection uses at most 64 reductions of two int64 cumulative counts, without
gathering reward vectors, trajectories or text. Only local reward keys are
sorted. Even medians use two exact ratios before one float64 rounding, avoiding
overflow and subnormal loss. Detailed selection runs only for reporting steps.

Mixed v1/v2 native loops reject during startup before entering different later
phases. Existing report counts remain
passing, including 131 empty-population and 17 loop-recovery cases per worker.
CI includes the new regression files; its actionlint log is retained separately.
Earlier findings are recorded in [failures/context.json](failures/context.json).
Runner inputs are archived as `.py.txt` data.

These synthetic tiny FP32 CPU/Gloo cases do not establish GPU/NCCL, multi-node,
FSDP/DeepSpeed, multi-rank Accelerate or model-quality improvement. Every live
peer must enter the same operation. Incompatible callback collectives, external
callback/environment state, process loss, diagnostic-history recovery and
separate task-boundary/final-flush reporting remain further requirements.
