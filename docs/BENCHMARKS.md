# Benchmarks and evidence

This page separates measured results from smoke tests, synthetic load tests,
and planned experiments. A number is publishable only when its raw artifact,
configuration, model/data revisions, hardware, seed, and producing commit are
retained in this repository or a linked immutable CI artifact.

## Evidence classes

| Class | Meaning | May support a performance claim? |
|---|---|---:|
| Measured benchmark | Real model/training execution with retained provenance | Yes, for the exact tested configuration |
| Live proof | Real hardware path proving an operation succeeds | Only correctness/convergence claims |
| Microbenchmark | Isolated implementation or kernel timing | Only the timed component |
| Smoke test | Small or stub execution checking wiring | No |
| Synthetic load test | Generated traffic or deliberate sleeps | No |
| Planned | Protocol exists but required runs are incomplete | No |

Synthetic and stub outputs must never be presented as model quality, training
throughput, framework superiority, GPU scaling, or production capacity.

## Currently supported claims

### Multi-turn customer-support improvement

The canonical first-party result uses Qwen2.5-0.5B-Instruct and three GSPO
seeds. The retained result reports mean LLM-judge improvement of `+0.0792`,
standard deviation `0.0577`, and positive agreement across all three seeds.

- Result: [`benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen25_05b_instruct.json`](../benchmark_results/whitepaper_v1/customer_support_3seed_judge_qwen25_05b_instruct.json)
- Methodology: [`docs/WHITEPAPER.md` §11.7](WHITEPAPER.md#117-first-party-reproduction-canonical-three-seed-result)
- Scope: one small model, one task, one trainer, first-party evaluation

This establishes a reproducible positive result. It does not establish
superiority over another framework or performance at 8B/70B scale.

### RunPod GPU training proof

The v0.42.2 GPU workflow completed QLoRA SFT and 40 real CUDA GSPO steps. The
target token probability increased from `0.0000281` to `0.124616`; both pods
were terminated and the cleanup canary found no remaining resources.

- Evidence ledger: [`docs/RELEASE_EVIDENCE.md`](RELEASE_EVIDENCE.md)
- Scope: proof of execution, learning signal, artifact creation, and cleanup

This is not a throughput comparison or a broad quality benchmark.

### StateSet orchestration parity with direct TRL GRPO

A three-seed RunPod shootout on one NVIDIA A40 used the same pinned
Qwen2.5-0.5B-Instruct revision, GSM8K revision, raw-model evaluator, optimizer,
scheduler, LoRA, precision, generation configuration, and four training steps.
StateSet 0.42.3 and direct TRL 1.9.1 were effectively identical on throughput
(`0.326 ± 0.010` vs `0.327 ± 0.006` samples/s), wall time (`196.4 ± 6.1` vs
`196.0 ± 3.9` seconds), and peak VRAM (`3436.7 ± 7.1` vs `3437.0 ± 7.5` MiB).

Both tiny runs started at `0.1875` GSM8K pass@1. StateSet ended at `0.1667 ±
0.0722`; TRL ended at `0.1875 ± 0.1083`. The protocol is too short and small
to support a learning-quality claim; it establishes that StateSet's TRL-backed
GRPO orchestration preserves upstream behavior without material overhead.

- [Validated report](../benchmark_results/framework_comparison/report/comparison.md)
- [Six per-seed evidence documents](../benchmark_results/framework_comparison/evidence/)
- Harness commit: `4173eee7be7187c0583390953b8ad79b55fb954f`

### Single-node DDP weak scaling

Three matched seeds on one RunPod host with eight identical NVIDIA RTX 5080
GPUs passed the predeclared monotonic-throughput and 50%-efficiency gate:

| GPUs | Samples/s | Speedup | Weak-scaling efficiency | Peak VRAM/GPU |
|---:|---:|---:|---:|---:|
| 1 | 322,574 ± 7,286 | 1.000× | 100.0% | 589.8 MiB |
| 2 | 672,581 ± 1,309 | 2.085× | 104.3% | 589.8 MiB |
| 4 | 1,291,535 ± 12,911 | 4.004× | 100.1% | 589.8 MiB |
| 8 | 2,606,530 ± 45,377 | 8.080× | 101.0% | 589.8 MiB |

The workload holds the per-device batch constant, so this is weak scaling.
The >100% observations reflect utilization/cache effects and are not a claim
of superlinear strong scaling. The earlier fixed-global-batch matrix failed
(`28.5%`, `10.7%`, and `2.9%` efficiency) and remains retained as a negative
diagnostic.

- [Passing weak-scaling report](../benchmark_results/scaling/report/scaling.md)
- [Twelve per-seed/topology evidence documents](../benchmark_results/scaling/evidence/)
- [Failed strong-scaling diagnostic](../benchmark_results/scaling/diagnostics/f6e7478-strong/report/scaling.md)
- Harness commit: `722f7e9fdafceec48723dc4392a212418cba9f2b`

### Checkpoint fault recovery

Nine CUDA runs (three seeds for each of worker exit, controller SIGKILL, and
live TCP heartbeat interruption) resumed from atomic model/optimizer
checkpoints. Every run reached step 12 with zero data-loss steps, duplicate
updates, or remaining child processes/sockets. Maximum measured recovery time
was `2.569` seconds.

- [Validated recovery report](../benchmark_results/reliability/report.json)
- [Nine raw evidence documents](../benchmark_results/reliability/evidence/)
- Harness commit: `52df33fa00d7eac4b2973fdeb4a5446f6278a8b1`

### Rust-core microbenchmarks

Rust-versus-Python results measure isolated advantage/ratio kernels only.
Generation and model forward/backward dominate normal LLM RL wall clock, so
kernel speedups must not be described as end-to-end framework speedups.

- Artifact: [`benchmark_results/whitepaper_v1/rust_vs_python_microbenchmark.json`](../benchmark_results/whitepaper_v1/rust_vs_python_microbenchmark.json)
- Interpretation: [`docs/WHITEPAPER.md` §7](WHITEPAPER.md#7-performance-and-scaling)

## Measurements that are not yet complete

The repository does **not** currently claim:

- faster training or lower memory than TRL, verl, NeMo RL, or OpenRLHF;
- broad superiority over TRL beyond the exact parity result above;
- multi-node or strong 2/4/8-GPU scaling efficiency;
- a measured GRPO/GSPO/DAPO/VAPO/GEPO winner on a shared protocol;
- Fireworks live training or serving success; or
- a completed 8B flagship result.

Those remain evidence gates. Protocols and validators exist so results can be
added without weakening the standard.

## Run the benchmark pipelines

### Fast Phase-0 smoke

```bash
make benchmark-smoke
```

This validates data loading, seeding, parsing, result serialization, and
aggregation. It does not train a real model and produces no publishable number.

### One measured Phase-0 run

```bash
python scripts/run_phase0_benchmark.py \
  --trainer gspo \
  --task customer_support \
  --model Qwen/Qwen3.5-8B-Instruct \
  --seed 42 \
  --train \
  --vllm \
  --output benchmark_results/flagship_v1/gspo_seed42_customer_support.json
```

Run all prescribed seeds; negative and null results must be retained. See
[`benchmarks/FLAGSHIP.md`](../benchmarks/FLAGSHIP.md) for the complete protocol.

### Measured algorithm comparison

```bash
python benchmarks/algorithm_comparison.py \
  benchmark_results/algorithm_comparison/evidence \
  --output-dir benchmark_results/algorithm_comparison/report
```

The command fails closed unless at least two algorithms have three unique,
matched seeds each. Evidence uses the schema documented in
[`benchmark_results/algorithm_comparison/README.md`](../benchmark_results/algorithm_comparison/README.md).

### Measured framework comparison

```bash
python benchmarks/framework_comparison.py \
  benchmark_results/framework_comparison/evidence \
  --output-dir benchmark_results/framework_comparison/report
```

The validator rejects simulated or estimated evidence, mismatched algorithm,
model, dataset or hardware revisions, duplicate seeds, mixed framework
versions, missing artifact digests, and fewer than three seeds per framework.
See the [`framework comparison schema`](../benchmark_results/framework_comparison/SCHEMA.md).

Use [`benchmarks/shootout.py`](../benchmarks/shootout.py) and the ready-to-fill
[`shootout manifest`](../benchmarks/shootout_manifest.example.json) to execute every
framework/seed from one neutral manifest. It rotates run order, measures wall
time outside the adapters, retains failure logs, hashes artifacts, and emits
validator-ready evidence. The adapter contract is documented in the
[`shootout manifest guide`](../benchmark_results/framework_comparison/MANIFEST.md).

The StateSet Phase-0 runner and an independent direct-upstream
[`TRL adapter`](../benchmarks/adapters/trl_grpo.py) implement that contract.
The shared manifest configuration is applied by both and attested by canonical
SHA-256 digest. For an individual StateSet adapter invocation:

```bash
python scripts/run_phase0_benchmark.py \
  --trainer gspo --task customer_support \
  --model Qwen/Qwen3.5-8B-Instruct \
  --model-revision FULL_40_CHARACTER_COMMIT \
  --dataset-revision FULL_40_CHARACTER_COMMIT \
  --seed 42 --train \
  --output raw-phase0.json \
  --output-dir adapter-artifact \
  --adapter-output adapter-result.json \
  --shootout-config-json 'CANONICAL_MANIFEST_CONFIG_JSON'
```

`--adapter-output` fails unless training, baseline evaluation, immutable model
and dataset revisions, exact shared configuration, real CUDA measurements, and
a saved final artifact all succeed.

### Local component performance tests

```bash
pytest tests/performance -m benchmark -n0 --benchmark-json=benchmark-results.json
```

These guard local regressions. They do not compare model-training frameworks.

### Measured distributed scaling

```bash
python benchmarks/run_scaling_matrix.py \
  --gpu-counts 1 2 4 8 \
  --seeds 42 1337 2026 \
  --output-dir benchmark_results/scaling

python benchmarks/scaling_comparison.py \
  benchmark_results/scaling/evidence \
  --gpu-counts 1 2 4 8 \
  --output-dir benchmark_results/scaling/report
```

The default gate requires the same three seeds and workload digest at every
topology, monotonic mean throughput, and at least 50% scaling efficiency. The
generated policy workload executes real BF16 optimization and DDP gradient
synchronization as a weak-scaling test with a fixed per-device batch. It
measures the single-node training path, not strong scaling, LLM quality, or
multi-node rollout serving. See
[`benchmark_results/scaling/README.md`](../benchmark_results/scaling/README.md).

### Measured fault recovery

```bash
python benchmarks/run_reliability_matrix.py \
  --device cuda \
  --seeds 42 1337 2026 \
  --output-dir benchmark_results/reliability

python benchmarks/reliability_evidence.py \
  benchmark_results/reliability/evidence \
  --max-data-loss-steps 10 \
  --output benchmark_results/reliability/report.json
```

The gate requires worker-exit, controller-restart, and network-interruption
evidence with exact checkpoint replay, no duplicate updates, bounded lost work,
successful completion, and zero leaked resources. See the
[`reliability evidence contract`](../benchmark_results/reliability/README.md).
The network case interrupts a real local TCP control-plane heartbeat; it is not
a claim of multi-node partition tolerance.

## Publication gates

A benchmark may be promoted into the README or release notes only if:

1. all prescribed seeds are present, including failures and negative results;
2. model and dataset revisions are immutable;
3. the exact command, full configuration, commit, hardware, and software stack
   are retained;
4. baseline and trained evaluation use identical prompts and decoding;
5. the evaluation metric and judge-stability protocol are documented;
6. the raw artifact or its SHA-256 digest is retained;
7. comparisons use identical protocols and resources; and
8. the claim states its scope and does not extrapolate beyond the evidence.

## Reproducibility boundary

GPU model, interconnect, driver, CUDA version, model revision, context length,
batching, rollout engine, quantization, and topology can materially alter the
result. Reports generated by the comparison tools are intentionally
descriptive; ecosystem maturity and developer experience are not converted
into pseudo-quantitative scores.
